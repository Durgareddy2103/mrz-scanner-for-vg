import os
import re
import numpy as np
import base64
import time
import asyncio
import logging
import datetime

from fastapi import FastAPI, APIRouter, Depends, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from typing import Annotated, Any
import secrets
from logging.handlers import TimedRotatingFileHandler
from concurrent.futures import ProcessPoolExecutor
import cv2
import onnxruntime as ort
from pydantic import BaseModel
from contextlib import asynccontextmanager

# ---------------------------------------------------------------
# MongoDB Log Handler
# ---------------------------------------------------------------

try:
    from pymongo import MongoClient
    from pymongo.errors import PyMongoError
    PYMONGO_AVAILABLE = True
except ImportError:
    PYMONGO_AVAILABLE = False


MONGO_CONNECTION_STRING = os.getenv("MONGO_CONNECTION_STRING")
if not MONGO_CONNECTION_STRING:
    raise RuntimeError("MONGO_CONNECTION_STRING environment variable is not set")
MONGO_DB_NAME         = os.getenv("MONGO_DB_NAME",         "mrz_db")
MONGO_COLLECTION_NAME = os.getenv("MONGO_COLLECTION_NAME", "mrz_scalable_logs")


class MongoDBLogHandler(logging.Handler):

    def __init__(self, connection_string, db_name, collection_name):
        super().__init__()
        self._collection = None
        if not PYMONGO_AVAILABLE:
            print("[MongoDBLogHandler] pymongo not installed Mongo logging disabled.")
            return
        try:
            client           = MongoClient(connection_string, serverSelectionTimeoutMS=5000)
            db               = client[db_name]
            self._collection = db[collection_name]
            # Lightweight connectivity check
            client.admin.command("ping")
            print(f"[MongoDBLogHandler] Connected ? {db_name}.{collection_name}")
        except Exception as exc:
            print(f"[MongoDBLogHandler] Connection failed: {exc} Mongo logging disabled.")
            self._collection = None

    def emit(self, record: logging.LogRecord):
        if self._collection is None:
            return
        try:
            log_entry = {
                "timestamp":  datetime.datetime.fromtimestamp(record.created, tz=datetime.timezone.utc),
                "level":      record.levelname,
                "message":    self.format(record),
                "logger":     record.name,
                "module":     record.module,
                "funcName":   record.funcName,
                "lineno":     record.lineno,
                "process":    record.process,
                "thread":     record.thread,
            }
            # Extract reference_id from message if present
            msg = record.getMessage()
            if msg.startswith("[REF-ID:"):
                end = msg.find("]")
                if end != -1:
                    log_entry["reference_id"] = msg[9:end].strip()

            self._collection.insert_one(log_entry)
        except Exception:
            # Never let a Mongo write failure propagate into the app
            self.handleError(record)


# ---------------------------------------------------------------
# Logging  — file (7-day rotation) + MongoDB, same logic
# ---------------------------------------------------------------

LOG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "mrz_logs"))
os.makedirs(LOG_DIR, exist_ok=True)
log_file = os.path.join(LOG_DIR, "mrz_service.log")
LOG_FORMAT = "%(asctime)s | %(levelname)s | %(message)s"


# --- Handler 1: existing 7-day rotating file (unchanged) ---
file_handler = TimedRotatingFileHandler(
    log_file, when="midnight", interval=1, backupCount=7
)
file_handler.setFormatter(logging.Formatter(LOG_FORMAT))

# --- Handler 2: MongoDB (same log records, pushed in parallel) ---
mongo_handler = MongoDBLogHandler(
    MONGO_CONNECTION_STRING, MONGO_DB_NAME, MONGO_COLLECTION_NAME
)
mongo_handler.setFormatter(logging.Formatter(LOG_FORMAT))

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)   # existing folder rotation
logger.addHandler(mongo_handler)  # new Mongo push


def log_prefix(ref_id):
    return f"[REF-ID: {ref_id}] " if ref_id else ""


# ---------------------------------------------------------------
# Field Name Constants
# ---------------------------------------------------------------

FIELD_STATUS               = "Status"
FIELD_DOCUMENT_NAME        = "Document Name"
FIELD_DOCUMENT_CLASS_CODE  = "Document Class Code"
FIELD_DOCUMENT_NUMBER      = "Document Number"
FIELD_IDENTITY_CARD_NUMBER = "Identity Card Number"
FIELD_FULL_NAME            = "Full Name"
FIELD_GIVEN_NAMES          = "Given Names"
FIELD_SURNAME              = "Surname"
FIELD_DATE_OF_BIRTH        = "Date of Birth"
FIELD_DATE_OF_EXPIRY       = "Date of Expiry"
FIELD_DATE_OF_ISSUE        = "Date of Issue"
FIELD_ISSUING_STATE_CODE   = "Issuing State Code"
FIELD_ISSUING_STATE_NAME   = "Issuing State Name"
FIELD_NATIONALITY          = "Nationality"
FIELD_SEX                  = "Sex"
FIELD_MRZ                  = "MRZ"
FIELD_IMAGES               = "Images"
FIELD_PORTRAIT_POSITION    = "Portrait Position"
FIELD_POSITION             = "Position"

MSG_EMPTY_BASE64   = "Empty base64 string"
MSG_INVALID_BASE64 = "Invalid base64 image"

# Scan status sentinels returned by _do_scan (distinct from mrz_type)
SCAN_STATUS_QUEUE_FULL = "QUEUE_FULL"
SCAN_STATUS_TIMEOUT    = "TIMEOUT"


# ---------------------------------------------------------------
# ONNX Pipeline Config
# ---------------------------------------------------------------

DETECTION_MODEL   = os.getenv("DETECTION_MODEL",   "mrz_detection_20250222_fp32.onnx")
RECOGNITION_MODEL = os.getenv("RECOGNITION_MODEL", "mrz_recognition_20250221_fp32.onnx")

DET_SIZE    = 256
REC_H, REC_W = 64, 640
_PROVIDERS  = ["CPUExecutionProvider"]

_SEP        = "<SEP>"
_VOCAB_KEYS = ["<PAD>", "<EOS>", _SEP] + list("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<")
VOCAB       = dict(enumerate(_VOCAB_KEYS))
VOCAB_SIZE  = len(_VOCAB_KEYS)
PAD_IDX     = 0
EOS_IDX     = 1


# ---------------------------------------------------------------
# ONNX Session Options
# ---------------------------------------------------------------

def _make_session_opts():
    intra_threads = int(os.getenv("MRZ_INTRA_THREADS", "1"))
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.execution_mode           = ort.ExecutionMode.ORT_SEQUENTIAL
    opts.intra_op_num_threads     = intra_threads
    opts.inter_op_num_threads     = 1
    opts.enable_mem_pattern       = False
    return opts


# ---------------------------------------------------------------
# ONNX Inference Helpers
# ---------------------------------------------------------------

def _ensure_landscape(img):
    h, w = img.shape[:2]
    if h > w:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img


def _square_pad(img):
    h, w = img.shape[:2]
    if h < w:
        pad    = (w - h) // 2
        bottom = (w - h) - pad
        padded = cv2.copyMakeBorder(img, pad, bottom, 0, 0, cv2.BORDER_CONSTANT)
        return padded, (0, pad)
    pad    = (h - w) // 2
    right  = (h - w) - pad
    padded = cv2.copyMakeBorder(img, 0, 0, pad, right, cv2.BORDER_CONSTANT)
    return padded, (pad, 0)


def _preprocess_detection(img):
    padded, shift = _square_pad(img)
    side    = padded.shape[0]
    resized = cv2.resize(padded, (DET_SIZE, DET_SIZE))
    blob    = (resized.astype(np.float32) / 255.0).transpose(2, 0, 1)[np.newaxis]
    return blob, side, shift


def _decode_detection_mask(mask, padded_side, shift):
    hmap = cv2.resize(np.uint8(mask * 255), (padded_side, padded_side))
    _, bin_mask = cv2.threshold(hmap, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    box = cv2.boxPoints(cv2.minAreaRect(cnt)).astype(np.float32)
    box[:, 0] -= shift[0]
    box[:, 1] -= shift[1]
    return box


def _warp_quadrangle(img, polygon):
    pts  = polygon.reshape(4, 2).astype(np.float32)
    s    = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)
    src  = np.array([
        pts[np.argmin(s)],
        pts[np.argmin(diff)],
        pts[np.argmax(s)],
        pts[np.argmax(diff)],
    ], dtype=np.float32)
    dst = np.array([[0, 0], [REC_W, 0], [REC_W, REC_H], [0, REC_H]], dtype=np.float32)
    M   = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, M, (REC_W, REC_H))


def _preprocess_recognition(crop):
    resized = cv2.resize(crop, (REC_W, REC_H))
    return (resized.astype(np.float32) / 255.0).transpose(2, 0, 1)[np.newaxis]


def _decode_recognition(logits):
    if logits.shape[1] != VOCAB_SIZE:
        logits = logits.T
    indices = np.argmax(logits, axis=-1)
    eos     = np.nonzero(indices == EOS_IDX)[0]
    if eos.size:
        indices = indices[:eos[0]]
    chars = [VOCAB[int(i)] for i in indices if int(i) != PAD_IDX]
    lines = "".join(chars).split(_SEP)
    return [ln for ln in lines if ln]


# ---------------------------------------------------------------
# ONNX Pipeline Class
# ---------------------------------------------------------------

class MRZPipeline:
    def __init__(self, det_path, rec_path):
        opts         = _make_session_opts()
        self.det     = ort.InferenceSession(det_path, sess_options=opts, providers=_PROVIDERS)
        self.rec     = ort.InferenceSession(rec_path, sess_options=opts, providers=_PROVIDERS)
        self.det_in  = self.det.get_inputs()[0].name
        self.rec_in  = self.rec.get_inputs()[0].name

    def run_inference(self, img):
        t0                = time.perf_counter()
        img               = _ensure_landscape(img)
        t1                = time.perf_counter()
        blob, side, shift = _preprocess_detection(img)
        t2                = time.perf_counter()
        mask              = self.det.run(None, {self.det_in: blob})[0][0]
        t3                = time.perf_counter()
        polygon           = _decode_detection_mask(mask, side, shift)
        t4                = time.perf_counter()

        if polygon is None:
            return None, None

        crop     = _warp_quadrangle(img, polygon)
        t5       = time.perf_counter()
        rec_blob = _preprocess_recognition(crop)
        t6       = time.perf_counter()
        logits   = self.rec.run(None, {self.rec_in: rec_blob})[0][0]
        t7       = time.perf_counter()
        lines    = _decode_recognition(logits)
        t8       = time.perf_counter()

        timing = {
            "landscape_ms":      int((t1 - t0) * 1000),
            "det_preprocess_ms": int((t2 - t1) * 1000),
            "det_inference_ms":  int((t3 - t2) * 1000),
            "det_decode_ms":     int((t4 - t3) * 1000),
            "warp_ms":           int((t5 - t4) * 1000),
            "rec_preprocess_ms": int((t6 - t5) * 1000),
            "rec_inference_ms":  int((t7 - t6) * 1000),
            "rec_decode_ms":     int((t8 - t7) * 1000),
            "total_ms":          int((t8 - t0) * 1000),
        }
        logging.info(
            f"[ONNX TIMING] "
            f"landscape={timing['landscape_ms']}ms | "
            f"det_preprocess={timing['det_preprocess_ms']}ms | "
            f"det_inference={timing['det_inference_ms']}ms | "
            f"det_decode={timing['det_decode_ms']}ms | "
            f"warp={timing['warp_ms']}ms | "
            f"rec_preprocess={timing['rec_preprocess_ms']}ms | "
            f"rec_inference={timing['rec_inference_ms']}ms | "
            f"rec_decode={timing['rec_decode_ms']}ms | "
            f"total={timing['total_ms']}ms"
        )
        return lines, timing


# ---------------------------------------------------------------
# Worker pool — one MRZPipeline per subprocess
# ---------------------------------------------------------------

_scanner = None


def _init_worker():
    """
    Called once per subprocess in the ProcessPoolExecutor.
    Each worker gets its OWN logger setup:
      - file handler  ? same rotating log file (7-day rotation)
      - mongo handler ? mrz_db.mrz_scalable_logs
    Without this, subprocess loggers have no handlers and logs never reach Mongo.
    """
    global _scanner

    # --- re-attach file handler in subprocess ---
    _worker_log_dir  = os.path.abspath(os.path.join(os.path.dirname(__file__), "mrz_logs"))
    os.makedirs(_worker_log_dir, exist_ok=True)
    _worker_log_file = os.path.join(_worker_log_dir, "mrz_service.log")

    _fmt = logging.Formatter(LOG_FORMAT)

    _fh = TimedRotatingFileHandler(_worker_log_file, when="midnight", interval=1, backupCount=7)
    _fh.setFormatter(_fmt)

    # --- re-attach mongo handler in subprocess ---
    _mh = MongoDBLogHandler(MONGO_CONNECTION_STRING, MONGO_DB_NAME, MONGO_COLLECTION_NAME)
    _mh.setFormatter(_fmt)

    _root = logging.getLogger()
    _root.setLevel(logging.INFO)
    # Clear any handlers inherited from fork (avoids duplicate logs)
    _root.handlers.clear()
    _root.addHandler(_fh)
    _root.addHandler(_mh)

    try:
        _scanner = MRZPipeline(DETECTION_MODEL, RECOGNITION_MODEL)
        intra    = int(os.getenv("MRZ_INTRA_THREADS", "1"))
        logging.info(f"[PID {os.getpid()}] MRZPipeline loaded. intra_threads={intra}")
    except Exception as e:
        _scanner = None
        logging.error(f"[PID {os.getpid()}] MRZPipeline load FAILED: {e}")


def _parse_by_type(mrz_type, lines, reference_id):
    """Route raw MRZ lines to the correct parser."""
    if mrz_type == "TD1":
        return parse_td1(lines)
    if mrz_type in ("TD2", "MRVB"):
        return parse_td2(lines)
    if mrz_type in ("TD3", "MRVA"):
        return parse_td3(lines)
    logging.error(log_prefix(reference_id) + "Unknown MRZ Type detected.")
    return {"error": "Unknown or unsupported MRZ type"}


def _decode_image_bytes(image_bytes, reference_id):
    """Returns (img, error_dict). On failure, error_dict is a ready-made result."""
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None or img.size == 0 or len(img.shape) != 3:
            raise ValueError("Invalid image")
        return img, None
    except Exception as e:
        logging.error(log_prefix(reference_id) + f"Image decode error: {e}")
        return None, {
            "success": False, "message": "Invalid or unsupported image",
            "result": None, "mrz_type": None, "timing": None,
        }


def _run_inference_safe(scanner, img, reference_id):
    """Returns (lines, timing, error_dict). On failure, error_dict is a ready-made result."""
    try:
        lines, timing = scanner.run_inference(img)
        return lines, timing, None
    except Exception as e:
        logging.error(log_prefix(reference_id) + f"Scanner crash: {e}")
        return None, None, {
            "success": False, "message": "Scanner error",
            "result": None, "mrz_type": None, "timing": None,
        }


def _run_scan_in_worker(image_bytes, reference_id):
    global _scanner
    if _scanner is None:
        return {"success": False, "message": "Scanner not initialized",
                "result": None, "mrz_type": None, "timing": None}

    img, err = _decode_image_bytes(image_bytes, reference_id)
    if err:
        return err

    lines, timing, err = _run_inference_safe(_scanner, img, reference_id)
    if err:
        return err

    if not lines:
        logging.error(log_prefix(reference_id) + "No MRZ lines detected.")
        return {"success": False, "message": "No MRZ detected in image",
                "result": None, "mrz_type": None, "timing": timing}

    logging.info(log_prefix(reference_id) + f"Raw MRZ lines: {lines}")
    mrz_type = detect_mrz_type(lines, reference_id)
    parsed   = _parse_by_type(mrz_type, lines, reference_id)
    final    = normalize_output(mrz_type, parsed)
    final["mrz_type"] = mrz_type
    final["timing"]   = timing
    return final


# ---------------------------------------------------------------
# Auth
# ---------------------------------------------------------------

security = HTTPBasic()


def authenticate(credentials: Annotated[HTTPBasicCredentials, Depends(security)]):
    correct_username = os.getenv("BASIC_AUTH_USERNAME")
    correct_password = os.getenv("BASIC_AUTH_PASSWORD")
    if not correct_username or not correct_password:
        raise RuntimeError("Basic auth env variables not set")
    if not (
        secrets.compare_digest(credentials.username, correct_username)
        and secrets.compare_digest(credentials.password, correct_password)
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized",
            headers={"WWW-Authenticate": "Basic"},
        )


# ---------------------------------------------------------------
# Request Models
# ---------------------------------------------------------------

class PassportBase64Request(BaseModel):
    passportImageBase64: str
    reference_id: str | None = None


class IDCardBase64Request(BaseModel):
    cardImageFrontBase64: str | None = None
    cardImageBackBase64: str
    reference_id: str | None = None


class BenchmarkRequest(BaseModel):
    passportImageBase64: str
    runs: int = 5


# ---------------------------------------------------------------
# MRZ Type Detection
# ---------------------------------------------------------------
    
def detect_mrz_type(lines, reference_id=None):
    if not lines:
        logging.warning(log_prefix(reference_id) + "MRZ lines empty cannot detect type.")
        return "UNKNOWN"

    line_count = len(lines)
    first_len  = len(lines[0])
    prefix     = lines[0][0] if lines[0] else ""

    if line_count == 3 and first_len == 30:
        mrz_type = "TD1"
    elif line_count == 2 and first_len == 44 and prefix == "P":
        mrz_type = "TD3"
    elif line_count == 2 and first_len == 44 and prefix == "V":
        mrz_type = "MRVA"
    elif line_count == 2 and first_len == 36 and prefix == "V":
        mrz_type = "MRVB"
    elif line_count == 2 and first_len == 36:
        mrz_type = "TD2"
    else:
        mrz_type = "UNKNOWN"

    logging.info(log_prefix(reference_id) + f"Detected MRZ Type: {mrz_type}")
    return mrz_type


# ---------------------------------------------------------------
# Field Helpers
# ---------------------------------------------------------------

def clean_field(val):
    return val.replace("<", "")


def format_mrz_date(yymmdd):
    if not yymmdd or len(yymmdd) != 6 or not yymmdd.isdigit():
        logging.warning(f"Invalid MRZ date: {yymmdd}")
        return ""
    yy      = int(yymmdd[0:2])
    mm      = yymmdd[2:4]
    dd      = yymmdd[4:6]
    century = "19" if yy > 50 else "20"
    yyyy    = century + yymmdd[0:2]
    return f"{dd}-{mm}-{yyyy}"


def validate_result(result_dict):
    required_fields = [
        "documentType", "documentCode", "issuingState",
        "secondaryIdentifier", "nationality", "documentNumber",
        "dateOfBirth", "gender", "dateOfExpiry",
    ]
    for field in required_fields:
        value = result_dict.get(field, "")
        if value is None or str(value).strip() == "":
            logging.error(f"Validation failed - missing: {field}")
            return False
    return True


def _split_names(raw):
    """Split a raw MRZ name field into (surname, given_names)."""
    clean = raw.strip("<")
    if "<<" in clean:
        parts = clean.split("<<")
        return (
            parts[0].replace("<", " ").strip(),
            (parts[1] if len(parts) > 1 else "").replace("<", " ").strip(),
        )
    return "", clean.replace("<", " ").strip()


# ---------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------

def parse_td1(lines):
    if len(lines) != 3:
        return {"error": "TD1 requires exactly 3 MRZ lines"}
    line1, line2, line3 = lines
    m1 = re.match(r"^([A-Z][A-Z0-9<])([A-Z]{3})([A-Z0-9<]{9})([0-9A-Z])([A-Z0-9<]{15})$", line1)
    if not m1:
        logging.error(f"TD1 Line1 parse error: {line1}")
        return {"error": "Line 1 does not match TD1 format"}
    doc_type, issuer, doc_number, doc_number_cd, optional1 = m1.groups()
    try:
        line2       = line2.ljust(30, "<")
        dob         = line2[0:6]
        dob_cd      = line2[6]
        sex         = line2[7]
        expiry      = line2[8:14]
        expiry_cd   = line2[14]
        nationality = line2[15:18]
        optional2   = line2[18:29]
        final_cd    = line2[29]
        if len(nationality) != 3 or not nationality.isalpha():
            nationality = clean_field(issuer)
    except Exception:
        logging.error(f"TD1 Line2 parsing crash: {line2}")
        return {"error": "Line 2 parsing failed"}
    surname, given_names = _split_names(line3)
    return {
        "document_type":         clean_field(doc_type),
        "issuing_state":         clean_field(issuer),
        "document_number":       clean_field(doc_number),
        "document_number_check": clean_field(doc_number_cd),
        "optional1":             clean_field(optional1),
        "date_of_birth":         clean_field(dob),
        "date_of_birth_check":   clean_field(dob_cd),
        "sex":                   clean_field(sex),
        "date_of_expiry":        clean_field(expiry),
        "date_of_expiry_check":  clean_field(expiry_cd),
        "nationality":           clean_field(nationality),
        "optional2":             clean_field(optional2),
        "final_check_digit":     clean_field(final_cd),
        "surname":               surname,
        "given_names":           given_names,
    }


def parse_td2(lines):
    if len(lines) != 2:
        return {"error": "TD2 requires exactly 2 MRZ lines"}
    line1, line2 = lines
    m1 = re.match(r"^([A-Z][A-Z0-9<])([A-Z]{3})([A-Z0-9<]{31})$", line1)
    if not m1:
        logging.error(f"TD2 Line1 parse error: {line1}")
        return {"error": "Line 1 does not match TD2 format"}
    doc_type, issuer, names_raw = m1.groups()
    surname, given_names = _split_names(names_raw)
    if len(line2) != 36:
        logging.error(f"TD2 Line2 invalid length: {len(line2)}")
        return {"error": f"TD2 line2 invalid length: {len(line2)}"}
    return {
        "document_type":         clean_field(doc_type),
        "issuing_state":         clean_field(issuer),
        "surname":               surname,
        "given_names":           given_names,
        "document_number":       clean_field(line2[0:9]),
        "document_number_check": clean_field(line2[9]),
        "nationality":           clean_field(line2[10:13]),
        "date_of_birth":         clean_field(line2[13:19]),
        "date_of_birth_check":   clean_field(line2[19]),
        "sex":                   clean_field(line2[20]),
        "date_of_expiry":        clean_field(line2[21:27]),
        "date_of_expiry_check":  clean_field(line2[27]),
        "optional_data":         clean_field(line2[28:35]),
        "final_check_digit":     clean_field(line2[35]),
    }


def _parse_td3_tail(tail):
    """Extract optional data and check digits from TD3 line2 tail."""
    if len(tail) == 16:
        return tail[0:14], tail[14], tail[15]
    if len(tail) == 15:
        return tail[0:14], "", tail[-1]
    if len(tail) == 0:
        return "", "", ""
    if len(tail) == 1:
        return "", "", tail[0]
    return tail[:-1], "", tail[-1]


def parse_td3(lines):
    if len(lines) != 2:
        return {"error": "TD3 requires exactly 2 MRZ lines"}
    line1, line2 = lines
    m1 = re.match(r"^([A-Z][A-Z0-9<])([A-Z]{3})([A-Z0-9<]{39})$", line1)
    if not m1:
        logging.error(f"TD3 Line1 parse error: {line1}")
        return {"error": "Line 1 does not match TD3 format"}
    doc_type, issuer, names_raw = m1.groups()
    surname, given_names = _split_names(names_raw)
    if len(line2) not in (43, 44):
        logging.error(f"TD3 Line2 invalid length: {len(line2)}")
        return {"error": f"TD3 line2 invalid length: {len(line2)}"}
    optional, optional_cd, final_cd = _parse_td3_tail(line2[28:])
    return {
        "document_type":         clean_field(doc_type),
        "issuing_state":         clean_field(issuer),
        "surname":               surname,
        "given_names":           given_names,
        "document_number":       clean_field(line2[0:9]),
        "document_number_check": clean_field(line2[9]),
        "nationality":           clean_field(line2[10:13]),
        "date_of_birth":         clean_field(line2[13:19]),
        "date_of_birth_check":   clean_field(line2[19]),
        "sex":                   clean_field(line2[20]),
        "date_of_expiry":        clean_field(line2[21:27]),
        "date_of_expiry_check":  clean_field(line2[27]),
        "optional_data":         clean_field(optional),
        "optional_data_check":   clean_field(optional_cd),
        "final_check_digit":     clean_field(final_cd),
    }


# ---------------------------------------------------------------
# Normalize Output
# ---------------------------------------------------------------

def normalize_output(mrz_type, parsed):
    if "error" in parsed or mrz_type not in ["TD1", "TD2", "TD3", "MRVA", "MRVB"]:
        return {"success": False, "message": "Unrecognized MRZ format", "result": None}

    def _base(doc_type_int):
        return {
            "documentType":        doc_type_int,
            "documentCode":        parsed.get("document_type", ""),
            "issuingState":        parsed.get("issuing_state", ""),
            "primaryIdentifier":   parsed.get("surname", ""),
            "secondaryIdentifier": parsed.get("given_names", ""),
            "nationality":         parsed.get("nationality", ""),
            "documentNumber":      parsed.get("document_number", ""),
            "dateOfBirth":         format_mrz_date(parsed.get("date_of_birth", "")),
            "gender":              parsed.get("sex", ""),
            "dateOfExpiry":        format_mrz_date(parsed.get("date_of_expiry", "")),
        }

    type_map = {"TD1": 1, "TD2": 2, "TD3": 3, "MRVA": 4, "MRVB": 5}
    result   = _base(type_map[mrz_type])
    if mrz_type == "TD1":
        result["optionalData1"] = parsed.get("optional1", "")
        result["optionalData2"] = parsed.get("optional2", "")
    elif mrz_type in ("TD2", "MRVB"):
        result["optionalData1"] = parsed.get("optional_data", "")
        result["optionalData2"] = parsed.get("final_check_digit", "")
    elif mrz_type in ("TD3", "MRVA"):
        result["optionalData1"] = parsed.get("optional_data", "")
        result["optionalData2"] = parsed.get("optional_data_check", "")

    if not validate_result(result):
        return {"success": False, "message": "Please capture again. we couldn't parse the mrz", "result": None}
    return {"success": True, "message": "Success", "result": result}


# ---------------------------------------------------------------
# VG Response Builders
# ---------------------------------------------------------------

_PASSPORT_FIELDS = [
    FIELD_STATUS, FIELD_DOCUMENT_NAME, FIELD_DOCUMENT_CLASS_CODE, FIELD_DOCUMENT_NUMBER,
    FIELD_FULL_NAME, FIELD_GIVEN_NAMES, FIELD_SURNAME, FIELD_DATE_OF_BIRTH, FIELD_DATE_OF_EXPIRY,
    FIELD_DATE_OF_ISSUE, FIELD_ISSUING_STATE_CODE, FIELD_ISSUING_STATE_NAME, FIELD_NATIONALITY,
    "Nationality-Hindi", FIELD_SEX, "Place of Birth", "Place of Issue", FIELD_MRZ,
    FIELD_IMAGES, FIELD_PORTRAIT_POSITION, FIELD_POSITION,
]

_IDCARD_FIELDS = [
    FIELD_STATUS, FIELD_DOCUMENT_NAME, FIELD_DOCUMENT_NUMBER, FIELD_IDENTITY_CARD_NUMBER,
    FIELD_FULL_NAME, "Full Name-Arabic (U.A.E.)", FIELD_DATE_OF_BIRTH, FIELD_DATE_OF_EXPIRY,
    FIELD_ISSUING_STATE_CODE, FIELD_ISSUING_STATE_NAME, FIELD_NATIONALITY,
    "Nationality-Arabic (U.A.E.)", FIELD_SEX, "Sex-Arabic (U.A.E.)", FIELD_MRZ,
    FIELD_IMAGES, FIELD_PORTRAIT_POSITION, FIELD_POSITION,
]


def _extract_name_parts(r):
    surname   = r.get("primaryIdentifier") or None
    given     = r.get("secondaryIdentifier") or None
    full_name = " ".join(filter(None, [surname, given])) or None
    return surname, given, full_name


def _build_passport_mrz_block(r, mrz_type, surname, given, full_name):
    return {
        "MRZ Type":               mrz_type,
        FIELD_DOCUMENT_CLASS_CODE: r.get("documentCode") or None,
        FIELD_DOCUMENT_NUMBER:    r.get("documentNumber") or None,
        FIELD_FULL_NAME:          full_name,
        FIELD_GIVEN_NAMES:        given,
        FIELD_SURNAME:            surname,
        FIELD_ISSUING_STATE_CODE: r.get("issuingState") or None,
        FIELD_ISSUING_STATE_NAME: None,
        "Nationality Code":       r.get("nationality") or None,
        FIELD_DATE_OF_BIRTH:      r.get("dateOfBirth") or None,
        FIELD_DATE_OF_EXPIRY:     r.get("dateOfExpiry") or None,
        FIELD_SEX:                r.get("gender") or None,
        "MRZ Code":               None,
        FIELD_IDENTITY_CARD_NUMBER: None,
        "Validation":             None,
    }


def build_vg_passport_response(internal_result, mrz_type):
    if not internal_result.get("success"):
        base = dict.fromkeys(_PASSPORT_FIELDS, None)
        base[FIELD_STATUS] = "Failed"
        return base

    r                         = internal_result["result"]
    surname, given, full_name = _extract_name_parts(r)
    mrz_block                 = _build_passport_mrz_block(r, mrz_type, surname, given, full_name)
    return {
        FIELD_STATUS:              "Success",
        FIELD_DOCUMENT_NAME:       None,
        FIELD_DOCUMENT_CLASS_CODE: r.get("documentCode") or None,
        FIELD_DOCUMENT_NUMBER:     r.get("documentNumber") or None,
        FIELD_FULL_NAME:           full_name,
        FIELD_GIVEN_NAMES:         given,
        FIELD_SURNAME:             surname,
        FIELD_DATE_OF_BIRTH:       r.get("dateOfBirth") or None,
        FIELD_DATE_OF_EXPIRY:      r.get("dateOfExpiry") or None,
        FIELD_DATE_OF_ISSUE:       None,
        FIELD_ISSUING_STATE_CODE:  r.get("issuingState") or None,
        FIELD_ISSUING_STATE_NAME:  None,
        FIELD_NATIONALITY:         r.get("nationality") or None,
        "Nationality-Hindi":       None,
        FIELD_SEX:                 r.get("gender") or None,
        "Place of Birth":          None,
        "Place of Issue":          None,
        FIELD_MRZ:                 mrz_block,
        FIELD_IMAGES:              None,
        FIELD_PORTRAIT_POSITION:   None,
        FIELD_POSITION:            None,
    }


def _build_idcard_mrz_block(r, mrz_type, surname, given, full_name):
    return {
        "MRZ Type":               mrz_type,
        FIELD_DOCUMENT_CLASS_CODE: r.get("documentCode") or None,
        FIELD_DOCUMENT_NUMBER:    r.get("documentNumber") or None,
        FIELD_IDENTITY_CARD_NUMBER: None,
        FIELD_FULL_NAME:          full_name,
        FIELD_GIVEN_NAMES:        given,
        FIELD_SURNAME:            surname,
        FIELD_ISSUING_STATE_CODE: r.get("issuingState") or None,
        FIELD_ISSUING_STATE_NAME: None,
        "Nationality Code":       r.get("nationality") or None,
        FIELD_DATE_OF_BIRTH:      r.get("dateOfBirth") or None,
        FIELD_DATE_OF_EXPIRY:     r.get("dateOfExpiry") or None,
        FIELD_SEX:                r.get("gender") or None,
        "MRZ Code":               None,
        "Validation":             None,
    }


def build_vg_idcard_response(internal_result, mrz_type):
    if not internal_result.get("success"):
        base = dict.fromkeys(_IDCARD_FIELDS, None)
        base[FIELD_STATUS] = "Failed"
        return base

    r                         = internal_result["result"]
    surname, given, full_name = _extract_name_parts(r)
    mrz_block                 = _build_idcard_mrz_block(r, mrz_type, surname, given, full_name)
    return {
        FIELD_STATUS:                  "Success",
        FIELD_DOCUMENT_NAME:           None,
        FIELD_DOCUMENT_NUMBER:         r.get("documentNumber") or None,
        FIELD_IDENTITY_CARD_NUMBER:    None,
        FIELD_FULL_NAME:               full_name,
        "Full Name-Arabic (U.A.E.)":   None,
        FIELD_DATE_OF_BIRTH:           r.get("dateOfBirth") or None,
        FIELD_DATE_OF_EXPIRY:          r.get("dateOfExpiry") or None,
        FIELD_ISSUING_STATE_CODE:      r.get("issuingState") or None,
        FIELD_ISSUING_STATE_NAME:      None,
        FIELD_NATIONALITY:             r.get("nationality") or None,
        "Nationality-Arabic (U.A.E.)": None,
        FIELD_SEX:                     r.get("gender") or None,
        "Sex-Arabic (U.A.E.)":         None,
        FIELD_MRZ:                     mrz_block,
        FIELD_IMAGES:                  None,
        FIELD_PORTRAIT_POSITION:       None,
        FIELD_POSITION:                None,
    }


# ---------------------------------------------------------------
# Base64 Helper
# ---------------------------------------------------------------

def decode_base64_image(b64_string, reference_id=None):
    if not b64_string:
        logging.error(log_prefix(reference_id) + MSG_EMPTY_BASE64)
        return None, MSG_EMPTY_BASE64
    try:
        return base64.b64decode(b64_string), None
    except Exception as e:
        logging.error(log_prefix(reference_id) + f"Base64 decode failed: {e}")
        return None, MSG_INVALID_BASE64


# ---------------------------------------------------------------
# Benchmark worker
# ---------------------------------------------------------------

def _run_benchmark_in_worker(image_bytes, runs):
    global _scanner
    if _scanner is None:
        return {"error": "Scanner not initialized"}
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return {"error": "Invalid image"}
    except Exception as e:
        return {"error": f"Decode failed: {e}"}

    timings = []
    for _ in range(runs):
        _, timing = _scanner.run_inference(img)
        if timing:
            timings.append(timing)

    if not timings:
        return {"error": "No successful scans"}

    keys = timings[0].keys()
    avg  = {k: int(sum(t[k] for t in timings) / len(timings)) for k in keys}
    mn   = {k: min(t[k] for t in timings) for k in keys}
    mx   = {k: max(t[k] for t in timings) for k in keys}
    return {
        "runs":                   len(timings),
        "average":                avg,
        "min":                    mn,
        "max":                    mx,
        "rps_per_process_at_avg": round(1000 / avg["total_ms"], 2) if avg["total_ms"] else 0,
    }


# ---------------------------------------------------------------
# Process Pool
# ---------------------------------------------------------------

WORKER_POOL_SIZE = int(os.getenv("MRZ_POOL_SIZE", "23"))
MAX_QUEUE        = int(os.getenv("MRZ_MAX_QUEUE", "1000"))

_pool          = None
_semaphore     = None
_queue_lock    = None
_queue_counter = 0
_inflight      = 0


def get_pool():
    return _pool


# ---------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ---- startup ----
    global _pool, _semaphore, _queue_lock
    intra = int(os.getenv("MRZ_INTRA_THREADS", "1"))
    logging.info(
        f"Starting ProcessPoolExecutor: pool={WORKER_POOL_SIZE} "
        f"intra_threads={intra} max_queue={MAX_QUEUE}"
    )
    _pool       = ProcessPoolExecutor(max_workers=WORKER_POOL_SIZE, initializer=_init_worker)
    _semaphore  = asyncio.Semaphore(WORKER_POOL_SIZE)
    _queue_lock = asyncio.Lock()
    logging.info(
        f"ProcessPoolExecutor ready. "
        f"theoretical_rps=~{int(WORKER_POOL_SIZE * 1000 / 300)} at 300ms/scan "
        f"(actual depends on hardware)"
    )
    yield
    # ---- shutdown ----
    if _pool:
        _pool.shutdown(wait=False)


app    = FastAPI(lifespan=lifespan)
router = APIRouter()


# ---------------------------------------------------------------
# ScanDispatcher — queue tracking and scan dispatch
# ---------------------------------------------------------------

class ScanDispatcher:

    @staticmethod
    def _is_queue_full():
        return _queue_counter >= MAX_QUEUE

    @staticmethod
    def _log_queue_full():
        logging.warning(
            f"Queue full: inflight={_inflight} queued={_queue_counter} max_queue={MAX_QUEUE}"
        )

    @staticmethod
    async def enqueue(reference_id) -> bool:
        global _queue_counter
        async with _queue_lock:
            if ScanDispatcher._is_queue_full():
                ScanDispatcher._log_queue_full()
                return False
            _queue_counter += 1
        logging.info(log_prefix(reference_id) + f"Request queued | inflight={_inflight} queued={_queue_counter}")
        return True
        
    @staticmethod
    async def _dequeue(reference_id):
        global _queue_counter
        async with _queue_lock:
            if _queue_counter > 0:
                _queue_counter -= 1
            else:
                logging.warning(log_prefix(reference_id) + "_dequeue called with queue_counter=0, ignoring")
        logging.info(log_prefix(reference_id) + f"Request dequeued | inflight={_inflight} queued={_queue_counter}")
        
        
    @staticmethod
    async def dispatch(image_bytes, reference_id):
        await ScanDispatcher._dequeue(reference_id)   # always runs, regardless of semaphore outcome
        try:
            async with _semaphore:
                return await ScanDispatcher.run(image_bytes, reference_id)
        except asyncio.TimeoutError:
            logging.error(log_prefix(reference_id) + "Scanner timeout after 60s")
            return None
    
    @staticmethod
    async def mark_started(reference_id):
        global _inflight
        async with _queue_lock:
            _inflight += 1
        logging.info(log_prefix(reference_id) + f"Scan started | inflight={_inflight} queued={_queue_counter}")
        
    
    @staticmethod
    async def run(image_bytes, reference_id):
        await ScanDispatcher.mark_started(reference_id)
        try:
            result = await ScanDispatcher._execute(image_bytes, reference_id)
        finally:
            await ScanDispatcher.mark_finished(reference_id)
        return result
        
    @staticmethod
    async def _execute(image_bytes, reference_id):
        loop = asyncio.get_running_loop()
        return await asyncio.wait_for(
            loop.run_in_executor(get_pool(), _run_scan_in_worker, image_bytes, reference_id),
            timeout=60.0,
        )
        
    @staticmethod
    async def mark_finished(reference_id):
        global _inflight
        async with _queue_lock:
            _inflight -= 1
        logging.info(log_prefix(reference_id) + f"Scan finished | inflight={_inflight} queued={_queue_counter}")


async def _do_scan(image_bytes, reference_id):
    """Returns (internal_result, mrz_type) on success,
    or (None, SCAN_STATUS_QUEUE_FULL/TIMEOUT) when the request cannot be processed."""
    if not await ScanDispatcher.enqueue(reference_id):
        return None, SCAN_STATUS_QUEUE_FULL
    internal = await ScanDispatcher.dispatch(image_bytes, reference_id)
    if internal is None:
        return None, SCAN_STATUS_TIMEOUT
    mrz_type = internal.pop("mrz_type", "UNKNOWN")
    internal.pop("timing", None)
    return internal, mrz_type


# ---------------------------------------------------------------
# Shared endpoint helpers
# ---------------------------------------------------------------

def _error_json(start_time, reference_id, message, http_status=200):
    """Build a standard failure JSONResponse."""
    return JSONResponse(status_code=http_status, content={
        FIELD_STATUS:       "Failed",
        "message":          message,
        "result":           None,
        "reference_id":     reference_id,
        "processingTimeMs": int((time.perf_counter() - start_time) * 1000),
    })


def _scan_error_response(scan_status, start_time, reference_id):
    """Return an error response for queue/timeout scan statuses, or None if scan succeeded."""
    if scan_status == SCAN_STATUS_QUEUE_FULL:
        return _error_json(start_time, reference_id, "Server busy, please retry", http_status=429)
    if scan_status == SCAN_STATUS_TIMEOUT:
        return _error_json(start_time, reference_id, "Processing timeout")
    return None


# ---------------------------------------------------------------
# Health
# ---------------------------------------------------------------

@router.get("/health")
async def health():
    intra = int(os.getenv("MRZ_INTRA_THREADS", "1"))
    return JSONResponse(status_code=200, content={
        "status":          "ok",
        "inflight":        _inflight,
        "queued":          _queue_counter,
        "pool_size":       WORKER_POOL_SIZE,
        "intra_threads":   intra,
        "max_queue":       MAX_QUEUE,
        "theoretical_rps": f"~{int(WORKER_POOL_SIZE * 1000 / 300)} at 300ms/scan (estimate only)",
    })


# ---------------------------------------------------------------
# Benchmark endpoint
# ---------------------------------------------------------------

@router.post("/benchmark")
async def benchmark(
    req: BenchmarkRequest,
    auth: Annotated[Any, Depends(authenticate)],
):
    runs = max(1, min(req.runs, 20))
    image_bytes, err = decode_base64_image(req.passportImageBase64)
    if err:
        return JSONResponse(status_code=400, content={"error": err})

    loop   = asyncio.get_running_loop()
    result = await loop.run_in_executor(get_pool(), _run_benchmark_in_worker, image_bytes, runs)

    if "error" in result:
        return JSONResponse(status_code=500, content=result)

    avg_ms       = result["average"]["total_ms"]
    rps_ceiling  = round(WORKER_POOL_SIZE * 1000 / avg_ms, 1) if avg_ms else 0
    threads_note = (
        f"To reach 140 RPS you need pool_size >= {int(140 * avg_ms / 1000) + 1} "
        f"(i.e. {int(140 * avg_ms / 1000) + 1} physical cores with MRZ_INTRA_THREADS=1)"
    )
    return JSONResponse(status_code=200, content={
        **result,
        "pool_size":   WORKER_POOL_SIZE,
        "rps_ceiling": rps_ceiling,
        "advice":      threads_note,
    })


# ---------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------

@router.post("/extract_passport_mrz_base64")
async def passport_mrz_base64(
    req: PassportBase64Request,
    auth: Annotated[Any, Depends(authenticate)],
):
    start_time = time.perf_counter()
    image_bytes, err = decode_base64_image(req.passportImageBase64, req.reference_id)
    if err:
        return _error_json(start_time, req.reference_id, err)

    internal, mrz_type = await _do_scan(image_bytes, req.reference_id)
    scan_err = _scan_error_response(mrz_type, start_time, req.reference_id)
    if scan_err:
        return scan_err

    response = build_vg_passport_response(internal, mrz_type)
    response["reference_id"]     = req.reference_id
    response["processingTimeMs"] = int((time.perf_counter() - start_time) * 1000)
    return JSONResponse(status_code=200, content=response)


@router.post("/extract_idcard_mrz_base64")
async def idcard_mrz_base64(
    req: IDCardBase64Request,
    auth: Annotated[Any, Depends(authenticate)],
):
    start_time = time.perf_counter()
    image_bytes, err = decode_base64_image(req.cardImageBackBase64, req.reference_id)
    if err:
        return _error_json(start_time, req.reference_id, err)

    internal, mrz_type = await _do_scan(image_bytes, req.reference_id)
    scan_err = _scan_error_response(mrz_type, start_time, req.reference_id)
    if scan_err:
        return scan_err

    response = build_vg_idcard_response(internal, mrz_type)
    response["reference_id"]     = req.reference_id
    response["processingTimeMs"] = int((time.perf_counter() - start_time) * 1000)
    return JSONResponse(status_code=200, content=response)

app.include_router(router)