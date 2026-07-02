import os
import re
import numpy as np
import base64
import time
import asyncio
from fastapi import FastAPI, APIRouter, Depends, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from typing import Annotated, Any, Optional
import secrets
import logging
from logging.handlers import TimedRotatingFileHandler
from concurrent.futures import ProcessPoolExecutor
import cv2
import onnxruntime as ort
from pydantic import BaseModel, Field, ConfigDict

# ---------------------------------------------------------------
# Logging
# ---------------------------------------------------------------
LOG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "mrz_logs"))
os.makedirs(LOG_DIR, exist_ok=True)
log_file = os.path.join(LOG_DIR, "mrz_service.log")
handler = TimedRotatingFileHandler(log_file, when="midnight", interval=1, backupCount=7)
handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(handler)

def log_prefix(ref_id):
    return f"[REF-ID: {ref_id}] " if ref_id else ""

# ---------------------------------------------------------------
# ONNX Pipeline Config
# ---------------------------------------------------------------
DETECTION_MODEL   = os.getenv("DETECTION_MODEL",   "mrz_detection_20250222_fp32.onnx")
RECOGNITION_MODEL = os.getenv("RECOGNITION_MODEL", "mrz_recognition_20250221_fp32.onnx")
DET_SIZE = 256
REC_H, REC_W = 64, 640
_PROVIDERS = ["CPUExecutionProvider"]
_SEP = "<SEP>"
_VOCAB_KEYS = ["<PAD>", "<EOS>", _SEP] + list("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<")
VOCAB = dict(enumerate(_VOCAB_KEYS))
VOCAB_SIZE = len(_VOCAB_KEYS)
PAD_IDX = 0
EOS_IDX = 1

# ---------------------------------------------------------------
# Constants
# ---------------------------------------------------------------
DOCUMENT_NAME        = "Document Name"
DOCUMENT_CLASS_CODE  = "Document Class Code"
DOCUMENT_NUMBER      = "Document Number"
FULL_NAME            = "Full Name"
GIVEN_NAMES          = "Given Names"
DATE_OF_BIRTH        = "Date of Birth"
DATE_OF_EXPIRY       = "Date of Expiry"
ISSUING_STATE_CODE   = "Issuing State Code"
ISSUING_STATE_NAME   = "Issuing State Name"
IDENTITY_CARD_NUMBER = "Identity Card Number"
PORTRAIT_POSITION    = "Portrait Position"
EXAMPLE_BASE64_IMAGE = "SGVsbG8="
INVALID_BASE64_IMAGE = "Invalid base64 image"

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
    else:
        pad   = (h - w) // 2
        right = (h - w) - pad
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
    eos = np.nonzero(indices == EOS_IDX)[0]
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
        t0 = time.perf_counter()
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
        crop              = _warp_quadrangle(img, polygon)
        t5                = time.perf_counter()
        rec_blob          = _preprocess_recognition(crop)
        t6                = time.perf_counter()
        logits            = self.rec.run(None, {self.rec_in: rec_blob})[0][0]
        t7                = time.perf_counter()
        lines             = _decode_recognition(logits)
        t8                = time.perf_counter()
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
# Worker pool - one MRZPipeline per subprocess
# ---------------------------------------------------------------
_scanner = None

def _init_worker():
    global _scanner
    try:
        _scanner = MRZPipeline(DETECTION_MODEL, RECOGNITION_MODEL)
        intra    = int(os.getenv("MRZ_INTRA_THREADS", "1"))
        logging.info(f"[PID {os.getpid()}] MRZPipeline loaded. intra_threads={intra}")
    except Exception as e:
        _scanner = None
        logging.error(f"[PID {os.getpid()}] MRZPipeline load FAILED: {e}")

def _run_scan_in_worker(image_bytes, reference_id):
    global _scanner
    if _scanner is None:
        return {"success": False, "message": "Scanner not initialized", "result": None,
                "mrz_type": None, "timing": None}
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None or img.size == 0 or len(img.shape) != 3:
            raise ValueError("Invalid image")
    except Exception as e:
        logging.error(log_prefix(reference_id) + f"Image decode error: {e}")
        return {"success": False, "message": "Invalid or unsupported image", "result": None,
                "mrz_type": None, "timing": None}
    try:
        lines, timing = _scanner.run_inference(img)
    except Exception as e:
        logging.error(log_prefix(reference_id) + f"Scanner crash: {e}")
        return {"success": False, "message": "Scanner error", "result": None,
                "mrz_type": None, "timing": None}
    if not lines:
        logging.error(log_prefix(reference_id) + "No MRZ lines detected.")
        return {"success": False, "message": "No MRZ detected in image", "result": None,
                "mrz_type": None, "timing": timing}
    logging.info(log_prefix(reference_id) + f"Raw MRZ lines: {lines}")
    mrz_type = detect_mrz_type(lines, reference_id)
    if mrz_type == "TD1":
        parsed = parse_td1(lines)
    elif mrz_type in ("TD2", "MRVB"):
        parsed = parse_td2(lines)
    elif mrz_type in ("TD3", "MRVA"):
        parsed = parse_td3(lines)
    else:
        logging.error(log_prefix(reference_id) + "Unknown MRZ Type detected.")
        parsed = {"error": "Unknown or unsupported MRZ type"}
    final = normalize_output(mrz_type, parsed)
    final["mrz_type"] = mrz_type
    final["timing"]   = timing
    return final

# ---------------------------------------------------------------
# Auth
# ---------------------------------------------------------------
security = HTTPBasic()

def authenticate(credentials: HTTPBasicCredentials = Depends(security)):
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
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "passportImageBase64": EXAMPLE_BASE64_IMAGE,
                "reference_id": "TEST123"
            }
        }
    )
    passportImageBase64: str = Field(..., min_length=1)
    reference_id: Optional[str] = None

class IDCardBase64Request(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "cardImageBackBase64": EXAMPLE_BASE64_IMAGE,
                "reference_id": "TEST123"
            }
        }
    )
    cardImageFrontBase64: Optional[str] = None
    cardImageBackBase64: str = Field(..., min_length=1)
    reference_id: Optional[str] = None

class BenchmarkRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "passportImageBase64": EXAMPLE_BASE64_IMAGE,
                "runs": 5
            }
        }
    )
    passportImageBase64: str = Field(..., min_length=1)
    runs: int = Field(default=5, ge=1, le=20)

# ---------------------------------------------------------------
# MRZ Type Detection
# ---------------------------------------------------------------
def detect_mrz_type(lines, reference_id=None):
    if not lines:
        logging.warning(log_prefix(reference_id) + "MRZ lines empty - cannot detect type.")
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
    raw = line3.strip("<")
    if "<<" in raw:
        parts       = raw.split("<<")
        surname_raw = parts[0]
        given_raw   = parts[1] if len(parts) > 1 else ""
    else:
        surname_raw, given_raw = "", raw
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
        "surname":               surname_raw.replace("<", " ").strip(),
        "given_names":           given_raw.replace("<", " ").strip(),
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
    clean_names = names_raw.strip("<")
    if "<<" in clean_names:
        parts       = clean_names.split("<<")
        surname_raw = parts[0]
        given_raw   = parts[1] if len(parts) > 1 else ""
    else:
        surname_raw, given_raw = "", clean_names
    if len(line2) != 36:
        logging.error(f"TD2 Line2 invalid length: {len(line2)}")
        return {"error": f"TD2 line2 invalid length: {len(line2)}"}
    return {
        "document_type":         clean_field(doc_type),
        "issuing_state":         clean_field(issuer),
        "surname":               surname_raw.replace("<", " ").strip(),
        "given_names":           given_raw.replace("<", " ").strip(),
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

def parse_td3(lines):
    if len(lines) != 2:
        return {"error": "TD3 requires exactly 2 MRZ lines"}
    line1, line2 = lines
    m1 = re.match(r"^([A-Z][A-Z0-9<])([A-Z]{3})([A-Z0-9<]{39})$", line1)
    if not m1:
        logging.error(f"TD3 Line1 parse error: {line1}")
        return {"error": "Line 1 does not match TD3 format"}
    doc_type, issuer, names_raw = m1.groups()
    clean_names = names_raw.strip("<")
    if "<<" in clean_names:
        parts       = clean_names.split("<<")
        surname_raw = parts[0]
        given_raw   = parts[1] if len(parts) > 1 else ""
    else:
        surname_raw, given_raw = "", clean_names
    if len(line2) not in (43, 44):
        logging.error(f"TD3 Line2 invalid length: {len(line2)}")
        return {"error": f"TD3 line2 invalid length: {len(line2)}"}
    doc_number    = line2[0:9]
    doc_number_cd = line2[9]
    nationality   = line2[10:13]
    dob           = line2[13:19]
    dob_cd        = line2[19]
    sex           = line2[20]
    expiry        = line2[21:27]
    expiry_cd     = line2[27]
    tail          = line2[28:]
    if len(tail) == 16:
        optional, optional_cd, final_cd = tail[0:14], tail[14], tail[15]
    elif len(tail) == 15:
        optional, optional_cd, final_cd = tail[0:14], "", tail[-1]
    elif len(tail) == 0:
        optional = optional_cd = final_cd = ""
    elif len(tail) == 1:
        optional = optional_cd = ""
        final_cd = tail[0]
    else:
        optional, optional_cd, final_cd = tail[:-1], "", tail[-1]
    return {
        "document_type":         clean_field(doc_type),
        "issuing_state":         clean_field(issuer),
        "surname":               surname_raw.replace("<", " ").strip(),
        "given_names":           given_raw.replace("<", " ").strip(),
        "document_number":       clean_field(doc_number),
        "document_number_check": clean_field(doc_number_cd),
        "nationality":           clean_field(nationality),
        "date_of_birth":         clean_field(dob),
        "date_of_birth_check":   clean_field(dob_cd),
        "sex":                   clean_field(sex),
        "date_of_expiry":        clean_field(expiry),
        "date_of_expiry_check":  clean_field(expiry_cd),
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
def get_name_fields(result):
    surname   = result.get("primaryIdentifier") or None
    given     = result.get("secondaryIdentifier") or None
    full_name = " ".join(filter(None, [surname, given])) or None
    return surname, given, full_name

def build_failed_passport_response():
    base = dict.fromkeys([
        "Status", DOCUMENT_NAME, DOCUMENT_CLASS_CODE, DOCUMENT_NUMBER,
        FULL_NAME, GIVEN_NAMES, "Surname", DATE_OF_BIRTH, DATE_OF_EXPIRY,
        "Date of Issue", ISSUING_STATE_CODE, ISSUING_STATE_NAME,
        "Nationality", "Nationality-Hindi", "Sex", "Place of Birth",
        "Place of Issue", "MRZ", "Images", PORTRAIT_POSITION, "Position",
    ], None)
    base["Status"] = "Failed"
    return base

def build_vg_passport_response(internal_result, mrz_type):
    if not internal_result.get("success"):
        return build_failed_passport_response()
    r = internal_result["result"]
    surname, given, full_name = get_name_fields(r)
    mrz_block = {
        "MRZ Type": mrz_type,
        DOCUMENT_CLASS_CODE: r.get("documentCode") or None,
        DOCUMENT_NUMBER: r.get("documentNumber") or None,
        FULL_NAME: full_name, GIVEN_NAMES: given, "Surname": surname,
        ISSUING_STATE_CODE: r.get("issuingState") or None,
        ISSUING_STATE_NAME: None,
        "Nationality Code": r.get("nationality") or None,
        DATE_OF_BIRTH: r.get("dateOfBirth") or None,
        DATE_OF_EXPIRY: r.get("dateOfExpiry") or None,
        "Sex": r.get("gender") or None,
        "MRZ Code": None, IDENTITY_CARD_NUMBER: None, "Validation": None,
    }
    return {
        "Status": "Success", DOCUMENT_NAME: None,
        DOCUMENT_CLASS_CODE: r.get("documentCode") or None,
        DOCUMENT_NUMBER: r.get("documentNumber") or None,
        FULL_NAME: full_name, GIVEN_NAMES: given, "Surname": surname,
        DATE_OF_BIRTH: r.get("dateOfBirth") or None,
        DATE_OF_EXPIRY: r.get("dateOfExpiry") or None,
        "Date of Issue": None,
        ISSUING_STATE_CODE: r.get("issuingState") or None,
        ISSUING_STATE_NAME: None,
        "Nationality": r.get("nationality") or None,
        "Nationality-Hindi": None,
        "Sex": r.get("gender") or None,
        "Place of Birth": None, "Place of Issue": None,
        "MRZ": mrz_block, "Images": None,
        PORTRAIT_POSITION: None, "Position": None,
    }

def build_vg_idcard_response(internal_result, mrz_type):
    if not internal_result.get("success"):
        base = dict.fromkeys([
            "Status", DOCUMENT_NAME, DOCUMENT_NUMBER, IDENTITY_CARD_NUMBER,
            FULL_NAME, "Full Name-Arabic (U.A.E.)", DATE_OF_BIRTH, DATE_OF_EXPIRY,
            ISSUING_STATE_CODE, ISSUING_STATE_NAME, "Nationality",
            "Nationality-Arabic (U.A.E.)", "Sex", "Sex-Arabic (U.A.E.)",
            "MRZ", "Images", PORTRAIT_POSITION, "Position",
        ], None)
        base["Status"] = "Failed"
        return base
    r = internal_result["result"]
    surname, given, full_name = get_name_fields(r)
    mrz_block = {
        "MRZ Type": mrz_type,
        DOCUMENT_CLASS_CODE: r.get("documentCode") or None,
        DOCUMENT_NUMBER: r.get("documentNumber") or None,
        IDENTITY_CARD_NUMBER: None, FULL_NAME: full_name,
        GIVEN_NAMES: given, "Surname": surname,
        ISSUING_STATE_CODE: r.get("issuingState") or None,
        ISSUING_STATE_NAME: None,
        "Nationality Code": r.get("nationality") or None,
        DATE_OF_BIRTH: r.get("dateOfBirth") or None,
        DATE_OF_EXPIRY: r.get("dateOfExpiry") or None,
        "Sex": r.get("gender") or None,
        "MRZ Code": None, "Validation": None,
    }
    return {
        "Status": "Success", DOCUMENT_NAME: None,
        DOCUMENT_NUMBER: r.get("documentNumber") or None,
        IDENTITY_CARD_NUMBER: None, FULL_NAME: full_name,
        "Full Name-Arabic (U.A.E.)": None,
        DATE_OF_BIRTH: r.get("dateOfBirth") or None,
        DATE_OF_EXPIRY: r.get("dateOfExpiry") or None,
        ISSUING_STATE_CODE: r.get("issuingState") or None,
        ISSUING_STATE_NAME: None,
        "Nationality": r.get("nationality") or None,
        "Nationality-Arabic (U.A.E.)": None,
        "Sex": r.get("gender") or None,
        "Sex-Arabic (U.A.E.)": None,
        "MRZ": mrz_block, "Images": None,
        PORTRAIT_POSITION: None, "Position": None,
    }

# ---------------------------------------------------------------
# Base64 Helper
# ---------------------------------------------------------------
def decode_base64_image(b64_string, reference_id=None):
    try:
        if not b64_string:
            return None, INVALID_BASE64_IMAGE
        decoded = base64.b64decode(b64_string, validate=True)
        if not decoded:
            return None, INVALID_BASE64_IMAGE
        return decoded, None
    except Exception as e:
        logging.error(log_prefix(reference_id) + f"Base64 decode failed: {e}")
        return None, INVALID_BASE64_IMAGE

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
        "runs": len(timings), "average": avg, "min": mn, "max": mx,
        "rps_per_process_at_avg": round(1000 / avg["total_ms"], 2) if avg["total_ms"] else 0,
    }

# ---------------------------------------------------------------
# Process Pool Config
#
# QUEUE DESIGN
# ============
# _worker_sem  (size = WORKER_POOL_SIZE):
#   Blocks when all workers are busy. Waiting coroutines park here
#   cheaply in asyncio's internal wait list — no threads consumed.
#
# _queued (int counter):
#   Tracks how many requests are currently waiting for a worker.
#   Checked atomically before admitting; if it would exceed
#   WAIT_QUEUE_DEPTH, return 429 immediately without waiting.
#
# Why no acquire_nowait():
#   asyncio.Semaphore does NOT have acquire_nowait(). The correct
#   non-blocking gate is an integer counter check, which is safe
#   in single-threaded asyncio (no race conditions between check
#   and increment since there's no preemption between awaits).
#
# Sizing guidance (tune via env vars):
#   MRZ_POOL_SIZE  = number of CPU cores (default 23)
#   MRZ_WAIT_QUEUE = max waiting requests (default 2000)
#
#   At 300ms/scan, 23 workers = ~76 RPS throughput.
#   2000 wait slots absorbs a burst of 2000 extra requests
#   which drain in ~26s. 429 only fires beyond that.
# ---------------------------------------------------------------
WORKER_POOL_SIZE = int(os.getenv("MRZ_POOL_SIZE",   "23"))
WAIT_QUEUE_DEPTH = int(os.getenv("MRZ_WAIT_QUEUE", "2000"))

_pool       = None
_worker_sem = None   # asyncio.Semaphore — blocks at worker capacity
_inflight   = 0      # currently running in subprocess
_queued     = 0      # waiting for a free worker

def get_pool():
    return _pool

# ---------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------
app    = FastAPI()
router = APIRouter()

@app.on_event("startup")
async def startup_event():
    global _pool, _worker_sem
    intra = int(os.getenv("MRZ_INTRA_THREADS", "1"))
    logging.info(
        f"Starting ProcessPoolExecutor: pool={WORKER_POOL_SIZE} "
        f"wait_queue={WAIT_QUEUE_DEPTH} intra_threads={intra}"
    )
    _pool       = ProcessPoolExecutor(
        max_workers=WORKER_POOL_SIZE,
        initializer=_init_worker,
    )
    _worker_sem = asyncio.Semaphore(WORKER_POOL_SIZE)
    logging.info(
        f"ProcessPoolExecutor ready. "
        f"rps_ceiling=~{int(WORKER_POOL_SIZE * 1000 / 300)} at 300ms/scan"
    )

@app.on_event("shutdown")
async def shutdown_event():
    global _pool
    if _pool:
        _pool.shutdown(wait=False)

# ---------------------------------------------------------------
# Core scan dispatcher
#
# Step 1: Integer counter check — O(1), never suspends.
#         If _queued >= WAIT_QUEUE_DEPTH ? return 429 immediately.
#         Safe without a lock: asyncio is single-threaded; there
#         is no context switch between the check and the increment.
#
# Step 2: Increment _queued, then await _worker_sem.acquire().
#         Coroutine parks here until a worker is free.
#         asyncio handles the wait list for free.
#
# Step 3: run_in_executor ? subprocess scan.
#
# Step 4: Release semaphore + decrement counters in finally blocks
#         (always executes, even on exception or timeout).
# ---------------------------------------------------------------
async def _do_scan(image_bytes, reference_id):
    global _inflight, _queued

    # --- Step 1: reject if waiting room is full ---
    if _queued >= WAIT_QUEUE_DEPTH:
        logging.warning(
            log_prefix(reference_id) +
            f"Queue full (queued={_queued} >= limit={WAIT_QUEUE_DEPTH}): "
            f"inflight={_inflight}"
        )
        return None, "QUEUE_FULL"


    # --- Step 2: admit to waiting room ---
    _queued += 1
    logging.info(log_prefix(reference_id) + f"Admitted | inflight={_inflight} queued={_queued}")
  
    loop = asyncio.get_running_loop()
    try:
        # Wait until a subprocess worker is free
        try:
            await _worker_sem.acquire()
        except asyncio.CancelledError:
            # Client disconnected / request cancelled while still waiting.
            # We never left the waiting room, so undo Step 2's increment.
            _queued -= 1
            logging.warning(log_prefix(reference_id) + f"Cancelled while waiting | queued={_queued}")
            raise
  
        _queued   -= 1
        _inflight += 1
        logging.info(log_prefix(reference_id) + f"Scan started | inflight={_inflight} queued={_queued}")
  
        try:
            # --- Step 3: run scan in subprocess ---
            internal = await asyncio.wait_for(
                loop.run_in_executor(get_pool(), _run_scan_in_worker, image_bytes, reference_id),
                timeout=60.0,
            )
        finally:
            # --- Step 4a: always release worker slot ---
            _worker_sem.release()
            _inflight -= 1
            logging.info(log_prefix(reference_id) + f"Scan finished | inflight={_inflight} queued={_queued}")
  
    except asyncio.TimeoutError:
        logging.error(log_prefix(reference_id) + "Scanner timeout after 60s")
        return None, "TIMEOUT"
  
    except asyncio.CancelledError:
        raise
  
    except Exception as e:
        logging.error(log_prefix(reference_id) + f"_do_scan unexpected error: {e}")
        return None, "ERROR"

    mrz_type = internal.pop("mrz_type", "UNKNOWN")
    internal.pop("timing", None)
    return internal, mrz_type

# ---------------------------------------------------------------
# Health
# ---------------------------------------------------------------
@router.get("/health")
async def health():
    intra = int(os.getenv("MRZ_INTRA_THREADS", "1"))
    return JSONResponse(status_code=200, content={
        "status":          "ok",
        "inflight":        _inflight,
        "queued":          _queued,
        "pool_size":       WORKER_POOL_SIZE,
        "wait_queue_max":  WAIT_QUEUE_DEPTH,
        "intra_threads":   intra,
        "theoretical_rps": f"~{int(WORKER_POOL_SIZE * 1000 / 300)} at 300ms/scan (estimate only)",
    })

# ---------------------------------------------------------------
# Benchmark endpoint
# ---------------------------------------------------------------
@router.post(
    "/benchmark",
    responses={
        200: {"description": "Benchmark completed"},
        400: {"description": "Invalid request/image"},
        401: {"description": "Unauthorized"},
        422: {"description": "Validation Error"},
    },
)
async def benchmark(
    req: BenchmarkRequest,
    auth: Annotated[Any, Depends(authenticate)],
):
    start_time = time.perf_counter()
    runs = max(1, min(req.runs, 20))
    image_bytes, err = decode_base64_image(req.passportImageBase64)
    if err:
        return JSONResponse(status_code=200, content={
            "Status": "Failed", "message": err, "result": None,
            "processingTimeMs": int((time.perf_counter() - start_time) * 1000),
        })
    loop   = asyncio.get_running_loop()
    result = await loop.run_in_executor(get_pool(), _run_benchmark_in_worker, image_bytes, runs)
    if "error" in result:
        return JSONResponse(status_code=200, content=result)
    avg_ms      = result["average"]["total_ms"]
    rps_ceiling = round(WORKER_POOL_SIZE * 1000 / avg_ms, 1) if avg_ms else 0
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
# Shared endpoint helper
# ---------------------------------------------------------------
async def _handle_scan_request(
    image_field: str,
    reference_id,
    build_response_function,
    start_time: float,
):
    image_bytes, err = decode_base64_image(image_field, reference_id)
    if err:
        return JSONResponse(status_code=200, content={
            "Status": "Failed", "message": err, "result": None,
            "reference_id": reference_id,
            "processingTimeMs": int((time.perf_counter() - start_time) * 1000),
        })
    internal, mrz_type = await _do_scan(image_bytes, reference_id)
    if mrz_type == "QUEUE_FULL":
        return JSONResponse(
            status_code=429,
            headers={"Retry-After": "2"},
            content={
                "Status": "Failed", "message": "Server busy, please retry",
                "result": None, "reference_id": reference_id,
                "processingTimeMs": int((time.perf_counter() - start_time) * 1000),
            },
        )
    if mrz_type in ("TIMEOUT", "ERROR"):
        return JSONResponse(status_code=200, content={
            "Status": "Failed",
            "message": "Processing timeout" if mrz_type == "TIMEOUT" else "Internal error",
            "result": None, "reference_id": reference_id,
            "processingTimeMs": int((time.perf_counter() - start_time) * 1000),
        })
    response = build_response_function(internal, mrz_type)
    response["reference_id"]     = reference_id
    response["processingTimeMs"] = int((time.perf_counter() - start_time) * 1000)
    return JSONResponse(status_code=200, content=response)


# ---------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------
@router.post(
    "/extract_passport_mrz_base64",
    responses={
        200: {"description": "Processed"},
        400: {"description": "Malformed JSON"},
        401: {"description": "Unauthorized"},
        422: {"description": "Validation Error"},
        429: {"description": "Queue Full"},
    },
)
async def passport_mrz_base64(
    req: PassportBase64Request,
    auth: Annotated[Any, Depends(authenticate)],
):
    return await _handle_scan_request(
        req.passportImageBase64,
        req.reference_id,
        build_vg_passport_response,
        time.perf_counter(),
    )

@router.post(
    "/extract_idcard_mrz_base64",
    responses={
        200: {"description": "Processed"},
        400: {"description": "Malformed JSON"},
        401: {"description": "Unauthorized"},
        422: {"description": "Validation Error"},
        429: {"description": "Queue Full"},
    },
)
async def idcard_mrz_base64(
    req: IDCardBase64Request,
    auth: Annotated[Any, Depends(authenticate)],
):
    return await _handle_scan_request(
        req.cardImageBackBase64,
        req.reference_id,
        build_vg_idcard_response,
        time.perf_counter(),
    )
app.include_router(router)
