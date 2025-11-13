# main.py
import os
import re
import time
import io
import firebase_admin
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException, Depends, Header, Request, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from pymongo import MongoClient, DESCENDING
from bson import ObjectId
from datetime import datetime
from firebase_admin import credentials, auth as fb_auth
from dotenv import load_dotenv

 
import requests
import pandas as pd

load_dotenv()
app = FastAPI()

origins = [
    "http://localhost:3000",
]

# -------------------------
# Config
# -------------------------
MONGO_URI = os.getenv("MONGODB_URI")
HF_API_TOKEN = os.getenv("HF_API_TOKEN")
HF_MODEL = os.getenv("HF_MODEL", "facebook/bart-large-mnli")
HF_TEXTNORMALIZER_MODEL = os.getenv("HF_TEXTNORMALIZER_MODEL", "vennify/t5-base-grammar-correction")
HF_TIMEOUT = int(os.getenv("HF_TIMEOUT", "30"))

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_API_URL = os.getenv("GEMINI_API_URL", f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent")

RATE_LIMIT_PER_MIN = int(os.getenv("RATE_LIMIT_PER_MIN", "20"))
DAILY_QUOTA = int(os.getenv("DAILY_QUOTA", "500"))

ADMIN_PASSCODE = os.getenv("ADMIN_PASSCODE", "admin")
MANAGER_PASSCODE = os.getenv("MANAGER_PASSCODE", "manager")

# Firebase key path (must exist)
cred_path = os.path.join(os.path.dirname(__file__), "firebase-key.json")

# -------------------------
# Initialize Firebase
# -------------------------
if not firebase_admin._apps:
    try:
        cred = credentials.Certificate(cred_path)
        firebase_admin.initialize_app(cred)
        print("✅ Firebase initialized.")
    except Exception as e:
        print("❌ Firebase init failed:", e)

# -------------------------
# Initialize MongoDB
# -------------------------
if not MONGO_URI:
    raise RuntimeError("MONGODB_URI not set in .env")
client = MongoClient(MONGO_URI)

db = client["isjps"]
logs = db["logs"]
users_col = db["users"]
chats_col = db['chats']
logs_col = db['logs']
notifications = db["notifications"]   # for manager/admin alerts
blacklist_col = db["blacklist"]

print("✅ MongoDB client initialized.")

# -------------------------
# FastAPI app
# -------------------------
from fastapi.security import HTTPBearer
from fastapi.openapi.models import SecuritySchemeType
from fastapi.openapi.utils import get_openapi

app = FastAPI(title="ISJPS Backend (Full)", version="2.0.0")
security = HTTPBearer()

def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title="ISJPS Backend (Full)",
        version="2.0.0",
        description="ISJPS API with Firebase Auth",
        routes=app.routes,
    )
    openapi_schema["components"]["securitySchemes"] = {
        "BearerAuth": {"type": SecuritySchemeType.http.value, "scheme": "bearer", "bearerFormat": "JWT"}
    }
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Models
# -------------------------
class PromptInput(BaseModel):
    prompt: str
    confirm: bool = False

# -------------------------
# Utilities: normalizer, sanitizer, HF/Gemini wrappers
# -------------------------
SLANG_FALLBACK = {
    "u": "you", "r": "are", "ur": "your", "pls": "please", "plz": "please",
    "thx": "thanks", "bcoz": "because", "idk": "I don't know", "im": "I'm",
    "dont": "don't", "wanna": "want to", "gonna": "going to"
}

def expand_slang_fallback(text: str) -> str:
    words = re.findall(r"\b\w+\b", text)
    return " ".join(SLANG_FALLBACK.get(w.lower(), w) for w in words)

def hf_normalize_text(prompt_text: str) -> str:
    """Try HF text normalizer, fallback to slang mapping."""
    if not HF_API_TOKEN:
        return expand_slang_fallback(prompt_text)
    url = f"https://router.huggingface.co/hf-inference/models/{HF_TEXTNORMALIZER_MODEL}"
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    try:
        r = requests.post(url, headers=headers, json={"inputs": prompt_text, "parameters": {"max_new_tokens": 120}}, timeout=HF_TIMEOUT)
        if r.status_code == 200:
            data = r.json()
            if isinstance(data, list) and data:
                if isinstance(data[0], dict) and "generated_text" in data[0]:
                    return data[0]["generated_text"].strip()
                if isinstance(data[0], str):
                    return data[0].strip()
            if isinstance(data, dict) and "generated_text" in data:
                return data["generated_text"].strip()
        # if anything fails, fallback
    except Exception as e:
        print("⚠️ HF normalization failed:", e)
    return expand_slang_fallback(prompt_text)

# Redaction patterns
EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
CREDITCARD_RE = re.compile(r"\b(?:\d[ -]*?){13,19}\b")
LONG_TOKEN_RE = re.compile(r"\b[0-9a-fA-F]{20,}\b")
IP_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")

def verify_admin_token(authorization: str = Header(None)):
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")

    try:
        token = authorization.split(" ")[1]
        decoded_token = fb_auth.verify_id_token(token)
        uid = decoded_token.get("uid")

        # 🔍 Check role from MongoDB instead of Firebase
        user = users_col.find_one({"uid": uid})
        if not user or user.get("role") != "admin":
            raise HTTPException(status_code=403, detail="Admin access required")

        return decoded_token
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Invalid token: {str(e)}")
def sanitize_text(text: str):
    count = 0
    def sub_and_count(pattern, repl, s):
        nonlocal count
        new, n = pattern.subn(repl, s)
        count += n
        return new
    t = text
    t = sub_and_count(EMAIL_RE, "[REDACTED_EMAIL]", t)
    t = sub_and_count(CREDITCARD_RE, "[REDACTED_CREDITCARD]", t)
    t = sub_and_count(LONG_TOKEN_RE, "[REDACTED_TOKEN]", t)
    t = sub_and_count(IP_RE, "[REDACTED_IP]", t)
    return t, count

# Hugging Face zero-shot via router
def hf_zero_shot_classify(text: str, candidate_labels):
    if not HF_API_TOKEN:
        raise RuntimeError("HF_API_TOKEN not configured")
    url = "https://router.huggingface.co/hf-inference/text-classification"
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    payload = {"model": HF_MODEL, "inputs": text, "parameters": {"candidate_labels": candidate_labels, "multi_label": False}}
    r = requests.post(url, headers=headers, json=payload, timeout=HF_TIMEOUT)
    if r.status_code != 200:
        raise RuntimeError(f"HF zero-shot error {r.status_code}: {r.text}")
    data = r.json()
    # data might be list or dict
    if isinstance(data, list) and data and isinstance(data[0], dict):
        labels = data[0].get("labels", [])
        scores = data[0].get("scores", [])
    else:
        labels = data.get("labels", [])
        scores = data.get("scores", [])
    if not labels or not scores:
        raise RuntimeError(f"Unexpected HF response: {data}")
    return labels[0], float(scores[0]), data

# Gemini wrapper
def gemini_generate(prompt_text: str):
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY not set in .env")
    headers = {"Content-Type": "application/json"}
    params = {"key": GEMINI_API_KEY}
    payload = {"contents": [{"role": "user", "parts": [{"text": prompt_text}]}]}
    r = requests.post(GEMINI_API_URL, headers=headers, params=params, json=payload, timeout=60)
    if r.status_code != 200:
        raise RuntimeError(f"Gemini generation error {r.status_code}: {r.text}")
    data = r.json()
    candidates = data.get("candidates", [])
    if candidates:
        c = candidates[0]
        # many Gemini outputs put text in content.parts[*].text
        if "content" in c and "parts" in c["content"] and c["content"]["parts"]:
            return c["content"]["parts"][0].get("text", "")
    # fallback
    return data

# Fall To HF
def hf_generate_text(prompt_text: str):
    """Generate text response using Gemini or Hugging Face, with graceful fallback."""
    # Try Gemini first
    try:
        if GEMINI_API_KEY:
            return gemini_generate(prompt_text)
    except Exception as e:
        print(f"⚠️ Gemini generation failed: {e}")

    # Then try Hugging Face if token exists
    if HF_API_TOKEN:
        try:
            url = f"https://router.huggingface.co/hf-inference/models/{HF_MODEL}"
            headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
            payload = {"inputs": prompt_text, "parameters": {"max_new_tokens": 200}}
            r = requests.post(url, headers=headers, json=payload, timeout=HF_TIMEOUT)
            if r.status_code == 200:
                data = r.json()
                # Normalize possible return shapes
                if isinstance(data, list) and data and isinstance(data[0], dict):
                    if "generated_text" in data[0]:
                        return data[0]["generated_text"].strip()
                if isinstance(data, dict) and "generated_text" in data:
                    return data["generated_text"].strip()
                if isinstance(data, list) and isinstance(data[0], str):
                    return data[0].strip()
            else:
                print(f"⚠️ HF generation error {r.status_code}: {r.text}")
        except Exception as e:
            print(f"⚠️ HF generation failed: {e}")

    # Fallback: simple echo so you can keep testing without external APIs
    return f"Echo: {prompt_text}"

# -------------------------
# Quotas / ensure user
# -------------------------
def ensure_user_record(uid: str, email: str):
    """Create user with defaults if missing. Adds metadata fields for faults/blocks."""
    now = datetime.utcnow()
    u = users_col.find_one({"uid": uid})
    if u:
        # normalize quota_reset_date if string
        reset = u.get("quota_reset_date")
        try:
            reset_dt = reset if isinstance(reset, datetime) else (datetime.fromisoformat(reset) if isinstance(reset, str) else None)
        except Exception:
            reset_dt = None
        if not reset_dt or reset_dt < now:
            users_col.update_one({"uid": uid}, {"$set": {"daily_quota_left": DAILY_QUOTA, "quota_reset_date": (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)}})
        return users_col.find_one({"uid": uid})
    # create new user
    user_count = users_col.count_documents({})
    role = "admin" if user_count == 0 else "user"
    doc = {
        "uid": uid,
        "email": email,
        "role": role,
        "display_name": email.split("@")[0],
        "daily_quota_left": DAILY_QUOTA,
        "quota_reset_date": (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0),
        "recent_timestamps": [],
        # monitoring fields:
        "fault_count": 0,
        "total_prompts": 0,
        "fault_percent": 0.0,
        "blocked": False,
        "deleted": False,
        "flagged": False,
    }
    users_col.insert_one(doc)
    print(f"➕ Created user record: {email} (role={role})")
    return users_col.find_one({"uid": uid})

def check_and_consume_quota(uid: str):
    u = users_col.find_one({"uid": uid})
    if not u:
        return False, "User record not found."
    now = time.time()
    stamps = [s for s in u.get("recent_timestamps", []) if now - s < 60]
    if len(stamps) >= RATE_LIMIT_PER_MIN:
        return False, f"Rate limit exceeded: {RATE_LIMIT_PER_MIN} requests per minute."
    if u.get("daily_quota_left", 0) <= 0:
        return False, "Daily quota exhausted."
    stamps.append(now)
    users_col.update_one({"uid": uid}, {"$set": {"recent_timestamps": stamps}, "$inc": {"daily_quota_left": -1, "total_prompts": 1}})
    return True, "OK"

# -------------------------
# Firebase verification
# -------------------------

async def verify_user(request: Request):
    auth_header = request.headers.get("Authorization")
    if not auth_header:
        raise HTTPException(status_code=401, detail="Missing Authorization header")

    parts = auth_header.strip().split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(status_code=401, detail=f"Invalid Authorization header format: {auth_header}")

    token = parts[1].strip()
    if not token:
        raise HTTPException(status_code=401, detail="Empty token")

    try:
        decoded = fb_auth.verify_id_token(token)
        ensure_user_record(decoded.get("uid"), decoded.get("email", "unknown"))
        return decoded
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Token invalid: {e}")

async def verify_manager(request: Request):
    decoded = await verify_user(request)
    uid = decoded.get("uid")

    user_doc = users_col.find_one({"uid": uid})
    if not user_doc or user_doc.get("role") != "manager":
        raise HTTPException(status_code=403, detail="Manager access required")

    return user_doc

# -------------------------
# Helper: flag, notify, block
# -------------------------
def flag_user_and_notify(uid: str, email: str, reason: str, prompt: str, severity: str="malicious"):
    """Mark user flagged, increment fault_count, create notification for manager/admin."""
    now = datetime.utcnow()
    users_col.update_one({"uid": uid}, {"$set": {"flagged": True}, "$inc": {"fault_count": 1}})
    u = users_col.find_one({"uid": uid})
    total = u.get("total_prompts", 1)
    faults = u.get("fault_count", 1)
    fault_percent = round((faults/total)*100, 2) if total else 100.0
    users_col.update_one({"uid": uid}, {"$set": {"fault_percent": fault_percent}})
    # create notification document
    notifications.insert_one({
        "uid": uid,
        "email": email,
        "reason": reason,
        "prompt": prompt,
        "severity": severity,
        "seen": False,
        "timestamp": now
    })
    # log
    logs.insert_one({
        "user_email": email,
        "uid": uid,
        "original": prompt,
        "label": severity,
        "reason": reason,
        "action": "flagged_and_notify",
        "timestamp": now
    })

def block_user(uid: str, actor_role: str, actor_uid: str):
    users_col.update_one({"uid": uid}, {"$set": {"blocked": True}})
    # log + notification
    u = users_col.find_one({"uid": uid})
    notifications.insert_one({
        "uid": uid,
        "email": u.get("email"),
        "reason": f"Blocked by {actor_role}",
        "prompt": None,
        "severity": "block",
        "seen": False,
        "timestamp": datetime.utcnow()
    })
    logs.insert_one({
        "user_email": u.get("email"),
        "uid": uid,
        "action": f"blocked_by_{actor_role}",
        "by_uid": actor_uid,
        "timestamp": datetime.utcnow()
    })

# -------------------------
# Endpoints
# -------------------------
def serialize_user(user):
    user_copy = dict(user)  # make a copy
    if "_id" in user_copy:
        user_copy["_id"] = str(user_copy["_id"])
    return user_copy

# -------------------------
# Hardened /user/history endpoint
# -------------------------

@app.get("/manager/history")
async def manager_history(request: Request):
    # Authenticate manager
    decoded = await verify_user(request)
    uid = decoded.get("uid")
    requester = users_col.find_one({"uid": uid})

    if not requester or requester.get("role") != "manager":
        raise HTTPException(status_code=403, detail="Manager access required")

    # fetch all non-admin users
    all_users = list(users_col.find({"role": {"$ne": "admin"}}))

    out = {}

    for user_doc in all_users:
        user_uid = user_doc.get("uid")
        user_name = user_doc.get("name", "-")
        email = user_doc.get("email", "-")

        # get fault logs for this user
        fault_logs = list(
            logs_col.find({
                "uid": user_uid,
                "label": {"$in": ["malicious", "dangerous"]}
            }).sort("timestamp", -1)
        )

        fault_prompts = []
        for log in fault_logs:
            fault_prompts.append({
                "_id": str(log.get("_id")),
                "uid": log.get("uid"),
                "original": log.get("original", ""),
                "label": log.get("label", "unknown"),
                "timestamp": log.get("timestamp"),
                "note": log.get("note", "")
            })

        if fault_prompts:  # only include if user has any
            out[user_uid] = {
                "name": user_name,
                "email": email,
                "fault_prompts": fault_prompts
            }

    return {"users": out, "count": len(out)}


@app.get("/manager/faults")
async def get_fault_summary(current_user: dict = Depends(verify_manager)):
    users_list = []
    all_users = list(users_col.find({"role": {"$ne": "admin"}}))

    for user in all_users:
        if "email" not in user:
            continue

        uid = user.get("uid")
        prompts = user.get("prompts", [])
        safe = sanitized = dangerous = malicious = 0

        for p in prompts:
            label = p.get("label") or p.get("status")
            if label == "safe":
                safe += 1
            elif label == "sanitized":
                sanitized += 1
            elif label == "dangerous":
                dangerous += 1
            elif label == "malicious":
                malicious += 1

        user_logs = logs.find({"uid": uid})
        for log_entry in user_logs:
            label = log_entry.get("label")
            if label == "safe":
                safe += 1
            elif label == "sanitized":
                sanitized += 1
            elif label == "dangerous":
                dangerous += 1
            elif label == "malicious":
                malicious += 1

        total = safe + sanitized + dangerous + malicious
        if total == 0:
            safe_pct = sanitized_pct = dangerous_pct = malicious_pct = 0
        else:
            safe_pct = round((safe / total) * 100, 2)
            sanitized_pct = round((sanitized / total) * 100, 2)
            dangerous_pct = round((dangerous / total) * 100, 2)
            malicious_pct = round((malicious / total) * 100, 2)

        users_list.append({
            "uid": uid,
            "email": user.get("email"),
            "fault_stats": {
                "safe": safe_pct,
                "sanitized": sanitized_pct,
                "dangerous": dangerous_pct,
                "malicious": malicious_pct
            }
        })

    # Make sure all ObjectIds are converted to strings
    users_list_serialized = [serialize_user(u) for u in users_list]

    return JSONResponse({"users": users_list_serialized})

# Manager: get a specific user's fault chat history

@app.get("/manager/faults/{user_uid}")
async def get_fault_history(user_uid: str, request: Request):
    # Verify requester is a manager
    decoded = await verify_user(request)
    uid = decoded.get("uid")

    requester = users_col.find_one({"uid": uid})
    if not requester or requester.get("role") != "manager":
        raise HTTPException(status_code=403, detail="Manager access required")

    # Verify target user exists and is not admin
    target_user = users_col.find_one({"uid": user_uid, "role": {"$ne": "admin"}})
    if not target_user:
        raise HTTPException(status_code=404, detail="User not found or admin cannot be queried")

    # Collect prompts from user
    prompts = target_user.get("prompts", [])
    safe = sanitized = dangerous = malicious = 0
    fault_prompts = []

    for p in prompts:
        label = p.get("label") or p.get("status")
        if label == "safe":
            safe += 1
        elif label == "sanitized":
            sanitized += 1
            fault_prompts.append(p)
        elif label == "dangerous":
            dangerous += 1
            fault_prompts.append(p)
        elif label == "malicious":
            malicious += 1
            fault_prompts.append(p)

    # Also include logs for this user
    user_logs = logs.find({"uid": user_uid})
    for log_entry in user_logs:
        label = log_entry.get("label")
        if label == "safe":
            safe += 1
        elif label == "sanitized":
            sanitized += 1
            fault_prompts.append(log_entry)
        elif label == "dangerous":
            dangerous += 1
            fault_prompts.append(log_entry)
        elif label == "malicious":
            malicious += 1
            fault_prompts.append(log_entry)

    # Calculate percentages
    total = safe + sanitized + dangerous + malicious
    if total == 0:
        safe_pct = sanitized_pct = dangerous_pct = malicious_pct = 0
    else:
        safe_pct = round((safe / total) * 100, 2)
        sanitized_pct = round((sanitized / total) * 100, 2)
        dangerous_pct = round((dangerous / total) * 100, 2)
        malicious_pct = round((malicious / total) * 100, 2)

    # Format fault prompts
    formatted_prompts = []
    for p in fault_prompts:
        formatted_prompts.append({
            "_id": str(p.get("_id")),
            "prompt": p.get("original") or "",
            "response": p.get("generation") or "",
            "status": p.get("label") or p.get("status") or "unknown",
            "timestamp": p.get("timestamp") or ""
        })

    return JSONResponse({
        "user": target_user.get("email") or target_user.get("name"),
        "fault_stats": {
            "safe": safe_pct,
            "sanitized": sanitized_pct,
            "dangerous": dangerous_pct,
            "malicious": malicious_pct
        },
        "fault_prompts": formatted_prompts
    })
@app.get("/user/history")
async def user_history(request: Request, limit: int = Query(20, ge=1, le=200)):
    decoded = await verify_user(request)
    uid = decoded.get("uid")
    u = users_col.find_one({"uid": uid})
    if not u:
        raise HTTPException(status_code=404, detail="User not found")
    if u.get("deleted", False):
        raise HTTPException(status_code=403, detail="Account deleted")
    if u.get("blocked", False):
        raise HTTPException(status_code=403, detail="Account blocked")
    
    docs = list(logs.find({"uid": uid}).sort("timestamp", DESCENDING).limit(limit))
    for d in docs:
        d["_id"] = str(d["_id"])
        if isinstance(d.get("timestamp"), datetime):
            d["timestamp"] = d["timestamp"].isoformat()
    return {"count": len(docs), "logs": docs}

# -------------------------
# Hardened /notifications endpoint
# -------------------------
@app.get("/notifications")
async def list_notifications(request: Request, limit: int = Query(100, ge=1, le=1000)):
    decoded = await verify_user(request)
    uid = decoded.get("uid")
    u = users_col.find_one({"uid": uid})
    if not u:
        raise HTTPException(status_code=404, detail="User not found")
    if u.get("deleted", False):
        raise HTTPException(status_code=403, detail="Account deleted")
    if u.get("blocked", False):
        raise HTTPException(status_code=403, detail="Account blocked")

    # managers/admins can see all notifications; users only their own
    if u.get("role") in ("admin", "manager"):
        docs = list(notifications.find().sort("timestamp", DESCENDING).limit(limit))
    else:
        docs = list(notifications.find({"uid": uid}).sort("timestamp", DESCENDING).limit(limit))
    out = []
    for d in docs:
        d["_id"] = str(d["_id"])
        if isinstance(d.get("timestamp"), datetime):
            d["timestamp"] = d["timestamp"].isoformat()
        out.append(d)
    return {"count": len(out), "notifications": out}

# -------------------------
# Hardened /analyze endpoint
# -------------------------

@app.post("/analyze")
async def analyze(prompt_in: PromptInput, request: Request):
    """
    Analyze user prompt -> classify into safe/sanitized/dangerous/malicious.
    Does NOT auto-block or delete anymore.
    Manager/Admin will handle those actions manually.
    """
    # --- Verify and setup ---
    uid = None
    email = "anonymous"
    try:
        decoded = await verify_user(request)
        uid = decoded.get("uid")
        email = decoded.get("email", "unknown")
    except HTTPException:
        decoded = None

    if uid:
        # Check blacklist
        if blacklist_col.find_one({"email": email}):
            raise HTTPException(status_code=403, detail="Account blacklisted. Contact admin.")
        # Check quota
        ok, msg = check_and_consume_quota(uid)
        if not ok:
            raise HTTPException(status_code=429, detail=msg)

    # --- Normalize & clean ---
    original = (prompt_in.prompt or "").strip()
    normalized = hf_normalize_text(original)
    cleaned = re.sub(r"\s+", " ", normalized).strip()

    # --- Classification ---
    candidate_labels = ["safe", "sanitized", "dangerous", "malicious"]
    label, confidence, hf_raw = "safe", 0.0, None

    try:
        best_label, best_score, hf_raw = hf_zero_shot_classify(cleaned, candidate_labels)
        label, confidence = best_label, best_score
    except Exception:
        lower = cleaned.lower()
        malicious_kw = [
            "bypass safety", "jailbreak", "root access", "secret key",
            "api_key", "delete /", "rm -rf /", "weapon", "harm", "kill"
        ]
        dangerous_kw = [
            "hurt myself", "suicide", "kill myself", "harm others",
            "abuse", "attack", "terror"
        ]
        if any(k in lower for k in malicious_kw):
            label, confidence = "malicious", 0.99
        elif any(k in lower for k in dangerous_kw):
            label, confidence = "dangerous", 0.95
        elif any(w in lower for w in ["pls", "plz", "bcoz", "idk", "wanna", "gonna"]):
            label, confidence = "sanitized", 0.6
        else:
            label, confidence = "safe", 0.5

    # --- Logging for all classifications ---
    entry = {
        "uid": uid,
        "email": email,
        "original": original,
        "cleaned": cleaned,
        "label": label,
        "confidence": confidence,
        "timestamp": datetime.utcnow(),
    }

    # --- Dangerous / Malicious Handling ---
    if label in ["dangerous", "malicious"]:
        reason = f"{label.capitalize()} prompt detected"
        severity = label

        # Notify and flag, but don't block/delete automatically
        if uid:
            flag_user_and_notify(uid, email, reason, original, severity=severity)

        entry.update({
            "action": f"flagged_{label}",
            "note": f"{label.capitalize()} prompt detected — flagged for review",
        })
        logs.insert_one(entry)

        msg = (
            "⚠️ Dangerous prompt detected. Managers have been notified."
            if label == "dangerous"
            else "🚫 Malicious prompt detected. Managers/Admins will review this."
        )
        raise HTTPException(status_code=403, detail=msg)

    # --- Sanitized Handling ---
    sanitized_text, redactions = None, 0
    if label == "sanitized":
        sanitized_text, redactions = sanitize_text(cleaned)
        if not prompt_in.confirm:
            logs.insert_one({
                **entry,
                "sanitized": sanitized_text,
                "redactions": redactions,
                "action": "require_confirm",
            })
            return {
                "status": "require_confirm",
                "message": "Prompt sanitized. Review and resend with confirm=true to continue.",
                "sanitized": sanitized_text,
                "label": label,
                "confidence": f"{confidence*100:.2f}%"
            }

    # --- Safe or confirmed sanitized: generate ---
    forward_text = sanitized_text if (label == "sanitized" and prompt_in.confirm) else cleaned
    try:
        generation = gemini_generate(forward_text)
        action = "forwarded_gemini"
    except Exception as e:
        print("⚠️ Gemini generation failed:", e)
        generation = hf_generate_text(forward_text)
        action = "forwarded_hf"

    # --- Update fault & usage stats ---
    if uid:
        u = users_col.find_one({"uid": uid})
        total = (u.get("total_prompts") or 0) + 1
        faults = u.get("fault_count") or 0
        users_col.update_one(
            {"uid": uid},
            {"$set": {
                "total_prompts": total,
                "fault_percent": round((faults / total) * 100, 2)
            }}
        )

    # --- Save final log ---
    logs.insert_one({
        **entry,
        "sanitized": sanitized_text,
        "generation_preview": str(generation)[:300],
        "action": action,
    })

    return {
        "status": "ok",
        "label": label,
        "confidence": f"{confidence*100:.2f}%",
        "generation": generation,
        "sanitized": sanitized_text
    }


# -------------------------
# Add /me endpoint
# -------------------------

@app.get("/me")
async def me(request: Request):
    """
    Returns the currently logged-in user info for frontend.
    """
    decoded = await verify_user(request)
    uid = decoded.get("uid")
    u = users_col.find_one({"uid": uid})
    if not u:
        raise HTTPException(status_code=404, detail="User not found")
    if u.get("deleted", False):
        raise HTTPException(status_code=403, detail="Account deleted")
    if u.get("blocked", False):
        raise HTTPException(status_code=403, detail="Account blocked")
    return {
        "uid": uid,
        "email": decoded.get("email"),
        "role": u.get("role"),
        "display_name": u.get("display_name", u.get("email").split("@")[0]),
        "daily_quota_left": u.get("daily_quota_left", DAILY_QUOTA)
    }

async def verify_user(request: Request):
    auth_header = request.headers.get("Authorization")
    if not auth_header:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    parts = auth_header.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(status_code=401, detail="Invalid Authorization header format")
    token = parts[1]
    try:
        decoded = fb_auth.verify_id_token(token)
        # ensure DB record exists
        ensure_user_record(decoded.get("uid"), decoded.get("email", "unknown"))
        return decoded
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Token invalid: {e}")

@app.get("/get_role")
async def get_role(request: Request):
    """
    Returns user's role after login for frontend redirect.
    """
    decoded = await verify_user(request)
    uid = decoded.get("uid")
    u = users_col.find_one({"uid": uid})
    if not u:
        raise HTTPException(status_code=404, detail="User not found")
    if u.get("deleted", False):
        raise HTTPException(status_code=403, detail="Account deleted")
    if u.get("blocked", False):
        raise HTTPException(status_code=403, detail="Account blocked")
    return {
        "uid": uid,
        "email": decoded.get("email"),
        "role": u.get("role"),
        "display_name": u.get("display_name", u.get("email").split("@")[0])
    }

@app.get("/")
def root():
    return {"status": "ISJPS running", "time": datetime.utcnow().isoformat()}

# get_role - frontend uses after login to know where to redirect
@app.get("/get_role")
async def get_role(request: Request):
    decoded = await verify_user(request)
    uid = decoded.get("uid")
    u = users_col.find_one({"uid": uid})
    if not u:
        raise HTTPException(status_code=404, detail="User not found")
    # if deleted or blacklisted, deny
    if u.get("deleted", False):
        raise HTTPException(status_code=403, detail="Account deleted")
    if u.get("blocked", False):
        raise HTTPException(status_code=403, detail="Account blocked")
    return {"uid": uid, "email": decoded.get("email"), "role": u.get("role"), "display_name": u.get("display_name", u.get("email").split("@")[0])}

@app.post("/update_name")
async def update_name(request: Request):
    """
    Update the user's display name in the database.
    Returns a clean JSON response with ok/message.
    """
    try:
        token = request.headers.get("Authorization")
        if not token:
            raise HTTPException(status_code=401, detail="Missing Authorization header")
        token = token.split("Bearer ")[-1].strip()

        decoded = fb_auth.verify_id_token(token)
        uid = decoded.get("uid")
        if not uid:
            raise HTTPException(status_code=400, detail="Invalid token UID")

        body = await request.json()
        new_name = body.get("name", "").strip()
        if not new_name:
            raise HTTPException(status_code=400, detail="Missing or empty 'name' field")

        users_col.update_one({"uid": uid}, {"$set": {"display_name": new_name}})
        print(f"✅ Updated display_name for {uid} to '{new_name}'")

        return {"ok": True, "message": "Display name updated successfully", "name": new_name}

    except Exception as e:
        print(f"⚠️ update_name failed: {e}")
        raise HTTPException(status_code=500, detail=f"Update failed: {e}")


# Analyze endpoint (main)
@app.post("/analyze")
async def analyze(prompt_in: PromptInput, request: Request):
    # Verify (optional)
    uid = None
    email = "anonymous"
    try:
        decoded = await verify_user(request)
        uid = decoded.get("uid")
        email = decoded.get("email", "unknown")
    except HTTPException:
        decoded = None

    # Check blacklist
    if uid:
        black = blacklist_col.find_one({"email": email})
        if black:
            raise HTTPException(status_code=403, detail="Account blacklisted. Contact admin.")

    # Quota
    if uid:
        ok, msg = check_and_consume_quota(uid)
        if not ok:
            raise HTTPException(status_code=429, detail=msg)

    original = (prompt_in.prompt or "").strip()
    normalized = hf_normalize_text(original)
    cleaned = re.sub(r'\s+', ' ', normalized).strip()

    # classification
    candidate_labels = ["safe", "suspicious", "malicious"]
    label = "safe"
    confidence = 0.0
    hf_raw = None
    try:
        best_label, best_score, hf_raw = hf_zero_shot_classify(cleaned, candidate_labels)
        label, confidence = best_label, best_score
    except Exception as e:
        # heuristics - expanded list
        lower = cleaned.lower()
        malicious_kws = [
            "bypass safety", "ignore safety", "bypass", "disable safety", "disable filters",
            "jailbreak", "override safety", "admin password", "root access", "dump database",
            "download /etc/passwd", "kill -9", "shutdown -h now", "format c:", "delete /", "rm -rf /",
            "steal token", "secret key", "api_key", "private_key", "exploit", "payload", "execute command",
            "harm", "injure", "kill", "assassinate", "bomb", "weaponize", "poison"
        ]
        suspicious_kws = ["roleplay", "act as", "system prompt", "hidden instructions", "reveal instructions", "how to bypass"]
        if any(k in lower for k in malicious_kws):
            label, confidence = "malicious", 0.99
        elif any(k in lower for k in suspicious_kws):
            label, confidence = "suspicious", 0.7
        else:
            label, confidence = "safe", 0.5

    # If malicious -> immediate block/flag + notify

    if label == "malicious":
        if uid:
            # increment fault, flag and notify admin/manager
            flag_user_and_notify(uid, email, "malicious prompt detected (heuristic/zero-shot)", original, severity="malicious")
            # optionally auto-block user immediately if severity high — here we flag and also block to be safe
            block_user(uid, actor_role="system", actor_uid="system")
        logs.insert_one({
            "user_email": email, "uid": uid, "original": original, "cleaned": cleaned,
            "label": label, "confidence": confidence, "action": "blocked_immediate", "timestamp": datetime.utcnow()
        })
        raise HTTPException(status_code=403, detail="Blocked: malicious prompt detected.")

    # If suspicious -> sanitize and require confirm
    sanitized_text = None
    redaction_count = 0
    if label == "suspicious":
        sanitized_text, redaction_count = sanitize_text(cleaned)
        if not prompt_in.confirm:
            logs.insert_one({
                "user_email": email, "uid": uid, "original": original, "cleaned": cleaned,
                "sanitized": sanitized_text, "label": label, "confidence": confidence,
                "action": "require_confirm", "redactions": redaction_count, "hf_raw": hf_raw, "timestamp": datetime.utcnow()
            })
            # increase fault counters mildly
            if uid:
                users_col.update_one({"uid": uid}, {"$inc": {"fault_count": 1}, "$set": {"flagged": True}})
            return {"status": "require_confirm", "message": "Prompt classified as suspicious. Send again with confirm=true to proceed.", "sanitized": sanitized_text, "label": label, "confidence": f"{confidence*100:.2f}%"}

    # Safe or confirmed suspicious -> forward to Gemini (or fallback)
    forward_text = sanitized_text if (label == "suspicious" and prompt_in.confirm) else cleaned
    action = "analyzed_local"
    generation = None
    try:
        # Prefer Gemini if available, else return an analyzed_local placeholder
        if GEMINI_API_KEY:
            generation = gemini_generate(forward_text)
            action = "forwarded_gemini"
        else:
            action = "analyzed_local"
            generation = "No external model configured."
    except Exception as e:
        generation = {"error": str(e)}
        action = "forward_failed"

    # final logging, update fault_percent if needed
    if uid:
        u = users_col.find_one({"uid": uid})
        total = u.get("total_prompts", 1)
        faults = u.get("fault_count", 0)
        try:
            fault_percent = round((faults / total) * 100, 2) if total else 0.0
        except Exception:
            fault_percent = 0.0
        users_col.update_one({"uid": uid}, {"$set": {"fault_percent": fault_percent}})

    logs.insert_one({
        "user_email": email, "uid": uid, "original": original, "cleaned": cleaned,
        "sanitized": sanitized_text, "label": label, "confidence": confidence,
        "action": action, "hf_raw": hf_raw, "generation_preview": (generation[:400] if isinstance(generation, str) else str(generation)[:400]),
        "timestamp": datetime.utcnow()
    })

    return {"status": "ok", "label": label, "confidence": f"{confidence*100:.2f}%", "action": action, "generation": generation, "sanitized": sanitized_text}

# -------------------------
# User endpoints
# -------------------------
@app.get("/user/history")
async def user_history(request: Request, limit: int = Query(20, ge=1, le=200)):
    decoded = await verify_user(request)
    uid = decoded.get("uid")
    docs = list(logs.find({"uid": uid}).sort("timestamp", DESCENDING).limit(limit))
    for d in docs:
        d["_id"] = str(d["_id"])
        if isinstance(d.get("timestamp"), datetime):
            d["timestamp"] = d["timestamp"].isoformat()
    return {"count": len(docs), "logs": docs}

# -------------------------
# Manager endpoints
# -------------------------
@app.get("/manager/flagged_users")
async def manager_flagged(request: Request, limit: int = Query(200, ge=1, le=2000)):
    decoded = await verify_user(request)
    uid = decoded.get("uid")
    u = users_col.find_one({"uid": uid})
    if not u or u.get("role") not in ("manager", "admin"):
        raise HTTPException(status_code=403, detail="Manager access required")
    docs = list(users_col.find({"flagged": True, "deleted": {"$ne": True}}).sort("fault_percent", DESCENDING).limit(limit))
    out = []
    for d in docs:
        d.pop("_id", None)
        out.append(d)
    return {"count": len(out), "users": out}

@app.post("/manager/block_user")
async def manager_block_user(payload: dict = Body(...), request: Request = None):
    decoded = await verify_user(request)
    uid = decoded.get("uid")
    actor = users_col.find_one({"uid": uid})
    
    if not actor or actor.get("role") != "manager":
        raise HTTPException(status_code=403, detail="Only managers can block/unblock users")
    
    target_uid = payload.get("uid")
    if not target_uid:
        raise HTTPException(status_code=400, detail="uid required")
    
    # cannot block/unblock admins
    target_user = users_col.find_one({"uid": target_uid})
    if not target_user:
        raise HTTPException(status_code=404, detail="Target user not found")
    if target_user.get("role") == "admin":
        raise HTTPException(status_code=403, detail="Cannot block/unblock admin")
    
    # Toggle blocked status
    new_status = not target_user.get("blocked", False)
    users_col.update_one({"uid": target_uid}, {"$set": {"blocked": new_status}})
    
    # Optional: log who blocked/unblocked
    block_user(target_uid, actor_role="manager", actor_uid=uid, blocked=new_status)
    
    return {"ok": True, "uid": target_uid, "blocked": new_status}

@app.get("/manager/users_to_block")
async def manager_users_to_block(request: Request):
    decoded = await verify_user(request)
    uid = decoded.get("uid")
    actor = users_col.find_one({"uid": uid})
    
    if not actor or actor.get("role") != "manager":
        raise HTTPException(status_code=403, detail="Only managers can view users to block")
    
    # Fetch all users except admins
    users = list(users_col.find({"role": {"$ne": "admin"}}))
    
    # Return relevant info
    result = [
        {
            "uid": u["uid"],
            "name": u.get("name", "-"),
            "email": u.get("email", "-"),
            "blocked": u.get("blocked", False),
        }
        for u in users
    ]
    
    return {"users": result}


# -------------------------
# Admin endpoints
# -------------------------

@app.get("/admin/chats")
async def get_all_chats(request: Request):
    """Return all chat logs from all users and managers (admins only)."""
    decoded = await verify_user(request)
    uid = decoded.get("uid")

    # ✅ Verify admin role from MongoDB
    user_doc = users_col.find_one({"uid": uid})
    if not user_doc or user_doc.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admins only")

    chats = []
    for log in logs.find(
        {"label": {"$in": ["safe", "sanitized"]}},
        {"_id": 0}
    ).sort("timestamp", -1):
        chats.append({
            "uid": log.get("uid"),
            "email": log.get("email"),
            "label": log.get("label"),
            "confidence": log.get("confidence"),
            "prompt": log.get("original"),
            "response": log.get("generation_preview", "")[:300],
            "timestamp": log.get("timestamp")
        })

    return {"chats": chats}

@app.get("/admin/members")
async def get_all_members(request: Request):
    """Return all users, managers, and admins with chat summaries."""
    decoded = await verify_user(request)
    uid = decoded.get("uid")

    # ✅ Get role from DB instead of token
    user_doc = users_col.find_one({"uid": uid})
    if not user_doc or user_doc.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admins only")

    members = []
    for user in users_col.find({}, {"_id": 0}):
        chat_logs = list(logs.find(
            {"uid": user["uid"], "label": {"$in": ["safe", "sanitized"]}},
            {"_id": 0, "original": 1, "generation_preview": 1, "timestamp": 1, "label": 1}
        ).sort("timestamp", -1))

        members.append({
            "uid": user.get("uid"),
            "email": user.get("email"),
            "role": user.get("role"),
            "display_name": user.get("display_name", ""),
            "fault_percent": user.get("fault_percent", 0),
            "total_prompts": user.get("total_prompts", 0),
            "blocked": user.get("blocked", False),
            "deleted": user.get("deleted", False),
            "flagged": user.get("flagged", False),
            "recent_chats": chat_logs[:5]
        })

    return {"members": members}

# ==========================================================
# ✅ Admin Manage Controls
# ==========================================================
@app.get("/admin/manage_overview")
async def admin_manage_overview(request: Request):
    """Return all users (non-deleted) and blacklisted emails for admin dashboard."""
    decoded = await verify_user(request)
    uid = decoded.get("uid")
    actor = users_col.find_one({"uid": uid})
    if not actor or actor.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin required")

    users = list(
        users_col.find({"deleted": {"$ne": True}}, {"_id": 0})
        .sort("fault_percent", DESCENDING)
    )
    blacklist = list(blacklist_col.find({}, {"_id": 0}))
    return {"users": users, "blacklist": blacklist}


@app.post("/admin/block_user")
async def admin_block_user(payload: dict = Body(...), request: Request = None):
    decoded = await verify_user(request)
    uid = decoded.get("uid")
    actor = users_col.find_one({"uid": uid})
    if not actor or actor.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin required")

    target_uid = payload.get("uid")
    if not target_uid:
        raise HTTPException(status_code=400, detail="uid required")

    users_col.update_one({"uid": target_uid}, {"$set": {"blocked": True}})
    logs.insert_one({
        "action": "blocked_by_admin",
        "uid": target_uid,
        "by_uid": uid,
        "timestamp": datetime.utcnow()
    })
    return {"ok": True, "message": f"User {target_uid} blocked."}


@app.post("/admin/unblock_user")
async def admin_unblock_user(payload: dict = Body(...), request: Request = None):
    decoded = await verify_user(request)
    uid = decoded.get("uid")
    actor = users_col.find_one({"uid": uid})
    if not actor or actor.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin required")

    target_uid = payload.get("uid")
    if not target_uid:
        raise HTTPException(status_code=400, detail="uid required")

    users_col.update_one({"uid": target_uid}, {"$set": {"blocked": False}})
    logs.insert_one({
        "action": "unblocked_by_admin",
        "uid": target_uid,
        "by_uid": uid,
        "timestamp": datetime.utcnow()
    })
    return {"ok": True, "message": f"User {target_uid} unblocked."}


@app.post("/admin/delete_user")
async def admin_delete_user(payload: dict = Body(...), request: Request = None):
    decoded = await verify_user(request)
    uid = decoded.get("uid")
    actor = users_col.find_one({"uid": uid})
    if not actor or actor.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin required")

    target_uid = payload.get("uid")
    if not target_uid:
        raise HTTPException(status_code=400, detail="uid required")

    users_col.update_one({"uid": target_uid}, {"$set": {"deleted": True, "blocked": True}})
    logs.insert_one({
        "action": "deleted_by_admin",
        "uid": target_uid,
        "by_uid": uid,
        "timestamp": datetime.utcnow()
    })
    return {"ok": True, "message": f"User {target_uid} deleted."}


@app.post("/admin/blacklist_add")
async def admin_blacklist_add(payload: dict = Body(...), request: Request = None):
    decoded = await verify_user(request)
    uid = decoded.get("uid")
    actor = users_col.find_one({"uid": uid})
    if not actor or actor.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin required")

    email = payload.get("email")
    if not email:
        raise HTTPException(status_code=400, detail="email required")

    blacklist_col.insert_one({
        "email": email,
        "added_by": uid,
        "timestamp": datetime.utcnow()
    })
    logs.insert_one({
        "action": "blacklist_add",
        "email": email,
        "by_uid": uid,
        "timestamp": datetime.utcnow()
    })
    return {"ok": True, "message": f"{email} blacklisted."}


@app.post("/admin/blacklist_remove")
async def admin_blacklist_remove(payload: dict = Body(...), request: Request = None):
    decoded = await verify_user(request)
    uid = decoded.get("uid")
    actor = users_col.find_one({"uid": uid})
    if not actor or actor.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin required")

    email = payload.get("email")
    if not email:
        raise HTTPException(status_code=400, detail="email required")

    blacklist_col.delete_many({"email": email})
    logs.insert_one({
        "action": "blacklist_remove",
        "email": email,
        "by_uid": uid,
        "timestamp": datetime.utcnow()
    })
    return {"ok": True, "message": f"{email} removed from blacklist."}

# notifications read/list
@app.get("/notifications")
async def list_notifications(request: Request, limit: int = Query(100, ge=1, le=1000)):
    decoded = await verify_user(request)
    uid = decoded.get("uid")
    u = users_col.find_one({"uid": uid})
    # managers/admins can see all notifications; users only their own
    if u.get("role") in ("admin", "manager"):
        docs = list(notifications.find().sort("timestamp", DESCENDING).limit(limit))
    else:
        docs = list(notifications.find({"uid": uid}).sort("timestamp", DESCENDING).limit(limit))
    out = []
    for d in docs:
        d["_id"] = str(d["_id"])
        if isinstance(d.get("timestamp"), datetime):
            d["timestamp"] = d["timestamp"].isoformat()
        out.append(d)
    return {"count": len(out), "notifications": out}

# admin logs + users endpoints (keep existing)

@app.get("/admin/logs")
def get_all_logs(current_user: dict = Depends(verify_admin_token)):
    """Fetch combined system logs and flagged notifications."""

    try:
        logs_cursor = db.logs.find().sort("timestamp", -1)
        notifications_cursor = db.notifications.find().sort("timestamp", -1)

        logs = []

        # Process system and user/manager logs
        for log in logs_cursor:
            logs.append({
                "_id": str(log.get("_id", ObjectId())),
                "actor_uid": log.get("by_uid") or "-",
                "target_uid": log.get("uid") or "-",
                "target_email": log.get("user_email") or "-",
                "action": log.get("action") or "UNKNOWN_ACTION",
                "timestamp": log.get("timestamp").isoformat() if isinstance(log.get("timestamp"), datetime) else str(log.get("timestamp")),
                "actor_role": (
                    "manager"
                    if log.get("action") and "manager" in log.get("action")
                    else "user"
                ),
                "details": (
                    f"Action: {log.get('action')} by {log.get('by_uid')}"
                    if log.get("action")
                    else "-"
                ),
            })

        # Process flagged notifications
        for notif in notifications_cursor:
            logs.append({
                "_id": str(notif.get("_id", ObjectId())),
                "actor_uid": notif.get("uid") or "-",
                "target_email": notif.get("email") or "-",
                "action": "FLAGGED_MALICIOUS",
                "timestamp": notif.get("timestamp").isoformat() if isinstance(notif.get("timestamp"), datetime) else str(notif.get("timestamp")),
                "actor_role": "system",
                "details": f"{notif.get('reason', '')} — {notif.get('prompt', '')}",
            })

        # Sort by newest first
        logs.sort(
            key=lambda x: datetime.fromisoformat(x["timestamp"].replace("Z", "")) 
            if x.get("timestamp") else datetime.min,
            reverse=True,
        )

        return {"logs": logs}

    except Exception as e:
        print("Error fetching logs:", e)
        raise HTTPException(status_code=500, detail="Failed to fetch logs")
@app.get("/admin/users")
def admin_list_users(request: Request):
    decoded = fb_auth.verify_id_token(request.headers.get("Authorization", "").split()[1])
    uid = decoded.get("uid")
    u = users_col.find_one({"uid": uid})
    if not u or u.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin required")
    docs = list(users_col.find({}))
    out = []
    for doc in docs:
        doc.pop("_id", None)
        out.append(doc)
    return {"count": len(out), "users": out}

# manager view users (no admins)
@app.get("/manager/users")
async def manager_list_users(request: Request):
    # verify the requester
    decoded = await verify_user(request)
    uid = decoded.get("uid")
    requester = users_col.find_one({"uid": uid})
    if not requester or requester.get("role") != "manager":
        raise HTTPException(status_code=403, detail="Manager access required")
    
    # fetch all non-admin users
    docs = list(users_col.find({"role": {"$ne": "admin"}}))
    
    # convert _id to string
    for doc in docs:
        doc["_id"] = str(doc["_id"])
    
    return {"count": len(docs), "users": docs}    # Only managers can access
    if current_user["role"] != "manager":
        raise HTTPException(status_code=401, detail="Unauthorized")

    # Fetch all users from MongoDB (or Firebase if you switch later)
    users = list(db.users.find({}, {"_id": 0}))  # exclude MongoDB _id
    return {"users": users}

# -------------------------
# Final
# -------------------------
if __name__ == "__main__":
    print("Run with uvicorn main:app --reload")
