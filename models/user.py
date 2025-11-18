# backend/models/user.py
from pydantic import BaseModel

class User(BaseModel):
    uid: str
    email: str
    role: str  # "user", "manager", or "admin"
    blocked: bool = False
    daily_quota_left: int = 10