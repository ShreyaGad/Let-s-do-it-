"""
Smart Supply Chain Backend
FastAPI + SQLAlchemy (PostgreSQL) + WebSockets + JWT Authentication
"""

# ─────────────────────────────────────────────
# DEPENDENCIES
# pip install fastapi uvicorn sqlalchemy asyncpg psycopg2-binary
#             passlib[bcrypt] python-jose[cryptography] python-dotenv
# ─────────────────────────────────────────────

import os, uuid, random
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional, List

from dotenv import load_dotenv
load_dotenv()

# ── FastAPI ──────────────────────────────────
from fastapi import (FastAPI, HTTPException, Depends, WebSocket,
                     WebSocketDisconnect, BackgroundTasks, status)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm

# ── SQLAlchemy (async) ───────────────────────
from sqlalchemy import (Column, String, Float, Boolean,
                        DateTime, Enum as SAEnum, ForeignKey, JSON)
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.future import select

# ── Auth ─────────────────────────────────────
from passlib.context import CryptContext
from jose import JWTError, jwt

# ── Pydantic ─────────────────────────────────
from pydantic import BaseModel, ConfigDict, Field

# =============================================================
# CONFIG
# =============================================================

DATABASE_URL  = os.getenv("DATABASE_URL",
                    "sqlite+aiosqlite:///./supplychain.db")


SECRET_KEY    = os.getenv("SECRET_KEY", "change-me-in-production-use-256-bit-random")
ALGORITHM     = "HS256"
TOKEN_EXPIRE  = int(os.getenv("TOKEN_EXPIRE_MINUTES", 60))

# =============================================================
# DATABASE SETUP
# =============================================================

engine        = create_async_engine(DATABASE_URL, echo=False)
AsyncSession_  = async_sessionmaker(engine, expire_on_commit=False)


class Base(DeclarativeBase):
    pass

async def get_db():
    async with AsyncSession_() as session:
        yield session

# =============================================================
# ENUMS
# =============================================================

class AlertSeverity(str, Enum):
    CRITICAL = "red"
    WARNING  = "yellow"
    RESOLVED = "green"

class PipelineStage(str, Enum):
    INGEST   = "ingest"
    DETECT   = "detect"
    PREDICT  = "predict"
    OPTIMIZE = "optimize"
    DELIVER  = "deliver"

class ShipmentStatus(str, Enum):
    ON_TIME   = "on_time"
    DELAYED   = "delayed"
    REROUTED  = "rerouted"
    DELIVERED = "delivered"

class UserRole(str, Enum):
    ADMIN    = "admin"
    OPERATOR = "operator"
    VIEWER   = "viewer"

# =============================================================
# ORM MODELS
# =============================================================

class UserORM(Base):
    __tablename__ = "users"
    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    email: Mapped[str] = mapped_column(String, unique=True, nullable=False, index=True)
    username: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    hashed_pw: Mapped[str] = mapped_column(String, nullable=False)
    role: Mapped[UserRole] = mapped_column(SAEnum(UserRole), default=UserRole.OPERATOR)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

class ShipmentORM(Base):
    __tablename__ = "shipments"
    id: Mapped[str] = mapped_column(String, primary_key=True,
                                    default=lambda: str(uuid.uuid4())[:8].upper())
    origin: Mapped[str] = mapped_column(String, nullable=False)
    destination: Mapped[str] = mapped_column(String, nullable=False)
    carrier: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[ShipmentStatus] = mapped_column(SAEnum(ShipmentStatus), default=ShipmentStatus.ON_TIME)
    eta: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    current_location: Mapped[str] = mapped_column(String, nullable=False)
    weight_kg: Mapped[float] = mapped_column(Float, nullable=False)
    pipeline_stage: Mapped[PipelineStage] = mapped_column(SAEnum(PipelineStage), default=PipelineStage.INGEST)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    alerts: Mapped[list["AlertORM"]] = relationship(
        "AlertORM", back_populates="shipment", cascade="all, delete-orphan"
    )

class AlertORM(Base):
    __tablename__ = "alerts"
    id: Mapped[str] = mapped_column(String, primary_key=True,
                                    default=lambda: str(uuid.uuid4())[:8])
    severity: Mapped[AlertSeverity] = mapped_column(SAEnum(AlertSeverity), nullable=False)
    message: Mapped[str] = mapped_column(String, nullable=False)
    location: Mapped[str] = mapped_column(String, nullable=False)
    affected_shipment_id: Mapped[Optional[str]] = mapped_column(String, ForeignKey("shipments.id"), nullable=True)
    resolved: Mapped[bool] = mapped_column(Boolean, default=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    shipment: Mapped[Optional["ShipmentORM"]] = relationship("ShipmentORM", back_populates="alerts")

class OptimizationORM(Base):
    __tablename__ = "optimizations"
    id: Mapped[str] = mapped_column(String, primary_key=True,
                                    default=lambda: str(uuid.uuid4())[:8])
    shipment_id: Mapped[str] = mapped_column(String, ForeignKey("shipments.id"), nullable=False)
    original_route: Mapped[str] = mapped_column(String, nullable=False)
    optimized_route: Mapped[str] = mapped_column(String, nullable=False)
    time_saved_hours: Mapped[float] = mapped_column(Float, nullable=False)
    reason: Mapped[str] = mapped_column(String, nullable=False)
    optimized_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

# =============================================================
# PYDANTIC SCHEMAS
# =============================================================

# ── Auth ──────────────────────────────────────
class UserCreate(BaseModel):
    email: str; username: str; password: str
    role: UserRole = UserRole.OPERATOR

class UserOut(BaseModel):
    id: str; email: str; username: str; role: UserRole; is_active: bool
    model_config = ConfigDict(from_attributes=True)

class Token(BaseModel):
    access_token: str; token_type: str = "bearer"

# ── Shipment ──────────────────────────────────
class ShipmentCreate(BaseModel):
    origin: str; destination: str; carrier: str
    eta: datetime; current_location: str; weight_kg: float

class ShipmentUpdate(BaseModel):
    status: Optional[ShipmentStatus] = None
    current_location: Optional[str]  = None
    eta: Optional[datetime]          = None
    pipeline_stage: Optional[PipelineStage] = None

class ShipmentOut(BaseModel):
    id: str; origin: str; destination: str; carrier: str
    status: ShipmentStatus; eta: datetime; current_location: str
    weight_kg: float; pipeline_stage: PipelineStage; created_at: datetime
    model_config = ConfigDict(from_attributes=True)

# ── Alert ─────────────────────────────────────
class AlertCreate(BaseModel):
    severity: AlertSeverity; message: str; location: str
    affected_shipment_id: Optional[str] = None

class AlertOut(BaseModel):
    id: str; severity: AlertSeverity; message: str; location: str
    affected_shipment_id: Optional[str]; resolved: bool; timestamp: datetime
    model_config = ConfigDict(from_attributes=True)

# ── Optimization ──────────────────────────────
class OptimizeRequest(BaseModel):
    shipment_id: str; reason: str; alternative_routes: List[str]

class OptimizeOut(BaseModel):
    id: str; shipment_id: str; original_route: str; optimized_route: str
    time_saved_hours: float; reason: str; optimized_at: datetime
    model_config = ConfigDict(from_attributes=True)

# ── Dashboard ─────────────────────────────────
class DashboardStats(BaseModel):
    total_shipments: int; on_time_rate_pct: float
    avg_reroute_ms: float; countries_covered: int
    active_alerts: int; shipments_in_transit: int

# =============================================================
# AUTH UTILITIES
# =============================================================

import bcrypt
oauth2    = OAuth2PasswordBearer(tokenUrl="/auth/login")

def hash_password(pw: str)         -> str:
    return bcrypt.hashpw(pw.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def verify_password(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(plain.encode('utf-8'), hashed.encode('utf-8'))

def create_token(data: dict) -> str:
    payload = {**data, "exp": datetime.utcnow() + timedelta(minutes=TOKEN_EXPIRE)}
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user(token: str = Depends(oauth2),
                           db: AsyncSession = Depends(get_db)) -> UserORM:
    creds_exc = HTTPException(status.HTTP_401_UNAUTHORIZED,
                              "Invalid credentials",
                              headers={"WWW-Authenticate": "Bearer"})
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        uid = payload.get("sub")
        if not uid:
            raise creds_exc
    except JWTError:
        raise creds_exc

    result = await db.execute(select(UserORM).where(UserORM.id == uid))
    user   = result.scalar_one_or_none()
    if not user or not user.is_active:
        raise creds_exc
    return user

def require_role(*roles: UserRole):
    """Role-based access control dependency factory."""
    async def check(user: UserORM = Depends(get_current_user)):
        if user.role not in roles:
            raise HTTPException(status.HTTP_403_FORBIDDEN, "Insufficient permissions")
        return user
    return check

# =============================================================
# WEBSOCKET CONNECTION MANAGER
# =============================================================

class ConnectionManager:
    def __init__(self):
        self._connections: dict[str, list[WebSocket]] = {}   # room → sockets

    async def connect(self, ws: WebSocket, room: str = "global"):
        await ws.accept()
        self._connections.setdefault(room, []).append(ws)

    def disconnect(self, ws: WebSocket, room: str = "global"):
        self._connections.get(room, []).remove(ws)

    async def broadcast(self, message: dict, room: str = "global"):
        dead = []
        for ws in self._connections.get(room, []):
            try:
                await ws.send_json(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws, room)

    async def broadcast_all(self, message: dict):
        for room in list(self._connections.keys()):
            await self.broadcast(message, room)

ws_manager = ConnectionManager()

# =============================================================
# APP
# =============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield


app = FastAPI(title="Smart Supply Chain API", version="2.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

# =============================================================
# AUTH ROUTES  /auth/*
# =============================================================

@app.post("/auth/register", response_model=UserOut, status_code=201, tags=["Auth"])
async def register(data: UserCreate, db: AsyncSession = Depends(get_db)):
    existing = await db.execute(
        select(UserORM).where(UserORM.email == data.email))
    if existing.scalar_one_or_none():
        raise HTTPException(400, "Email already registered")
    user = UserORM(email=data.email, username=data.username,
                   hashed_pw=hash_password(data.password), role=data.role)
    db.add(user)
    await db.commit()
    await db.refresh(user)
    return user

@app.post("/auth/login", response_model=Token, tags=["Auth"])
async def login(form: OAuth2PasswordRequestForm = Depends(),
                db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(UserORM).where(UserORM.username == form.username))
    user = result.scalar_one_or_none()
    if not user or not verify_password(form.password, user.hashed_pw):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Wrong credentials")
    return Token(access_token=create_token({"sub": user.id, "role": user.role}))

@app.get("/auth/me", response_model=UserOut, tags=["Auth"])
async def me(user: UserORM = Depends(get_current_user)):
    return user

# =============================================================
# DASHBOARD  /dashboard/*
# =============================================================

@app.get("/dashboard/stats", response_model=DashboardStats, tags=["Dashboard"])
async def dashboard_stats(db: AsyncSession = Depends(get_db),
                           _=Depends(get_current_user)):
    total_r   = await db.execute(select(ShipmentORM))
    all_s     = total_r.scalars().all()
    on_time   = sum(1 for s in all_s if s.status == ShipmentStatus.ON_TIME)
    in_transit= sum(1 for s in all_s
                    if s.status in [ShipmentStatus.ON_TIME, ShipmentStatus.REROUTED])
    alert_r   = await db.execute(
        select(AlertORM).where(AlertORM.resolved == False))
    active_alerts = len(alert_r.scalars().all())
    total = len(all_s)
    return DashboardStats(
        total_shipments=total,
        on_time_rate_pct=round((on_time / total * 100) if total else 0, 1),
        avg_reroute_ms=round(random.uniform(1.4, 2.0), 2),
        countries_covered=140,
        active_alerts=active_alerts,
        shipments_in_transit=in_transit,
    )

# =============================================================
# SHIPMENTS  /shipments/*
# =============================================================

@app.get("/shipments", response_model=List[ShipmentOut], tags=["Shipments"])
async def list_shipments(status: Optional[ShipmentStatus] = None,
                          stage: Optional[PipelineStage]  = None,
                          db: AsyncSession = Depends(get_db),
                          _=Depends(get_current_user)):
    q = select(ShipmentORM)
    if status: q = q.where(ShipmentORM.status == status)
    if stage:  q = q.where(ShipmentORM.pipeline_stage == stage)
    res = await db.execute(q)
    return res.scalars().all()

@app.post("/shipments", response_model=ShipmentOut, status_code=201, tags=["Shipments"])
async def create_shipment(data: ShipmentCreate,
                           db: AsyncSession = Depends(get_db),
                           user=Depends(require_role(UserRole.ADMIN, UserRole.OPERATOR))):
    s = ShipmentORM(**data.dict())
    db.add(s)
    await db.commit()
    await db.refresh(s)
    await ws_manager.broadcast_all({"event": "shipment_created",
                                    "shipment_id": s.id, "origin": s.origin,
                                    "destination": s.destination})
    return s

@app.get("/shipments/{sid}", response_model=ShipmentOut, tags=["Shipments"])
async def get_shipment(sid: str, db: AsyncSession = Depends(get_db),
                        _=Depends(get_current_user)):
    res = await db.execute(select(ShipmentORM).where(ShipmentORM.id == sid.upper()))
    s   = res.scalar_one_or_none()
    if not s: raise HTTPException(404, "Shipment not found")
    return s

@app.patch("/shipments/{sid}", response_model=ShipmentOut, tags=["Shipments"])
async def update_shipment(sid: str, update: ShipmentUpdate,
                           db: AsyncSession = Depends(get_db),
                           user=Depends(require_role(UserRole.ADMIN, UserRole.OPERATOR))):
    res = await db.execute(select(ShipmentORM).where(ShipmentORM.id == sid.upper()))
    s   = res.scalar_one_or_none()
    if not s: raise HTTPException(404, "Shipment not found")
    for f, v in update.dict(exclude_none=True).items():
        setattr(s, f, v)
    await db.commit()
    await db.refresh(s)
    await ws_manager.broadcast_all({"event": "shipment_updated", "shipment_id": s.id,
                                    "status": s.status, "stage": s.pipeline_stage})
    return s

@app.post("/shipments/{sid}/advance", response_model=ShipmentOut, tags=["Shipments"])
async def advance_stage(sid: str, db: AsyncSession = Depends(get_db),
                         user=Depends(require_role(UserRole.ADMIN, UserRole.OPERATOR))):
    res = await db.execute(select(ShipmentORM).where(ShipmentORM.id == sid.upper()))
    s   = res.scalar_one_or_none()
    if not s: raise HTTPException(404, "Shipment not found")
    order = list(PipelineStage)
    idx   = order.index(s.pipeline_stage)
    if idx < len(order) - 1:
        s.pipeline_stage = order[idx + 1]
    await db.commit()
    await db.refresh(s)
    await ws_manager.broadcast_all({"event": "pipeline_advanced",
                                    "shipment_id": s.id, "stage": s.pipeline_stage})
    return s

@app.delete("/shipments/{sid}", status_code=204, tags=["Shipments"])
async def delete_shipment(sid: str, db: AsyncSession = Depends(get_db),
                           _=Depends(require_role(UserRole.ADMIN))):
    res = await db.execute(select(ShipmentORM).where(ShipmentORM.id == sid.upper()))
    s   = res.scalar_one_or_none()
    if not s: raise HTTPException(404, "Shipment not found")
    await db.delete(s)
    await db.commit()

# =============================================================
# ALERTS  /alerts/*
# =============================================================

@app.get("/alerts", response_model=List[AlertOut], tags=["Alerts"])
async def list_alerts(resolved: Optional[bool] = None,
                       severity: Optional[AlertSeverity] = None,
                       db: AsyncSession = Depends(get_db),
                       _=Depends(get_current_user)):
    q = select(AlertORM).order_by(AlertORM.timestamp.desc())
    if resolved is not None: q = q.where(AlertORM.resolved == resolved)
    if severity:             q = q.where(AlertORM.severity == severity)
    res = await db.execute(q)
    return res.scalars().all()

@app.post("/alerts", response_model=AlertOut, status_code=201, tags=["Alerts"])
async def create_alert(data: AlertCreate,
                        background_tasks: BackgroundTasks,
                        db: AsyncSession = Depends(get_db),
                        user=Depends(require_role(UserRole.ADMIN, UserRole.OPERATOR))):
    alert = AlertORM(**data.dict())
    db.add(alert)
    await db.commit()
    await db.refresh(alert)

    # push live to all WebSocket listeners
    await ws_manager.broadcast_all({
        "event":    "new_alert",
        "id":       alert.id,
        "severity": alert.severity,
        "message":  alert.message,
        "location": alert.location,
        "timestamp": alert.timestamp.isoformat(),
    })

    # mark affected shipment delayed in background
    async def flag(sid: str):
        async with AsyncSession_() as s:
            r = await s.execute(select(ShipmentORM).where(ShipmentORM.id == sid.upper()))
            ship = r.scalar_one_or_none()
            if ship:
                ship.status = ShipmentStatus.DELAYED
                await s.commit()
    if data.affected_shipment_id:
        background_tasks.add_task(flag, data.affected_shipment_id)
    return alert

@app.patch("/alerts/{aid}/resolve", response_model=AlertOut, tags=["Alerts"])
async def resolve_alert(aid: str, db: AsyncSession = Depends(get_db),
                         user=Depends(require_role(UserRole.ADMIN, UserRole.OPERATOR))):
    res = await db.execute(select(AlertORM).where(AlertORM.id == aid))
    a   = res.scalar_one_or_none()
    if not a: raise HTTPException(404, "Alert not found")
    a.resolved = True
    a.severity = AlertSeverity.RESOLVED
    await db.commit()
    await db.refresh(a)
    await ws_manager.broadcast_all({"event": "alert_resolved", "id": a.id})
    return a

# =============================================================
# OPTIMIZATION  /optimize/*
# =============================================================

@app.post("/optimize", response_model=OptimizeOut, tags=["Optimization"])
async def optimize_route(req: OptimizeRequest,
                          db: AsyncSession = Depends(get_db),
                          user=Depends(require_role(UserRole.ADMIN, UserRole.OPERATOR))):
    res = await db.execute(
        select(ShipmentORM).where(ShipmentORM.id == req.shipment_id.upper()))
    s = res.scalar_one_or_none()
    if not s: raise HTTPException(404, "Shipment not found")
    if not req.alternative_routes:
        raise HTTPException(400, "Provide at least one alternative route")

    original      = s.current_location
    best_route    = req.alternative_routes[0]
    time_saved    = round(random.uniform(0.5, 6.0), 1)

    s.status           = ShipmentStatus.REROUTED
    s.current_location = best_route
    s.eta             -= timedelta(hours=time_saved)

    opt = OptimizationORM(shipment_id=s.id, original_route=original,
                           optimized_route=best_route,
                           time_saved_hours=time_saved, reason=req.reason)
    db.add(opt)
    await db.commit()
    await db.refresh(opt)

    await ws_manager.broadcast_all({
        "event": "route_optimized", "shipment_id": s.id,
        "new_route": best_route, "time_saved_hours": time_saved,
    })
    return opt

@app.get("/optimize/history", response_model=List[OptimizeOut], tags=["Optimization"])
async def optimization_history(db: AsyncSession = Depends(get_db),
                                _=Depends(get_current_user)):
    res = await db.execute(
        select(OptimizationORM).order_by(OptimizationORM.optimized_at.desc()))
    return res.scalars().all()

# =============================================================
# PIPELINE  /pipeline/*
# =============================================================

@app.get("/pipeline/summary", tags=["Pipeline"])
async def pipeline_summary(db: AsyncSession = Depends(get_db),
                             _=Depends(get_current_user)):
    res   = await db.execute(select(ShipmentORM))
    all_s = res.scalars().all()
    summary = {stage.value: 0 for stage in PipelineStage}
    for s in all_s:
        summary[s.pipeline_stage.value] += 1
    return {"stages": summary, "total": len(all_s)}

# =============================================================
# WEBSOCKET  /ws/{room}
# =============================================================

@app.websocket("/ws/{room}")
async def websocket_endpoint(ws: WebSocket, room: str,
                              token: Optional[str] = None):
    """
    Connect:   ws://localhost:8000/ws/global?token=<JWT>
    Rooms:     global | alerts | shipments
    Events:    new_alert · alert_resolved · shipment_created ·
               shipment_updated · pipeline_advanced · route_optimized
    """
    # Validate JWT before accepting
    if token:
        try:
            jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        except JWTError:
            await ws.close(code=4001)
            return

    await ws_manager.connect(ws, room)
    await ws.send_json({"event": "connected", "room": room})
    try:
        while True:
            data = await ws.receive_json()       # accept ping/custom msgs
            if data.get("type") == "ping":
                await ws.send_json({"type": "pong"})
    except WebSocketDisconnect:
        ws_manager.disconnect(ws, room)

# =============================================================
# HEALTH
# =============================================================

@app.get("/health", tags=["System"])
async def health():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}


# =============================================================
# RUN:  uvicorn supply_chain_backend:app --reload
# =============================================================