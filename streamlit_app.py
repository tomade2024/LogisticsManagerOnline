# streamlit_app.py
# Run: streamlit run streamlit_app.py

import os
import uuid
import datetime as dt

import pandas as pd
import streamlit as st

# dotenv optional (kein Dependency-Zwang)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ModuleNotFoundError:
    pass

from sqlalchemy import (
    create_engine, String, Integer, BigInteger, DateTime, ForeignKey, Numeric, Boolean
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker, joinedload


# =========================================================
# Config / DB
# =========================================================
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+psycopg2://lm:lm@localhost:5432/lm")

def make_engine():
    """
    PostgreSQL bevorzugt, aber falls nicht erreichbar: SQLite-Fallback.
    Damit läuft das Spiel auch ohne Docker/DB sofort.
    """
    try:
        eng = create_engine(DATABASE_URL, pool_pre_ping=True)
        with eng.connect() as c:
            c.exec_driver_sql("SELECT 1")
        return eng
    except Exception:
        return create_engine("sqlite:///lm_demo.sqlite", future=True)

engine = make_engine()
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

class Base(DeclarativeBase):
    pass


# =========================================================
# Models
# =========================================================
class Player(Base):
    __tablename__ = "players"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    display_name: Mapped[str] = mapped_column(String, unique=True, index=True)
    money_cents: Mapped[int] = mapped_column(BigInteger, default=100000)  # 10000€
    level: Mapped[int] = mapped_column(Integer, default=1)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=dt.datetime.utcnow)

    sites = relationship("PlayerSite", back_populates="player")
    ledger = relationship("LedgerEntry", back_populates="player")


class Plot(Base):
    __tablename__ = "plots"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    name: Mapped[str] = mapped_column(String)
    price_cents: Mapped[int] = mapped_column(BigInteger)
    width: Mapped[int] = mapped_column(Integer)
    height: Mapped[int] = mapped_column(Integer)
    bonus_type: Mapped[str] = mapped_column(String, default="HIGHWAY")
    bonus_value: Mapped[float] = mapped_column(Numeric(8, 4), default=0.10)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)

    sites = relationship("PlayerSite", back_populates="plot")


class PlayerSite(Base):
    __tablename__ = "player_sites"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    player_id: Mapped[str] = mapped_column(String, ForeignKey("players.id"), unique=True, index=True)
    plot_id: Mapped[str] = mapped_column(String, ForeignKey("plots.id"), index=True)
    site_name: Mapped[str] = mapped_column(String)
    purchased_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=dt.datetime.utcnow)

    player = relationship("Player", back_populates="sites")
    plot = relationship("Plot", back_populates="sites")

    objects = relationship("PlacedObject", back_populates="site")
    shipments = relationship("InboundShipment", back_populates="site")
    orders = relationship("Order", back_populates="site")


class ObjectDef(Base):
    __tablename__ = "object_defs"
    object_type: Mapped[str] = mapped_column(String, primary_key=True)
    name: Mapped[str] = mapped_column(String)
    buy_cost_cents: Mapped[int] = mapped_column(BigInteger, default=0)
    w: Mapped[int] = mapped_column(Integer, default=1)
    h: Mapped[int] = mapped_column(Integer, default=1)
    rack_slots: Mapped[int] = mapped_column(Integer, default=0)


class PlacedObject(Base):
    __tablename__ = "placed_objects"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    site_id: Mapped[str] = mapped_column(String, ForeignKey("player_sites.id"), index=True)
    object_type: Mapped[str] = mapped_column(String, ForeignKey("object_defs.object_type"))
    x: Mapped[int] = mapped_column(Integer)
    y: Mapped[int] = mapped_column(Integer)
    w: Mapped[int] = mapped_column(Integer)
    h: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=dt.datetime.utcnow)

    site = relationship("PlayerSite", back_populates="objects")
    obj_def = relationship("ObjectDef")


class Item(Base):
    __tablename__ = "items"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    sku: Mapped[str] = mapped_column(String, unique=True, index=True)
    name: Mapped[str] = mapped_column(String)
    base_price_cents: Mapped[int] = mapped_column(BigInteger, default=1000)


class Inventory(Base):
    __tablename__ = "inventories"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    site_id: Mapped[str] = mapped_column(String, ForeignKey("player_sites.id"), index=True)
    item_id: Mapped[str] = mapped_column(String, ForeignKey("items.id"), index=True)
    qty: Mapped[int] = mapped_column(Integer, default=0)

    item = relationship("Item")


class InboundShipment(Base):
    __tablename__ = "inbound_shipments"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    site_id: Mapped[str] = mapped_column(String, ForeignKey("player_sites.id"), index=True)
    status: Mapped[str] = mapped_column(String, default="SCHEDULED")  # SCHEDULED, ARRIVED, DONE
    scheduled_arrival_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=dt.datetime.utcnow)
    arrived_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    accepted_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=dt.datetime.utcnow)

    site = relationship("PlayerSite", back_populates="shipments")
    lines = relationship("InboundLine", back_populates="shipment", cascade="all, delete-orphan")


class InboundLine(Base):
    __tablename__ = "inbound_lines"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    shipment_id: Mapped[str] = mapped_column(String, ForeignKey("inbound_shipments.id"), index=True)
    item_id: Mapped[str] = mapped_column(String, ForeignKey("items.id"))
    qty: Mapped[int] = mapped_column(Integer)

    shipment = relationship("InboundShipment", back_populates="lines")
    item = relationship("Item")


class Order(Base):
    __tablename__ = "orders"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    site_id: Mapped[str] = mapped_column(String, ForeignKey("player_sites.id"), index=True)
    status: Mapped[str] = mapped_column(String, default="OPEN")  # OPEN, SHIPPED
    due_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=dt.datetime.utcnow)
    reward_cents: Mapped[int] = mapped_column(BigInteger, default=5000)

    site = relationship("PlayerSite", back_populates="orders")
    lines = relationship("OrderLine", back_populates="order", cascade="all, delete-orphan")


class OrderLine(Base):
    __tablename__ = "order_lines"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    order_id: Mapped[str] = mapped_column(String, ForeignKey("orders.id"), index=True)
    item_id: Mapped[str] = mapped_column(String, ForeignKey("items.id"))
    qty: Mapped[int] = mapped_column(Integer)

    order = relationship("Order", back_populates="lines")
    item = relationship("Item")


class Employee(Base):
    __tablename__ = "employees"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    site_id: Mapped[str] = mapped_column(String, ForeignKey("player_sites.id"), index=True)
    role: Mapped[str] = mapped_column(String, default="WAREHOUSE_WORKER")
    efficiency: Mapped[float] = mapped_column(Numeric(6, 3), default=1.0)
    wage_cents_per_hour: Mapped[int] = mapped_column(BigInteger, default=1500)
    status: Mapped[str] = mapped_column(String, default="ACTIVE")


class LedgerEntry(Base):
    __tablename__ = "ledger_entries"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    player_id: Mapped[str] = mapped_column(String, ForeignKey("players.id"), index=True)
    entry_type: Mapped[str] = mapped_column(String)
    amount_cents: Mapped[int] = mapped_column(BigInteger)
    ref_type: Mapped[str | None] = mapped_column(String, nullable=True)
    ref_id: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=dt.datetime.utcnow)

    player = relationship("Player", back_populates="ledger")


# =========================================================
# DB init + seed
# =========================================================
def init_db():
    Base.metadata.create_all(bind=engine)

def ensure_seed(db):
    # Plots
    if db.query(Plot).count() == 0:
        db.add_all([
            Plot(id=str(uuid.uuid4()), name="A-01 Autobahn", price_cents=200000, width=20, height=20, bonus_type="HIGHWAY", bonus_value=0.10),
            Plot(id=str(uuid.uuid4()), name="B-02 Stadt", price_cents=260000, width=24, height=24, bonus_type="CITY", bonus_value=0.08),
            Plot(id=str(uuid.uuid4()), name="C-03 Land", price_cents=150000, width=30, height=20, bonus_type="CHEAP_LAND", bonus_value=0.00),
        ])

    # Items
    if db.query(Item).count() == 0:
        db.add_all([
            Item(id=str(uuid.uuid4()), sku="ELEC-001", name="Router", base_price_cents=8000),
            Item(id=str(uuid.uuid4()), sku="ELEC-002", name="Laptop", base_price_cents=90000),
            Item(id=str(uuid.uuid4()), sku="ELEC-003", name="Netzteil", base_price_cents=2500),
        ])

    # Object definitions
    if db.query(ObjectDef).count() == 0:
        db.add_all([
            ObjectDef(object_type="GATE_IN", name="Wareneingangstor", buy_cost_cents=15000, w=2, h=1, rack_slots=0),
            ObjectDef(object_type="GATE_OUT", name="Warenausgangstor", buy_cost_cents=15000, w=2, h=1, rack_slots=0),
            ObjectDef(object_type="RACK_STD", name="Palettenregal", buy_cost_cents=20000, w=2, h=2, rack_slots=8),
            ObjectDef(object_type="PACK_TABLE", name="Packtisch", buy_cost_cents=8000, w=2, h=1, rack_slots=0),
        ])

    db.commit()


# =========================================================
# Business Logic
# =========================================================
def get_or_create_player(db, display_name: str) -> Player:
    name = (display_name or "Player").strip()
    p = db.query(Player).filter(Player.display_name == name).first()
    if p:
        return p
    p = Player(id=str(uuid.uuid4()), display_name=name, money_cents=100000)
    db.add(p)
    db.commit()
    db.refresh(p)
    return p

def get_site(db, player_id: str):
    return db.query(PlayerSite).options(joinedload(PlayerSite.plot)).filter(PlayerSite.player_id == player_id).first()

def ledger(db, player_id: str, entry_type: str, amount_cents: int, ref_type=None, ref_id=None):
    db.add(LedgerEntry(
        id=str(uuid.uuid4()),
        player_id=player_id,
        entry_type=entry_type,
        amount_cents=amount_cents,
        ref_type=ref_type,
        ref_id=ref_id
    ))

def buy_site(db, player_id: str, plot_id: str, site_name: str):
    if get_site(db, player_id):
        raise ValueError("Spieler besitzt bereits einen Standort.")

    plot = db.query(Plot).filter(Plot.id == plot_id).first()
    if not plot:
        raise ValueError("Plot nicht gefunden.")

    player = db.query(Player).filter(Player.id == player_id).first()
    if player.money_cents < plot.price_cents:
        raise ValueError("Nicht genug Geld.")

    player.money_cents -= plot.price_cents
    site = PlayerSite(
        id=str(uuid.uuid4()),
        player_id=player_id,
        plot_id=plot_id,
        site_name=(site_name.strip() or "Mein Lager")
    )
    db.add(site)
    ledger(db, player_id, "PLOT_PURCHASE", -int(plot.price_cents), "PLOT", plot_id)
    db.commit()

def overlaps(ax, ay, aw, ah, bx, by, bw, bh):
    return not (ax + aw <= bx or bx + bw <= ax or ay + ah <= by or by + bh <= ay)

def place_object(db, player_id: str, site: PlayerSite, obj_type: str, x: int, y: int):
    obj_def = db.query(ObjectDef).filter(ObjectDef.object_type == obj_type).first()
    if not obj_def:
        raise ValueError("Objekt-Typ unbekannt.")

    # bounds
    if x < 0 or y < 0 or x + obj_def.w > site.plot.width or y + obj_def.h > site.plot.height:
        raise ValueError("Außerhalb des Rasters.")

    # collision
    existing = db.query(PlacedObject).filter(PlacedObject.site_id == site.id).all()
    for e in existing:
        if overlaps(x, y, obj_def.w, obj_def.h, e.x, e.y, e.w, e.h):
            raise ValueError("Kollision: Objekt überlappt ein bestehendes Objekt.")

    # money
    player = db.query(Player).filter(Player.id == player_id).first()
    if player.money_cents < obj_def.buy_cost_cents:
        raise ValueError("Nicht genug Geld.")

    player.money_cents -= obj_def.buy_cost_cents
    o = PlacedObject(
        id=str(uuid.uuid4()),
        site_id=site.id,
        object_type=obj_type,
        x=x, y=y,
        w=obj_def.w, h=obj_def.h
    )
    db.add(o)
    ledger(db, player_id, "COST_BUILD", -int(obj_def.buy_cost_cents), "OBJECT", o.id)
    db.commit()

def delete_object(db, player_id: str, site_id: str, object_id: str, refund_ratio: float = 0.5):
    obj = db.query(PlacedObject).filter(
        PlacedObject.id == object_id,
        PlacedObject.site_id == site_id
    ).first()
    if not obj:
        raise ValueError("Objekt nicht gefunden.")

    obj_def = db.query(ObjectDef).filter(ObjectDef.object_type == obj.object_type).first()
    refund = 0
    if obj_def and refund_ratio > 0:
        refund = int(obj_def.buy_cost_cents * refund_ratio)

    db.delete(obj)

    if refund > 0:
        player = db.query(Player).filter(Player.id == player_id).first()
        player.money_cents += refund
        ledger(db, player_id, "REFUND_BUILD", refund, "OBJECT", object_id)

    db.commit()

def compute_capacity(db, site_id: str):
    placed = db.query(PlacedObject).options(joinedload(PlacedObject.obj_def)).filter(PlacedObject.site_id == site_id).all()
    total = sum(int(p.obj_def.rack_slots) for p in placed if p.obj_def)
    used = db.query(Inventory).filter(Inventory.site_id == site_id).with_entities(Inventory.qty).all()
    used_qty = sum(int(q[0]) for q in used)
    return total, used_qty

def generate_shipment(db, site_id: str):
    now = dt.datetime.utcnow()
    item = db.query(Item).first()
    if not item:
        raise ValueError("Keine Items vorhanden.")

    s = InboundShipment(
        id=str(uuid.uuid4()),
        site_id=site_id,
        status="ARRIVED",
        scheduled_arrival_at=now,
        arrived_at=now,
        created_at=now
    )
    s.lines = [InboundLine(id=str(uuid.uuid4()), item_id=item.id, qty=5)]
    db.add(s)
    db.commit()

def accept_shipment(db, site_id: str, shipment_id: str):
    s = db.query(InboundShipment).options(joinedload(InboundShipment.lines)).filter(
        InboundShipment.id == shipment_id,
        InboundShipment.site_id == site_id
    ).first()
    if not s or s.status != "ARRIVED":
        raise ValueError("Lieferung nicht annehmbar.")

    total, used = compute_capacity(db, site_id)
    incoming = sum(ln.qty for ln in s.lines)
    if used + incoming > total:
        raise ValueError("Nicht genug Kapazität. Baue mehr Regale.")

    for ln in s.lines:
        inv = db.query(Inventory).filter(Inventory.site_id == site_id, Inventory.item_id == ln.item_id).first()
        if not inv:
            inv = Inventory(id=str(uuid.uuid4()), site_id=site_id, item_id=ln.item_id, qty=0)
            db.add(inv)
        inv.qty += ln.qty

    s.status = "DONE"
    s.accepted_at = dt.datetime.utcnow()
    db.commit()

def generate_order(db, site_id: str):
    now = dt.datetime.utcnow()
    item = db.query(Item).first()
    if not item:
        raise ValueError("Keine Items vorhanden.")

    o = Order(
        id=str(uuid.uuid4()),
        site_id=site_id,
        status="OPEN",
        due_at=now + dt.timedelta(hours=2),
        reward_cents=10000
    )
    o.lines = [OrderLine(id=str(uuid.uuid4()), item_id=item.id, qty=3)]
    db.add(o)
    db.commit()

def start_order(db, player_id: str, site_id: str, order_id: str):
    o = db.query(Order).options(joinedload(Order.lines)).filter(
        Order.id == order_id,
        Order.site_id == site_id
    ).first()
    if not o or o.status != "OPEN":
        raise ValueError("Auftrag nicht startbar.")

    for ln in o.lines:
        inv = db.query(Inventory).filter(Inventory.site_id == site_id, Inventory.item_id == ln.item_id).first()
        if not inv or inv.qty < ln.qty:
            raise ValueError("Nicht genug Bestand.")

    for ln in o.lines:
        inv = db.query(Inventory).filter(Inventory.site_id == site_id, Inventory.item_id == ln.item_id).first()
        inv.qty -= ln.qty

    o.status = "SHIPPED"

    player = db.query(Player).filter(Player.id == player_id).first()
    player.money_cents += o.reward_cents
    ledger(db, player_id, "REVENUE_ORDER", int(o.reward_cents), "ORDER", o.id)

    db.commit()

def hire_workers(db, site_id: str, n: int, wage_cents_per_hour: int):
    for _ in range(n):
        db.add(Employee(
            id=str(uuid.uuid4()),
            site_id=site_id,
            role="WAREHOUSE_WORKER",
            efficiency=1.0,
            wage_cents_per_hour=wage_cents_per_hour,
            status="ACTIVE"
        ))
    db.commit()


# -----------------------------
# Auto-Events (C)
# -----------------------------
def ensure_auto_events(db, site_id: str, inbound_every_min: int, order_every_min: int):
    """
    Streamlit hat keinen Background-Job. Das läuft pro App-Run.
    """
    now = dt.datetime.utcnow()
