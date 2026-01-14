# Logistics Manager (Streamlit) — MVP v2 with time-based simulation (Jobs)
# Run: streamlit run streamlit_app.py

import os
import uuid
import json
import datetime as dt

import pandas as pd
import streamlit as st

# dotenv optional
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
DATABASE_URL = os.getenv("DATABASE_URL", "").strip()

def make_engine():
    # Prefer DATABASE_URL if set; otherwise SQLite fallback.
    if not DATABASE_URL:
        return create_engine("sqlite:///lm_demo.sqlite", future=True)

    # PostgreSQL with short timeout to avoid "blank app" (hang on connect)
    try:
        eng = create_engine(
            DATABASE_URL,
            pool_pre_ping=True,
            connect_args={"connect_timeout": 3},
        )
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
    money_cents: Mapped[int] = mapped_column(BigInteger, default=100000)
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
    status: Mapped[str] = mapped_column(String, default="ARRIVED")  # ARRIVED, PROCESSING, DONE
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
    status: Mapped[str] = mapped_column(String, default="OPEN")  # OPEN, PROCESSING, SHIPPED
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

class Job(Base):
    __tablename__ = "jobs"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    site_id: Mapped[str] = mapped_column(String, index=True)
    player_id: Mapped[str] = mapped_column(String, index=True)
    kind: Mapped[str] = mapped_column(String)  # RECEIVE, PUTAWAY, PICK, PACK, SHIP
    status: Mapped[str] = mapped_column(String, default="RUNNING")  # RUNNING, DONE
    ref_type: Mapped[str] = mapped_column(String)  # SHIPMENT, ORDER
    ref_id: Mapped[str] = mapped_column(String)
    payload_json: Mapped[str] = mapped_column(String, default="{}")
    started_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=dt.datetime.utcnow)
    finishes_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=dt.datetime.utcnow)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=dt.datetime.utcnow)


# =========================================================
# DB init + seed
# =========================================================
def init_db():
    Base.metadata.create_all(bind=engine)

def ensure_seed(db):
    if db.query(Plot).count() == 0:
        db.add_all([
            Plot(id=str(uuid.uuid4()), name="A-01 Autobahn", price_cents=200000, width=20, height=20, bonus_type="HIGHWAY", bonus_value=0.10),
            Plot(id=str(uuid.uuid4()), name="B-02 Stadt", price_cents=260000, width=24, height=24, bonus_type="CITY", bonus_value=0.08),
            Plot(id=str(uuid.uuid4()), name="C-03 Land", price_cents=150000, width=30, height=20, bonus_type="CHEAP_LAND", bonus_value=0.00),
        ])

    if db.query(Item).count() == 0:
        db.add_all([
            Item(id=str(uuid.uuid4()), sku="ELEC-001", name="Router", base_price_cents=8000),
            Item(id=str(uuid.uuid4()), sku="ELEC-002", name="Laptop", base_price_cents=90000),
            Item(id=str(uuid.uuid4()), sku="ELEC-003", name="Netzteil", base_price_cents=2500),
        ])

    if db.query(ObjectDef).count() == 0:
        db.add_all([
            ObjectDef(object_type="GATE_IN", name="Wareneingangstor", buy_cost_cents=15000, w=2, h=1, rack_slots=0),
            ObjectDef(object_type="GATE_OUT", name="Warenausgangstor", buy_cost_cents=15000, w=2, h=1, rack_slots=0),
            ObjectDef(object_type="RACK_STD", name="Palettenregal", buy_cost_cents=20000, w=2, h=2, rack_slots=8),
            ObjectDef(object_type="PACK_TABLE", name="Packtisch", buy_cost_cents=8000, w=2, h=1, rack_slots=0),
        ])
    db.commit()


# =========================================================
# Helpers / Business
# =========================================================
def money_fmt(cents: int) -> str:
    return f"{cents/100:,.2f} €"

def get_or_create_player(db, display_name: str) -> Player:
    # Option B: existing players are topped up to at least START_MONEY
    START_MONEY = 5_000_000  # 50.000 €
    name = (display_name or "Player").strip()

    p = db.query(Player).filter(Player.display_name == name).first()
    if p:
        if p.money_cents < START_MONEY:
            p.money_cents = START_MONEY
            db.commit()
            db.refresh(p)
        return p

    p = Player(id=str(uuid.uuid4()), display_name=name, money_cents=START_MONEY)
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
    if x < 0 or y < 0 or x + obj_def.w > site.plot.width or y + obj_def.h > site.plot.height:
        raise ValueError("Außerhalb des Rasters.")
    existing = db.query(PlacedObject).filter(PlacedObject.site_id == site.id).all()
    for e in existing:
        if overlaps(x, y, obj_def.w, obj_def.h, e.x, e.y, e.w, e.h):
            raise ValueError("Kollision: Objekt überlappt ein bestehendes Objekt.")
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
    refund = int(obj_def.buy_cost_cents * refund_ratio) if obj_def and refund_ratio > 0 else 0
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

def get_worker_factor(db, site_id: str) -> float:
    emps = db.query(Employee).filter(Employee.site_id == site_id, Employee.status == "ACTIVE").all()
    factor = sum(float(e.efficiency) for e in emps) if emps else 1.0
    return max(1.0, factor)

JOB_BASE_SECONDS_PER_UNIT = {
    "RECEIVE": 3.0,
    "PUTAWAY": 5.0,
    "PICK": 6.0,
    "PACK": 4.0,
    "SHIP": 3.0,
}

def create_job(db, player_id: str, site_id: str, kind: str, ref_type: str, ref_id: str, payload: dict, units: int):
    now = dt.datetime.utcnow()
    worker_factor = get_worker_factor(db, site_id)
    base = JOB_BASE_SECONDS_PER_UNIT.get(kind, 5.0)
    seconds = max(3.0, (base * max(1, units)) / worker_factor)

    j = Job(
        id=str(uuid.uuid4()),
        site_id=site_id,
        player_id=player_id,
        kind=kind,
        status="RUNNING",
        ref_type=ref_type,
        ref_id=ref_id,
        payload_json=json.dumps(payload, ensure_ascii=False),
        started_at=now,
        finishes_at=now + dt.timedelta(seconds=seconds),
        created_at=now,
    )
    db.add(j)
    db.commit()

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
        created_at=now,
    )
    s.lines = [InboundLine(id=str(uuid.uuid4()), item_id=item.id, qty=5)]
    db.add(s)
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
        reward_cents=10000,
    )
    o.lines = [OrderLine(id=str(uuid.uuid4()), item_id=item.id, qty=3)]
    db.add(o)
    db.commit()

def start_inbound_processing(db, player_id: str, site_id: str, shipment_id: str):
    s = db.query(InboundShipment).options(joinedload(InboundShipment.lines)).filter(
        InboundShipment.id == shipment_id, InboundShipment.site_id == site_id
    ).first()
    if not s or s.status != "ARRIVED":
        raise ValueError("Lieferung nicht startbar.")

    # Capacity check at start (so we don't do work we can't store)
    total, used = compute_capacity(db, site_id)
    incoming = sum(int(ln.qty) for ln in s.lines)
    if used + incoming > total:
        raise ValueError("Nicht genug Kapazität. Baue mehr Regale.")

    s.status = "PROCESSING"
    db.commit()

    payload = {"shipment_id": s.id}
    create_job(db, player_id, site_id, "RECEIVE", "SHIPMENT", s.id, payload, units=incoming)

def start_order_processing(db, player_id: str, site_id: str, order_id: str):
    o = db.query(Order).options(joinedload(Order.lines)).filter(
        Order.id == order_id, Order.site_id == site_id
    ).first()
    if not o or o.status != "OPEN":
        raise ValueError("Auftrag nicht startbar.")

    # Stock check at start
    for ln in o.lines:
        inv = db.query(Inventory).filter(Inventory.site_id == site_id, Inventory.item_id == ln.item_id).first()
        if not inv or inv.qty < ln.qty:
            raise ValueError("Nicht genug Bestand.")

    o.status = "PROCESSING"
    db.commit()

    units = sum(int(ln.qty) for ln in o.lines)
    create_job(db, player_id, site_id, "PICK", "ORDER", o.id, {"order_id": o.id}, units=units)

def tick_simulation(db, player_id: str, site_id: str):
    now = dt.datetime.utcnow()
    due = db.query(Job).filter(
        Job.site_id == site_id,
        Job.player_id == player_id,
        Job.status == "RUNNING",
        Job.finishes_at <= now
    ).order_by(Job.finishes_at.asc()).all()

    for j in due:
        payload = {}
        try:
            payload = json.loads(j.payload_json or "{}")
        except Exception:
            payload = {}

        # Mark job done
        j.status = "DONE"
        db.commit()

        # Chain / Effects
        if j.ref_type == "SHIPMENT":
            shipment_id = j.ref_id
            s = db.query(InboundShipment).options(joinedload(InboundShipment.lines)).filter(
                InboundShipment.id == shipment_id, InboundShipment.site_id == site_id
            ).first()
            if not s:
                continue

            if j.kind == "RECEIVE":
                # After receiving, start putaway
                incoming = sum(int(ln.qty) for ln in s.lines)
                create_job(
                    db, player_id, site_id, "PUTAWAY", "SHIPMENT", s.id,
                    {"shipment_id": s.id}, units=incoming
                )

            elif j.kind == "PUTAWAY":
                # Apply inventory changes and close shipment
                total, used = compute_capacity(db, site_id)
                incoming = sum(int(ln.qty) for ln in s.lines)
                if used + incoming > total:
                    # If capacity changed in the meantime, stop here (player must fix)
                    s.status = "ARRIVED"
                    db.commit()
                    continue

                for ln in s.lines:
                    inv = db.query(Inventory).filter(
                        Inventory.site_id == site_id, Inventory.item_id == ln.item_id
                    ).first()
                    if not inv:
                        inv = Inventory(id=str(uuid.uuid4()), site_id=site_id, item_id=ln.item_id, qty=0)
                        db.add(inv)
                    inv.qty += int(ln.qty)

                s.status = "DONE"
                s.accepted_at = now
                db.commit()

        elif j.ref_type == "ORDER":
            order_id = j.ref_id
            o = db.query(Order).options(joinedload(Order.lines)).filter(
                Order.id == order_id, Order.site_id == site_id
            ).first()
            if not o:
                continue

            if j.kind == "PICK":
                # Consume inventory
                for ln in o.lines:
                    inv = db.query(Inventory).filter(
                        Inventory.site_id == site_id, Inventory.item_id == ln.item_id
                    ).first()
                    if not inv or inv.qty < ln.qty:
                        # Stock got reduced meanwhile: revert order to OPEN
                        o.status = "OPEN"
                        db.commit()
                        break
                else:
                    for ln in o.lines:
                        inv = db.query(Inventory).filter(
                            Inventory.site_id == site_id, Inventory.item_id == ln.item_id
                        ).first()
                        inv.qty -= int(ln.qty)
                    db.commit()

                    units = sum(int(ln.qty) for ln in o.lines)
                    create_job(db, player_id, site_id, "PACK", "ORDER", o.id, {"order_id": o.id}, units=units)

            elif j.kind == "PACK":
                units = sum(int(ln.qty) for ln in o.lines)
                create_job(db, player_id, site_id, "SHIP", "ORDER", o.id, {"order_id": o.id}, units=units)

            elif j.kind == "SHIP":
                o.status = "SHIPPED"
                player = db.query(Player).filter(Player.id == player_id).first()
                player.money_cents += int(o.reward_cents)
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
# Auto-Events (simple)
# -----------------------------
def ensure_auto_events(db, site_id: str, inbound_every_min: int, order_every_min: int):
    now = dt.datetime.utcnow()

    # Inbound: not if any ARRIVED or PROCESSING exists
    if inbound_every_min and inbound_every_min > 0:
        open_inbound = db.query(InboundShipment).filter(
            InboundShipment.site_id == site_id,
            InboundShipment.status.in_(["ARRIVED", "PROCESSING"])
        ).first()

        if not open_inbound:
            last_ship = db.query(InboundShipment).filter(
                InboundShipment.site_id == site_id
            ).order_by(InboundShipment.created_at.desc()).first()

            due = True
            if last_ship:
                due = (now - last_ship.created_at).total_seconds() >= inbound_every_min * 60
            if due:
                generate_shipment(db, site_id)

    # Orders: not if any OPEN or PROCESSING exists
    if order_every_min and order_every_min > 0:
        open_order = db.query(Order).filter(
            Order.site_id == site_id,
            Order.status.in_(["OPEN", "PROCESSING"])
        ).first()

        if not open_order:
            last_order = db.query(Order).filter(
                Order.site_id == site_id
            ).order_by(Order.due_at.desc()).first()

            due = True
            if last_order:
                approx_created = last_order.due_at.replace(tzinfo=None) - dt.timedelta(hours=2)
                due = (now - approx_created).total_seconds() >= order_every_min * 60
            if due:
                generate_order(db, site_id)


# =========================================================
# UI
# =========================================================
st.set_page_config(page_title="Logistics Manager (Jobs)", layout="wide")
st.title("Logistics Manager (MVP v2)")
st.caption("Zeitbasierte Simulation: Lieferungen & Aufträge erzeugen Jobs, die Zeit benötigen.")

init_db()
with SessionLocal() as db:
    ensure_seed(db)

st.sidebar.title("Logistics Manager")

display_name = st.sidebar.text_input("Spielername (Login MVP)", value=st.session_state.get("display_name", "Tester"))
st.session_state["display_name"] = display_name

st.sidebar.subheader("Auto-Events")
auto_events = st.sidebar.checkbox("Automatisch Lieferungen & Aufträge erzeugen", value=True)
inbound_every = st.sidebar.number_input("Lieferung alle X Minuten", min_value=0, max_value=120, value=3, step=1)
order_every = st.sidebar.number_input("Auftrag alle Y Minuten", min_value=0, max_value=120, value=4, step=1)
st.session_state["auto_events"] = auto_events
st.session_state["inbound_every"] = int(inbound_every)
st.session_state["order_every"] = int(order_every)

st.sidebar.subheader("Anzeige")
auto_refresh = st.sidebar.checkbox("Auto-Refresh (15s)", value=False)

page = st.sidebar.radio(
    "Navigation",
    ["Home", "Weltkarte", "Lager", "Wareneingang", "Aufträge", "Job Center", "Personal", "Finanzen", "Rangliste"]
)

if auto_refresh:
    # light-weight refresh without extra libs
    st.markdown(
        """<script>
        setTimeout(() => { window.parent.location.reload(); }, 15000);
        </script>""",
        unsafe_allow_html=True
    )

try:
    with SessionLocal() as db:
        player = get_or_create_player(db, display_name)
        site = get_site(db, player.id)

        # Run simulation tick (complete due jobs)
        if site:
            tick_simulation(db, player.id, site.id)

        # Auto events (create new shipments/orders)
        if site and st.session_state.get("auto_events", True):
            ensure_auto_events(
                db,
                site_id=site.id,
                inbound_every_min=st.session_state.get("inbound_every", 3),
                order_every_min=st.session_state.get("order_every", 4),
            )

        # Sidebar status
        st.sidebar.metric("Geld", money_fmt(player.money_cents))
        if site:
            st.sidebar.success(f"Standort: {site.site_name}")
            st.sidebar.caption(f"Worker-Faktor: {get_worker_factor(db, site.id):.2f}")
        else:
            st.sidebar.info("Kein Standort → Weltkarte")

        # Pages
        if page == "Home":
            st.subheader("Core Loop (v2)")
            st.write(
                """1) Standort kaufen

2) Regale bauen (Kapazität)

3) Wareneingang: Lieferung starten → Jobs laufen → Putaway füllt Bestand

4) Aufträge: Starten → Jobs laufen → Versand bringt Geld
"""
            )
            st.info("Tipp: Stelle 1–3 Mitarbeiter ein, dann laufen Jobs deutlich schneller.")

        elif page == "Weltkarte":
            st.subheader("Weltkarte / Grundstücke")
            if site:
                st.success(f"Du besitzt bereits einen Standort: {site.site_name}")
            else:
                plots = db.query(Plot).filter(Plot.is_active == True).all()
                rows = [{
                    "plot_id": p.id,
                    "Name": p.name,
                    "Preis (€)": p.price_cents/100,
                    "Größe": f"{p.width}x{p.height}",
                    "Bonus": f"{p.bonus_type} {float(p.bonus_value):+.0%}",
                } for p in plots]
                st.dataframe(rows, hide_index=True, use_container_width=True)

                pick = st.selectbox(
                    "Plot auswählen",
                    options=[r["plot_id"] for r in rows],
                    format_func=lambda pid: next(r["Name"] for r in rows if r["plot_id"] == pid),
                )
                site_name = st.text_input("Standortname", value="Mein Lager")
                if st.button("Grundstück kaufen", type="primary"):
                    try:
                        buy_site(db, player.id, pick, site_name)
                        st.success("Kauf erfolgreich. Gehe zu Lager.")
                        st.rerun()
                    except Exception as e:
                        st.error(str(e))

        elif page == "Lager":
            st.subheader("Lager")
            if not site:
                st.warning("Bitte zuerst Standort kaufen (Weltkarte).")
            else:
                site = get_site(db, player.id)
                left, right = st.columns([2, 1])

                with right:
                    st.markdown("### Bauen")
                    defs = db.query(ObjectDef).all()
                    obj_type = st.selectbox(
                        "Objekt",
                        options=[d.object_type for d in defs],
                        format_func=lambda t: next(d.name for d in defs if d.object_type == t)
                    )
                    x = st.number_input("X", min_value=0, max_value=site.plot.width - 1, value=0, step=1)
                    y = st.number_input("Y", min_value=0, max_value=site.plot.height - 1, value=0, step=1)

                    d = next(dd for dd in defs if dd.object_type == obj_type)
                    st.caption(f"Kosten: {money_fmt(d.buy_cost_cents)} | Größe: {d.w}x{d.h} | Slots: {d.rack_slots}")

                    if st.button("Platzieren", type="primary"):
                        try:
                            place_object(db, player.id, site, obj_type, int(x), int(y))
                            st.success("Platziert.")
                            st.rerun()
                        except Exception as e:
                            st.error(str(e))

                    total, used = compute_capacity(db, site.id)
                    st.metric("Kapazität (Slots)", f"{used} / {total}")

                with left:
                    placed = db.query(PlacedObject).filter(
                        PlacedObject.site_id == site.id
                    ).order_by(PlacedObject.created_at.asc()).all()

                    st.markdown("### Raster")
                    width, height = site.plot.width, site.plot.height
                    grid = [["" for _ in range(width)] for __ in range(height)]
                    for o in placed:
                        label = o.object_type.replace("RACK_", "R")
                        for dy in range(o.h):
                            for dx in range(o.w):
                                gx, gy = o.x + dx, o.y + dy
                                if 0 <= gx < width and 0 <= gy < height:
                                    grid[gy][gx] = label
                    st.dataframe(pd.DataFrame(grid), use_container_width=True, height=420)

                    st.markdown("### Objekte (inkl. Löschen)")
                    refund_ratio = st.slider("Rückerstattung beim Löschen (%)", 0, 100, 50) / 100

                    obj_rows = [{
                        "ID": o.id,
                        "Typ": o.object_type,
                        "Pos": f"({o.x},{o.y})",
                        "Größe": f"{o.w}x{o.h}",
                        "Erstellt": o.created_at.strftime("%Y-%m-%d %H:%M"),
                    } for o in placed]
                    st.dataframe(obj_rows, hide_index=True, use_container_width=True)

                    delete_id = st.selectbox("Objekt-ID zum Löschen", options=[""] + [r["ID"] for r in obj_rows])
                    if delete_id and st.button("Objekt löschen", type="secondary"):
                        try:
                            delete_object(db, player.id, site.id, delete_id, refund_ratio=refund_ratio)
                            st.success("Objekt gelöscht.")
                            st.rerun()
                        except Exception as e:
                            st.error(str(e))

        elif page == "Wareneingang":
            st.subheader("Wareneingang (zeitbasiert)")
            if not site:
                st.warning("Bitte zuerst Standort kaufen (Weltkarte).")
            else:
                shipments = db.query(InboundShipment).options(
                    joinedload(InboundShipment.lines).joinedload(InboundLine.item)
                ).filter(
                    InboundShipment.site_id == site.id
                ).order_by(InboundShipment.created_at.desc()).all()

                if not shipments:
                    st.info("Noch keine Lieferungen. Auto-Events erzeugen welche.")
                for s in shipments:
                    with st.expander(f"{s.status} | {s.created_at:%Y-%m-%d %H:%M} | ID: {s.id}"):
                        st.dataframe(
                            [{"SKU": ln.item.sku, "Artikel": ln.item.name, "Menge": ln.qty} for ln in s.lines],
                            hide_index=True
                        )
                        if s.status == "ARRIVED":
                            if st.button(f"Bearbeitung starten ({s.id})", key=f"start_in_{s.id}", type="primary"):
                                try:
                                    start_inbound_processing(db, player.id, site.id, s.id)
                                    st.success("Inbound gestartet: RECEIVE läuft.")
                                    st.rerun()
                                except Exception as e:
                                    st.error(str(e))

        elif page == "Aufträge":
            st.subheader("Aufträge (zeitbasiert)")
            if not site:
                st.warning("Bitte zuerst Standort kaufen (Weltkarte).")
            else:
                orders = db.query(Order).options(
                    joinedload(Order.lines).joinedload(OrderLine.item)
                ).filter(
                    Order.site_id == site.id
                ).order_by(Order.due_at.asc()).all()

                if not orders:
                    st.info("Noch keine Aufträge. Auto-Events erzeugen welche.")
                for o in orders:
                    with st.expander(f"{o.status} | Reward {money_fmt(o.reward_cents)} | Fällig {o.due_at:%Y-%m-%d %H:%M} | ID {o.id}"):
                        st.dataframe(
                            [{"SKU": ln.item.sku, "Artikel": ln.item.name, "Menge": ln.qty} for ln in o.lines],
                            hide_index=True
                        )
                        if o.status == "OPEN":
                            if st.button(f"Kommissionierung starten ({o.id})", key=f"start_o_{o.id}", type="primary"):
                                try:
                                    start_order_processing(db, player.id, site.id, o.id)
                                    st.success("Order gestartet: PICK läuft.")
                                    st.rerun()
                                except Exception as e:
                                    st.error(str(e))

        elif page == "Job Center":
            st.subheader("Job Center")
            if not site:
                st.warning("Bitte zuerst Standort kaufen.")
            else:
                now = dt.datetime.utcnow()
                jobs = db.query(Job).filter(
                    Job.site_id == site.id, Job.player_id == player.id
                ).order_by(Job.created_at.desc()).limit(200).all()

                if not jobs:
                    st.info("Keine Jobs. Starte eine Lieferung oder einen Auftrag.")
                else:
                    rows = []
                    for j in jobs:
                        remaining = (j.finishes_at - now).total_seconds()
                        rows.append({
                            "Status": j.status,
                            "Kind": j.kind,
                            "Ref": f"{j.ref_type}:{j.ref_id}",
                            "Start": j.started_at.strftime("%H:%M:%S"),
                            "Fertig": j.finishes_at.strftime("%H:%M:%S"),
                            "Rest (s)": max(0, int(remaining)) if j.status == "RUNNING" else 0,
                        })
                    st.dataframe(rows, hide_index=True, use_container_width=True)

                    running = [r for r in rows if r["Status"] == "RUNNING"]
                    st.metric("Laufende Jobs", len(running))

        elif page == "Personal":
            st.subheader("Personal")
            if not site:
                st.warning("Bitte zuerst Standort kaufen.")
            else:
                n = st.number_input("Anzahl Lagerarbeiter", min_value=1, max_value=50, value=1, step=1)
                wage = st.number_input("Lohn €/h", min_value=8.0, max_value=40.0, value=15.0, step=0.5)
                if st.button("Einstellen", type="primary"):
                    hire_workers(db, site.id, int(n), int(wage * 100))
                    st.success("Eingestellt.")
                    st.rerun()

                emps = db.query(Employee).filter(Employee.site_id == site.id).all()
                st.dataframe([{
                    "ID": e.id,
                    "Rolle": e.role,
                    "Effizienz": float(e.efficiency),
                    "Lohn €/h": e.wage_cents_per_hour/100,
                    "Status": e.status
                } for e in emps], hide_index=True, use_container_width=True)

        elif page == "Finanzen":
            st.subheader("Finanzen / Ledger")
            st.metric("Kontostand", money_fmt(player.money_cents))
            led = db.query(LedgerEntry).filter(
                LedgerEntry.player_id == player.id
            ).order_by(LedgerEntry.created_at.desc()).limit(200).all()
            if not led:
                st.info("Noch keine Buchungen.")
            else:
                st.dataframe([{
                    "Zeit": e.created_at.strftime("%Y-%m-%d %H:%M"),
                    "Typ": e.entry_type,
                    "Betrag (€)": e.amount_cents/100,
                    "Ref": f"{e.ref_type or ''} {e.ref_id or ''}".strip()
                } for e in led], hide_index=True, use_container_width=True)

        elif page == "Rangliste":
            st.subheader("Rangliste (MVP: Kontostand)")
            top = db.query(Player).order_by(Player.money_cents.desc()).limit(50).all()
            st.dataframe([{
                "Spieler": p.display_name,
                "Geld (€)": p.money_cents/100,
                "Level": p.level
            } for p in top], hide_index=True, use_container_width=True)

except Exception as e:
    st.error("App-Fehler (Details):")
    st.exception(e)
