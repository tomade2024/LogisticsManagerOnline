# app.py
import os
import uuid
import datetime as dt

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from sqlalchemy import (
    create_engine, String, Integer, BigInteger, DateTime, ForeignKey, Numeric, Boolean
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker, joinedload

# -----------------------------
# Config / DB
# -----------------------------
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+psycopg2://lm:lm@localhost:5432/lm")

def make_engine():
    try:
        eng = create_engine(DATABASE_URL, pool_pre_ping=True)
        # quick connectivity check (lazy but effective)
        with eng.connect() as c:
            c.execute("SELECT 1")
        return eng
    except Exception:
        # fallback for quick local demo (no docker)
        return create_engine("sqlite:///lm_demo.sqlite", future=True)

engine = make_engine()
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

class Base(DeclarativeBase):
    pass

# -----------------------------
# Models
# -----------------------------
class Player(Base):
    __tablename__ = "players"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    display_name: Mapped[str] = mapped_column(String, unique=True, index=True)
    money_cents: Mapped[int] = mapped_column(BigInteger, default=100000)  # 1000€
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
    status: Mapped[str] = mapped_column(String, default="SCHEDULED")  # SCHEDULED, ARRIVED, ACCEPTED, DONE
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

# -----------------------------
# DB init + seed
# -----------------------------
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

# -----------------------------
# Helpers / Business
# -----------------------------
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
        id=str(uuid.uuid4()), player_id=player_id, entry_type=entry_type,
        amount_cents=amount_cents, ref_type=ref_type, ref_id=ref_id
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
    site = PlayerSite(id=str(uuid.uuid4()), player_id=player_id, plot_id=plot_id, site_name=site_name.strip() or "Mein Lager")
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
            raise ValueError("Kollision: Überlappt ein Objekt.")
    player = db.query(Player).filter(Player.id == player_id).first()
    if player.money_cents < obj_def.buy_cost_cents:
        raise ValueError("Nicht genug Geld.")
    player.money_cents -= obj_def.buy_cost_cents
    o = PlacedObject(
        id=str(uuid.uuid4()), site_id=site.id, object_type=obj_type,
        x=x, y=y, w=obj_def.w, h=obj_def.h
    )
    db.add(o)
    ledger(db, player_id, "COST_BUILD", -int(obj_def.buy_cost_cents), "OBJECT", o.id)
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
        id=str(uuid.uuid4()), site_id=site_id, status="ARRIVED",
        scheduled_arrival_at=now, arrived_at=now
    )
    s.lines = [InboundLine(id=str(uuid.uuid4()), item_id=item.id, qty=5)]
    db.add(s)
    db.commit()

def accept_shipment(db, player_id: str, site_id: str, shipment_id: str):
    s = db.query(InboundShipment).options(joinedload(InboundShipment.lines)).filter(
        InboundShipment.id == shipment_id, InboundShipment.site_id == site_id
    ).first()
    if not s or s.status != "ARRIVED":
        raise ValueError("Lieferung nicht annehmbar.")
    total, used = compute_capacity(db, site_id)
    incoming = sum(ln.qty for ln in s.lines)
    if used + incoming > total:
        raise ValueError("Nicht genug Kapazität. Baue mehr Regale.")
    # putaway instant
    for ln in s.lines:
        inv = db.query(Inventory).filter(Inventory.site_id == site_id, Inventory.item_id == ln.item_id).first()
        if not inv:
            inv = Inventory(id=str(uuid.uuid4()), site_id=site_id, item_id=ln.item_id, qty=0)
            db.add(inv)
        inv.qty += ln.qty
    s.status = "DONE"
    db.commit()

def generate_order(db, site_id: str):
    now = dt.datetime.utcnow()
    item = db.query(Item).first()
    if not item:
        raise ValueError("Keine Items vorhanden.")
    o = Order(id=str(uuid.uuid4()), site_id=site_id, status="OPEN", due_at=now + dt.timedelta(hours=2), reward_cents=10000)
    o.lines = [OrderLine(id=str(uuid.uuid4()), item_id=item.id, qty=3)]
    db.add(o)
    db.commit()

def start_order(db, player_id: str, site_id: str, order_id: str):
    o = db.query(Order).options(joinedload(Order.lines)).filter(Order.id == order_id, Order.site_id == site_id).first()
    if not o or o.status != "OPEN":
        raise ValueError("Auftrag nicht startbar.")
    # check & consume
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
            id=str(uuid.uuid4()), site_id=site_id, role="WAREHOUSE_WORKER",
            efficiency=1.0, wage_cents_per_hour=wage_cents_per_hour, status="ACTIVE"
        ))
    db.commit()

# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="Logistics Manager (Single File)", layout="wide")
init_db()

with SessionLocal() as db:
    ensure_seed(db)

st.sidebar.title("Logistics Manager (MVP)")
display_name = st.sidebar.text_input("Spielername (Login MVP)", value=st.session_state.get("display_name", "Tester"))
st.session_state["display_name"] = display_name

page = st.sidebar.radio(
    "Navigation",
    ["Home", "Weltkarte", "Lager", "Wareneingang", "Aufträge", "Personal", "Finanzen", "Rangliste"],
)

with SessionLocal() as db:
    player = get_or_create_player(db, display_name)
    st.session_state["player_id"] = player.id
    site = get_site(db, player.id)

    st.sidebar.metric("Geld", f"{player.money_cents/100:,.2f} €")
    if site:
        st.sidebar.success(f"Standort: {site.site_name}")
    else:
        st.sidebar.info("Kein Standort. → Weltkarte")

    if page == "Home":
        st.title("Logistics Manager Online — Streamlit Single-File MVP")
        st.write("MVP: Standort kaufen, Lager bauen (X/Y), Lieferungen annehmen, Aufträge erfüllen, Ledger/Rangliste.")
        st.info("Start: Weltkarte → Plot kaufen → Lager → Regale platzieren → Wareneingang → Lieferung erzeugen/annehmen → Aufträge.")
    elif page == "Weltkarte":
        st.title("Weltkarte / Grundstücke")
        if site:
            st.success(f"Du besitzt bereits einen Standort: {site.site_name}")
        else:
            plots = db.query(Plot).filter(Plot.is_active == True).all()
            rows = [{
                "plot_id": p.id,
                "Name": p.name,
                "Preis (€)": p.price_cents/100,
                "Größe": f"{p.width}x{p.height}",
                "Bonus": f"{p.bonus_type} {float(p.bonus_value):+.0%}"
            } for p in plots]
            st.dataframe(rows, hide_index=True, use_container_width=True)

            pick = st.selectbox("Plot auswählen", options=[r["plot_id"] for r in rows],
                                format_func=lambda pid: next(r["Name"] for r in rows if r["plot_id"] == pid))
            site_name = st.text_input("Standortname", value="Mein Lager")
            if st.button("Grundstück kaufen", type="primary"):
                try:
                    buy_site(db, player.id, pick, site_name)
                    st.success("Kauf erfolgreich. Gehe zu Lager.")
                except Exception as e:
                    st.error(str(e))
    elif page == "Lager":
        st.title("Lager")
        if not site:
            st.warning("Bitte zuerst Standort kaufen (Weltkarte).")
        else:
            site = get_site(db, player.id)  # refresh with plot
            left, right = st.columns([2, 1])

            with right:
                st.subheader("Bauen")
                defs = db.query(ObjectDef).all()
                obj_type = st.selectbox("Objekt", options=[d.object_type for d in defs],
                                        format_func=lambda t: next(d.name for d in defs if d.object_type == t))
                x = st.number_input("X", min_value=0, max_value=site.plot.width - 1, value=0, step=1)
                y = st.number_input("Y", min_value=0, max_value=site.plot.height - 1, value=0, step=1)

                d = next(dd for dd in defs if dd.object_type == obj_type)
                st.caption(f"Kosten: {d.buy_cost_cents/100:,.2f} € | Größe: {d.w}x{d.h} | Slots: {d.rack_slots}")
                if st.button("Platzieren", type="primary"):
                    try:
                        place_object(db, player.id, site, obj_type, int(x), int(y))
                        st.success("Platziert.")
                    except Exception as e:
                        st.error(str(e))

                total, used = compute_capacity(db, site.id)
                st.metric("Kapazität (Slots)", f"{used} / {total}")
                st.metric("Geld", f"{player.money_cents/100:,.2f} €")

            with left:
                st.subheader("Raster")
                placed = db.query(PlacedObject).filter(PlacedObject.site_id == site.id).order_by(PlacedObject.created_at.asc()).all()
                width, height = site.plot.width, site.plot.height
                grid = [["" for _ in range(width)] for __ in range(height)]
                for o in placed:
                    label = o.object_type.replace("RACK_", "R")
                    for dy in range(o.h):
                        for dx in range(o.w):
                            gx, gy = o.x + dx, o.y + dy
                            if 0 <= gx < width and 0 <= gy < height:
                                grid[gy][gx] = label
                st.dataframe(pd.DataFrame(grid), use_container_width=True, height=600)

                st.subheader("Objekte")
                st.dataframe([{
                    "Typ": o.object_type, "Pos": f"({o.x},{o.y})", "Größe": f"{o.w}x{o.h}",
                    "Erstellt": o.created_at.strftime("%Y-%m-%d %H:%M")
                } for o in placed], hide_index=True, use_container_width=True)
    elif page == "Wareneingang":
        st.title("Wareneingang")
        if not site:
            st.warning("Bitte zuerst Standort kaufen (Weltkarte).")
        else:
            if st.button("Test: Lieferung erzeugen"):
                generate_shipment(db, site.id)
                st.success("Lieferung erzeugt.")
            shipments = db.query(InboundShipment).options(joinedload(InboundShipment.lines).joinedload(InboundLine.item)).filter(
                InboundShipment.site_id == site.id
            ).order_by(InboundShipment.created_at.desc()).all()
            if not shipments:
                st.info("Keine Lieferungen vorhanden.")
            for s in shipments:
                with st.expander(f"{s.status} | {s.created_at:%Y-%m-%d %H:%M} | ID: {s.id}"):
                    st.dataframe([{"SKU": ln.item.sku, "Artikel": ln.item.name, "Menge": ln.qty} for ln in s.lines], hide_index=True)
                    if s.status == "ARRIVED":
                        if st.button(f"Annehmen ({s.id})", key=f"acc_{s.id}", type="primary"):
                            try:
                                accept_shipment(db, player.id, site.id, s.id)
                                st.success("Angenommen und eingelagert.")
                            except Exception as e:
                                st.error(str(e))
    elif page == "Aufträge":
        st.title("Aufträge")
        if not site:
            st.warning("Bitte zuerst Standort kaufen (Weltkarte).")
        else:
            if st.button("Test: Auftrag erzeugen"):
                generate_order(db, site.id)
                st.success("Auftrag erzeugt.")
            orders = db.query(Order).options(joinedload(Order.lines).joinedload(OrderLine.item)).filter(
                Order.site_id == site.id
            ).order_by(Order.due_at.asc()).all()
            if not orders:
                st.info("Keine Aufträge vorhanden.")
            for o in orders:
                with st.expander(f"{o.status} | Reward {o.reward_cents/100:,.2f} € | Fällig {o.due_at:%Y-%m-%d %H:%M} | ID {o.id}"):
                    st.dataframe([{"SKU": ln.item.sku, "Artikel": ln.item.name, "Menge": ln.qty} for ln in o.lines], hide_index=True)
                    if o.status == "OPEN":
                        if st.button(f"Starten ({o.id})", key=f"start_{o.id}", type="primary"):
                            try:
                                start_order(db, player.id, site.id, o.id)
                                st.success("Auftrag erfüllt und bezahlt.")
                            except Exception as e:
                                st.error(str(e))
    elif page == "Personal":
        st.title("Personal")
        if not site:
            st.warning("Bitte zuerst Standort kaufen (Weltkarte).")
        else:
            n = st.number_input("Anzahl Lagerarbeiter", min_value=1, max_value=20, value=1, step=1)
            wage = st.number_input("Lohn €/h", min_value=8.0, max_value=40.0, value=15.0, step=0.5)
            if st.button("Einstellen", type="primary"):
                hire_workers(db, site.id, int(n), int(wage * 100))
                st.success("Eingestellt.")
            emps = db.query(Employee).filter(Employee.site_id == site.id).all()
            st.dataframe([{
                "ID": e.id, "Rolle": e.role, "Effizienz": float(e.efficiency),
                "Lohn €/h": e.wage_cents_per_hour/100, "Status": e.status
            } for e in emps], hide_index=True, use_container_width=True)
            st.caption("MVP: Payroll/Arbeitsleistung wird im Single-File noch nicht zeitbasiert simuliert.")
    elif page == "Finanzen":
        st.title("Finanzen / Ledger")
        st.metric("Kontostand", f"{player.money_cents/100:,.2f} €")
        led = db.query(LedgerEntry).filter(LedgerEntry.player_id == player.id).order_by(LedgerEntry.created_at.desc()).limit(200).all()
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
        st.title("Rangliste (MVP: Kontostand)")
        top = db.query(Player).order_by(Player.money_cents.desc()).limit(20).all()
        st.dataframe([{"Spieler": p.display_name, "Geld (€)": p.money_cents/100, "Level": p.level} for p in top],
                     hide_index=True, use_container_width=True)
