#!/usr/bin/env python3
"""
Lead Review Bot
===============
FastAPI webhook that receives new real estate leads from Zapier,
looks up the property on Redfin (listing status, sale history, rental history),
analyzes it with Claude against 6 specific flags, and posts to Slack.

Six flags checked on every lead:
  1. Currently active on MLS
  2. Estimated value >= $500,000
  3. Rental history detected
  4. Direct link to listing photos (Redfin / Zillow)
  5. Last sold within 36 months
  6. Active or expired listing within last 36 months

Setup:
  cp .env.example .env   # fill in keys
  python lead_bot.py
"""

import os
import re
import json
import logging
import psycopg2
import psycopg2.extras
from datetime import datetime, timedelta

import anthropic
import uvicorn
from curl_cffi import requests as cffi_requests
from dotenv import load_dotenv
from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse

load_dotenv(override=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ANTHROPIC_API_KEY   = os.getenv("ANTHROPIC_API_KEY", "")
SLACK_WEBHOOK_URL   = os.getenv("SLACK_WEBHOOK_URL", "")
MIN_VALUE_THRESHOLD = int(os.getenv("MIN_VALUE_THRESHOLD", "500000"))
PORT                = int(os.getenv("PORT", "8000"))

if not ANTHROPIC_API_KEY:
    raise RuntimeError("ANTHROPIC_API_KEY is not set in .env")
if not SLACK_WEBHOOK_URL:
    raise RuntimeError("SLACK_WEBHOOK_URL is not set in .env")

claude = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

DATABASE_URL = os.getenv("DATABASE_URL", "")

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

def _db():
    """Return a new Postgres connection."""
    conn = psycopg2.connect(DATABASE_URL, cursor_factory=psycopg2.extras.RealDictCursor)
    return conn


def init_db():
    """Create the leads table if it doesn't exist."""
    conn = _db()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS leads (
                    id              SERIAL PRIMARY KEY,
                    address_raw     TEXT,
                    address_norm    TEXT,
                    zpid            TEXT,
                    name            TEXT,
                    source          TEXT,
                    zestimate       REAL,
                    last_sold_price REAL,
                    last_sold_date  TEXT,
                    flag_count      INTEGER,
                    flags_hit       TEXT,
                    slack_sent      INTEGER DEFAULT 0,
                    processed_at    TEXT,
                    lead_raw        TEXT,
                    prop_raw        TEXT
                )
            """)
            cur.execute("CREATE INDEX IF NOT EXISTS idx_address_norm ON leads(address_norm)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_zpid ON leads(zpid)")
        conn.commit()
    finally:
        conn.close()


def _norm_address(address: str) -> str:
    """Normalize an address for dedup comparison."""
    a = address.lower().strip()
    a = re.sub(r'[^a-z0-9\s]', '', a)
    a = re.sub(r'\s+', ' ', a)
    return a


def is_duplicate(address: str, zpid: str = None) -> dict | None:
    """
    Return the existing lead row if this address/zpid was already processed,
    otherwise return None.
    """
    norm = _norm_address(address)
    conn = _db()
    try:
        with conn.cursor() as cur:
            if zpid:
                cur.execute(
                    "SELECT * FROM leads WHERE zpid = %s ORDER BY processed_at DESC LIMIT 1",
                    (zpid,)
                )
                row = cur.fetchone()
                if row:
                    return dict(row)
            cur.execute(
                "SELECT * FROM leads WHERE address_norm = %s ORDER BY processed_at DESC LIMIT 1",
                (norm,)
            )
            row = cur.fetchone()
            return dict(row) if row else None
    finally:
        conn.close()


def record_lead(lead: dict, prop: dict, analysis: dict, slack_sent: bool):
    """Persist a processed lead to the database."""
    flags_hit = analysis.get("flags_hit", [])
    conn = _db()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO leads
                    (address_raw, address_norm, zpid, name, source,
                     zestimate, last_sold_price, last_sold_date,
                     flag_count, flags_hit, slack_sent, processed_at,
                     lead_raw, prop_raw)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                lead.get("address", ""),
                _norm_address(lead.get("address", "")),
                prop.get("zpid"),
                lead.get("name") or (lead.get("first_name","") + " " + lead.get("last_name","")).strip(),
                lead.get("source", ""),
                prop.get("zestimate"),
                prop.get("last_sold_price"),
                prop.get("last_sold_date"),
                analysis.get("flag_count", 0),
                json.dumps(flags_hit),
                int(slack_sent),
                datetime.now().isoformat(),
                json.dumps(lead),
                json.dumps({k: v for k, v in prop.items() if isinstance(v, (str, int, float, bool, type(None)))}),
            ))
        conn.commit()
    finally:
        conn.close()


init_db()

REDFIN_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    ),
    "Referer": "https://www.redfin.com/",
    "Accept-Language": "en-US,en;q=0.9",
}

THIRTY_SIX_MONTHS_AGO = datetime.now() - timedelta(days=36 * 30)

# ---------------------------------------------------------------------------
# Redfin lookup helpers
# ---------------------------------------------------------------------------

def _redfin_get(session, path, params=None):
    """GET a Redfin API endpoint and strip the {}&& prefix."""
    r = session.get(
        f"https://www.redfin.com{path}",
        params=params or {},
        headers=REDFIN_HEADERS,
        timeout=15,
    )
    raw = r.text
    if raw.startswith("{}&&"):
        raw = raw[4:]
    return json.loads(raw)


def _parse_redfin_date(ms_timestamp):
    """Convert Redfin's millisecond epoch timestamp to a datetime."""
    if not ms_timestamp:
        return None
    try:
        return datetime.fromtimestamp(int(ms_timestamp) / 1000)
    except Exception:
        return None


def _extract_zip(address: str) -> str:
    """Pull a 5-digit zip code out of an address string."""
    m = re.search(r'\b(\d{5})\b', address)
    return m.group(1) if m else ""


def _get_region_for_zip(session, zip_code: str):
    """
    Fetch the Redfin zip page and extract region_id, region_type, market.
    Returns (region_id, region_type, market) or (None, None, None).
    """
    try:
        r = session.get(
            f"https://www.redfin.com/zipcode/{zip_code}",
            headers=REDFIN_HEADERS,
            timeout=15,
        )
        html = r.text
        rid   = re.search(r"['\"]region_id['\"]\s*:\s*['\"](\d+)['\"]", html)
        rtype = re.search(r"['\"]region_type_id['\"]\s*:\s*['\"](\d+)['\"]", html)
        mkt   = re.search(r"var searchMarket\s*=\s*['\"]([^'\"]+)['\"]", html)
        if rid and rtype:
            return rid.group(1), rtype.group(1), mkt.group(1) if mkt else None
    except Exception as exc:
        log.warning("Region lookup error for zip %s: %s", zip_code, exc)
    return None, None, None


def redfin_lookup(address, zip_code=""):
    """
    Look up a property on Redfin using the zip code to search the area,
    then match the street address in the results.

    Searches active listings first, then recently sold (last 3 years) as fallback.
    Returns a structured dict of everything found.
    """
    session = cffi_requests.Session(impersonate="chrome120")
    result  = {"address_searched": address}

    # Extract zip from address string if not passed directly
    zip_code = zip_code or _extract_zip(address)
    if not zip_code:
        log.warning("No zip code found in address: %s", address)
        return result

    # ── Step 1: Get Redfin region ID for this zip ──────────────────────────
    region_id, region_type, market = _get_region_for_zip(session, zip_code)
    if not region_id:
        log.warning("Could not find Redfin region for zip: %s", zip_code)
        return result

    # ── Helpers ────────────────────────────────────────────────────────────
    def _norm(s):
        s = s.lower().strip()
        s = re.sub(r'\b(unit|apt|#|ste|suite)\s*[\w-]+', '', s)
        s = re.sub(r'[^a-z0-9\s]', '', s)
        return s.split(',')[0].strip()

    def _search_gis(extra_params):
        base = [
            ("al", "1"), ("region_id", region_id), ("region_type", region_type),
            ("uipt", "1"), ("uipt", "2"), ("uipt", "3"),
            ("uipt", "4"), ("uipt", "5"), ("uipt", "6"),
            ("v", "8"), ("num_homes", "500"),
        ]
        if market:
            base.append(("market", market))
        data = _redfin_get(session, "/stingray/api/gis", base + extra_params)
        return data.get("payload", {}).get("homes", [])

    def _match(homes):
        for h in homes:
            street_val = (h.get("streetLine") or {}).get("value", "")
            unit_val   = (h.get("unitNumber") or {}).get("value", "") or ""
            candidate  = _norm(f"{street_val} {unit_val}")
            if target_street and candidate and (
                target_street in candidate or candidate in target_street
            ):
                return h
        return None

    def gv(obj, *keys):
        for k in keys:
            if not isinstance(obj, dict):
                return None
            obj = obj.get(k)
        return obj.get("value") if isinstance(obj, dict) else obj

    target_street = _norm(address)
    matched  = None
    is_sold  = False

    # ── Step 2a: Search active listings ───────────────────────────────────
    try:
        matched = _match(_search_gis([("status", "1")]))
    except Exception as exc:
        log.warning("Active GIS search error: %s", exc)

    # ── Step 2b: Search recently sold (last 3 years) if not active ─────────
    if not matched:
        try:
            sold_homes = _search_gis([("status", "2"), ("sold_within_days", "1095")])
            matched = _match(sold_homes)
            if matched:
                is_sold = True
        except Exception as exc:
            log.warning("Sold GIS search error: %s", exc)

    # ── Step 3: Extract data from matched home ─────────────────────────────
    if matched:
        street   = gv(matched, "streetLine") or ""
        unit     = gv(matched, "unitNumber") or ""
        city     = matched.get("city", "")
        state    = matched.get("state", "")
        zip_out  = matched.get("zip") or matched.get("postalCode") or zip_code
        addr_str = ", ".join(p for p in [f"{street} {unit}".strip(), city, state, str(zip_out)] if p)

        url_path   = matched.get("url", "")
        redfin_url = f"https://www.redfin.com{url_path}" if url_path else ""

        price = gv(matched, "price")
        sqft  = gv(matched, "sqFt")
        ppsf  = gv(matched, "pricePerSqFt")
        mls_status = matched.get("mlsStatus", "")

        # Sold date / within-36-months check
        last_sold_date       = None
        last_sold_within_36mo = False
        sold_date_ms = matched.get("soldDate") or matched.get("lastSoldDate")
        if sold_date_ms:
            last_sold_date = _parse_redfin_date(sold_date_ms)
            if last_sold_date and last_sold_date >= THIRTY_SIX_MONTHS_AGO:
                last_sold_within_36mo = True

        # If we found it via the sold search, it was definitely sold recently
        if is_sold and not last_sold_within_36mo:
            last_sold_within_36mo = True   # sold_within_days=1095 guarantees this

        result.update({
            "display_name":          addr_str or address,
            "redfin_url":            redfin_url,
            "has_photos":            bool(redfin_url),
            "price":                 float(price) if price else None,
            "beds":                  float(matched["beds"]) if matched.get("beds") is not None else None,
            "baths":                 float(matched["baths"]) if matched.get("baths") is not None else None,
            "sqft":                  float(sqft) if sqft else None,
            "price_per_sqft":        float(ppsf) if ppsf else None,
            "year_built":            float(matched["yearBuilt"]) if matched.get("yearBuilt") else None,
            "dom":                   float(gv(matched, "dom")) if gv(matched, "dom") is not None else None,
            "mls_status":            mls_status,
            "property_type":         matched.get("propertyType", ""),
            "last_sold_within_36mo": last_sold_within_36mo,
            "last_sold_date":        last_sold_date.strftime("%Y-%m-%d") if last_sold_date else None,
            "last_sold_price":       float(price) if (is_sold and price) else None,
            # Found on Redfin at all = it has had listing/sale activity
            "listed_or_expired_36mo": True,
        })
        log.info(
            "Matched on Redfin (%s): %s — %s",
            "sold" if is_sold else "active", addr_str, mls_status
        )
    else:
        result["display_name"] = address
        result["mls_status"]   = "Not Listed"
        log.info("Not found on Redfin for zip %s: %s", zip_code, address)

    # ── Step 4: Rental search (same zip) ──────────────────────────────────
    try:
        rental_params = [
            ("al", "1"), ("region_id", region_id), ("region_type", region_type),
            ("isRentals", "true"), ("v", "8"), ("num_homes", "50"),
        ]
        rental_data  = _redfin_get(session, "/stingray/api/gis", rental_params)
        rental_homes = rental_data.get("payload", {}).get("homes", [])
        for h in rental_homes:
            street = _norm((h.get("streetLine") or {}).get("value", ""))
            if street and (target_street in street or street in target_street):
                result["rental_history_detected"] = True
                result["rental_notes"] = ["Currently or recently listed as rental on Redfin"]
                break
    except Exception as exc:
        log.warning("Rental search error: %s", exc)

    return result


# ---------------------------------------------------------------------------
# Zillow — URL builder + data lookup
# ---------------------------------------------------------------------------

def zillow_url(address: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9\s]", "", address)
    slug = re.sub(r"\s+", "-", slug.strip())
    return f"https://www.zillow.com/homes/{slug}_rb/"


SCRAPEAK_API_KEY = os.getenv("SCRAPEAK_API_KEY", "")
SCRAPEAK_BASE    = "https://app.scrapeak.com/v1/scrapers/zillow"


def zillow_lookup(address: str) -> dict:
    """
    Look up a property on Zillow via the Scrapeak API.

    Two calls per lead (20 credits):
      1. zpidByAddress  — resolve street/city/state/zip → zpid
      2. property       — full details using that zpid

    Returns a dict with property data, or {"zillow_url": search_url} on
    failure/missing key.  Caller should treat missing "zpid" as not found.
    """
    search_url = zillow_url(address)
    result = {"zillow_url": search_url}

    if not SCRAPEAK_API_KEY:
        log.debug("SCRAPEAK_API_KEY not set — skipping Zillow lookup")
        return result

    session = cffi_requests.Session(impersonate="chrome120")

    # ── Parse address components ───────────────────────────────────────────
    # Expect "123 Main St, City, ST, 12345" or similar
    parts  = [p.strip() for p in address.split(",")]
    street = parts[0] if len(parts) > 0 else address
    city   = parts[1] if len(parts) > 1 else ""
    state  = parts[2] if len(parts) > 2 else ""
    zip_   = parts[3] if len(parts) > 3 else _extract_zip(address)

    # ── Step 1: address → zpid ─────────────────────────────────────────────
    try:
        r = session.get(
            f"{SCRAPEAK_BASE}/zpidByAddress",
            params={"api_key": SCRAPEAK_API_KEY, "street": street,
                    "city": city, "state": state, "zipcode": zip_},
            timeout=20,
        )
        data = r.json()
        if not data.get("is_success"):
            log.info("Scrapeak zpid lookup failed for %s: %s", address, data.get("message"))
            return result
        matches = data.get("data") or []
        if isinstance(matches, dict):
            matches = [matches]
        if not matches:
            log.info("Scrapeak: no zpid for %s", address)
            return result
        # First result is the closest address match
        best    = matches[0]
        zpid    = str(best.get("zpid", ""))
        prop_url = best.get("url", "")
        if prop_url:
            result["zillow_url"] = prop_url
        if not zpid:
            log.info("Scrapeak: no zpid for %s", address)
            return result
    except Exception as exc:
        log.warning("Scrapeak zpid error for %s: %s", address, exc)
        return result

    # ── Step 2: zpid → full property details (one retry on timeout) ──────
    prop = None
    for attempt in range(2):
        try:
            r = session.get(
                f"{SCRAPEAK_BASE}/property",
                params={"api_key": SCRAPEAK_API_KEY, "zpid": zpid},
                timeout=60,
            )
            data = r.json()
            if not data.get("is_success"):
                log.info("Scrapeak property fetch failed for zpid %s: %s", zpid, data.get("message"))
                return result
            prop = data.get("data", {})
            break
        except Exception as exc:
            log.warning("Scrapeak property attempt %d failed for zpid %s: %s", attempt + 1, zpid, exc)
            if attempt == 1:
                return result

    # ── Extract fields ─────────────────────────────────────────────────────
    addr_data     = prop.get("address") or {}
    price         = prop.get("price")
    zestimate     = prop.get("zestimate")
    status        = prop.get("homeStatus") or prop.get("homeStatusForHDP") or ""
    photos        = prop.get("photos") or prop.get("originalPhotos") or []
    price_history = prop.get("priceHistory") or []
    reso          = prop.get("resoFacts") or {}

    # Build canonical Zillow URL
    url_path = prop.get("hdpUrl") or prop.get("url") or ""
    if url_path:
        result["zillow_url"] = (
            f"https://www.zillow.com{url_path}" if url_path.startswith("/") else url_path
        )

    # ── Parse price history for sold date and listing activity ────────────
    last_sold_date        = None
    last_sold_price       = None
    last_sold_within_36mo = False
    listed_or_expired_36mo = False

    for event in price_history:
        event_type = (event.get("event") or "").lower()
        date_str   = event.get("date") or ""
        ep         = event.get("price")

        if not date_str:
            continue
        try:
            event_dt = datetime.strptime(date_str[:10], "%Y-%m-%d")
        except ValueError:
            continue

        if "sold" in event_type:
            if last_sold_date is None:          # priceHistory is newest-first
                last_sold_date  = date_str[:10]
                last_sold_price = float(ep) if ep else None
                if event_dt >= THIRTY_SIX_MONTHS_AGO:
                    last_sold_within_36mo = True

        if any(x in event_type for x in ("listed", "list", "price change", "relisted")):
            if event_dt >= THIRTY_SIX_MONTHS_AGO:
                listed_or_expired_36mo = True

    # Fall back to top-level lastSoldPrice if history had no price
    if last_sold_price is None:
        lsp = prop.get("lastSoldPrice")
        if lsp:
            last_sold_price = float(lsp)

    rent_zestimate = prop.get("rentZestimate")
    lot_size       = reso.get("lotSize") or prop.get("lotAreaValue") or prop.get("lotSize")
    lot_units      = prop.get("lotAreaUnits") or ""
    home_type      = prop.get("homeType") or prop.get("propertyTypeDimension") or ""
    hoa_fee        = prop.get("monthlyHoaFee") or prop.get("hoaFee")
    days_on_zillow = prop.get("daysOnZillow")
    raw_photo_url  = prop.get("desktopWebHdpImageLink") or ""
    photo_url      = raw_photo_url

    # Garage
    has_garage       = reso.get("hasGarage") or reso.get("hasAttachedGarage")
    parking_features = reso.get("parkingFeatures") or []
    garage_capacity  = reso.get("garageParkingCapacity")
    if not has_garage and parking_features:
        has_garage = any("garage" in f.lower() for f in parking_features)
    garage_str = None
    if has_garage:
        cap = f" ({int(garage_capacity)}-car)" if garage_capacity else ""
        attached = any("attached" in f.lower() for f in parking_features)
        garage_str = f"Garage{cap}" + (" – Attached" if attached else "")

    # Basement
    basement_raw = reso.get("basement") or reso.get("basementYN")
    basement_str = None
    if basement_raw and str(basement_raw).lower() not in ("none", "false", "0", ""):
        basement_str = basement_raw if isinstance(basement_raw, str) else "Yes"

    result.update({
        "source":                 "zillow",
        "zpid":                   zpid,
        "zestimate":              float(zestimate) if zestimate else None,
        "rent_zestimate":         float(rent_zestimate) if rent_zestimate else None,
        "price":                  float(price) if price else None,
        "beds":                   prop.get("bedrooms"),
        "baths":                  prop.get("bathrooms"),
        "sqft":                   prop.get("livingArea"),
        "lot_size":               str(lot_size) if lot_size else None,
        "lot_units":              lot_units,
        "year_built":             prop.get("yearBuilt"),
        "home_type":              home_type,
        "hoa_fee":                float(hoa_fee) if hoa_fee else None,
        "days_on_zillow":         int(days_on_zillow) if days_on_zillow else None,
        "mls_status":             status,
        "has_photos":             bool(photos) or bool(url_path),
        "photo_count":            len(photos) if isinstance(photos, list) else 0,
        "last_sold_price":        last_sold_price,
        "last_sold_date":         last_sold_date,
        "last_sold_within_36mo":  last_sold_within_36mo,
        "listed_or_expired_36mo": listed_or_expired_36mo or status.upper() in (
                                      "FOR_SALE", "RECENTLY_SOLD", "SOLD",
                                      "PENDING", "COMING_SOON",
                                  ),
        "display_name":           addr_data.get("streetAddress", street),
        "photo_url":              photo_url,
        "garage":                 garage_str,
        "basement":               basement_str,
        "lot_size":               str(lot_size) if lot_size else None,
        "price_history":          price_history,
    })
    log.info(
        "Scrapeak/Zillow found: zpid=%s status=%s zestimate=%s last_sold=%s photos=%s",
        zpid, status, zestimate, last_sold_date, result["photo_count"]
    )
    return result


# ---------------------------------------------------------------------------
# Claude analysis
# ---------------------------------------------------------------------------

def analyze_lead(lead: dict, prop: dict) -> dict:
    """
    Ask Claude to evaluate the lead against the 6 flags.
    Returns {"verdict": "...", "analysis": "...", "flags_hit": [...]}
    """

    # Pre-compute flag states to give Claude clean input
    price    = prop.get("price")
    estimate = prop.get("redfin_estimate") or prop.get("zestimate")
    value    = price or estimate or 0

    year_built      = prop.get("year_built")
    last_sold_price = prop.get("last_sold_price")
    zestimate       = prop.get("zestimate") or prop.get("redfin_estimate")
    low_equity      = False
    high_equity     = False
    if last_sold_price and zestimate and zestimate > 0:
        low_equity  = abs(zestimate - last_sold_price) / zestimate <= 0.20
        high_equity = last_sold_price <= zestimate * 0.50

    flags = {
        "active_mls_listing":        bool(prop.get("mls_status", "").lower() in
                                          ("active", "for sale", "coming soon", "active under contract",
                                           "for_sale", "pending")),
        "value_at_or_above_500k":    value >= MIN_VALUE_THRESHOLD,
        "rental_history":            bool(prop.get("rental_history_detected")),
        "listing_photos_available":  (prop.get("photo_count") or 0) > 1,
        "sold_within_36_months":     bool(prop.get("last_sold_within_36mo")),
        "listed_or_expired_36mo":    bool(prop.get("listed_or_expired_36mo")),
        "newer_build":               bool(year_built and int(year_built) >= 1990),
        "low_equity":                low_equity,
        "high_equity":               high_equity,
        "unknown_bed_count":         prop.get("beds") is None,
    }
    flags_hit = [k for k, v in flags.items() if v]

    # Format price history for the prompt
    history = prop.get("price_history") or []
    if history:
        history_lines = "\n".join(
            f"  {e.get('date', '')[:10]}  {e.get('event', ''):<30}  ${e.get('price') or 0:,.0f}"
            for e in history
        )
    else:
        history_lines = "  No listing history available"

    prompt = f"""You are summarizing the listing history of a property for a real estate investment team.

Rules:
- For any listing or sale activity within the last 5 years: include specific dates, prices, and events.
- For activity older than 5 years: mention it briefly in general terms only (e.g. "listed in 2015 but did not sell") — no specific prices or amounts.
- No interpretation, no commentary, no conclusions. Facts only. 2-3 sentences max.
- If there is no meaningful listing history, say so in one sentence.

Today's year: {datetime.now().year}

LISTING HISTORY (newest first):
{history_lines}

MLS Status: {prop.get("mls_status") if prop.get("mls_status", "").upper() not in ("", "OTHER") else "Off-market"}
Rental history: {"Yes — " + "; ".join(prop.get("rental_notes", [])) if prop.get("rental_history_detected") else "None detected"}"""

    response = claude.messages.create(
        model="claude-opus-4-6",
        max_tokens=600,
        thinking={"type": "adaptive"},
        messages=[{"role": "user", "content": prompt}],
    )

    analysis_text = next(
        (b.text for b in response.content if b.type == "text"),
        "Analysis unavailable."
    )

    return {
        "text":      analysis_text,
        "flags":     flags,
        "flags_hit": flags_hit,
        "flag_count": len(flags_hit),
    }


# ---------------------------------------------------------------------------
# Slack notification
# ---------------------------------------------------------------------------

def _fmt(val, prefix="", suffix="", fallback="N/A"):
    if val is None or val == "":
        return fallback
    if isinstance(val, (int, float)):
        return f"{prefix}{val:,.0f}{suffix}"
    return f"{prefix}{val}{suffix}"


FLAG_EMOJI = {
    "active_mls_listing":       "📋",
    "value_at_or_above_500k":   "💰",
    "rental_history":           "🏘",
    "listing_photos_available": "📸",
    "sold_within_36_months":    "🔄",
    "listed_or_expired_36mo":   "📅",
    "newer_build":              "🏗",
    "low_equity":               "📉",
    "high_equity":              "📈",
    "unknown_bed_count":        "❓",
}

FLAG_LABEL = {
    "active_mls_listing":       "Active MLS listing",
    "value_at_or_above_500k":   f"Value >= ${MIN_VALUE_THRESHOLD:,}",
    "rental_history":           "Rental history detected",
    "listing_photos_available": "Listing photos available",
    "sold_within_36_months":    "Sold within last 36 months",
    "listed_or_expired_36mo":   "Listed/expired in last 36 months",
    "newer_build":              "Newer build (1990+)",
    "low_equity":               "Low equity (sold price within 20% of Zestimate)",
    "high_equity":              "High equity (purchased at <=50% of current value)",
    "unknown_bed_count":        "Unknown bed count (Zillow shows --)",
}


def send_slack(lead: dict, prop: dict, analysis: dict, zillow_link: str):
    name    = (lead.get("name") or
               (lead.get("first_name", "") + " " + lead.get("last_name", "")).strip()
               or "Unknown")
    source  = lead.get("source", "CRM")
    # Prefer full address from lead (has city/state/zip), fall back to prop display_name
    address = lead.get("address") or prop.get("display_name") or "Unknown address"

    flag_count  = analysis["flag_count"]
    flags       = analysis["flags"]
    total_flags = len(flags)

    # ── Build detail section (property info + flags all together) ──────────
    lsp   = prop.get("last_sold_price")
    zest  = prop.get("zestimate")
    photo_count = prop.get("photo_count") or 0

    # Owner name + address line with Zestimate inline
    detail_lines = [f"*{name}*"]
    addr_line = f"*📍 {address}*"
    if zest:
        addr_line += f"  •  Zestimate: *${zest:,.0f}*"
    detail_lines.append(addr_line)

    # Beds / baths / sqft / lot / year / type
    beds       = prop.get("beds")
    baths      = prop.get("baths")
    sqft       = prop.get("sqft")
    year_built = prop.get("year_built")
    lot_size   = prop.get("lot_size")
    raw_type   = prop.get("home_type") or prop.get("property_type") or ""
    # Clean up CamelCase → readable (e.g. SingleFamily → Single Family)
    import re as _re
    home_type_str = _re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', raw_type).replace("_", " ").strip()
    parts = []
    if home_type_str: parts.append(home_type_str)
    parts.append(f"{int(beds)} bd" if beds is not None else "-- bd")
    parts.append(f"{baths:.1g} ba" if baths is not None else "-- ba")
    if sqft:     parts.append(f"{int(sqft):,} sqft")
    if lot_size: parts.append(f"lot {lot_size}")
    if year_built: parts.append(f"built {int(year_built)}")
    detail_lines.append("🏡 " + "  /  ".join(parts))

    # Garage / basement
    extra = []
    if prop.get("garage"):   extra.append(prop["garage"])
    if prop.get("basement"): extra.append(f"Basement: {prop['basement']}")
    if extra:
        detail_lines.append("🔑 " + "  •  ".join(extra))

    # MLS status
    mls = prop.get("mls_status")
    if mls and mls.upper() not in ("", "OTHER", "NOT LISTED"):
        detail_lines.append(f"🏷 {mls}")

    # Last sold
    if prop.get("last_sold_date"):
        sold = f"🔄 Last sold: {prop['last_sold_date']}"
        if lsp:
            sold += f"  •  ${lsp:,.0f}"
        if prop.get("last_sold_within_36mo"):
            sold += "  ⚠️ within 36mo"
        detail_lines.append(sold)

    # Equity lines
    if lsp and zest and zest > 0:
        pct = int((zest - lsp) / zest * 100)
        if pct >= 50:
            detail_lines.append(f"📈 Purchased at {100 - pct}% of current value")
        elif abs(pct) <= 20:
            detail_lines.append(f"📉 Low equity — paid ${lsp:,.0f}, Zestimate ${zest:,.0f}")

    # Flags folded in
    for k, v in flags.items():
        if not v:
            continue
        if k in ("listing_photos_available", "low_equity", "high_equity", "newer_build"):
            continue  # handled separately or silently triggers only
        detail_lines.append(f"• {FLAG_LABEL.get(k, k)}")

    # Photos link
    if photo_count > 1 and zillow_link:
        detail_lines.append(f"📸 <{zillow_link}|{photo_count} photos on Zillow>")

    detail_text = "\n".join(detail_lines)

    photo_url = prop.get("photo_url", "")
    blocks = [
        {"type": "divider"},
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": detail_text}
        },
        {"type": "divider"},
    ]

    if photo_url:
        blocks.append({
            "type": "section",
            "text": {"type": "mrkdwn", "text": analysis["text"]},
            "accessory": {
                "type": "image",
                "image_url": photo_url,
                "alt_text": address,
            },
        })
    else:
        blocks.append({
            "type": "section",
            "text": {"type": "mrkdwn", "text": analysis["text"]}
        })

    if lead.get("message"):
        blocks.append({
            "type": "context",
            "elements": [{"type": "mrkdwn", "text": f"💬 _{lead['message']}_"}]
        })

    if prop.get("rental_notes"):
        blocks.append({
            "type": "context",
            "elements": [{"type": "mrkdwn",
                          "text": f"🏘 {' | '.join(prop['rental_notes'])}"}]
        })

    if zillow_link:
        blocks.append({"type": "actions", "elements": [{
            "type": "button",
            "text": {"type": "plain_text", "text": "View Details on Zillow"},
            "url": zillow_link,
            "style": "primary",
        }]})

    r = cffi_requests.Session(impersonate="chrome120").post(
        SLACK_WEBHOOK_URL,
        json={"blocks": blocks},
        headers={"Content-Type": "application/json"},
        timeout=10,
    )
    if r.status_code != 200:
        log.error("Slack error: %s %s", r.status_code, r.text)
    else:
        log.info("Slack sent — %s flags for %s", flag_count, name)


# ---------------------------------------------------------------------------
# Lead pipeline
# ---------------------------------------------------------------------------

def process_lead(lead: dict):
    # Normalize field names from various CRM/Zapier formats
    if not lead.get("address"):
        if lead.get("full_address"):
            lead["address"] = lead["full_address"]
        elif lead.get("address1"):
            parts = [lead.get("address1", ""), lead.get("city", ""),
                     lead.get("state", ""), lead.get("postal_code", "")]
            lead["address"] = ", ".join(p for p in parts if p)
    if not lead.get("name"):
        lead["name"] = (lead.get("full_name") or
                        f"{lead.get('first_name', '')} {lead.get('last_name', '')}".strip())

    address = lead.get("address", "")
    log.info("Processing lead — %s", address or "(no address)")

    if not address:
        return

    # ── 0. Duplicate check (address only — zpid checked again after lookup) ─
    existing = is_duplicate(address)
    if existing:
        log.info(
            "Duplicate lead skipped — already processed %s on %s (flags: %s)",
            address, existing["processed_at"], existing["flags_hit"]
        )
        return

    # ── 1. Try Zillow first (direct property match + Zestimate) ────────────
    prop = zillow_lookup(address)

    # ── 1b. Dedup check by zpid now that we have it ────────────────────────
    if prop.get("zpid"):
        existing = is_duplicate(address, zpid=prop["zpid"])
        if existing:
            log.info(
                "Duplicate lead skipped (zpid %s) — already processed %s on %s",
                prop["zpid"], address, existing["processed_at"]
            )
            return

    # ── 2. Fill in gaps with Redfin GIS (active listings / recent sales) ───
    if not prop.get("zpid"):
        # Zillow didn't find it; try Redfin
        zip_code = _extract_zip(address)
        redfin   = redfin_lookup(address, zip_code=zip_code)
        # Merge: keep Zillow URL, overlay Redfin data
        zlink = prop.get("zillow_url") or zillow_url(address)
        prop  = redfin
        prop["zillow_url"] = zlink
    else:
        # Zillow found it; still grab Redfin active-listing status if missing
        if not prop.get("mls_status"):
            zip_code = _extract_zip(address)
            redfin   = redfin_lookup(address, zip_code=zip_code)
            if redfin.get("mls_status") and redfin["mls_status"] != "Not Listed":
                prop["mls_status"] = redfin["mls_status"]
                prop["redfin_url"] = redfin.get("redfin_url", "")

    zlink = prop.get("zillow_url") or zillow_url(address)
    prop["zillow_url"] = zlink

    # ── 3. Pre-check: only call Claude if at least one data flag is present ─
    price    = prop.get("price") or prop.get("zestimate") or 0
    has_data = (
        prop.get("zpid") or
        prop.get("mls_status") not in (None, "", "Not Listed") or
        prop.get("last_sold_within_36mo") or
        float(price) >= MIN_VALUE_THRESHOLD
    )
    if not has_data:
        log.info("No meaningful property data found — skipping AI for %s", address)
        return

    try:
        analysis = analyze_lead(lead, prop)
    except Exception as exc:
        log.error("Claude error: %s", exc)
        analysis = {"text": f"Analysis unavailable: {exc}",
                    "flags": {}, "flags_hit": [], "flag_count": 0}

    if analysis["flag_count"] == 0:
        log.info("No flags — skipping Slack for %s", address)
        record_lead(lead, prop, analysis, slack_sent=False)
        return

    slack_ok = False
    try:
        send_slack(lead, prop, analysis, zlink)
        slack_ok = True
    except Exception as exc:
        log.error("Slack error: %s", exc)

    record_lead(lead, prop, analysis, slack_sent=slack_ok)


# ---------------------------------------------------------------------------
# FastAPI
# ---------------------------------------------------------------------------

app = FastAPI(title="Lead Review Bot", version="2.0")


@app.get("/health")
def health():
    return {"status": "ok", "threshold": MIN_VALUE_THRESHOLD}


@app.get("/api/leads")
def api_leads(limit: int = 50, slack_sent: int = None, min_flags: int = 0):
    """Query processed leads from the database."""
    query  = "SELECT * FROM leads WHERE flag_count >= %s"
    params = [min_flags]
    if slack_sent is not None:
        query  += " AND slack_sent = %s"
        params.append(slack_sent)
    query += " ORDER BY processed_at DESC LIMIT %s"
    params.append(limit)
    conn = _db()
    try:
        with conn.cursor() as cur:
            cur.execute(query, params)
            rows = cur.fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


@app.post("/webhook/lead")
async def webhook_lead(request: Request, background_tasks: BackgroundTasks):
    content_type = request.headers.get("content-type", "")
    try:
        if "application/json" in content_type:
            lead = await request.json()
        elif "form" in content_type:
            form = await request.form()
            lead = dict(form)
        else:
            body = await request.body()
            lead = json.loads(body) if body else {}
    except Exception as exc:
        raise HTTPException(400, f"Could not parse body: {exc}")

    if not lead:
        raise HTTPException(400, "Empty payload")

    log.info("Lead received: %s", lead)
    background_tasks.add_task(process_lead, lead)
    return JSONResponse({"status": "received"})


if __name__ == "__main__":
    log.info("Lead Review Bot — threshold $%s — port %s", f"{MIN_VALUE_THRESHOLD:,}", PORT)
    uvicorn.run(app, host="0.0.0.0", port=PORT)
