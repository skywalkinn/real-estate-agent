#!/usr/bin/env python3
"""
Batch Lead Tester
=================
Run your historical lead log through the bot pipeline and see results
in the terminal — without sending to Slack unless you pass --slack.

Usage:
    python batch_test.py leads.csv               # dry run, no Slack
    python batch_test.py leads.csv --slack        # send matches to Slack
    python batch_test.py leads.csv --limit 20     # first 20 only

CSV format (any subset of these columns works):
    address, name, email, phone, message, source

If your CSV uses different column names, just rename the headers before running.
"""

import os
import sys
import csv
import json
import time
import argparse
from datetime import datetime

# Load env before importing lead_bot
from dotenv import load_dotenv
load_dotenv()

# Patch out Slack if dry run (set before import so the module sees it)
_original_slack_url = os.getenv("SLACK_WEBHOOK_URL", "")

import lead_bot
from rich.console import Console
from rich.table import Table
from rich import box

console = Console()


def run_batch(csv_path: str, send_to_slack: bool = False, limit: int = None, args_random: int = None):

    # Read leads — override headers since the CSV is missing the first 3 column names
    FIELDNAMES = [
        "Event Category", "Event Type", "First Name", "Last Name",
        "Phone", "Email", "Address", "City", "State", "ZIP",
        "# Beds", "# Baths", "Square Feet", "Year Built", "Owner Occupied",
        "Date", "Account", "week", "Weekday", "Day of year", "date", "Year",
    ]
    try:
        with open(csv_path, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f, fieldnames=FIELDNAMES)
            next(reader)   # skip the broken header row
            leads  = [row for row in reader if any(row.values())]
    except FileNotFoundError:
        console.print(f"[red]File not found: {csv_path}[/red]")
        sys.exit(1)

    if limit:
        leads = leads[:limit]
    if args_random and len(leads) > args_random:
        import random as _random
        leads = _random.sample(leads, args_random)

    total = len(leads)
    console.print(f"\n[cyan]Processing {total} leads from {csv_path}[/cyan]")
    if not send_to_slack:
        console.print("[yellow]Dry run — Slack notifications disabled. Pass --slack to enable.[/yellow]\n")

    results = []

    for i, raw_lead in enumerate(leads, 1):
        # Map your CRM columns
        street = raw_lead.get("Address", "").strip()
        city   = raw_lead.get("City", "").strip()
        state  = raw_lead.get("State", "").strip()
        zip_   = raw_lead.get("ZIP", "").strip()
        full_address = ", ".join(p for p in [street, city, state, zip_] if p)

        lead = {
            "address":      full_address,
            "name":         f"{raw_lead.get('First Name','').strip()} {raw_lead.get('Last Name','').strip()}".strip(),
            "email":        raw_lead.get("Email", "").strip(),
            "phone":        raw_lead.get("Phone", "").strip(),
            "source":       raw_lead.get("Account") or raw_lead.get("Event Type") or "Lead Log",
            "event_type":   raw_lead.get("Event Type", "").strip(),
            "event_category": raw_lead.get("Event Category", "").strip(),
            "date":         raw_lead.get("Date") or raw_lead.get("date", ""),
            "owner_occupied": raw_lead.get("Owner Occupied", "").strip(),
            # Pre-fill from CSV — used if Redfin lookup finds nothing
            "_beds":        raw_lead.get("# Beds", "").strip(),
            "_baths":       raw_lead.get("# Baths", "").strip(),
            "_sqft":        raw_lead.get("Square Feet", "").strip(),
            "_year_built":  raw_lead.get("Year Built", "").strip(),
        }

        address = lead["address"]
        name    = lead["name"] or f"Lead #{i}"

        console.print(f"[dim]({i}/{total})[/dim] {address or name} ... ", end="")

        # Property lookup — Zillow first, Redfin as fallback
        zip_code = raw_lead.get("ZIP", "").strip()
        if address:
            prop  = lead_bot.zillow_lookup(address)
            zlink = prop.get("zillow_url") or lead_bot.zillow_url(address)
            if not prop.get("zpid"):
                # Zillow blocked/not found — try Redfin
                redfin = lead_bot.redfin_lookup(address, zip_code=zip_code)
                prop   = redfin
                prop["zillow_url"] = zlink
        else:
            prop  = {}
            zlink = ""

        # Fill in blanks from CSV if neither source found them
        def _to_num(v):
            try: return float(str(v).replace(",", ""))
            except: return None

        if not prop.get("beds")       and lead.get("_beds"):       prop["beds"]       = _to_num(lead["_beds"])
        if not prop.get("baths")      and lead.get("_baths"):      prop["baths"]      = _to_num(lead["_baths"])
        if not prop.get("sqft")       and lead.get("_sqft"):       prop["sqft"]       = _to_num(lead["_sqft"])
        if not prop.get("year_built") and lead.get("_year_built"): prop["year_built"] = _to_num(lead["_year_built"])

        # AI analysis
        try:
            analysis = lead_bot.analyze_lead(lead, prop)
        except Exception as exc:
            console.print(f"[red]AI error: {exc}[/red]")
            analysis = {"text": str(exc), "flags": {}, "flags_hit": [], "flag_count": 0}

        flag_count = analysis["flag_count"]
        flags_hit  = analysis["flags_hit"]

        if flag_count == 0:
            console.print("[dim]0 flags — skip[/dim]")
        else:
            console.print(f"[green]{flag_count}/7 flags:[/green] {', '.join(flags_hit)}")
            if send_to_slack:
                try:
                    lead_bot.send_slack(lead, prop, analysis, zlink)
                    console.print(f"  [cyan]→ Slack sent[/cyan]")
                except Exception as exc:
                    console.print(f"  [red]→ Slack error: {exc}[/red]")

        results.append({
            "name":       name,
            "address":    address,
            "flag_count": flag_count,
            "flags_hit":  ", ".join(flags_hit) if flags_hit else "—",
            "price":      lead_bot._fmt(prop.get("price"), "$"),
            "estimate":   lead_bot._fmt(prop.get("redfin_estimate"), "$"),
            "mls_status": prop.get("mls_status") or "—",
            "analysis":   analysis["text"][:120] + "..." if len(analysis["text"]) > 120 else analysis["text"],
        })

        # Small delay to avoid hammering Redfin
        time.sleep(1)

    # ── Summary table ───────────────────────────────────────────────────────
    flagged = [r for r in results if r["flag_count"] > 0]

    console.print(f"\n[bold]Results: {len(flagged)}/{total} leads had at least one flag[/bold]\n")

    if flagged:
        tbl = Table(
            title="Flagged Leads",
            box=box.ROUNDED,
            show_lines=True,
            header_style="bold cyan",
        )
        tbl.add_column("#",          width=4,  justify="right", style="dim")
        tbl.add_column("Name",       min_width=16)
        tbl.add_column("Address",    min_width=28)
        tbl.add_column("Flags",      width=6,  justify="center")
        tbl.add_column("Flags Hit",  min_width=30)
        tbl.add_column("Price",      min_width=10, justify="right")
        tbl.add_column("Estimate",   min_width=10, justify="right")
        tbl.add_column("MLS Status", min_width=12)

        for i, r in enumerate(sorted(flagged, key=lambda x: -x["flag_count"]), 1):
            tbl.add_row(
                str(i),
                r["name"],
                r["address"],
                str(r["flag_count"]),
                r["flags_hit"],
                r["price"],
                r["estimate"],
                r["mls_status"],
            )
        console.print(tbl)

    # Save results to CSV
    out_path = f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["name","address","flag_count","flags_hit",
                                                "price","estimate","mls_status","analysis"])
        writer.writeheader()
        writer.writerows(results)

    console.print(f"\n[green]Full results saved to:[/green] [bold]{out_path}[/bold]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch lead tester")
    parser.add_argument("csv",            help="Path to your leads CSV file")
    parser.add_argument("--slack",          action="store_true", help="Send flagged leads to Slack")
    parser.add_argument("--limit", "-n",    type=int, default=None, help="Only process first N leads")
    parser.add_argument("--random", "-r",   type=int, default=None, help="Pick N leads at random", dest="random_n")
    args = parser.parse_args()

    if not args.slack:
        lead_bot.SLACK_WEBHOOK_URL = ""

    run_batch(args.csv, send_to_slack=args.slack, limit=args.limit, args_random=args.random_n)
