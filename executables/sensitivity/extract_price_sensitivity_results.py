#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Price Sensitivity Analysis — Results Extraction & Excel Aggregation

Reads every reactor's *_Comprehensive_TEA_Summary.txt (and hourly opt CSV)
for 6 price scenarios + the cs1 baseline, and writes a multi-sheet Excel
workbook organised by the 4 manuscript cases:

  Case 1 — Existing nuclear, no H2 (w/ and w/o 45U)
  Case 2 — Existing nuclear + retrofit H2/battery (w/ and w/o 45U)
  Case 3 — Greenfield 60-yr new build (Baseline / 45Y PTC / 48E ITC)
  Case 4 — Greenfield 80-yr new build (Baseline / 45Y PTC / 48E ITC)

Output:
  output/tea/sa_price/price_sensitivity_results_<timestamp>.xlsx

Sheets:
  All_Results          — one row per (scenario, reactor), all metrics
  C1_Nuclear_Baseline  — Case 1 metrics pivoted across price scenarios
  C2_Retrofit_Hybrid   — Case 2 metrics pivoted across price scenarios
  C3_Greenfield_60yr   — Case 3 (60yr) pivoted across price scenarios
  C4_Greenfield_80yr   — Case 4 (80yr) pivoted across price scenarios
  Scenario_Summary     — aggregate stats per price scenario
  ISO_Summary          — aggregate stats per (ISO, scenario)
  Change_vs_Baseline   — absolute & % change vs. baseline price
  LCOH_Breakdown       — LCOH component decomposition per reactor × scenario
"""

import re
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
TEA_SA_DIR   = PROJECT_ROOT / "output" / "tea" / "sa_price"
TEA_CS1_DIR  = PROJECT_ROOT / "output" / "tea" / "cs1"
OPT_SA_DIR   = PROJECT_ROOT / "output" / "opt" / "sa_price"
OPT_CS1_DIR  = PROJECT_ROOT / "output" / "opt" / "cs1"
EXCEL_OUT    = TEA_SA_DIR / f"price_sensitivity_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"

SCENARIO_ORDER = [
    "baseline",
    "price_m10pct",
    "price_m5pct",
    "price_p5pct",
    "price_p10pct",
    "price_p20pct",
    "price_p30pct",
]
SCENARIO_LABELS = {
    "baseline":     "Baseline (0%)",
    "price_m10pct": "−10%",
    "price_m5pct":  "−5%",
    "price_p5pct":  "+5%",
    "price_p10pct": "+10%",
    "price_p20pct": "+20%",
    "price_p30pct": "+30%",
}

REACTOR_PATTERN = re.compile(r'^(.+?)_(\d+)_([A-Z]+)_(\d+)$')

# Greenfield policy short names → column key
GF_POLICY_MAP = {
    "baseline": "baseline",
    "45y ptc":  "45y_ptc",
    "45y":      "45y_ptc",
    "48e itc":  "48e_itc",
    "48e":      "48e_itc",
}


# ---------------------------------------------------------------------------
# Low-level parsing helpers
# ---------------------------------------------------------------------------
def _num(s) -> float | None:
    """Convert a string to float; return None if N/A or unparseable."""
    if s is None:
        return None
    s = str(s).strip().replace(",", "").replace("$", "").replace("%", "")
    if s.upper() in ("N/A", "NA", "NAN", "NONE", ""):
        return None
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


def _find(pattern: str, text: str, group: int = 1) -> str | None:
    m = re.search(pattern, text, re.IGNORECASE)
    return m.group(group).strip() if m else None


def _extract_section(text: str, start_pat: str, end_pat: str | None = None) -> str:
    """Return the substring of *text* from the first match of *start_pat*
    to the first match of *end_pat* (exclusive), or to end-of-string."""
    m = re.search(start_pat, text, re.IGNORECASE | re.DOTALL)
    if not m:
        return ""
    start = m.start()
    tail = text[start:]
    if end_pat:
        e = re.search(end_pat, tail, re.IGNORECASE | re.DOTALL)
        if e:
            return tail[: e.start()]
    return tail


def _npv_irr_payback(block: str, prefix: str, d: dict):
    """Extract NPV / IRR / Payback from a text block and store with *prefix*."""
    npv_raw = _find(r'NPV\s*:\s*\$?([-\d,]+)', block)
    irr_raw = _find(r'IRR\s*:\s*([-\d.]+)%', block)
    pb_raw  = _find(r'Payback\s*:\s*([-\d.]+)\s*years?', block)
    d[f"{prefix}_npv"]     = _num(npv_raw)
    d[f"{prefix}_irr_pct"] = _num(irr_raw)
    d[f"{prefix}_payback_yrs"] = _num(pb_raw)


# ---------------------------------------------------------------------------
# Greenfield table parser  (used for Cases 3 & 4)
# ---------------------------------------------------------------------------
def _parse_greenfield_table(text: str) -> dict:
    """
    Parse the 60yr/80yr × 3-policy comparison table from the Comprehensive
    TEA Summary.  Returns dict keyed by e.g. "60yr_baseline", "80yr_45y_ptc".
    """
    results: dict = {}
    # Each data row looks like:
    # "  60 years  | Baseline     |    -7821.7 |   -5.1% |           N/A |  $17.790/kg |   $81.83/MWh |          N/A"
    row_re = re.compile(
        r'(\d+)\s*years?\s*\|\s*(.+?)\s*\|'       # lifecycle | policy label
        r'\s*([-\d.]+)\s*\|'                        # NPV ($M)
        r'\s*([-\d.%NA/]+)\s*\|'                    # IRR
        r'\s*([-\d.NA/\s]+?)\s*\|'                  # Payback
        r'\s*\$?\s*([\d.]+)/kg\s*\|'               # LCOH
        r'\s*\$?\s*([\d.]+)/MWh',                   # LCOE
        re.IGNORECASE,
    )
    for m in row_re.finditer(text):
        lifecycle = int(m.group(1))
        policy_raw = m.group(2).strip().lower()
        npv_m   = _num(m.group(3))
        irr     = _num(re.sub(r'[%NA/]', '', m.group(4)))
        payback = None if "n/a" in m.group(5).lower() else _num(m.group(5))
        lcoh    = _num(m.group(6))
        lcoe    = _num(m.group(7))

        # Normalise policy name
        policy_key = None
        for raw_key, norm_key in GF_POLICY_MAP.items():
            if raw_key in policy_raw:
                policy_key = norm_key
                break
        if policy_key is None:
            policy_key = policy_raw.replace(" ", "_")

        table_key = f"{lifecycle}yr_{policy_key}"
        results[table_key] = {
            "npv_m": npv_m, "irr_pct": irr,
            "payback_yrs": payback, "lcoh": lcoh, "lcoe_mwh": lcoe,
        }
    return results


# ---------------------------------------------------------------------------
# LCOH breakdown extractor
# ---------------------------------------------------------------------------
def _extract_lcoh_breakdown(text: str) -> dict:
    """Extract LCOH component breakdown ($/kg and %) from comprehensive text."""
    d: dict = {}
    d["c2_lcoh_total"] = _num(_find(r'Total LCOH:\s*\$?([\d.]+)/kg', text))
    comp_re = re.compile(
        r'^\s{6,}(.+?)\s*:\s*\$?\s*([-\d.]+)/kg\s*\(\s*([\d.]+)%\)',
        re.MULTILINE,
    )
    for m in comp_re.finditer(text):
        name = m.group(1).strip().lower().replace(" ", "_").replace("/", "_")
        # Skip duplicate entries by keeping only first occurrence
        key_usd = f"lcoh_{name}_usd"
        if key_usd not in d:
            d[key_usd] = _num(m.group(2))
            d[f"lcoh_{name}_pct"] = _num(m.group(3))
    return d


# ---------------------------------------------------------------------------
# Main case extractor — reads *_Comprehensive_TEA_Summary.txt
# ---------------------------------------------------------------------------
def extract_all_cases(comp_path: Path) -> dict:
    """
    Extract all manuscript-case metrics from one
    *_Comprehensive_TEA_Summary.txt file.
    Returns a flat dict with c1_/c2_/c3_/c4_ prefixed keys.
    """
    if not comp_path or not comp_path.exists():
        return {}

    text = comp_path.read_text(encoding="utf-8", errors="replace")
    d: dict = {}

    # ---- Plant / project configuration ----
    d["project_lifetime_yrs"] = _num(_find(
        r'Project Lifecycle\s*:\s*(\d+)\s*years', text))
    d["discount_rate_pct"]    = _num(_find(
        r'Discount Rate\s*:\s*([\d.]+)%', text))
    d["turbine_capacity_mw"]  = _num(_find(
        r'Nuclear Unit Capacity \(MW\)\s*:\s*([\d,.]+)', text))
    d["thermal_capacity_mwt"] = _num(_find(
        r'Nuclear Unit Thermal Capacity \(MWt\)\s*:\s*([\d,.]+)', text))
    d["thermal_efficiency"]   = _num(_find(
        r'Nuclear Unit Thermal Efficiency\s*:\s*([\d.]+)\s*\(', text))

    # ── isolate sections ──────────────────────────────────────────────────
    # (use heading patterns that are unique enough in the file)
    sec1 = _extract_section(text,
        r'\n2\. Case 1:',
        r'\n3\. Case 2:')
    sec2 = _extract_section(text,
        r'\n3\. Case 2:',
        r'\n4\. Case 3:')
    sec7 = _extract_section(text,
        r'\n7\. Detailed Performance',
        r'\n8\. Core Economic')

    # ── CASE 1 ─────────────────────────────────────────────────────────────
    # Without 45U — isolate by ending at the 45U policy block
    blk1_no45u = _extract_section(sec1,
        r'Financial Metrics \(NPV.*?without 45U\)',
        r'45U PTC Policy')
    _npv_irr_payback(blk1_no45u, "c1_no45u", d)

    # LCOE
    d["c1_lcoe_mwh"] = _num(_find(
        r'LCOE \(Nuclear OPEX only\)\s*:\s*\$?([\d.]+)/MWh', sec1))

    # With 45U — start from the "45U PTC Policy Impact:" header so we never
    # accidentally match the "without 45U" block above
    blk1_policy = _extract_section(sec1,
        r'45U PTC Policy Impact:',
        r'Key Annual Operating')
    d["c1_with45u_npv"]         = _num(_find(r'NPV\s*:\s*\$?([-\d,]+)', blk1_policy))
    d["c1_with45u_irr_pct"]     = _num(_find(r'IRR\s*:\s*([-\d.]+)%', blk1_policy))
    d["c1_with45u_payback_yrs"] = _num(_find(
        r'Payback\s*:\s*([-\d.]+)\s*years?', blk1_policy))

    # 45U programme totals
    d["c1_45u_npv_improvement"] = _num(_find(
        r'NPV Improvement\s*[^$\n]*:\s*\$?([\d,]+)', sec1))
    d["c1_45u_credits_total"]   = _num(_find(
        r'Total 45U Credits\s*:\s*\$?([\d,]+)', sec1))

    # Annual operating data
    d["c1_annual_gen_mwh"]   = _num(_find(
        r'Annual Generation\s*:\s*([\d,]+)\s*MWh', sec1))
    d["c1_annual_revenue"]   = _num(_find(
        r'Annual Revenue\s*:\s*\$?([\d,]+)', sec1))
    d["c1_annual_opex"]      = _num(_find(
        r'Total Annual OPEX\s*:\s*\$?([\d,]+)', sec1))

    # ── CASE 2 ─────────────────────────────────────────────────────────────
    # System capacities
    d["c2_electrolyzer_mw"]   = _num(_find(
        r'Electrolyzer Capacity\s*:\s*([\d,.]+)\s*MW', sec2))
    d["c2_h2_storage_kg"]     = _num(_find(
        r'H2 Storage Capacity\s*:\s*([\d,]+)\s*kg', sec2))
    d["c2_battery_energy_mwh"]= _num(_find(
        r'Battery Energy Capacity\s*:\s*([\d.]+)\s*MWh', sec2))
    d["c2_battery_power_mw"]  = _num(_find(
        r'Battery Power Capacity\s*:\s*([\d.]+)\s*MW', sec2))

    # Financial — without 45U
    blk2_no45u = _extract_section(sec2,
        r'Financial Metrics \(Integrated System.*?without 45U\)',
        r'45U PTC Policy Impact on Nuclear')
    _npv_irr_payback(blk2_no45u, "c2_no45u", d)

    # Financial — with 45U
    blk2_with45u = _extract_section(sec2,
        r'45U PTC Policy Impact on Nuclear',
        r'LCOH \(Detailed Composition\)')
    _npv_irr_payback(blk2_with45u, "c2_with45u", d)

    # LCOH total
    d["c2_lcoh_usd_per_kg"] = _num(_find(
        r'Total LCOH:\s*\$?([\d.]+)/kg', sec2))

    # Annual operating
    d["c2_total_revenue"]   = _num(_find(
        r'Total Annual Revenue\s*:\s*\$?([\d,]+)', sec2))
    d["c2_total_opex"]      = _num(_find(
        r'Total System OPEX\s*:\s*\$?([\d,]+)', sec2))
    d["c2_annual_h2_kg"]    = _num(_find(
        r'Annual H2 Production\s*:\s*([\d,]+)\s*kg', sec2))
    d["c2_as_revenue"]      = _num(_find(
        r'Ancillary Services Revenue\s*:\s*\$?([\d,]+)', sec2))

    # Performance (Section 7)
    d["c2_electrolyzer_cf_pct"]    = _num(_find(
        r'Electrolyzer Capacity Factor\s*:\s*([\d.]+)%', sec7))
    d["c2_turbine_cf_pct"]         = _num(_find(
        r'Turbine Capacity Factor\s*:\s*([\d.]+)%', sec7))
    d["c2_avg_electricity_price"]  = _num(_find(
        r'Average Electricity Price\s*:\s*\$?([\d.]+)/MWh', sec7))
    d["c2_annual_nuclear_gen_mwh"] = _num(_find(
        r'Total Nuclear Generation\s*:\s*([\d,]+)\s*MWh', sec7))
    d["c2_as_revenue_sec7"]        = _num(_find(
        r'Total AS Revenue\s*:\s*\$?([\d,]+)', sec7))

    # CAPEX (Section 7 cost composition)
    sec7b = _extract_section(sec7,
        r'B\. Detailed Cost Composition',
        r'C\. MACRS')
    d["c2_total_capex"]          = _num(_find(
        r'Total CAPEX\s*:\s*\$?([\d,]+)', sec7b))
    d["c2_electrolyzer_capex"]   = _num(_find(
        r'Electrolyzer System\s*:\s*\$?([\d,]+)\s*\(', sec7b))
    d["c2_h2_storage_capex"]     = _num(_find(
        r'H2 Storage System\s*:\s*\$?([\d,]+)\s*\(', sec7b))
    d["c2_nuclear_fixed_om"]     = _num(_find(
        r'Nuclear Fixed O&M\s*:\s*\$?([\d,]+)', sec7b))
    d["c2_h2_battery_opex"]      = _num(_find(
        r'H2/Battery VOM \(Electrolyzer\)\s*:\s*\$?([\d,]+)', sec7b))

    # ── CASES 3 & 4 (Greenfield) ───────────────────────────────────────────
    # Extracted from the Case 5 comparison table in the comprehensive file
    gf = _parse_greenfield_table(text)

    gf_map = [
        ("60", "baseline",  "c3"),
        ("60", "45y_ptc",   "c3"),
        ("60", "48e_itc",   "c3"),
        ("80", "baseline",  "c4"),
        ("80", "45y_ptc",   "c4"),
        ("80", "48e_itc",   "c4"),
    ]
    for lifecycle, policy_key, case_prefix in gf_map:
        tbl_key = f"{lifecycle}yr_{policy_key}"
        row = gf.get(tbl_key, {})
        col_prefix = f"{case_prefix}_{policy_key}"
        d[f"{col_prefix}_npv_m"]       = row.get("npv_m")
        d[f"{col_prefix}_irr_pct"]     = row.get("irr_pct")
        d[f"{col_prefix}_payback_yrs"] = row.get("payback_yrs")
        d[f"{col_prefix}_lcoh"]        = row.get("lcoh")
        d[f"{col_prefix}_lcoe_mwh"]    = row.get("lcoe_mwh")

    return d


# ---------------------------------------------------------------------------
# Opt hourly CSV statistics
# ---------------------------------------------------------------------------
def extract_opt_stats(csv_path: Path | None) -> dict:
    d: dict = {}
    if not csv_path or not csv_path.exists():
        return d
    try:
        df = pd.read_csv(csv_path)
        sums = [
            ("opt_annual_profit_usd",      "Profit_Hourly_USD"),
            ("opt_energy_revenue_usd",     "Revenue_Energy_USD"),
            ("opt_h2_revenue_usd",         "Revenue_Hydrogen_Sales_USD"),
            ("opt_as_revenue_usd",         "Revenue_Ancillary_USD"),
            ("opt_total_revenue_usd",      "Revenue_Total_USD"),
            ("opt_annual_opex_usd",        "Cost_HourlyOpex_Total_USD"),
            ("opt_h2_battery_opex_usd",    "Cost_HourlyOpex_H2_Battery_USD"),
            ("opt_annual_h2_production_kg","mHydrogenProduced_kg_hr"),
        ]
        for out_col, in_col in sums:
            if in_col in df.columns:
                d[out_col] = df[in_col].sum()
        if "EnergyPrice_LMP_USDperMWh" in df.columns:
            d["opt_avg_electricity_price"] = df["EnergyPrice_LMP_USDperMWh"].mean()
        for out_col, in_col in [
            ("opt_electrolyzer_capacity_mw", "Electrolyzer_Capacity_MW"),
            ("opt_h2_storage_capacity_kg",   "H2_Storage_Capacity_kg"),
            ("opt_battery_capacity_mwh",     "Battery_Capacity_MWh"),
        ]:
            if in_col in df.columns:
                vals = df[in_col].dropna()
                d[out_col] = float(vals.iloc[0]) if not vals.empty else None
    except Exception as exc:
        print(f"  Warning: cannot read opt CSV {csv_path.name}: {exc}")
    return d


# ---------------------------------------------------------------------------
# Directory discovery helpers
# ---------------------------------------------------------------------------
def find_reactor_dirs(tea_dir: Path) -> list:
    """Returns list of (plant_name, gen_id, iso, rem_yrs, dir_path)."""
    out = []
    if not tea_dir.exists():
        return out
    for d in sorted(tea_dir.iterdir()):
        if not d.is_dir():
            continue
        m = REACTOR_PATTERN.match(d.name)
        if m:
            out.append((m.group(1), m.group(2), m.group(3), m.group(4), d))
    return out


def find_comprehensive_file(reactor_dir: Path, iso: str) -> Path | None:
    for name in [
        f"{iso}_Comprehensive_TEA_Summary.txt",
        f"{iso}_TEA_Summary_Report.txt",
    ]:
        p = reactor_dir / name
        if p.exists():
            return p
    for p in reactor_dir.glob("*_Comprehensive_TEA_Summary.txt"):
        return p
    return None


def find_opt_csv(scenario: str, plant_name: str, gen_id: str,
                 iso: str, rem_yrs: str) -> Path | None:
    label = f"{plant_name}_{gen_id}_{iso}_{rem_yrs}_hourly_results.csv"
    base = OPT_CS1_DIR if scenario == "baseline" else OPT_SA_DIR / scenario
    p = base / label
    return p if p.exists() else None


# ---------------------------------------------------------------------------
# Collect all records
# ---------------------------------------------------------------------------
def collect_all_results() -> tuple[list, list]:
    """Returns (all_records, lcoh_records)."""
    all_records:  list[dict] = []
    lcoh_records: list[dict] = []

    for scenario in SCENARIO_ORDER:
        label = SCENARIO_LABELS[scenario]
        tea_dir = TEA_CS1_DIR if scenario == "baseline" else TEA_SA_DIR / scenario
        reactors = find_reactor_dirs(tea_dir)
        if not reactors:
            print(f"  [{scenario}] No reactor directories found under {tea_dir}")
            continue
        print(f"  [{scenario}] {len(reactors)} reactors")

        for plant_name, gen_id, iso, rem_yrs, rdir in reactors:
            comp_file = find_comprehensive_file(rdir, iso)
            opt_csv   = find_opt_csv(scenario, plant_name, gen_id, iso, rem_yrs)

            case_metrics = extract_all_cases(comp_file)
            opt_metrics  = extract_opt_stats(opt_csv)

            # Build LCOH breakdown record separately
            lcoh_d: dict = {}
            if comp_file and comp_file.exists():
                text = comp_file.read_text(encoding="utf-8", errors="replace")
                lcoh_d = _extract_lcoh_breakdown(text)

            record = {
                "scenario":        scenario,
                "scenario_label":  label,
                "plant_name":      plant_name,
                "generator_id":    gen_id,
                "iso_region":      iso,
                "remaining_years": int(rem_yrs),
                "reactor_label":   f"{plant_name}_{gen_id}",
                "comp_file_found": comp_file is not None,
                "opt_csv_found":   opt_csv is not None,
                **case_metrics,
                **opt_metrics,
            }
            all_records.append(record)

            lcoh_rec = {
                "scenario":        scenario,
                "scenario_label":  label,
                "plant_name":      plant_name,
                "generator_id":    gen_id,
                "iso_region":      iso,
                "remaining_years": int(rem_yrs),
                "reactor_label":   f"{plant_name}_{gen_id}",
                **lcoh_d,
            }
            lcoh_records.append(lcoh_rec)

    return all_records, lcoh_records


# ---------------------------------------------------------------------------
# Pivot helpers for focused case sheets
# ---------------------------------------------------------------------------
def _pivot_columns(df: pd.DataFrame, metric_cols: list[str]) -> pd.DataFrame:
    """
    Pivot df so index = (reactor_label, iso_region, remaining_years)
    and columns = scenario_label × metric (only *metric_cols* that exist).
    """
    available = [c for c in metric_cols if c in df.columns]
    if not available:
        return pd.DataFrame()
    pivot = df.pivot_table(
        index=["reactor_label", "iso_region", "remaining_years"],
        columns="scenario_label",
        values=available,
        aggfunc="first",
    )
    # Flatten: outer level = metric, inner = scenario → "metric | scenario"
    ordered_scenarios = [SCENARIO_LABELS[s] for s in SCENARIO_ORDER
                         if SCENARIO_LABELS[s] in pivot.columns.get_level_values(1)]
    pivot = pivot.reindex(ordered_scenarios, axis=1, level=1)
    pivot.columns = [f"{m} | {s}" for m, s in pivot.columns]
    return pivot.reset_index()


C1_METRICS = [
    "c1_no45u_npv", "c1_no45u_irr_pct", "c1_no45u_payback_yrs",
    "c1_lcoe_mwh",
    "c1_with45u_npv", "c1_with45u_irr_pct",
    "c1_45u_credits_total", "c1_45u_npv_improvement",
    "c1_annual_gen_mwh", "c1_annual_revenue", "c1_annual_opex",
]

C2_METRICS = [
    "c2_electrolyzer_mw", "c2_h2_storage_kg",
    "c2_no45u_npv", "c2_no45u_irr_pct", "c2_no45u_payback_yrs",
    "c2_with45u_npv", "c2_with45u_irr_pct", "c2_with45u_payback_yrs",
    "c2_lcoh_usd_per_kg",
    "c2_total_revenue", "c2_total_opex",
    "c2_annual_h2_kg", "c2_as_revenue",
    "c2_electrolyzer_cf_pct", "c2_turbine_cf_pct",
    "c2_avg_electricity_price",
    "c2_total_capex", "c2_electrolyzer_capex", "c2_h2_storage_capex",
    "opt_avg_electricity_price",
]

C3_METRICS = [
    "c3_baseline_npv_m", "c3_baseline_irr_pct", "c3_baseline_lcoh", "c3_baseline_lcoe_mwh",
    "c3_45y_ptc_npv_m",  "c3_45y_ptc_irr_pct",  "c3_45y_ptc_lcoh",  "c3_45y_ptc_lcoe_mwh",
    "c3_48e_itc_npv_m",  "c3_48e_itc_irr_pct",  "c3_48e_itc_lcoh",  "c3_48e_itc_lcoe_mwh",
]

C4_METRICS = [
    "c4_baseline_npv_m", "c4_baseline_irr_pct", "c4_baseline_lcoh", "c4_baseline_lcoe_mwh",
    "c4_45y_ptc_npv_m",  "c4_45y_ptc_irr_pct",  "c4_45y_ptc_lcoh",  "c4_45y_ptc_lcoe_mwh",
    "c4_48e_itc_npv_m",  "c4_48e_itc_irr_pct",  "c4_48e_itc_lcoh",  "c4_48e_itc_lcoe_mwh",
]


# ---------------------------------------------------------------------------
# Scenario summary (aggregate stats per price scenario)
# ---------------------------------------------------------------------------
SUMMARY_METRICS = [
    "c1_no45u_npv", "c1_with45u_npv",
    "c2_no45u_npv", "c2_no45u_irr_pct", "c2_no45u_payback_yrs",
    "c2_with45u_npv", "c2_with45u_irr_pct",
    "c2_lcoh_usd_per_kg",
    "c2_electrolyzer_mw", "c2_annual_h2_kg",
    "c2_total_revenue", "c2_avg_electricity_price",
    "c2_electrolyzer_cf_pct",
    "c3_baseline_npv_m", "c3_45y_ptc_npv_m", "c3_48e_itc_npv_m",
    "c4_baseline_npv_m", "c4_45y_ptc_npv_m", "c4_48e_itc_npv_m",
]

def build_scenario_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scenario in SCENARIO_ORDER:
        sdf = df[df["scenario"] == scenario]
        if sdf.empty:
            continue
        row = {
            "scenario":        scenario,
            "scenario_label":  SCENARIO_LABELS[scenario],
            "n_reactors":      len(sdf),
            "n_c2_npv_positive": int((sdf["c2_no45u_npv"].dropna() > 0).sum())
                if "c2_no45u_npv" in sdf else None,
            "n_c2_npv_negative": int((sdf["c2_no45u_npv"].dropna() <= 0).sum())
                if "c2_no45u_npv" in sdf else None,
        }
        for m in SUMMARY_METRICS:
            if m not in sdf.columns:
                continue
            col = sdf[m].dropna()
            row[f"{m}_mean"]   = col.mean()   if not col.empty else None
            row[f"{m}_median"] = col.median() if not col.empty else None
            row[f"{m}_min"]    = col.min()    if not col.empty else None
            row[f"{m}_max"]    = col.max()    if not col.empty else None
            row[f"{m}_std"]    = col.std()    if not col.empty else None
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# ISO summary
# ---------------------------------------------------------------------------
ISO_METRICS = [
    "c2_no45u_npv", "c2_no45u_irr_pct",
    "c2_lcoh_usd_per_kg", "c2_electrolyzer_mw",
    "c2_annual_h2_kg", "c2_total_revenue",
    "c2_avg_electricity_price", "c2_electrolyzer_cf_pct",
]

def build_iso_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scenario in SCENARIO_ORDER:
        for iso in sorted(df["iso_region"].unique()):
            sdf = df[(df["scenario"] == scenario) & (df["iso_region"] == iso)]
            if sdf.empty:
                continue
            row = {
                "scenario":       scenario,
                "scenario_label": SCENARIO_LABELS[scenario],
                "iso_region":     iso,
                "n_reactors":     len(sdf),
            }
            for m in ISO_METRICS:
                if m not in sdf.columns:
                    continue
                col = sdf[m].dropna()
                row[f"{m}_mean"] = col.mean() if not col.empty else None
                row[f"{m}_min"]  = col.min()  if not col.empty else None
                row[f"{m}_max"]  = col.max()  if not col.empty else None
            rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Change vs baseline
# ---------------------------------------------------------------------------
CHANGE_METRICS = [
    "c2_no45u_npv", "c2_no45u_irr_pct", "c2_no45u_payback_yrs",
    "c2_with45u_npv", "c2_with45u_irr_pct",
    "c2_lcoh_usd_per_kg",
    "c2_electrolyzer_mw", "c2_annual_h2_kg",
    "c2_total_revenue", "c2_avg_electricity_price",
    "c2_electrolyzer_cf_pct",
    "c1_no45u_npv", "c1_with45u_npv",
    "c3_baseline_npv_m", "c3_45y_ptc_npv_m",
    "c4_baseline_npv_m", "c4_45y_ptc_npv_m",
]

def build_change_vs_baseline(df: pd.DataFrame) -> pd.DataFrame:
    avail = [c for c in CHANGE_METRICS if c in df.columns]
    baseline_df = df[df["scenario"] == "baseline"].set_index("reactor_label")
    rows = []
    for scenario in SCENARIO_ORDER[1:]:
        sdf = df[df["scenario"] == scenario]
        for _, row in sdf.iterrows():
            reactor = row["reactor_label"]
            if reactor not in baseline_df.index:
                continue
            base = baseline_df.loc[reactor]
            rec = {
                "scenario":        scenario,
                "scenario_label":  SCENARIO_LABELS[scenario],
                "plant_name":      row["plant_name"],
                "generator_id":    row["generator_id"],
                "iso_region":      row["iso_region"],
                "remaining_years": row["remaining_years"],
                "reactor_label":   reactor,
            }
            for m in avail:
                v_new  = row.get(m)
                v_base = base.get(m)
                rec[f"{m}_baseline"] = v_base
                rec[f"{m}_new"]      = v_new
                try:
                    if (v_base not in (None, 0)
                            and v_new is not None
                            and not (isinstance(v_base, float) and np.isnan(v_base))
                            and not (isinstance(v_new,  float) and np.isnan(v_new))):
                        rec[f"{m}_abs_chg"] = v_new - v_base
                        rec[f"{m}_pct_chg"] = (v_new - v_base) / abs(v_base) * 100
                    else:
                        rec[f"{m}_abs_chg"] = rec[f"{m}_pct_chg"] = None
                except Exception:
                    rec[f"{m}_abs_chg"] = rec[f"{m}_pct_chg"] = None
            rows.append(rec)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Excel writer
# ---------------------------------------------------------------------------
def write_excel(all_df: pd.DataFrame, lcoh_df: pd.DataFrame) -> Path:
    c1_pivot   = _pivot_columns(all_df, C1_METRICS)
    c2_pivot   = _pivot_columns(all_df, C2_METRICS)
    c3_pivot   = _pivot_columns(all_df, C3_METRICS)
    c4_pivot   = _pivot_columns(all_df, C4_METRICS)
    summary_df = build_scenario_summary(all_df)
    iso_df     = build_iso_summary(all_df)
    change_df  = build_change_vs_baseline(all_df)

    EXCEL_OUT.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(EXCEL_OUT, engine="openpyxl") as writer:

        def _sheet(df: pd.DataFrame, name: str, freeze: str = "A2"):
            if df.empty:
                pd.DataFrame({"(no data)": []}).to_excel(
                    writer, sheet_name=name, index=False)
                return
            df.to_excel(writer, sheet_name=name, index=False)
            ws = writer.sheets[name]
            ws.freeze_panes = freeze
            for col_cells in ws.columns:
                max_len = max(
                    len(str(c.value)) if c.value is not None else 0
                    for c in col_cells
                )
                ws.column_dimensions[col_cells[0].column_letter].width = min(
                    max_len + 2, 50)

        _sheet(all_df,    "All_Results",          freeze="C2")
        _sheet(c1_pivot,  "C1_Nuclear_Baseline",  freeze="D2")
        _sheet(c2_pivot,  "C2_Retrofit_Hybrid",   freeze="D2")
        _sheet(c3_pivot,  "C3_Greenfield_60yr",   freeze="D2")
        _sheet(c4_pivot,  "C4_Greenfield_80yr",   freeze="D2")
        _sheet(summary_df,"Scenario_Summary",      freeze="B2")
        _sheet(iso_df,    "ISO_Summary",            freeze="C2")
        _sheet(change_df, "Change_vs_Baseline",    freeze="H2")
        _sheet(lcoh_df,   "LCOH_Breakdown",        freeze="C2")

    return EXCEL_OUT


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("  Price Sensitivity — Results Extraction (4-Case Structure)")
    print("=" * 70)
    print(f"TEA SA dir  : {TEA_SA_DIR}")
    print(f"TEA CS1 dir : {TEA_CS1_DIR}")
    print(f"Output      : {EXCEL_OUT}\n")

    print("Collecting results …")
    all_records, lcoh_records = collect_all_results()

    if not all_records:
        print("\nNo results found.")
        return

    all_df  = pd.DataFrame(all_records)
    lcoh_df = pd.DataFrame(lcoh_records)

    print(f"\nTotal records : {len(all_df)}")
    print(f"  Scenarios   : {all_df['scenario'].nunique()}")
    print(f"  Reactors    : {all_df['reactor_label'].nunique()}")
    print(f"  ISO regions : {sorted(all_df['iso_region'].unique())}")

    print("\nWriting Excel workbook …")
    out_path = write_excel(all_df, lcoh_df)

    print(f"\nDone → {out_path}")
    print("\nSheets:")
    print("  All_Results          — all metrics, one row per (scenario, reactor)")
    print("  C1_Nuclear_Baseline  — Case 1: nuclear-only (±45U) × price scenario")
    print("  C2_Retrofit_Hybrid   — Case 2: retrofit H2/battery (±45U) × price scenario")
    print("  C3_Greenfield_60yr   — Case 3: new build 60yr (Baseline/45Y/48E) × price scenario")
    print("  C4_Greenfield_80yr   — Case 4: new build 80yr (Baseline/45Y/48E) × price scenario")
    print("  Scenario_Summary     — aggregate stats per price scenario")
    print("  ISO_Summary          — aggregate stats per (ISO, scenario)")
    print("  Change_vs_Baseline   — absolute & % change vs. baseline price")
    print("  LCOH_Breakdown       — LCOH component decomposition")


if __name__ == "__main__":
    main()
