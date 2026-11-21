"""Unified PPMI curated-extract loader, PD subtyping feature panels, external 2-year anchors, and stratified reserve."""
from __future__ import annotations

import csv
import hashlib
import math
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from xml.etree.ElementTree import iterparse
from zipfile import ZipFile

import openpyxl
from openpyxl.utils import get_column_letter

_HERE = Path(__file__).resolve().parent
PROJECT = _HERE.parent if _HERE.name == "code" else _HERE

VERSIONS = (
    "20230612",
    "20240129",
    "20240729",
    "20241211",
    "20250321",
    "20250714",
    "20251112",
    "20260223",
    "20260319",
    "20260511",
    "20260810",
)
REFERENCE = VERSIONS[-1]
RESERVE_SEED = "P26-subtyping-eval-20261121"
RESERVE_FRACTION = 0.30
MIN_OBSERVED_FRACTION = 0.80

HIGHER_IS_BETTER = frozenset({"moca", "SDMTOTAL", "VLTANIM", "HVLTRDLY", "upsit"})


@dataclass(frozen=True)
class FeaturePanel:
    id: str
    label: str
    features: tuple[str, ...]

    @property
    def min_observed(self) -> int:
        return math.ceil(MIN_OBSERVED_FRACTION * len(self.features))


PANELS: tuple[FeaturePanel, ...] = (
    FeaturePanel(
        "P1_multidomain_11",
        "Multi-domain clinical (11 features)",
        (
            "updrs2_score",
            "updrs3_score",
            "pigd",
            "moca",
            "SDMTOTAL",
            "VLTANIM",
            "HVLTRDLY",
            "scopa",
            "gds",
            "ess",
            "rem",
        ),
    ),
    FeaturePanel(
        "P2_compact_7",
        "Compact clinical (7 features)",
        ("updrs2_score", "updrs3_score", "moca", "scopa", "gds", "ess", "rem"),
    ),
    FeaturePanel(
        "P3_multidomain_plus_upsit_12",
        "Multi-domain clinical + UPSIT (12 features)",
        (
            "updrs2_score",
            "updrs3_score",
            "pigd",
            "moca",
            "SDMTOTAL",
            "VLTANIM",
            "HVLTRDLY",
            "scopa",
            "gds",
            "ess",
            "rem",
            "upsit",
        ),
    ),
)

ALGORITHMS = ("kmeans", "ward", "gmm")
CLUSTER_COUNTS = (2, 3, 4)

EXTERNAL_ANCHORS = (
    ("2y_moca", "Two-year MoCA", "continuous"),
    ("2y_updrs3", "Two-year MDS-UPDRS III (OFF/untreated)", "continuous"),
    ("2y_updrs2", "Two-year MDS-UPDRS II", "continuous"),
    ("2y_adl", "Two-year Schwab-England ADL", "continuous"),
    ("2y_pm_any", "Two-year any progression milestone", "binary"),
    ("2y_cogstate_abn", "Two-year cognitive impairment (MCI/dementia)", "binary"),
    ("2y_adl_lt80", "Two-year Schwab-England ADL <80", "binary"),
)

BOUNDS = {
    "age": (18, 120),
    "SEX": (0, 1),
    "EDUCYRS": (0, 40),
    "moca": (0, 30),
    "SDMTOTAL": (0, 120),
    "VLTANIM": (0, 100),
    "HVLTRDLY": (0, 12),
    "updrs2_score": (0, 52),
    "updrs3_score": (0, 132),
    "pigd": (0, 20),
    "scopa": (0, 69),
    "gds": (0, 15),
    "ess": (0, 24),
    "rem": (0, 13),
    "upsit": (0, 40),
    "MSEADLG": (0, 100),
    "cogstate": (1, 3),
    "pm_any": (0, 1),
}

REQUIRED_COLUMNS = (
    {"PATNO", "COHORT", "EVENT_ID", "visit_date", "subgroup", "age", "SEX", "EDUCYRS", "MSEADLG", "pm_any", "cogstate"}
    | set().union(*(p.features for p in PANELS))
)


def extract_filename(version: str) -> str:
    if version == "20230612":
        return "PPMI_Curated_Data_Cut_Public_20230612_rev.csv"
    return f"PPMI_Curated_Data_Cut_Public_{version}.xlsx"


def resolve_data_dir(cli_dir: str | Path | None = None) -> Path:
    """Resolve curated PPMI data directory: CLI --data-dir > PPMI_DATA_DIR > local data/raw."""
    candidates: list[Path] = []
    if cli_dir is not None:
        base = Path(cli_dir)
        candidates.extend([base, base / "curated"])
    env = os.environ.get("PPMI_DATA_DIR")
    if env:
        base = Path(env)
        candidates.extend([base, base / "curated"])
    candidates.extend([PROJECT / "data" / "raw", PROJECT / "data" / "raw" / "curated"])
    marker = extract_filename(REFERENCE)
    for path in candidates:
        if (path / marker).is_file():
            return path
    searched = ", ".join(str(p) for p in candidates)
    raise FileNotFoundError(
        f"Curated PPMI extract '{marker}' not found. Pass --data-dir, set PPMI_DATA_DIR, "
        f"or place the 11 official curated files in data/raw/. Checked: {searched}"
    )


def snapshot_month(version: str) -> int:
    return int(version[:4]) * 12 + int(version[4:6]) - 2


def calendar_month(index: int) -> str:
    year, month_idx = divmod(index, 12)
    return f"{year:04d}-{month_idx + 1:02d}"


def key(value) -> str:
    return str(value).strip().removesuffix(".0")


def number(value) -> float | None:
    try:
        x = float(value)
        return x if math.isfinite(x) else None
    except (ValueError, TypeError):
        return None


def parse_month(value) -> int | None:
    if isinstance(value, (date, datetime)):
        return value.year * 12 + value.month - 1
    raw = str(value).strip()
    for fmt in ("%m/%Y", "%Y-%m", "%b%Y", "%Y-%m-%d", "%m/%d/%Y", "%Y-%m-%d %H:%M:%S"):
        try:
            d = datetime.strptime(raw, fmt)
            return d.year * 12 + d.month - 1
        except ValueError:
            pass
    return None


def text(value) -> str | None:
    s = "" if value is None else str(value).strip()
    return None if s.upper() in {"", ".", "NA", "N/A", "NAN", "NULL", "UNKNOWN", "-"} else s


def agree(values):
    items = list(values)
    return items[0] if items and None not in items and len(set(items)) == 1 else None


def normalized(field: str, raw):
    raw = text(raw)
    if raw is None:
        return None
    if field == "visit_date":
        return parse_month(raw)
    if field == "PATNO":
        return key(raw)
    if field in {"EVENT_ID", "subgroup"}:
        return raw.upper()
    val = number(raw)
    lo, hi = BOUNDS.get(field, (0.0, math.inf))
    return val if val is not None and lo <= val <= hi else None


def selected_rows(data_dir: Path, version: str, columns: set[str]):
    path = data_dir / extract_filename(version)
    if version == "20230612":
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            headers = set(reader.fieldnames or ())
            yield headers
            sel = headers & columns
            for row in reader:
                yield {f: normalized(f, row[f]) for f in sel if text(row[f]) is not None}
        return

    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    try:
        sheet = next(s for s in wb if re.fullmatch(r"\d{8}", s.title))
        header = next(sheet.values)
        yield set(header) - {None}
        lookup = {get_column_letter(i): f for i, f in enumerate(header, 1) if f in columns}
        with ZipFile(path) as archive, archive.open(sheet._worksheet_path) as stream:
            for _, elem in iterparse(stream, events=("end",)):
                if not elem.tag.endswith("}row"):
                    continue
                if elem.get("r") != "1":
                    row = {}
                    for cell in elem:
                        col = lookup.get(cell.attrib.get("r", "").rstrip("0123456789"))
                        if col is None:
                            continue
                        val = cell.find("{*}v")
                        kind = cell.get("t")
                        if val is not None:
                            raw = sheet._shared_strings[int(val.text)] if kind == "s" else val.text
                        elif kind == "inlineStr":
                            raw = "".join(part.text or "" for part in cell.iterfind("{*}is//{*}t"))
                        else:
                            continue
                        row[col] = normalized(col, raw)
                    yield row
                elem.clear()
    finally:
        wb.close()


def combine(rows: list[dict]) -> dict:
    fields = set().union(*(r.keys() for r in rows)) if rows else set()
    return {f: agree(r.get(f) for r in rows) for f in fields}


def load_pd_release(data_dir: Path, version: str, columns: set[str] = REQUIRED_COLUMNS):
    """Parse one curated extract for PD (COHORT == 1) with unified V[0-9]+ scheduled 2-year follow-up."""
    cutoff = snapshot_month(version)
    stream = selected_rows(data_dir, version, columns)
    headers = next(stream)
    by_person = defaultdict(list)
    for row in stream:
        pid = row.get("PATNO")
        if pid is not None:
            by_person[pid].append(row)

    pd_people = {}
    for pid, recs in by_person.items():
        cohort = agree(r.get("COHORT") for r in recs)
        if cohort != 1:
            continue
        bls = [r for r in recs if r.get("EVENT_ID") == "BL"]
        bm = agree(r.get("visit_date") for r in bls)
        if bm is None or bm > cutoff:
            continue
        months = defaultdict(list)
        for r in recs:
            vm = r.get("visit_date")
            if vm is not None and bm < vm <= cutoff and re.fullmatch(r"V[0-9]+", r.get("EVENT_ID") or ""):
                months[vm].append(r)
        idx_m = min((m for m in months if bm + 21 <= m <= bm + 27), key=lambda m: (abs(m - bm - 24), m), default=None)
        fu = combine(months[idx_m]) if idx_m is not None else {}
        pd_people[pid] = {
            "baseline_month": int(bm),
            "mature_24m": (bm + 27 <= cutoff),
            "baseline": combine(bls),
            "followup_24m": fu,
            "followup_month": int(idx_m) if idx_m is not None else None,
        }
    return headers, pd_people, cutoff


def classify_pd_subgroup(raw_sub: str | None) -> str:
    if not raw_sub:
        return "Unlabeled"
    if "LRRK2" in raw_sub:
        return "LRRK2"
    if "GBA" in raw_sub:
        return "GBA"
    if "SPORADIC" in raw_sub or raw_sub == "PD NORMOSMIC":
        return "Sporadic"
    return "Other_Genetic_or_Mixed"


def stratified_pd_reserve(pd_by_cut: dict[str, dict], versions: tuple[str, ...] = VERSIONS) -> tuple[set[str], dict]:
    """Single-pass 30% deterministic stratified PD evaluation reserve by earliest contiguous cut."""
    reference = versions[-1]
    still_common = set(pd_by_cut[reference])
    earliest: dict[str, str] = {}
    for v in reversed(versions):
        still_common &= set(pd_by_cut[v])
        for pid in still_common:
            earliest[pid] = v

    groups: dict[str, list[str]] = defaultdict(list)
    for pid, first in earliest.items():
        if first != reference:
            groups[first].append(pid)

    reserved: set[str] = set()
    strata: dict[str, dict[str, int]] = {}
    for first, pids in sorted(groups.items()):
        ordered = sorted(pids, key=lambda p: (hashlib.sha256(f"{RESERVE_SEED}:{p}".encode("utf-8")).hexdigest(), p))
        take = math.floor(RESERVE_FRACTION * len(ordered) + 0.5)
        reserved.update(ordered[:take])
        strata[first] = {"eligible": len(ordered), "reserved": take}

    return reserved, {
        "seed": RESERVE_SEED,
        "fraction": RESERVE_FRACTION,
        "eligible_pool": sum(len(v) for v in groups.values()),
        "total_reserved": len(reserved),
        "strata": strata,
    }


def extract_panel_eligible(people: dict, panel: FeaturePanel) -> dict[str, tuple[int, tuple]]:
    """Return {pid: (baseline_month, feature_tuple)} for PD with >=80% observed panel features."""
    out = {}
    min_obs = panel.min_observed
    for pid, rec in people.items():
        bl = rec["baseline"]
        vals = tuple(bl.get(f) for f in panel.features)
        if sum(v is not None for v in vals) >= min_obs:
            out[pid] = (rec["baseline_month"], vals)
    return out


def extract_external_anchors(ref_people: dict, eval_ids: list[str]) -> dict[str, list[float]]:
    """Extract held-out 2-year clinical outcomes (never used in clustering) for fixed evaluation participants."""
    anchors: dict[str, list[float]] = {name: [] for name, _, _ in EXTERNAL_ANCHORS}
    for pid in eval_ids:
        rec = ref_people[pid]
        bl = rec["baseline"]
        fu = rec["followup_24m"]
        mature = rec["mature_24m"]

        def cont(col: str) -> float:
            if not mature:
                return math.nan
            v = fu.get(col)
            return float(v) if v is not None else math.nan

        anchors["2y_moca"].append(cont("moca"))
        anchors["2y_updrs3"].append(cont("updrs3_score"))
        anchors["2y_updrs2"].append(cont("updrs2_score"))
        anchors["2y_adl"].append(cont("MSEADLG"))

        # Binary at-risk milestones
        if mature and bl.get("pm_any") == 0 and fu.get("pm_any") is not None:
            anchors["2y_pm_any"].append(1.0 if fu["pm_any"] == 1 else 0.0)
        else:
            anchors["2y_pm_any"].append(math.nan)

        if mature and bl.get("cogstate") == 1 and fu.get("cogstate") in (1, 2, 3):
            anchors["2y_cogstate_abn"].append(1.0 if fu["cogstate"] in (2, 3) else 0.0)
        else:
            anchors["2y_cogstate_abn"].append(math.nan)

        if mature and bl.get("MSEADLG") is not None and bl["MSEADLG"] >= 80 and fu.get("MSEADLG") is not None:
            anchors["2y_adl_lt80"].append(1.0 if fu["MSEADLG"] < 80 else 0.0)
        else:
            anchors["2y_adl_lt80"].append(math.nan)
    return anchors


def severity_signs(panel: FeaturePanel) -> tuple[float, ...]:
    """Return +1 (higher = worse) or -1 (higher = better) for canonical severity ordering of reference clusters."""
    return tuple(-1.0 if f in HIGHER_IS_BETTER else 1.0 for f in panel.features)


def subgroup_summary(people: dict, ids: list[str]) -> dict[str, int]:
    cnt = Counter(classify_pd_subgroup(people[p]["baseline"].get("subgroup")) for p in ids)
    return dict(sorted(cnt.items()))
