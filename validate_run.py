# validate_run.py
# Validate one pipeline run folder: outputs/run_YYYYMMDD_HHMMSS/
# Produces:
#   - validation_report.json
#   - samples_for_labeling.xlsx
#   - suspects_unit_mismatch.xlsx (optional)
#   - suspects_duplicates.xlsx (optional)

import re
import json
import argparse
from pathlib import Path
from collections import Counter, defaultdict

import pandas as pd


NUM_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")  # crude numeric detector


def has_number(s: str) -> bool:
    if s is None:
        return False
    return bool(NUM_RE.search(str(s)))


def norm_unit(u: str) -> str:
    """Loose normalize to compare company vs catalog unit."""
    if u is None:
        return ""
    u = str(u).strip().lower()
    u = u.replace(" ", "")
    u = u.replace("co2e", "co₂e")  # optional
    u = u.replace("tco2", "tco₂")
    # common aliases
    aliases = {
        "tons": "tonnes",
        "ton": "tonnes",
        "tonne": "tonnes",
        "t": "tonnes",
        "kwh": "kwh",
        "mwh": "mwh",
        "gj": "gj",
        "usd": "usd",
        "$": "usd",
        "percent": "%",
        "percentage": "%",
        "ratio": "ratio",
        "number": "number",
        "count": "number",
    }
    return aliases.get(u, u)


def unit_family(u: str) -> str:
    """Map unit into coarse family for mismatch detection."""
    u = norm_unit(u)
    if not u:
        return ""
    if u in ["%", "ratio"]:
        return "ratio"
    if "co₂e" in u or "ghg" in u:
        return "emissions"
    if u in ["mwh", "kwh", "gj"]:
        return "energy"
    if u in ["m³", "m3", "l", "litre", "liters", "litres"]:
        return "water"
    if u in ["tonnes", "kg", "g"]:
        return "mass"
    if u in ["usd", "cny", "eur", "gbp"]:
        return "currency"
    if u in ["hours", "hr", "h"]:
        return "time"
    if u in ["employees", "fte"]:
        return "people"
    if u in ["number"]:
        return "count"
    return "other"


def safe_read_excel(path: Path) -> dict:
    xls = pd.read_excel(path, sheet_name=None)
    # normalize column names a bit
    for k, df in xls.items():
        df.columns = [str(c).strip() for c in df.columns]
        xls[k] = df
    return xls


def make_labeling_sample(df_val: pd.DataFrame, df_um: pd.DataFrame, n_each_bucket=40, n_unmatched=80, seed=42):
    sample_frames = []

    # Matched buckets by match_confidence
    if not df_val.empty and "match_confidence" in df_val.columns:
        d = df_val.copy()
        d["match_confidence"] = pd.to_numeric(d["match_confidence"], errors="coerce").fillna(-1)

        buckets = [
            ("high_conf(>=0.90)", d[d["match_confidence"] >= 0.90]),
            ("mid_conf(0.75-0.90)", d[(d["match_confidence"] >= 0.75) & (d["match_confidence"] < 0.90)]),
            ("low_conf(<0.75)", d[(d["match_confidence"] >= 0) & (d["match_confidence"] < 0.75)]),
        ]
        for name, part in buckets:
            if part.empty:
                continue
            samp = part.sample(n=min(n_each_bucket, len(part)), random_state=seed)
            samp = samp.copy()
            samp.insert(0, "sample_group", name)
            samp.insert(1, "human_label", "")  # CORRECT / WRONG / UNSURE
            samp.insert(2, "human_note", "")
            sample_frames.append(samp)

    # Unmatched sample
    if df_um is not None and not df_um.empty:
        samp_um = df_um.sample(n=min(n_unmatched, len(df_um)), random_state=seed).copy()
        samp_um.insert(0, "sample_group", "unmatched")
        samp_um.insert(1, "human_label", "")  # SHOULD_MATCH / TRUE_UNMATCH / UNSURE
        samp_um.insert(2, "human_note", "")
        sample_frames.append(samp_um)

    if not sample_frames:
        return pd.DataFrame()

    # keep useful columns only (avoid super wide)
    keep_cols = [
        "sample_group", "human_label", "human_note",
        "metric_id", "catalog_metric_name", "catalog_unit",
        "raw_metric_name", "value", "unit_reported", "period", "evidence",
        "file", "pages", "match_method", "match_confidence", "match_reason",
        "top_candidates", "best_rule_score", "llm_confidence", "llm_reason",
    ]
    out = pd.concat(sample_frames, ignore_index=True)
    cols = [c for c in keep_cols if c in out.columns]
    return out[cols]


def build_report(df_cat: pd.DataFrame, df_val: pd.DataFrame, df_um: pd.DataFrame):
    report = {}

    # Basic sizes
    report["sizes"] = {
        "catalog_rows": int(len(df_cat)),
        "matched_rows": int(len(df_val)),
        "unmatched_rows": int(len(df_um)),
        "coverage_matched_rate": float(len(df_val) / max(1, (len(df_val) + len(df_um)))),
    }

    # Schema sanity
    expected_cat = ["metric_id", "metric_name", "unit", "hierarchy", "sources"]
    expected_val = ["metric_id", "catalog_metric_name", "raw_metric_name", "value", "file", "pages"]
    expected_um = ["raw_metric_name", "value", "file", "pages"]

    report["schema"] = {
        "catalog_missing_cols": [c for c in expected_cat if c not in df_cat.columns],
        "companyvalues_missing_cols": [c for c in expected_val if c not in df_val.columns],
        "unmatched_missing_cols": [c for c in expected_um if c not in df_um.columns],
    }

    # Hard sanity checks for extracted values
    if not df_val.empty:
        val = df_val.copy()
        val["value_has_number"] = val["value"].map(has_number)
        report["companyvalues_sanity"] = {
            "value_missing_rate": float((val["value"].isna() | (val["value"].astype(str).str.strip() == "")).mean()),
            "value_no_number_rate": float((~val["value_has_number"]).mean()),
            "file_missing_rate": float((val["file"].isna() | (val["file"].astype(str).str.strip() == "")).mean()),
            "pages_missing_rate": float((val["pages"].isna() | (val["pages"].astype(str).str.strip() == "")).mean()),
        }

        # Match confidence distribution
        if "match_confidence" in val.columns:
            mc = pd.to_numeric(val["match_confidence"], errors="coerce").fillna(-1)
            report["match_confidence"] = {
                "p50": float(mc.quantile(0.5)),
                "p10": float(mc.quantile(0.1)),
                "p90": float(mc.quantile(0.9)),
                "count_lt_0_75": int((mc >= 0).sum() - (mc >= 0.75).sum()),
                "count_ge_0_75": int((mc >= 0.75).sum()),
                "count_ge_0_90": int((mc >= 0.90).sum()),
            }

    # Unit mismatch suspects (coarse family)
    unit_mismatch_rows = []
    if not df_val.empty and ("unit_reported" in df_val.columns) and ("catalog_unit" in df_val.columns):
        for _, r in df_val.iterrows():
            ur = str(r.get("unit_reported", "") or "")
            cu = str(r.get("catalog_unit", "") or "")
            if not ur or not cu:
                continue
            fam_r = unit_family(ur)
            fam_c = unit_family(cu)
            if fam_r and fam_c and fam_r != fam_c:
                unit_mismatch_rows.append({
                    "file": r.get("file",""),
                    "pages": r.get("pages",""),
                    "raw_metric_name": r.get("raw_metric_name",""),
                    "value": r.get("value",""),
                    "unit_reported": ur,
                    "catalog_metric_name": r.get("catalog_metric_name",""),
                    "catalog_unit": cu,
                    "match_confidence": r.get("match_confidence",""),
                    "match_reason": r.get("match_reason",""),
                    "unit_family_reported": fam_r,
                    "unit_family_catalog": fam_c,
                })

    report["unit_mismatch"] = {
        "count": len(unit_mismatch_rows),
        "rate_over_matched": float(len(unit_mismatch_rows) / max(1, len(df_val))),
        "top_examples": unit_mismatch_rows[:20],  # first 20 examples
    }

    # Duplicates suspects (same file+raw_name+period+value)
    dup_rows = []
    if not df_val.empty:
        keys = ["file", "raw_metric_name", "period", "value"]
        keys = [k for k in keys if k in df_val.columns]
        if keys:
            g = df_val.groupby(keys, dropna=False).size().reset_index(name="cnt")
            g = g[g["cnt"] >= 2].sort_values("cnt", ascending=False)
            for _, r in g.head(50).iterrows():
                dup_rows.append(r.to_dict())

    report["duplicates"] = {
        "count_groups": len(dup_rows),
        "top_groups": dup_rows[:20],
    }

    # Unmatched analysis: top by file + top candidate patterns
    if not df_um.empty:
        top_files = df_um["file"].fillna("").value_counts().head(15).to_dict() if "file" in df_um.columns else {}
        report["unmatched_top_files"] = top_files

        if "top_candidates" in df_um.columns:
            # count how often certain catalog KPI appears in top candidates
            cand_counter = Counter()
            for s in df_um["top_candidates"].fillna("").astype(str).tolist():
                parts = [p.strip() for p in s.split(";") if p.strip()]
                for p in parts:
                    # format: metric_id|metric_name
                    if "|" in p:
                        _, name = p.split("|", 1)
                        cand_counter[name.strip()] += 1
            report["unmatched_common_candidates"] = cand_counter.most_common(20)

    return report, unit_mismatch_rows, dup_rows


def find_latest_run(outputs_dir: Path) -> Path:
    """Find the most recent run_YYYYMMDD_HHMMSS folder in outputs."""
    run_dirs = sorted(
        [d for d in outputs_dir.iterdir() if d.is_dir() and d.name.startswith("run_")],
        key=lambda x: x.name,
        reverse=True
    )
    if not run_dirs:
        raise FileNotFoundError(f"No run_* folders found in {outputs_dir}")
    return run_dirs[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", default=None, help="outputs/run_YYYYMMDD_HHMMSS (default: latest run)")
    ap.add_argument("--out", default=None, help="output dir for validation artifacts (default: run_dir)")
    ap.add_argument("--sample_each_bucket", type=int, default=40)
    ap.add_argument("--sample_unmatched", type=int, default=80)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # Auto-detect latest run if not specified
    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        outputs_dir = Path(__file__).parent / "outputs"
        run_dir = find_latest_run(outputs_dir)
        print(f"📂 Auto-detected latest run: {run_dir.name}")
    out_dir = Path(args.out) if args.out else run_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    final_xlsx = run_dir / "final.xlsx"
    if not final_xlsx.exists():
        raise FileNotFoundError(f"final.xlsx not found in {run_dir}")

    sheets = safe_read_excel(final_xlsx)
    df_cat = sheets.get("Catalog", pd.DataFrame())
    df_val = sheets.get("CompanyValues", pd.DataFrame())
    df_um = sheets.get("Unmatched", pd.DataFrame())

    report, unit_mismatch_rows, dup_rows = build_report(df_cat, df_val, df_um)

    # write report json
    (out_dir / "validation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    # write labeling samples
    sample_df = make_labeling_sample(
        df_val, df_um,
        n_each_bucket=args.sample_each_bucket,
        n_unmatched=args.sample_unmatched,
        seed=args.seed,
    )
    if not sample_df.empty:
        sample_path = out_dir / "samples_for_labeling.xlsx"
        with pd.ExcelWriter(sample_path, engine="openpyxl") as w:
            sample_df.to_excel(w, index=False, sheet_name="Samples")

    # write suspects
    if unit_mismatch_rows:
        pd.DataFrame(unit_mismatch_rows).to_excel(out_dir / "suspects_unit_mismatch.xlsx", index=False)
    if dup_rows:
        pd.DataFrame(dup_rows).to_excel(out_dir / "suspects_duplicates.xlsx", index=False)

    print("✅ Validation done:")
    print(f"  - {out_dir / 'validation_report.json'}")
    if not sample_df.empty:
        print(f"  - {out_dir / 'samples_for_labeling.xlsx'}")
    if unit_mismatch_rows:
        print(f"  - {out_dir / 'suspects_unit_mismatch.xlsx'}")
    if dup_rows:
        print(f"  - {out_dir / 'suspects_duplicates.xlsx'}")


if __name__ == "__main__":
    main()
