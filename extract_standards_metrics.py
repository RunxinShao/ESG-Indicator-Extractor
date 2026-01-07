# pipeline_catalog_fill.py
# ============================================================
# Two-phase ESG KPI pipeline:
# Phase A) Build Master KPI Catalog from Standards (Excel + PDF), measurable-only
# Phase B) Extract company-specific KPI values from PDFs and map them to Catalog
#
# Output: one Excel with sheets:
#   - Catalog
#   - CompanyValues
#   - Unmatched
#
# Precision-first:
#   - hard-guards for category/narrative
#   - strict LLM measurable filter (KEEP/DROP + confidence)
#   - merge duplicates for catalog
#   - company extraction requires numeric value
#   - matching: cheap Top-K candidates + optional LLM judge
#
# Dependencies:
#   pip install pdfplumber pandas tqdm openai openpyxl
#
# Usage example:
#   python pipeline_catalog_fill.py --standards GRI PRI --pdf company_specific --use_openai --keep_conf 0.85
#
# Folder layout (relative to this script):
#   ./GRI/*.xlsx, ./GRI/*.pdf
#   ./PRI/*.xlsx, ./PRI/*.pdf
#   ./company_specific/*.pdf

import os
import re
import json
import time
import hashlib
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

import pdfplumber
import pandas as pd
from tqdm import tqdm
from difflib import SequenceMatcher
from openai import OpenAI

# ============================================================
# Config
# ============================================================

class Config:
    def __init__(
        self,
        use_ollama: bool = False,
        ollama_base_url: str = "http://localhost:11434/v1",
        ollama_model: str = "llama3.1:8b",
        openai_api_key: Optional[str] = None,
        openai_model: str = "gpt-5.2",
        max_chunk_chars: int = 14000,
        out_dir: str = "outputs",

        # Precision knobs
        keep_confidence_threshold: float = 0.80,
        post_merge_refilter: bool = True,

        # Matching knobs
        match_use_llm: bool = True,
        match_top_k: int = 8,
        match_confidence_threshold: float = 0.75,   # LLM matching confidence threshold
        rule_match_score_threshold: float = 0.72,   # if not using LLM, rule match threshold
    ):
        self.use_ollama = use_ollama
        self.ollama_base_url = ollama_base_url
        self.ollama_model = ollama_model
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.openai_model = openai_model
        self.max_chunk_chars = max_chunk_chars
        self.out_dir = out_dir

        self.keep_confidence_threshold = keep_confidence_threshold
        self.post_merge_refilter = post_merge_refilter

        self.match_use_llm = match_use_llm
        self.match_top_k = match_top_k
        self.match_confidence_threshold = match_confidence_threshold
        self.rule_match_score_threshold = rule_match_score_threshold


# ============================================================
# LLM Client
# ============================================================

class LLMClient:
    def __init__(self, cfg: Config):
        if cfg.use_ollama:
            self.client = OpenAI(base_url=cfg.ollama_base_url, api_key="ollama")
            self.model = cfg.ollama_model
        else:
            self.client = OpenAI(api_key=cfg.openai_api_key)
            self.model = cfg.openai_model

    def chat(self, prompt: str, max_retries: int = 3) -> str:
        for attempt in range(max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                )
                return resp.choices[0].message.content or ""
            except Exception as e:
                s = str(e)
                if "429" in s or "rate_limit" in s.lower():
                    wait = 2 ** attempt
                    print(f"      [RATE LIMIT] wait {wait}s ({attempt+1}/{max_retries})")
                    time.sleep(wait)
                    continue
                raise
        # final attempt
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        return resp.choices[0].message.content or ""


# ============================================================
# JSON parsing helpers
# ============================================================

def parse_json_array(text: str) -> List[Any]:
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return obj
    except:
        pass
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group())
            if isinstance(obj, list):
                return obj
        except:
            pass
    return []

def parse_json_obj(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except:
        pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group())
            if isinstance(obj, dict):
                return obj
        except:
            pass
    return {}


# ============================================================
# Utility
# ============================================================

def stable_hash(s: str) -> str:
    return hashlib.md5((s or "").encode("utf-8")).hexdigest()

def normalize_metric_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (name or "").lower().strip())

def tokenize(s: str) -> List[str]:
    return [t for t in re.findall(r"[a-z0-9]+", (s or "").lower()) if t]

def get_hierarchy_from_source(source: str) -> str:
    """
    Map source folder paths to hierarchy levels.
    Supports both old format (e.g., "GRI") and new format (e.g., "global/GRI").
    """
    source_lower = (source or "").lower()

    # Normalize path separators and extract components
    path_parts = source_lower.replace("\\", "/").split("/")
    folder_name = path_parts[-1].strip()
    category = path_parts[0] if len(path_parts) > 1 else ""

    # NEW: Category-based hierarchy (e.g., "global/GRI" -> "Global")
    if category:
        if category == "global":
            return "Global"
        elif category == "national":
            return "National"
        elif category == "local":
            return "Local"
        elif category == "industry":
            return "Industry"
        elif category in ["company_specific", "company"]:
            return "Company-specific"

    # Default fallback
    return "Global"

def make_metric_id(metric_name: str, hierarchy: str) -> str:
    base = normalize_metric_name(metric_name)
    h = (hierarchy or "").lower()
    return stable_hash(f"{h}::{base}")


# ============================================================
# Hard guards (precision-first)
# ============================================================

def looks_like_category_title(name: str) -> bool:
    if not name:
        return True
    n = name.strip()
    nl = n.lower()

    bad_keywords = [
        "environment", "environmental", "social", "governance",
        "responsible investment", "stewardship", "reporting", "disclosure",
        "performance", "metrics", "metric", "pillar", "topic", "category", "categories",
        "overview", "general", "framework", "assessment", "index", "appendix", "glossary",
        "asset allocation", "portfolio", "thematic bonds", "real estate esg", "infrastructure esg",
    ]

    if re.search(r"\b(metrics?|performance|reporting|disclosure|overview)\b", nl):
        good_tokens = ["emission", "energy", "water", "waste", "injur", "turnover", "divers",
                       "salary", "revenue", "ghg", "scope", "intensity", "ratio", "number", "amount", "%"]
        if any(t in nl for t in good_tokens):
            return False
        return True

    if any(k in nl for k in bad_keywords):
        good_tokens = ["emission", "energy", "water", "waste", "injur", "turnover", "divers",
                       "salary", "revenue", "ghg", "scope", "intensity", "ratio", "number", "amount", "%"]
        if any(t in nl for t in good_tokens):
            return False
        return True

    words = re.findall(r"[A-Za-z]+", n)
    if len(words) <= 3 and n == n.title() and not re.search(r"\d", n):
        return True
    if len(words) == 1 and not re.search(r"\d", n):
        return True
    return False

def looks_like_narrative_disclosure(name: str) -> bool:
    if not name:
        return True
    nl = name.lower()
    bad = [
        "process", "processes",
        "approach", "management approach",
        "polic", "strategy", "governance",
        "identification", "assessment", "risk assessment",
        "incident investigation", "stakeholder engagement",
        "remediate", "remediation",
        "due diligence", "grievance",
        "mechanism", "procedures", "controls",
        "oversight", "roles and responsibilities",
        "training program", "awareness program",
    ]
    scalar_signals = ["number of", "total", "amount", "percentage", "%", "ratio", "rate",
                      "count", "volume", "tons", "tco2", "mwh", "usd", "hours"]
    if any(s in nl for s in bad) and not any(s in nl for s in scalar_signals):
        return True
    if re.search(r"\bprocess(es)?\b", nl) and not any(s in nl for s in scalar_signals):
        return True
    return False


# ============================================================
# Unit inference (cached)
# ============================================================

_UNIT_CACHE: Dict[str, str] = {}

def unit_type_to_unit(unit_type: str) -> str:
    ut = (unit_type or "").strip().lower()
    mapping = {
        "%": "%",
        "percent": "%",
        "percentage": "%",
        "ratio": "ratio",
        "count": "number",
        "number": "number",
        "employees": "employees",
        "currency": "USD",
        "usd": "USD",
        "emissions": "tCO₂e",
        "tco2e": "tCO₂e",
        "energy": "MWh",
        "water": "m³",
        "volume": "m³",
        "waste": "tonnes",
        "weight": "tonnes",
        "time": "hours",
        "hours": "hours",
        "area": "m²",
    }
    return mapping.get(ut, "")

def infer_unit_from_metric_name(llm: LLMClient, metric_name: str) -> str:
    if not metric_name or not metric_name.strip():
        return ""
    key = stable_hash(metric_name.strip().lower())
    if key in _UNIT_CACHE:
        return _UNIT_CACHE[key]
    prompt = f"""Based on the ESG metric name, infer the most likely unit of measurement.

Metric name: "{metric_name}"

Return ONLY the unit (e.g., "tCO₂e", "MWh", "%", "number", "hours", "USD", "employees").
If the metric is a ratio, percentage, or count, return "%", "ratio", or "number" respectively.
If you cannot determine a unit, return "".

Return only the unit:
"""
    try:
        resp = llm.chat(prompt)
        unit = (resp or "").strip().strip('"').strip("'")
        if unit.lower() in ["none", "n/a", "na", "null", ""]:
            unit = ""
        if len(unit) > 20 or ("\n" in unit) or (unit.count(".") > 1):
            unit = ""
        _UNIT_CACHE[key] = unit
        return unit
    except Exception:
        _UNIT_CACHE[key] = ""
        return ""


# ============================================================
# PDF Processing
# ============================================================

def load_pdf_text(pdf_path: Path) -> List[Dict[str, Any]]:
    pages = []
    try:
        with pdfplumber.open(str(pdf_path)) as pdf:
            for i, page in enumerate(pdf.pages):
                text = (page.extract_text() or "").strip()
                if text:
                    pages.append({"page": i + 1, "text": text})
    except Exception as e:
        print(f"⚠️ Failed to read PDF: {pdf_path} -> {e}")
    return pages

def chunk_pages(pages: List[Dict[str, Any]], max_chars: int) -> List[Dict[str, Any]]:
    chunks = []
    current_text = ""
    current_pages = []
    for page in pages:
        addition = ("\n\n" + page["text"]) if current_text else page["text"]
        if len(current_text) + len(addition) > max_chars and current_text:
            chunks.append({"text": current_text.strip(), "pages": current_pages})
            current_text = page["text"]
            current_pages = [page["page"]]
        else:
            current_text += addition
            current_pages.append(page["page"])
    if current_text.strip():
        chunks.append({"text": current_text.strip(), "pages": current_pages})
    return chunks


# ============================================================
# Standards extraction (PDF chunk -> KPI defs)
# ============================================================

def extract_standard_metrics_from_chunk(llm: LLMClient, chunk_text: str, source: str) -> List[Dict[str, Any]]:
    prompt = f"""Extract ONLY directly quantifiable ESG metrics (single scalar KPIs) from the text.

IMPORTANT: The input text may be in any language (Chinese, Korean, Spanish, etc.).
Always output metric_name and definition in ENGLISH, regardless of input language.

A KPI is KEEPABLE only if it can be reported as ONE numeric value (possibly with unit/type), e.g.:
- total emissions (tCO2e)
- energy consumption (MWh/GJ)
- number of injuries
- turnover rate (%)
- amount of fines (currency)
- percentage of suppliers screened (%)

DO NOT extract:
- category headings/pillars/modules (e.g., "Environmental Metrics")
- narrative disclosures or processes (e.g., "Processes to remediate impacts", "Risk assessment process")
- management approach disclosures
- standard/version names (e.g., "GRI 2", "ISO 14001")
- disclosure titles that are not a KPI

Return JSON array with:
- metric_name (in English)
- definition (in English, only what the KPI measures; no extra assumptions)

Return [] if none.

Text:
\"\"\"{chunk_text[:12000]}\"\"\"
"""
    resp = llm.chat(prompt)
    results = parse_json_array(resp)

    metrics = []
    for r in results:
        if not isinstance(r, dict):
            continue
        name = str(r.get("metric_name", "")).strip()
        if not name:
            continue
        if looks_like_category_title(name) or looks_like_narrative_disclosure(name):
            continue
        unit = infer_unit_from_metric_name(llm, name)
        metrics.append({
            "metric_name": name,
            "definition": str(r.get("definition", "")).strip(),
            "unit": unit,
            "source": source,
            "hierarchy": get_hierarchy_from_source(source),
            "value": "",
        })
    return metrics


# ============================================================
# Company extraction (PDF chunk -> KPI values)
# ============================================================

def extract_company_metrics_from_chunk(llm: LLMClient, chunk_text: str, source: str) -> List[Dict[str, Any]]:
    prompt = f"""Extract ONLY measurable (quantitative) ESG metrics from this text.

IMPORTANT: The input text may be in any language (Chinese, Korean, Spanish, etc.).
Always output metric_name, definition, and evidence in ENGLISH, regardless of input language.
Keep numeric values and units as-is (do not translate numbers).

Each metric MUST have a numeric value. Do NOT extract policies or commitments.

Return JSON array with fields:
- metric_name (in English)
- definition (in English, what it measures)
- value (keep original number)
- unit (or "")
- period (or "")
- evidence (in English, ≤20 words)

Return [] if none.

Text:
\"\"\"{chunk_text[:12000]}\"\"\"
"""
    resp = llm.chat(prompt)
    results = parse_json_array(resp)

    metrics = []
    for r in results:
        if not isinstance(r, dict):
            continue
        name = str(r.get("metric_name", "")).strip()
        value = str(r.get("value", "")).strip()
        if not name or not value:
            continue
        if looks_like_category_title(name) or looks_like_narrative_disclosure(name):
            continue
        metrics.append({
            "raw_metric_name": name,
            "definition": str(r.get("definition", "")).strip(),
            "value": value,
            "unit_reported": str(r.get("unit", "")).strip(),
            "period": str(r.get("period", "")).strip(),
            "evidence": str(r.get("evidence", "")).strip(),
            "source": source,
            "hierarchy": get_hierarchy_from_source(source),
        })
    return metrics


# ============================================================
# Excel Processing (standards)
# ============================================================

def validate_unit_column(df: pd.DataFrame, col: str) -> bool:
    if col not in df.columns:
        return False
    sample = df[col].dropna().head(30)
    if sample.empty:
        return False
    short_count = 0
    non_empty = 0
    for val in sample:
        s = str(val).strip()
        if not s or s.lower() in ["nan", "none", "n/a", "na"]:
            continue
        non_empty += 1
        if len(s) <= 15 and s.count(" ") <= 2:
            short_count += 1
    if non_empty == 0:
        return False
    return (short_count / non_empty) >= 0.75

def find_metric_column(df: pd.DataFrame, llm: Optional[LLMClient] = None) -> Optional[str]:
    col_samples = {}
    for col in df.columns:
        samples = df[col].dropna().head(12).tolist()
        col_samples[col] = [str(s)[:140] for s in samples]

    if llm is not None:
        prompt = f"""Pick the column that contains measurable KPI NAMES (not categories).

KPI name examples:
- "Scope 1 GHG emissions"
- "Number of workplace injuries"
- "Employee turnover rate"
- "Percentage of suppliers screened"
- "Total fines for non-compliance"
- "Energy consumption"

NOT KPI:
- categories/pillars (e.g., "Environmental Metrics")
- narrative/process items ("Processes to remediate impacts")
- standard/version names
- disclosure titles that are not KPI

If no KPI-name column exists, return null.

Columns:
{json.dumps(col_samples, indent=2, ensure_ascii=False)}

Return JSON only: {{"metric_column": <col or null>}}
"""
        try:
            resp = llm.chat(prompt)
            obj = parse_json_obj(resp)
            mc = obj.get("metric_column")
            if mc and mc in df.columns:
                return mc
        except Exception:
            pass

    best_col, best_score = None, -1e9
    for col in df.columns:
        series = df[col].dropna().astype(str).map(lambda x: x.strip())
        series = series[series.str.lower().ne("nan")]
        if series.empty:
            continue
        uniq = series.nunique()
        sample_vals = series.head(30).tolist()
        cat_hits = sum(1 for v in sample_vals if looks_like_category_title(v))
        narr_hits = sum(1 for v in sample_vals if looks_like_narrative_disclosure(v))
        score = uniq - 3 * cat_hits - 2 * narr_hits
        if score > best_score:
            best_score = score
            best_col = col
    return best_col

def find_definition_unit_columns(df: pd.DataFrame, llm: Optional[LLMClient] = None) -> Tuple[Optional[str], Optional[str]]:
    def_col, unit_col = None, None
    col_samples = {}
    for col in df.columns:
        samples = df[col].dropna().head(12).tolist()
        col_samples[col] = [str(s)[:140] for s in samples]

    if llm is not None:
        prompt = f"""Identify columns:
- definition_column: explains what the KPI measures (sentences)
- unit_column: short unit strings (%/tCO2e/MWh/USD/hours/number/etc.)

If unsure return null.

Columns:
{json.dumps(col_samples, indent=2, ensure_ascii=False)}

Return JSON only:
{{"definition_column": <col or null>, "unit_column": <col or null>}}
"""
        try:
            resp = llm.chat(prompt)
            obj = parse_json_obj(resp)
            d = obj.get("definition_column")
            u = obj.get("unit_column")
            def_col = d if d and d in df.columns else None
            if u and u in df.columns and validate_unit_column(df, u):
                unit_col = u
        except Exception:
            pass

    cols_lower = {c.lower(): c for c in df.columns}
    if def_col is None:
        for k in ["definition", "description", "meaning", "explanation"]:
            if k in cols_lower:
                def_col = cols_lower[k]
                break
    if unit_col is None:
        for k in ["unit", "units"]:
            if k in cols_lower and validate_unit_column(df, cols_lower[k]):
                unit_col = cols_lower[k]
                break
    return def_col, unit_col

def load_excel_metrics(xlsx_path: Path, source: str, llm: Optional[LLMClient] = None) -> List[Dict[str, Any]]:
    metrics: List[Dict[str, Any]] = []
    hierarchy = get_hierarchy_from_source(source)
    try:
        sheets = pd.read_excel(str(xlsx_path), sheet_name=None)
        for sheet_name, df in sheets.items():
            if df is None or df.empty:
                continue
            metric_col = find_metric_column(df, llm)
            if not metric_col:
                continue
            def_col, unit_col = find_definition_unit_columns(df, llm)

            for _, row in df.iterrows():
                raw_name = str(row.get(metric_col, "")).strip()
                if not raw_name or raw_name.lower() == "nan":
                    continue
                if looks_like_category_title(raw_name) or looks_like_narrative_disclosure(raw_name):
                    continue

                raw_def = ""
                if def_col:
                    raw_def = str(row.get(def_col, "")).strip()
                    if raw_def.lower() in ["nan", "none", "n/a", "na"]:
                        raw_def = ""

                unit = ""
                if unit_col:
                    ru = str(row.get(unit_col, "")).strip()
                    if ru and ru.lower() not in ["nan", "none", "n/a", "na", ""]:
                        if len(ru) <= 20 and ru.count(" ") <= 2:
                            unit = ru

                if llm and not unit:
                    unit = infer_unit_from_metric_name(llm, raw_name)

                metrics.append({
                    "metric_name": raw_name,
                    "definition": raw_def,
                    "unit": unit,
                    "source": source,
                    "hierarchy": hierarchy,
                    "sheet": sheet_name,
                    "value": "",
                })
    except Exception as e:
        print(f"⚠️ Failed to read Excel: {xlsx_path} -> {e}")
    return metrics


# ============================================================
# STRICT Quantitative Filter (precision-first)
# ============================================================

_STRICT_FILTER_CACHE: Dict[str, List[Dict[str, Any]]] = {}

def strict_quant_filter(llm: LLMClient, metrics: List[Dict[str, Any]], keep_conf_threshold: float = 0.80) -> List[Dict[str, Any]]:
    if not metrics:
        return []

    signature = stable_hash(json.dumps(
        [{"n": m.get("metric_name",""), "d": (m.get("definition","") or "")[:200]} for m in metrics],
        ensure_ascii=False
    ))
    if signature in _STRICT_FILTER_CACHE:
        return _STRICT_FILTER_CACHE[signature]

    batch_size = 30
    kept: List[Dict[str, Any]] = []

    for i in range(0, len(metrics), batch_size):
        batch = metrics[i:i+batch_size]
        items = [{"name": m.get("metric_name",""), "definition": m.get("definition","")} for m in batch]

        prompt = f"""You are filtering ESG items into DIRECTLY-QUANTIFIABLE KPIs (precision-first).

KEEP only if the item is a measurable KPI that can be reported as ONE numeric value (with unit/type), such as:
- total/amount/number/percentage/rate/ratio of something
Examples: "Scope 1 GHG emissions", "Number of injuries", "Employee turnover rate", "Total fines (USD)".

DROP if the item is:
- category/pillar/module (e.g., "Environmental Metrics", "ESG Disclosure and Reporting")
- narrative/process/management approach (e.g., "Processes to remediate impacts", "Risk assessment process")
- a disclosure title that is not itself a scalar KPI
- standard/version name

IMPORTANT (precision):
- If you are not highly confident it is a DIRECT scalar KPI, set keep=false.
- Do NOT invent new KPIs that are not implied by the name/definition.

Return JSON array, one per input item (name must match exactly):
- name: exact input name
- keep: true/false
- confidence: number 0..1
- unit_type: one of ["emissions","energy","water","waste","time","area","currency","percent","count","ratio","other",""]
- canonical_name: if keep=true, a cleaned KPI name close to the original (do not turn it into a category)
- reason: short reason

Items:
{json.dumps(items, indent=2, ensure_ascii=False)}
"""
        resp = llm.chat(prompt)
        results = parse_json_array(resp)

        name_to_decision: Dict[str, Dict[str, Any]] = {}
        for r in results:
            if isinstance(r, dict) and r.get("name") is not None:
                name_to_decision[str(r.get("name",""))] = r

        for m in batch:
            name = m.get("metric_name","")
            dec = name_to_decision.get(name, {})
            keep = bool(dec.get("keep", False))
            conf = float(dec.get("confidence", 0.0) or 0.0)
            canon = str(dec.get("canonical_name","") or "").strip()
            unit_type = str(dec.get("unit_type","") or "").strip()

            if (not keep) or conf < keep_conf_threshold:
                continue

            if not canon:
                canon = name
            if looks_like_category_title(canon) or looks_like_narrative_disclosure(canon):
                continue

            mm = m.copy()
            mm["metric_name"] = canon

            unit = str(mm.get("unit","") or "").strip()
            if not unit:
                unit = unit_type_to_unit(unit_type)
            if not unit:
                unit = infer_unit_from_metric_name(llm, canon)
            if unit and len(unit) > 25:
                unit = ""
            mm["unit"] = unit

            kept.append(mm)

    _STRICT_FILTER_CACHE[signature] = kept
    return kept


# ============================================================
# Merge duplicates (Catalog only)
# ============================================================

def merge_metrics(llm: LLMClient, metrics: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if len(metrics) <= 1:
        return metrics

    # first pass: exact normalized dedup
    seen: Dict[str, Dict[str, Any]] = {}
    for m in metrics:
        k = normalize_metric_name(m.get("metric_name",""))
        if not k:
            continue
        if k not in seen:
            seen[k] = m
        else:
            ex = seen[k]
            ex_sources = ex.get("sources", [ex.get("source","")])
            ns = m.get("source","")
            if ns and ns not in ex_sources:
                ex["sources"] = ex_sources + [ns]

    unique = list(seen.values())
    if len(unique) <= 1:
        return unique

    batch_size = 50
    merged: List[Dict[str, Any]] = []
    name_to_metric = {m["metric_name"]: m for m in unique if m.get("metric_name")}
    used = set()

    for i in range(0, len(unique), batch_size):
        batch = unique[i:i+batch_size]
        items = [{"name": m.get("metric_name",""), "definition": m.get("definition","")} for m in batch]

        prompt = f"""Group semantically identical ESG KPIs (ONLY direct scalar metrics).

Return JSON:
{{
  "groups": [
    {{
      "canonical_name": "KPI name (must be measurable, NOT a category)",
      "members": ["name1","name2",...]
    }}
  ]
}}

Rules:
- Only group if they measure the SAME scalar KPI.
- canonical_name MUST be a measurable KPI name (not "Environmental Metrics", not "Reporting", not "Performance").
- Each input metric must appear in exactly one group.

Metrics:
{json.dumps(items, indent=2, ensure_ascii=False)}
"""
        try:
            resp = llm.chat(prompt)
            obj = parse_json_obj(resp)
            groups = obj.get("groups", [])
            if not isinstance(groups, list):
                groups = []

            for g in groups:
                if not isinstance(g, dict):
                    continue
                canon = str(g.get("canonical_name","") or "").strip()
                members = g.get("members", [])
                if not canon or not isinstance(members, list) or not members:
                    continue

                if looks_like_category_title(canon) or looks_like_narrative_disclosure(canon):
                    for mem in members:
                        if mem in name_to_metric and not looks_like_category_title(mem) and not looks_like_narrative_disclosure(mem):
                            canon = mem
                            break

                if looks_like_category_title(canon) or looks_like_narrative_disclosure(canon):
                    continue

                base = None
                sources: List[str] = []
                for mem in members:
                    if mem in name_to_metric and mem not in used:
                        if base is None:
                            base = name_to_metric[mem].copy()
                        mm = name_to_metric[mem]
                        if isinstance(mm.get("sources"), list):
                            sources.extend(mm.get("sources", []))
                        else:
                            s = mm.get("source","")
                            if s:
                                sources.append(s)
                        used.add(mem)

                if base is not None:
                    base["metric_name"] = canon
                    base["sources"] = list(set([s for s in sources if s]))
                    merged.append(base)

        except Exception as e:
            print(f"      [DEBUG] merge batch failed: {e}")

    for m in unique:
        nm = m.get("metric_name","")
        if nm and nm not in used:
            merged.append(m)

    return merged


# ============================================================
# Catalog building
# ============================================================

def build_catalog(cfg: Config, llm: LLMClient, standard_folders: List[str]) -> List[Dict[str, Any]]:
    script_dir = Path(__file__).resolve().parent
    all_std_metrics: List[Dict[str, Any]] = []

    for folder in standard_folders:
        folder_path = script_dir / folder
        if not folder_path.exists():
            print(f"⚠️ Folder not found: {folder_path}")
            continue

        excel_files = list(folder_path.glob("*.xlsx")) + list(folder_path.glob("*.xls"))
        pdf_files = list(folder_path.glob("*.pdf"))
        print(f"\n📁 [STANDARDS] {folder}: {len(excel_files)} Excel, {len(pdf_files)} PDF")

        for xlsx in tqdm(excel_files, desc=f"Excel {folder}", leave=False):
            all_std_metrics.extend(load_excel_metrics(xlsx, source=folder, llm=llm))

        if pdf_files:
            print(f"   Processing {len(pdf_files)} PDFs in {folder} for KPI definitions...")
            for pdf in tqdm(pdf_files, desc=f"PDF {folder}", leave=False):
                pages = load_pdf_text(pdf)
                chunks = chunk_pages(pages, cfg.max_chunk_chars)
                for ch in chunks:
                    ms = extract_standard_metrics_from_chunk(llm, ch["text"], source=folder)
                    for m in ms:
                        m["pages"] = ch["pages"]
                        m["file"] = pdf.name
                    all_std_metrics.extend(ms)

    if not all_std_metrics:
        return []

    print(f"\n🔍 [CATALOG] strict quantitative filter on {len(all_std_metrics)} items...")
    all_std_metrics = strict_quant_filter(llm, all_std_metrics, keep_conf_threshold=cfg.keep_confidence_threshold)
    print(f"   kept {len(all_std_metrics)}")

    print("🔗 [CATALOG] merging duplicates...")
    all_std_metrics = merge_metrics(llm, all_std_metrics)
    print(f"   after merge: {len(all_std_metrics)}")

    if cfg.post_merge_refilter:
        print("🧽 [CATALOG] post-merge strict re-filter...")
        all_std_metrics = strict_quant_filter(llm, all_std_metrics, keep_conf_threshold=cfg.keep_confidence_threshold)
        print(f"   after post-merge: {len(all_std_metrics)}")

    # assign metric_id
    for m in all_std_metrics:
        h = m.get("hierarchy", get_hierarchy_from_source(m.get("source","")))
        m["hierarchy"] = h
        m["metric_id"] = make_metric_id(m.get("metric_name",""), h)

    return all_std_metrics


# ============================================================
# Company extraction
# ============================================================

def extract_company_values(cfg: Config, llm: LLMClient, pdf_folders: List[str]) -> List[Dict[str, Any]]:
    script_dir = Path(__file__).resolve().parent
    out: List[Dict[str, Any]] = []

    for folder in pdf_folders:
        folder_path = script_dir / folder
        if not folder_path.exists():
            print(f"⚠️ Folder not found: {folder_path}")
            continue

        pdf_files = list(folder_path.glob("*.pdf"))
        print(f"\n📁 [COMPANY] {folder}: {len(pdf_files)} PDFs (extract KPI values)")

        for pdf in tqdm(pdf_files, desc=f"PDF {folder}", leave=False):
            pages = load_pdf_text(pdf)
            chunks = chunk_pages(pages, cfg.max_chunk_chars)
            for ch in chunks:
                ms = extract_company_metrics_from_chunk(llm, ch["text"], source=folder)
                for m in ms:
                    m["pages"] = ch["pages"]
                    m["file"] = pdf.name
                out.extend(ms)

    print(f"\n📊 [COMPANY] extracted {len(out)} KPI-value rows")
    return out


# ============================================================
# Matching: company metric -> catalog metric_id
# ============================================================

def rule_similarity(a: str, b: str) -> float:
    """
    Cheap similarity:
    - character similarity (SequenceMatcher)
    - token Jaccard overlap
    """
    a = (a or "").lower().strip()
    b = (b or "").lower().strip()
    if not a or not b:
        return 0.0

    sm = SequenceMatcher(None, a, b).ratio()
    ta = set(tokenize(a))
    tb = set(tokenize(b))
    if not ta or not tb:
        jac = 0.0
    else:
        jac = len(ta & tb) / max(1, len(ta | tb))

    return 0.65 * sm + 0.35 * jac


def build_catalog_index(catalog: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    idx = []
    for m in catalog:
        idx.append({
            "metric_id": m.get("metric_id",""),
            "metric_name": m.get("metric_name",""),
            "definition": m.get("definition",""),
            "unit": m.get("unit",""),
            "hierarchy": m.get("hierarchy",""),
            "source": m.get("source",""),
        })
    return idx


def llm_match_to_catalog(llm: LLMClient, company_item: Dict[str, Any], candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Let LLM pick best candidate among Top-K.
    Return: {"metric_id": <id or None>, "confidence": 0..1, "reason": "..."}
    """
    comp_name = company_item.get("raw_metric_name","")
    comp_def = company_item.get("definition","")
    comp_val = company_item.get("value","")
    comp_unit = company_item.get("unit_reported","")
    comp_period = company_item.get("period","")
    comp_evi = company_item.get("evidence","")

    # keep candidates concise
    cand_lines = []
    for i, c in enumerate(candidates, 1):
        cand_lines.append({
            "rank": i,
            "metric_id": c.get("metric_id",""),
            "metric_name": c.get("metric_name",""),
            "definition": (c.get("definition","") or "")[:220],
            "unit": c.get("unit",""),
            "hierarchy": c.get("hierarchy",""),
        })

    prompt = f"""You are mapping a company-reported ESG metric to a master KPI catalog.

Company metric:
- name: "{comp_name}"
- definition: "{comp_def}"
- value: "{comp_val}"
- unit: "{comp_unit}"
- period: "{comp_period}"
- evidence: "{comp_evi}"

Choose the best matching catalog KPI from the candidates below.
Return JSON only: {{"metric_id": "<id or null>", "confidence": 0..1, "reason": "short"}}.
If none match confidently, return metric_id=null.

Candidates:
{json.dumps(cand_lines, indent=2, ensure_ascii=False)}
"""
    resp = llm.chat(prompt)
    obj = parse_json_obj(resp)
    mid = obj.get("metric_id", None)
    conf = obj.get("confidence", 0.0)
    reason = obj.get("reason", "")
    try:
        conf = float(conf or 0.0)
    except:
        conf = 0.0
    if isinstance(mid, str) and mid.strip().lower() in ["null", "none", ""]:
        mid = None
    if not isinstance(mid, str):
        mid = None
    return {"metric_id": mid, "confidence": conf, "reason": str(reason or "")}


def match_company_to_catalog(cfg: Config, llm: LLMClient, catalog_idx: List[Dict[str, Any]], company_items: List[Dict[str, Any]]):
    """
    returns:
      matched_rows: list
      unmatched_rows: list
    """
    matched = []
    unmatched = []

    for item in tqdm(company_items, desc="🔎 Matching to catalog", leave=False):
        raw_name = item.get("raw_metric_name","")
        if not raw_name:
            continue

        # Top-K by rule similarity
        scored = []
        for c in catalog_idx:
            s = rule_similarity(raw_name, c.get("metric_name",""))
            # small bonus if unit looks aligned
            unit_rep = (item.get("unit_reported","") or "").lower()
            unit_cat = (c.get("unit","") or "").lower()
            if unit_rep and unit_cat and unit_rep in unit_cat:
                s += 0.03
            scored.append((s, c))
        scored.sort(key=lambda x: x[0], reverse=True)
        top = [c for _, c in scored[:max(1, cfg.match_top_k)]]

        # if no LLM matching, just rule-based decision
        if not cfg.match_use_llm:
            best_score = scored[0][0] if scored else 0.0
            best = scored[0][1] if scored else None
            if best and best_score >= cfg.rule_match_score_threshold:
                matched.append({
                    "metric_id": best.get("metric_id",""),
                    "catalog_metric_name": best.get("metric_name",""),
                    "catalog_unit": best.get("unit",""),
                    "match_confidence": best_score,
                    "match_method": "rule",
                    **item
                })
            else:
                unmatched.append({
                    "match_method": "rule",
                    "best_score": best_score,
                    "top_candidates": "; ".join([f"{c.get('metric_id','')}|{c.get('metric_name','')}" for c in top[:5]]),
                    **item
                })
            continue

        # LLM judge among Top-K
        dec = llm_match_to_catalog(llm, item, top)
        mid = dec.get("metric_id", None)
        conf = float(dec.get("confidence", 0.0) or 0.0)

        if mid and conf >= cfg.match_confidence_threshold:
            # find matched candidate info
            cand = next((c for c in top if c.get("metric_id") == mid), None)
            matched.append({
                "metric_id": mid,
                "catalog_metric_name": (cand.get("metric_name","") if cand else ""),
                "catalog_unit": (cand.get("unit","") if cand else ""),
                "match_confidence": conf,
                "match_method": "llm",
                "match_reason": dec.get("reason",""),
                **item
            })
        else:
            best_score = scored[0][0] if scored else 0.0
            unmatched.append({
                "match_method": "llm",
                "llm_confidence": conf,
                "llm_reason": dec.get("reason",""),
                "best_rule_score": best_score,
                "top_candidates": "; ".join([f"{c.get('metric_id','')}|{c.get('metric_name','')}" for c in top[:5]]),
                **item
            })

    return matched, unmatched


# ============================================================
# Export
# ============================================================

def export_excel(out_path: Path, catalog: List[Dict[str, Any]], matched: List[Dict[str, Any]], unmatched: List[Dict[str, Any]]):
    # Catalog sheet
    cat_rows = []
    for m in catalog:
        sources = m.get("sources", [m.get("source","")]) if isinstance(m.get("sources"), list) else [m.get("source","")]
        sources = [s for s in sources if s]
        src_str = ", ".join(sources)
        if m.get("sheet"):
            src_str = f"{src_str} - {m.get('sheet')}"
        if m.get("file"):
            src_str = f"{src_str} - {m.get('file')}"
        if m.get("pages"):
            src_str = f"{src_str} - pages:{m.get('pages')}"

        cat_rows.append({
            "metric_id": m.get("metric_id",""),
            "metric_name": m.get("metric_name",""),
            "definition": m.get("definition",""),
            "unit": m.get("unit",""),
            "hierarchy": m.get("hierarchy",""),
            "sources": src_str,
        })
    df_cat = pd.DataFrame(cat_rows)

    # CompanyValues sheet
    val_rows = []
    for r in matched:
        val_rows.append({
            "metric_id": r.get("metric_id",""),
            "catalog_metric_name": r.get("catalog_metric_name",""),
            "raw_metric_name": r.get("raw_metric_name",""),
            "value": r.get("value",""),
            "unit_reported": r.get("unit_reported",""),
            "catalog_unit": r.get("catalog_unit",""),
            "period": r.get("period",""),
            "evidence": r.get("evidence",""),
            "pages": str(r.get("pages","")),
            "file": r.get("file",""),
            "match_method": r.get("match_method",""),
            "match_confidence": r.get("match_confidence",""),
            "match_reason": r.get("match_reason",""),
        })
    df_val = pd.DataFrame(val_rows)

    # Unmatched sheet
    um_rows = []
    for r in unmatched:
        um_rows.append({
            "raw_metric_name": r.get("raw_metric_name",""),
            "value": r.get("value",""),
            "unit_reported": r.get("unit_reported",""),
            "period": r.get("period",""),
            "evidence": r.get("evidence",""),
            "pages": str(r.get("pages","")),
            "file": r.get("file",""),
            "match_method": r.get("match_method",""),
            "llm_confidence": r.get("llm_confidence",""),
            "llm_reason": r.get("llm_reason",""),
            "best_rule_score": r.get("best_rule_score",""),
            "top_candidates": r.get("top_candidates",""),
        })
    df_um = pd.DataFrame(um_rows)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        df_cat.to_excel(writer, sheet_name="Catalog", index=False)
        df_val.to_excel(writer, sheet_name="CompanyValues", index=False)
        df_um.to_excel(writer, sheet_name="Unmatched", index=False)


# ============================================================
# Main runner
# ============================================================

def run_pipeline(
    standard_folders: List[str],
    pdf_folders: List[str],
    config: Config,
):
    cfg = config
    llm = LLMClient(cfg)

    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(cfg.out_dir) / f"run_{run_ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Phase A: Catalog
    catalog = build_catalog(cfg, llm, standard_folders)
    print(f"\n✅ Catalog size: {len(catalog)}")

    # Phase B: Company extraction
    company_items = extract_company_values(cfg, llm, pdf_folders)

    # Match
    catalog_idx = build_catalog_index(catalog)
    matched, unmatched = match_company_to_catalog(cfg, llm, catalog_idx, company_items)
    print(f"\n✅ Matched: {len(matched)} | Unmatched: {len(unmatched)}")

    # Export
    out_xlsx = out_dir / "final.xlsx"
    export_excel(out_xlsx, catalog, matched, unmatched)

    # Also dump JSON for debugging if you want
    with open(out_dir / "catalog.json", "w", encoding="utf-8") as f:
        json.dump(catalog, f, ensure_ascii=False, indent=2)
    with open(out_dir / "company_matched.json", "w", encoding="utf-8") as f:
        json.dump(matched, f, ensure_ascii=False, indent=2)
    with open(out_dir / "company_unmatched.json", "w", encoding="utf-8") as f:
        json.dump(unmatched, f, ensure_ascii=False, indent=2)

    print(f"\nDone! Output folder: {out_dir}")
    print(f"   - final.xlsx (Catalog / CompanyValues / Unmatched)")
    print(f"   - catalog.json / company_matched.json / company_unmatched.json")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ESG KPI Catalog + Company Fill Pipeline (Precision-first)")
    parser.add_argument(
        "--standards",
        nargs="*",
        default=["global/GRI", "global/PRI", "global/TCFD"],
        help="Standard folders (build master catalog from Excel + PDF)."
    )
    parser.add_argument(
        "--pdf",
        nargs="*",
        default=["company_specific"],
        help="Company report PDF folders (extract KPI values and match to catalog)."
    )
    parser.add_argument(
        "--use_openai",
        action="store_true",
        help="Use OpenAI API instead of local Ollama."
    )
    parser.add_argument(
        "--out",
        default="outputs",
        help="Output directory."
    )
    parser.add_argument(
        "--keep_conf",
        type=float,
        default=0.80,
        help="KEEP threshold for strict KPI filter confidence (higher => fewer, cleaner)."
    )
    parser.add_argument(
        "--no_llm_match",
        action="store_true",
        help="Disable LLM matching, use rule-based matching instead (default: use LLM)."
    )
    parser.add_argument(
        "--match_conf",
        type=float,
        default=0.75,
        help="Confidence threshold for LLM matching to accept."
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=8,
        help="Top-K catalog candidates for matching."
    )
    args = parser.parse_args()

    cfg = Config(
        use_ollama=not args.use_openai,
        out_dir=args.out,
        keep_confidence_threshold=args.keep_conf,
        match_use_llm=not args.no_llm_match,  # 默认用 LLM 匹配
        match_top_k=args.top_k,
        match_confidence_threshold=args.match_conf,
    )

    run_pipeline(
        standard_folders=args.standards,
        pdf_folders=args.pdf,
        config=cfg,
    )
