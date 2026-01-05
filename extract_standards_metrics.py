# pipeline_simplified_fixed_v2.py
# Simplified ESG metrics extraction pipeline - Fixed v2
#
# What you asked:
# - Metric column should be measurable indicator names (quantitative)
# - Definition should explain what the metric means
#
# Key improvements:
# 1) Much stricter prompt for finding metric column (avoid categories/pillars)
# 2) Definition/unit column prompt improved
# 3) Excel row fallback: if metric cell looks like a category, use LLM to extract a measurable metric name from the row
# 4) Caching for unit inference and row-extraction to reduce LLM calls
# 5) Kept the "filter_measurable" name-based mapping fix

import os
import re
import json
import hashlib
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

import pdfplumber
import pandas as pd
from tqdm import tqdm
from openai import OpenAI


# ============================================================
# Config
# ============================================================

class Config:
    def __init__(
        self,
        use_ollama: bool = True,
        ollama_base_url: str = "http://localhost:11434/v1",
        ollama_model: str = "llama3.1:8b",
        openai_api_key: Optional[str] = None,
        openai_model: str = "gpt-4o",
        max_chunk_chars: int = 14000,
        out_dir: str = "outputs",
    ):
        self.use_ollama = use_ollama
        self.ollama_base_url = ollama_base_url
        self.ollama_model = ollama_model
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.openai_model = openai_model
        self.max_chunk_chars = max_chunk_chars
        self.out_dir = out_dir


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

    def chat(self, prompt: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        return resp.choices[0].message.content or ""


def parse_json_array(text: str) -> List[Any]:
    """Extract JSON array from LLM response."""
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
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group())
            if isinstance(obj, list):
                return obj
        except:
            pass
    return []


def parse_json_obj(text: str) -> Dict[str, Any]:
    """Extract JSON object from LLM response."""
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
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group())
            if isinstance(obj, dict):
                return obj
        except:
            pass
    return {}


# ============================================================
# Utility Functions
# ============================================================

def normalize_metric_name(name: str) -> str:
    """Normalize metric name for matching/deduplication."""
    return re.sub(r"[^a-z0-9]", "", (name or "").lower().strip())


def get_hierarchy_from_source(source: str) -> str:
    """Determine hierarchy level based on source folder name."""
    source_lower = (source or "").lower()
    if source_lower in ["gri", "ifrs", "tcfd", "pri", "unsdg"]:
        return "Global"
    elif source_lower in ["china", "india", "korea", "mexico", "singapore", "uk", "us", "vietnam", "japan"]:
        return "National"
    elif source_lower in ["company_specific", "company"]:
        return "Company-specific"
    elif source_lower in ["sasb"]:
        return "Industry"
    else:
        if "local" in source_lower or "city" in source_lower or "municipal" in source_lower:
            return "Local"
        elif "industry" in source_lower or "sector" in source_lower:
            return "Industry"
        return "Global"


def stable_hash(s: str) -> str:
    return hashlib.md5((s or "").encode("utf-8")).hexdigest()


def looks_like_category_title(name: str) -> bool:
    """
    Heuristic: detect category/pillar/module labels instead of real metrics.
    Examples: "Environmental Metrics", "Governance and Reporting", "Social", "Metrics", "Performance"
    """
    if not name:
        return True
    n = name.strip()
    nl = n.lower()

    # Very common category words
    bad_keywords = [
        "environment", "environmental", "social", "governance",
        "metrics", "metric", "performance", "reporting", "disclosure",
        "pillar", "topic", "category", "categories", "overview", "general",
        "management approach", "strategy", "policies", "policy",
        "appendix", "table", "index", "glossary"
    ]
    if any(k in nl for k in bad_keywords):
        # BUT real metrics can contain some keywords like "emissions" etc.
        # If it also contains strong metric tokens, allow it
        good_tokens = ["emission", "energy", "water", "waste", "injur", "turnover", "divers", "salary", "revenue", "ghg", "scope", "tco2", "%", "ratio"]
        if any(t in nl for t in good_tokens):
            return False
        return True

    # Too short and purely title case words
    words = re.findall(r"[A-Za-z]+", n)
    if len(words) <= 3 and n == n.title() and not re.search(r"\d", n):
        return True

    # Single word, no numbers, not typical metric word
    if len(words) == 1 and not re.search(r"\d", n):
        return True

    return False


# ============================================================
# Unit inference (cached)
# ============================================================

_UNIT_CACHE: Dict[str, str] = {}

def infer_unit_from_metric_name(llm: LLMClient, metric_name: str) -> str:
    """Infer unit of measurement from metric name using LLM (cached)."""
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

Examples:
- "Scope 1 GHG Emissions" -> "tCO₂e"
- "Total Energy Consumption" -> "MWh" or "GJ"
- "Water Withdrawal" -> "m³" or "liters"
- "Employee Turnover Rate" -> "%"
- "Board Gender Diversity" -> "%"
- "Number of Employees" -> "number" or "employees"

Return only the unit, nothing else:
"""
    try:
        resp = llm.chat(prompt)
        unit = (resp or "").strip().strip('"').strip("'")
        if unit.lower() in ["none", "n/a", "na", "null", ""]:
            unit = ""
        # reject if looks like sentence
        if len(unit) > 20 or (unit.count(".") > 1) or ("\n" in unit):
            unit = ""
        _UNIT_CACHE[key] = unit
        return unit
    except Exception as e:
        print(f"      [DEBUG] Unit inference failed for '{metric_name}': {e}")
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
# Metric Extraction (PDF)
# ============================================================

def extract_metrics_definitions_from_chunk(llm: LLMClient, chunk_text: str, source: str) -> List[Dict[str, Any]]:
    """
    Standards PDFs: extract measurable metrics + definitions (no values).
    """
    prompt = f"""Extract measurable (quantitative) ESG metrics and their definitions from the text.

A metric is measurable ONLY if it can be reported as a number with a unit (count/ratio/%/currency/weight/energy/etc.).

DO NOT extract:
- category headings or pillars (e.g., "Environmental Metrics", "Governance")
- section titles, chapter names, module names
- disclosure requirement titles that are not metrics
- standard/version names (e.g., "GRI 2", "ISO 14001")
- policies/commitments without a measurable indicator

Return JSON array with fields:
- metric_name: a SPECIFIC measurable indicator name (if the text only describes it but doesn't name it, create a concise standard English name)
- definition: explanation of what the metric measures (scope/meaning)

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

        # Extra guard: avoid categories
        if looks_like_category_title(name):
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


def extract_metrics_from_chunk(llm: LLMClient, chunk_text: str, source: str) -> List[Dict[str, Any]]:
    """
    Company PDFs: extract metrics WITH values.
    """
    prompt = f"""Extract ONLY measurable (quantitative) ESG metrics from this text.

Each metric MUST have a numeric value. Do NOT extract policies or commitments.

Return JSON array with fields:
- metric_name: standard English name
- definition: short definition from text (what it measures)
- value: numeric value as shown (e.g., "12,345", "12.3%")
- unit: unit or "" if not specified
- period: reporting period (e.g., "2023", "FY2022") or ""
- evidence: short quote (≤20 words) supporting the value

Return [] if no measurable metrics found.

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
        if name and value:
            hierarchy = get_hierarchy_from_source(source)
            metrics.append({
                "metric_name": name,
                "definition": str(r.get("definition", "")).strip(),
                "value": value,
                "unit": str(r.get("unit", "")).strip(),
                "period": str(r.get("period", "")).strip(),
                "evidence": str(r.get("evidence", "")).strip(),
                "source": source,
                "hierarchy": hierarchy,
            })
    return metrics


# ============================================================
# Excel Processing
# ============================================================

def validate_unit_column(df: pd.DataFrame, col: str) -> bool:
    if col not in df.columns:
        return False
    sample = df[col].dropna().head(20)
    if sample.empty:
        return False

    short_count = 0
    non_empty = 0
    for val in sample:
        val_str = str(val).strip()
        if not val_str or val_str.lower() in ["nan", "none", "n/a", "na"]:
            continue
        non_empty += 1
        if len(val_str) <= 15 and val_str.count(" ") <= 2:
            short_count += 1

    if non_empty == 0:
        return False
    return (short_count / non_empty) >= 0.7


def find_metric_column(df: pd.DataFrame, llm: Optional[LLMClient] = None) -> Optional[str]:
    """
    Find the column most likely to contain measurable metric names.
    """
    # Build samples
    col_samples = {}
    for col in df.columns:
        samples = df[col].dropna().head(10).tolist()
        col_samples[col] = [str(s)[:120] for s in samples]

    if llm is not None:
        prompt = f"""Analyze the following Excel columns and identify which column contains measurable (quantitative) ESG metric names.

A "measurable metric name" must be a specific indicator that can be reported as a number with a unit, such as:
- "Scope 1 GHG emissions"
- "Total energy consumption"
- "Water withdrawal"
- "Employee turnover rate"
- "Direct economic value generated and distributed"
- "Number of workplace injuries"

DO NOT choose columns that are:
- high-level categories or pillars (e.g., "Environmental Metrics", "Governance and Reporting", "ESG Disclosure")
- module/section headings, table headings, or grouping labels
- narrative descriptions that are not naming a specific indicator
- standard names or versions (e.g., "GRI 2", "ISO 14001")
- disclosure IDs/titles that are not metrics (e.g., "2-1 Organizational details", "GRI 201: Economic Performance")

If the sheet does not contain a column of measurable metric NAMES, return null.

Columns and their sample values:
{json.dumps(col_samples, indent=2, ensure_ascii=False)}

Return JSON only:
{{"metric_column": <column name or null>}}
"""
        try:
            resp = llm.chat(prompt)
            obj = parse_json_obj(resp)
            metric_col = obj.get("metric_column")
            if metric_col and metric_col in df.columns:
                return metric_col
        except Exception as e:
            print(f"      [DEBUG] AI metric column detection failed: {e}")

    # Fallback heuristics:
    # Prefer columns with many unique non-empty values AND not mostly category titles.
    best_col = None
    best_score = -1

    for col in df.columns:
        series = df[col].dropna().astype(str).map(lambda x: x.strip())
        series = series[series.str.lower().ne("nan")]
        if series.empty:
            continue

        uniq = series.nunique()
        sample_vals = series.head(20).tolist()
        # penalty if many samples look like categories
        cat_hits = sum(1 for v in sample_vals if looks_like_category_title(v))
        score = uniq - 2 * cat_hits

        # small bonus if column name includes metric-ish words
        col_l = col.lower()
        if any(k in col_l for k in ["metric", "indicator", "measure", "item", "requirement", "disclosure", "topic"]):
            score += 3

        if score > best_score:
            best_score = score
            best_col = col

    return best_col


def find_definition_unit_columns(df: pd.DataFrame, llm: Optional[LLMClient] = None) -> Tuple[Optional[str], Optional[str]]:
    def_col, unit_col = None, None

    col_samples = {}
    for col in df.columns:
        samples = df[col].dropna().head(10).tolist()
        col_samples[col] = [str(s)[:120] for s in samples]

    if llm is not None:
        prompt = f"""Analyze the following Excel columns and identify which columns contain metric definitions and units.

definition_column should contain EXPLANATIONS of what the metric means, how it is defined, calculated, or scoped.
- It is usually longer text (phrases/sentences).
- It should NOT be the same as the metric name column (not just repeating the title).

unit_column should contain SHORT unit strings like "%", "tCO2e", "MWh", "hours", "USD", "employees", "number".
- Do NOT select columns with long sentences or full metric names.
- If unsure, return null.

Columns and their sample values:
{json.dumps(col_samples, indent=2, ensure_ascii=False)}

Return JSON only:
{{"definition_column": <column name or null>, "unit_column": <column name or null>}}
"""
        try:
            resp = llm.chat(prompt)
            obj = parse_json_obj(resp)
            d = obj.get("definition_column")
            u = obj.get("unit_column")

            def_col = d if d and d in df.columns else None
            if u and u in df.columns and validate_unit_column(df, u):
                unit_col = u
            else:
                unit_col = None
        except Exception as e:
            print(f"      [DEBUG] AI def/unit detection failed: {e}")

    # Fallback keywords
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


_ROW_EXTRACT_CACHE: Dict[str, Dict[str, str]] = {}

def extract_measurable_metric_from_row(
    llm: LLMClient,
    raw_title: str,
    row_dict: Dict[str, Any],
    metric_col: str,
    def_col: Optional[str],
) -> Dict[str, str]:
    """
    If the 'metric_col' cell looks like category, infer real measurable metric name from row content.
    Returns {"metric_name": "...", "definition": "..."} or empty strings.
    Cached by (raw_title + row_json).
    """
    # Build compact row text from other columns
    parts = []
    for k, v in row_dict.items():
        if k == metric_col:
            continue
        s = str(v).strip()
        if not s or s.lower() in ["nan", "none", "n/a", "na"]:
            continue
        parts.append(f"{k}: {s[:300]}")

    definition_hint = ""
    if def_col and def_col in row_dict:
        definition_hint = str(row_dict.get(def_col, "")).strip()

    row_text = "\n".join(parts)[:2000]
    cache_key = stable_hash((raw_title or "") + "||" + row_text)

    if cache_key in _ROW_EXTRACT_CACHE:
        return _ROW_EXTRACT_CACHE[cache_key]

    prompt = f"""You will be given a row from an ESG standards/disclosure table.

Your job:
1) Output a measurable metric name (quantitative indicator) that this row is actually about.
2) Output a short definition explaining what the metric measures.

Rules:
- metric_name MUST be a specific measurable indicator (not a category/pillar/module)
- It should be something that can be reported as a number with a unit (%/count/currency/energy/water/emissions/etc.)
- If the row does not define any measurable metric, return {{"metric_name":"","definition":""}}

Row:
- raw_title_cell: "{raw_title}"
- other_cells:
{row_text}

Return JSON only:
{{"metric_name":"...", "definition":"..."}}
"""
    resp = llm.chat(prompt)
    obj = parse_json_obj(resp)
    metric_name = str(obj.get("metric_name", "")).strip()
    definition = str(obj.get("definition", "")).strip()

    # Extra safety: if model still returns category, drop it
    if metric_name and looks_like_category_title(metric_name):
        metric_name = ""
    if definition and len(definition) > 1500:
        definition = definition[:1500]

    out = {"metric_name": metric_name, "definition": definition}
    _ROW_EXTRACT_CACHE[cache_key] = out
    return out


def load_excel_metrics(xlsx_path: Path, source: str, llm: Optional[LLMClient] = None) -> List[Dict[str, Any]]:
    """
    Load metrics from Excel file.
    Goal: metric_name should be measurable indicator; definition explains meaning.
    """
    metrics: List[Dict[str, Any]] = []
    hierarchy = get_hierarchy_from_source(source)

    try:
        sheets = pd.read_excel(str(xlsx_path), sheet_name=None)
        for sheet_name, df in sheets.items():
            if df is None or df.empty:
                continue

            metric_col = find_metric_column(df, llm)
            if not metric_col:
                print(f"      [DEBUG] Could not identify metric column in sheet '{sheet_name}'")
                continue

            def_col, unit_col = find_definition_unit_columns(df, llm)

            print(f"      [DEBUG] Sheet '{sheet_name}': metric_col='{metric_col}', def_col='{def_col}', unit_col='{unit_col}'")

            for _, row in df.iterrows():
                raw_name = str(row.get(metric_col, "")).strip()
                if not raw_name or raw_name.lower() == "nan":
                    continue

                # Definition from definition column (if any)
                raw_def = ""
                if def_col:
                    raw_def = str(row.get(def_col, "")).strip()
                    if raw_def.lower() in ["nan", "none", "n/a", "na"]:
                        raw_def = ""

                # If raw_name looks like category title, try LLM row extraction
                final_name = raw_name
                final_def = raw_def

                if llm and looks_like_category_title(raw_name):
                    extracted = extract_measurable_metric_from_row(
                        llm=llm,
                        raw_title=raw_name,
                        row_dict=row.to_dict(),
                        metric_col=metric_col,
                        def_col=def_col,
                    )
                    if extracted.get("metric_name"):
                        final_name = extracted["metric_name"]
                        # If extracted definition exists, prefer it; else keep raw_def
                        if extracted.get("definition"):
                            final_def = extracted["definition"]

                # If after extraction still bad, skip
                if not final_name or final_name.lower() == "nan" or looks_like_category_title(final_name):
                    continue

                # Unit from excel unit column (validated)
                unit = ""
                if unit_col:
                    raw_unit = str(row.get(unit_col, "")).strip()
                    if raw_unit and raw_unit.lower() not in ["nan", "none", "n/a", "na", ""]:
                        if len(raw_unit) <= 20 and raw_unit.count(" ") <= 2:
                            unit = raw_unit

                # Infer unit if none
                if llm and not unit:
                    unit = infer_unit_from_metric_name(llm, final_name)

                metrics.append({
                    "metric_name": final_name,
                    "definition": final_def,
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
# Filter Measurable Metrics (LLM) - name-based mapping
# ============================================================

def filter_measurable(llm: LLMClient, metrics: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Use LLM to filter only measurable (quantitative) metrics.
    Name-based mapping avoids index misalignment.
    """
    if not metrics:
        return []

    batch_size = 30
    measurable: List[Dict[str, Any]] = []

    for i in range(0, len(metrics), batch_size):
        batch = metrics[i:i + batch_size]

        name_to_metric: Dict[str, Dict[str, Any]] = {}
        for m in batch:
            key = normalize_metric_name(m.get("metric_name", ""))
            if key and key not in name_to_metric:
                name_to_metric[key] = m

        items = [{"name": m.get("metric_name", ""), "definition": m.get("definition", "")} for m in batch]

        prompt = f"""Classify each metric as measurable (quantitative) or not.

A metric is measurable ONLY if:
1) It represents a quantifiable ESG performance indicator
2) It can be reported with a numeric value and unit (count/%/ratio/currency/weight/energy/etc.)
3) It is NOT a category/pillar/section heading
4) It is NOT a standard/version name (e.g., "GRI 2", "ISO 14001")
5) It is NOT a disclosure ID/title that is not a metric (e.g., "2-1 Organizational details")
6) It is NOT a policy statement/commitment without measurable indicator

Return JSON array:
- name: metric name (MUST match input exactly)
- measurable: true/false
- reason: brief reason

Items:
{json.dumps(items, indent=2, ensure_ascii=False)}
"""
        resp = llm.chat(prompt)
        results = parse_json_array(resp)

        for r in results:
            if not isinstance(r, dict):
                continue
            if not r.get("measurable"):
                continue
            returned_name = r.get("name", "")
            key = normalize_metric_name(returned_name)
            if key in name_to_metric:
                measurable.append(name_to_metric[key])
                del name_to_metric[key]

    return measurable


# ============================================================
# Merge Duplicate Metrics
# ============================================================

def merge_metrics(llm: LLMClient, metrics: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if len(metrics) <= 1:
        return metrics

    # Simple dedup by normalized name
    seen: Dict[str, Dict[str, Any]] = {}
    for m in metrics:
        key = normalize_metric_name(m.get("metric_name", ""))
        if not key:
            continue
        if key not in seen:
            seen[key] = m
        else:
            existing = seen[key]
            existing_sources = existing.get("sources", [existing.get("source", "")])
            new_source = m.get("source", "")
            if new_source and new_source not in existing_sources:
                existing["sources"] = existing_sources + [new_source]

    unique = list(seen.values())

    # LLM semantic merge in batches
    batch_size = 50
    merged: List[Dict[str, Any]] = []
    name_to_metric = {m["metric_name"]: m for m in unique if m.get("metric_name")}
    used = set()

    for i in range(0, len(unique), batch_size):
        batch = unique[i:i + batch_size]
        items = [{"name": m.get("metric_name", ""), "definition": m.get("definition", "")} for m in batch]

        prompt = f"""Group semantically identical ESG metrics.

Return JSON:
{{
  "groups": [
    {{
      "canonical_name": "best standard name",
      "members": ["name1", "name2", ...]
    }}
  ]
}}

Rules:
- Only group metrics that measure the SAME thing
- canonical_name should be the clearest, most standard name
- Each metric should appear in exactly one group
- Single metrics should still be in a group with just themselves

Metrics:
{json.dumps(items, indent=2, ensure_ascii=False)}
"""
        try:
            resp = llm.chat(prompt)
            result = parse_json_obj(resp)
            groups = result.get("groups", [])
            if not isinstance(groups, list):
                groups = []

            for group in groups:
                if not isinstance(group, dict):
                    continue
                canonical = str(group.get("canonical_name", "")).strip()
                members = group.get("members", [])
                if not canonical or not isinstance(members, list) or not members:
                    continue

                base = None
                sources: List[str] = []
                for member in members:
                    if member in name_to_metric and member not in used:
                        if base is None:
                            base = name_to_metric[member].copy()
                        mm = name_to_metric[member]
                        if isinstance(mm.get("sources"), list):
                            sources.extend(mm.get("sources", []))
                        else:
                            s = mm.get("source", "")
                            if s:
                                sources.append(s)
                        used.add(member)

                if base is not None:
                    base["metric_name"] = canonical
                    base["sources"] = list(set([s for s in sources if s]))
                    merged.append(base)

        except Exception as e:
            print(f"      [DEBUG] Merge batch {i//batch_size + 1} failed: {e}, keeping originals")
            for m in batch:
                nm = m.get("metric_name", "")
                if nm and nm not in used:
                    merged.append(m)
                    used.add(nm)

    # Add any unused
    for m in unique:
        nm = m.get("metric_name", "")
        if nm and nm not in used:
            merged.append(m)

    return merged


# ============================================================
# Main Pipeline
# ============================================================

def run_pipeline(
    standard_folders: List[str] = None,
    pdf_folders: List[str] = None,
    config: Config = None,
):
    cfg = config or Config()
    llm = LLMClient(cfg)
    script_dir = Path(__file__).resolve().parent

    standard_folders = standard_folders or []
    pdf_folders = pdf_folders or []

    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(cfg.out_dir) / f"run_{run_ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_metrics: List[Dict[str, Any]] = []

    # Standards: Excel + PDF definitions
    for folder in standard_folders:
        folder_path = script_dir / folder
        if not folder_path.exists():
            print(f"⚠️ Folder not found: {folder_path}")
            continue

        excel_files = list(folder_path.glob("*.xlsx")) + list(folder_path.glob("*.xls"))
        pdf_files = list(folder_path.glob("*.pdf"))

        print(f"\n📁 {folder}: {len(excel_files)} Excel files, {len(pdf_files)} PDF files")

        # Excel
        for xlsx in tqdm(excel_files, desc=f"Excel {folder}", leave=False):
            metrics = load_excel_metrics(xlsx, source=folder, llm=llm)
            all_metrics.extend(metrics)

        # PDF (definitions only)
        if pdf_files:
            print(f"   Processing {len(pdf_files)} PDF files from {folder} (extracting metric definitions)...")
            for pdf in tqdm(pdf_files, desc=f"PDF {folder}", leave=False):
                pages = load_pdf_text(pdf)
                chunks = chunk_pages(pages, cfg.max_chunk_chars)
                for chunk in chunks:
                    metrics = extract_metrics_definitions_from_chunk(llm, chunk["text"], source=folder)
                    for m in metrics:
                        m["pages"] = chunk["pages"]
                    all_metrics.extend(metrics)

    # Filter measurable
    if all_metrics:
        print(f"\n🔍 Filtering {len(all_metrics)} metrics for measurability...")
        all_metrics = filter_measurable(llm, all_metrics)
        print(f"   Kept {len(all_metrics)} measurable metrics")

    # Company PDFs: metrics with values
    for folder in pdf_folders:
        folder_path = script_dir / folder
        if not folder_path.exists():
            print(f"⚠️ Folder not found: {folder_path}")
            continue

        pdf_files = list(folder_path.glob("*.pdf"))
        print(f"\n📁 {folder}: {len(pdf_files)} PDF files (extracting metrics with values)")

        for pdf in tqdm(pdf_files, desc=f"PDF {folder}"):
            pages = load_pdf_text(pdf)
            chunks = chunk_pages(pages, cfg.max_chunk_chars)

            for chunk in chunks:
                metrics = extract_metrics_from_chunk(llm, chunk["text"], source=folder)
                hierarchy = get_hierarchy_from_source(folder)
                for m in metrics:
                    m["pages"] = chunk["pages"]
                    m["hierarchy"] = hierarchy
                all_metrics.extend(metrics)

    print(f"\n📊 Total metrics before merge: {len(all_metrics)}")

    # Merge duplicates
    if all_metrics:
        print("🔗 Merging duplicate metrics...")
        all_metrics = merge_metrics(llm, all_metrics)
        print(f"   After merge: {len(all_metrics)}")

    # Format output
    output_rows = []
    for m in all_metrics:
        hierarchy = m.get("hierarchy", get_hierarchy_from_source(m.get("source", "")))

        # Example value only for company-specific
        example_value = ""
        is_company_specific = (hierarchy == "Company-specific" or "company" in (m.get("source", "") or "").lower())

        if is_company_specific:
            value = str(m.get("value", "")).strip()
            unit_v = str(m.get("unit", "")).strip()
            period = str(m.get("period", "")).strip()
            if value:
                parts = [value]
                if unit_v:
                    parts.append(unit_v)
                if period:
                    parts.append(f"({period})")
                example_value = " ".join(parts)

        # Clean unit
        unit = str(m.get("unit", "")).strip()
        if unit.lower() in ["nan", "none", "n/a", "na"]:
            unit = ""
        if len(unit) > 25:
            unit = ""

        # Sources
        source = m.get("source", "")
        sources = m.get("sources", [source]) if isinstance(m.get("sources"), list) else [source]
        sources_str = ", ".join([s for s in sources if s])
        if m.get("sheet"):
            sources_str = f"{sources_str} - {m.get('sheet')}"

        output_rows.append({
            "Metric (Standardized Name)": str(m.get("metric_name", "")).strip(),
            "Definition": str(m.get("definition", "")).strip(),
            "Sources": sources_str,
            "Hierarchy": hierarchy,
            "Example Value (if present)": example_value,
            "Unit": unit,
        })

    df = pd.DataFrame(output_rows)
    df.to_excel(out_dir / "metrics.xlsx", index=False)
    df.to_csv(out_dir / "metrics.csv", index=False)

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Done! Output: {out_dir}")
    print(f"   - metrics.xlsx ({len(output_rows)} rows)")
    print(f"   - metrics.csv ({len(output_rows)} rows)")
    print(f"   - metrics.json (raw data)")

    return all_metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ESG Metrics Extraction Pipeline (Fixed v2)")
    parser.add_argument(
        "--standards",
        nargs="*",
        default=["GRI", "PRI"],
        help="Standard folders (extract definitions only, no values). Can contain Excel and/or PDF files."
    )
    parser.add_argument(
        "--pdf",
        nargs="*",
        default=["company_specific"],
        help="Company report PDF folders (extract metrics WITH numeric values)"
    )
    parser.add_argument(
        "--use_openai",
        action="store_true",
        help="Use OpenAI API instead of local Ollama"
    )
    parser.add_argument(
        "--out",
        default="outputs",
        help="Output directory"
    )
    args = parser.parse_args()

    cfg = Config(
        use_ollama=not args.use_openai,
        out_dir=args.out,
    )

    run_pipeline(
        standard_folders=args.standards,
        pdf_folders=args.pdf,
        config=cfg,
    )
