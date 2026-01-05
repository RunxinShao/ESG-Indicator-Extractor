import os
import pdfplumber
import pandas as pd
import json
import re
import time
from typing import List, Dict, Any
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

LEVEL_ENUM = ["global standard", "national", "local", "industry", "company specific"]

# -----------------------------
# 1) 提取 PDF 段落 + 页码
# -----------------------------
def extract_paragraphs_from_pdf(pdf_path: str) -> List[Dict[str, Any]]:
    paras = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if not text:
                continue
            for para in text.split("\n"):
                cleaned = (para or "").strip()
                if len(cleaned) < 5:
                    continue
                if re.fullmatch(r"[\W_]+", cleaned):
                    continue
                paras.append({"page": i + 1, "text": cleaned})
    return paras

def clean_json_output(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return "[]"
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    return text.strip()

# -----------------------------
# 2) Yes/No 分类（带上下文）
# -----------------------------
def build_yesno_prompt(paragraph: str, context_prev: str, context_next: str) -> str:
    return f"""
You are an ESG analyst. Determine whether the CURRENT paragraph likely contains quantitative or structural ESG indicator data
(e.g., emissions Scope 1/2/3, energy use, water, waste, recycling, training hours, injury rates, board composition, targets, KPIs).

Answer ONLY "Yes" or "No".

Previous context:
\"\"\"{context_prev}\"\"\"

Current paragraph:
\"\"\"{paragraph}\"\"\"

Next context:
\"\"\"{context_next}\"\"\"
""".strip()

def is_potential_esg_paragraph(
    paragraph: str,
    context_prev: str = "",
    context_next: str = "",
    model: str = "gpt-4o",
) -> bool:
    prompt = build_yesno_prompt(paragraph, context_prev, context_next)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    ans = (resp.choices[0].message.content or "").strip().lower()
    return ans.startswith("yes")

# -----------------------------
# 3) Yes 的段落才抽取 rows（含 definition/level/page）
# -----------------------------
def build_extraction_prompt(paragraph: str, context_prev: str, context_next: str, page: int) -> str:
    return f"""
You are an ESG analyst. You will be given a paragraph from a corporate sustainability report, plus surrounding context.

Task:
Extract ALL quantitative or structural ESG indicators mentioned (ONLY those with clearly stated values or counts/percentages/targets).
For EACH indicator, output one JSON object with the following fields:
- "indicator_name": short human-readable name
- "definition": 1–2 sentences based ONLY on paragraph + context. If not defined, keep minimal and do NOT invent details.
- "values": exact value string as stated (preserve units)
- "page": integer page number (use the provided page)
- "level": choose EXACTLY ONE from:
  ["global standard", "national", "local", "industry", "company specific"]

How to assign "level":
- "global standard": explicitly tied to global frameworks/standards (GHG Protocol, GRI, SASB, ISSB, TCFD, CDP, UN SDGs, ISO, etc.)
- "national": national laws/regulations/country-level mandatory reporting
- "local": state/province/city/municipal/local authority requirements
- "industry": sector/industry standards or supply-chain frameworks (e.g., RBA)
- "company specific": internal KPIs/targets/figures without external standard/regulation anchors

Output format:
Return ONLY a valid JSON array. If nothing extractable, return [].

Page: {page}

Previous context:
\"\"\"{context_prev}\"\"\"

Current paragraph:
\"\"\"{paragraph}\"\"\"

Next context:
\"\"\"{context_next}\"\"\"
""".strip()

def extract_esg_rows_from_paragraph(
    paragraph: str,
    context_prev: str,
    context_next: str,
    page: int,
    model: str = "gpt-4o",
    max_retries: int = 2,
    sleep_sec: float = 1.0,
) -> List[Dict[str, Any]]:
    prompt = build_extraction_prompt(paragraph, context_prev, context_next, page)

    last_err = None
    for _ in range(max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            content = clean_json_output(resp.choices[0].message.content)
            parsed = json.loads(content)

            if not isinstance(parsed, list):
                return []

            normalized = []
            for r in parsed:
                if not isinstance(r, dict):
                    continue
                indicator_name = str(r.get("indicator_name", "")).strip()
                definition = str(r.get("definition", "")).strip()
                values = str(r.get("values", "")).strip()
                level = str(r.get("level", "")).strip()

                if not indicator_name or not values:
                    continue
                if level not in LEVEL_ENUM:
                    level = "company specific"

                normalized.append({
                    "indicator_name": indicator_name,
                    "definition": definition,
                    "values": values,
                    "page": int(page),
                    "level": level,
                })

            return normalized

        except Exception as e:
            last_err = e
            time.sleep(sleep_sec)

    raise RuntimeError(f"LLM extraction failed after retries: {last_err}")

# -----------------------------
# 4) 保存
# -----------------------------
def save_intermediate_results(rows: List[Dict[str, Any]], output_path: str) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows, columns=["indicator_name", "definition", "values", "page", "level"])
    df.to_excel(output_path, index=False)

# -----------------------------
# 5) 主流程：先 Yes/No，再抽取
# -----------------------------
def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pdf_path = os.path.join(script_dir, "luxshare_sustainability_report_2024.pdf")
    output_path = os.path.join(script_dir, "esg_extracted_results_yesno_post.xlsx")

    if not os.path.exists(pdf_path):
        print(f"❌ 找不到 PDF 文件: {pdf_path}")
        return

    paras = extract_paragraphs_from_pdf(pdf_path)
    print(f"📄 段落数: {len(paras)}")

    extracted_rows: List[Dict[str, Any]] = []

    processed = 0
    yes_count = 0
    no_count = 0
    error_count = 0

    # 上下文窗口：前后各 N 段（你可以改成 2）
    context_window = 1

    # 保存频率：每累计多少条“指标 row”保存一次
    save_interval_rows = 30

    for i, item in enumerate(paras):
        processed += 1
        page = item["page"]
        text = item["text"]

        # 组上下文
        prev_texts = []
        next_texts = []
        for k in range(1, context_window + 1):
            if i - k >= 0:
                prev_texts.append(paras[i - k]["text"])
            if i + k < len(paras):
                next_texts.append(paras[i + k]["text"])

        context_prev = "\n".join(reversed(prev_texts))
        context_next = "\n".join(next_texts)

        try:
            # 先 Yes/No
            if not is_potential_esg_paragraph(
                paragraph=text,
                context_prev=context_prev,
                context_next=context_next,
                model="gpt-4o",   # 想省钱可换成更便宜的小模型（如果你账号可用）
            ):
                no_count += 1
                if no_count % 200 == 0:
                    print(f"⏩ 已判 No: {no_count}")
                continue

            yes_count += 1
            print(f"✅ Yes | 第 {page} 页 | 开始抽取...")

            # Yes 才抽取 rows
            rows = extract_esg_rows_from_paragraph(
                paragraph=text,
                context_prev=context_prev,
                context_next=context_next,
                page=page,
                model="gpt-4o",
            )

            if rows:
                before = len(extracted_rows)
                extracted_rows.extend(rows)
                added = len(extracted_rows) - before
                print(f"   ✓ 抽到 {added} 条指标（累计 {len(extracted_rows)}）")
                for r in rows[:2]:
                    print(f"   • {r['indicator_name']} | {r['values']} | {r['level']}")
            else:
                print(f"   ⚠️ Yes 但没抽到指标（第 {page} 页）")

            # 增量保存
            if len(extracted_rows) > 0 and (len(extracted_rows) % save_interval_rows) < max(len(rows), 1):
                save_intermediate_results(extracted_rows, output_path)
                print(f"💾 已自动保存: {output_path}")

            if processed % 50 == 0:
                print(f"⏳ processed {processed}/{len(paras)} | Yes {yes_count} | No {no_count} | rows {len(extracted_rows)} | errors {error_count}")

        except Exception as e:
            error_count += 1
            print(f"❌ 出错 | 第 {page} 页: {e}")

    # 最终保存
    if extracted_rows:
        save_intermediate_results(extracted_rows, output_path)
        print("\n🎉 完成！")
        print(f"   Yes: {yes_count}, No: {no_count}, Errors: {error_count}")
        print(f"   指标行数: {len(extracted_rows)}")
        print(f"   输出: {output_path}")
    else:
        df = pd.DataFrame(columns=["indicator_name", "definition", "values", "page", "level"])
        df.to_excel(output_path, index=False)
        print("\n⚠️ 没抽到任何指标，已保存空表头:", output_path)

if __name__ == "__main__":
    main()
