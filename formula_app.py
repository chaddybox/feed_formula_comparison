import streamlit as st
import camelot
import pandas as pd
import json
import re
import io
import tempfile
from openai import OpenAI

# -------------------------------
# Initialize OpenAI
# -------------------------------
client = OpenAI(api_key=st.secrets["openai"]["api_key"])

# Initialize session state
if "df1" not in st.session_state:
    st.session_state.df1 = None
if "df2" not in st.session_state:
    st.session_state.df2 = None
if "file1_pages" not in st.session_state:
    st.session_state.file1_pages = None
if "file1_path" not in st.session_state:
    st.session_state.file1_path = None
if "file2_path" not in st.session_state:
    st.session_state.file2_path = None

# -------------------------------
# Helpers
# -------------------------------

def extract_number(x):
    """Extract numeric value from a string, return float or None."""
    if pd.isna(x):
        return None
    x = str(x).strip()
    x = x.replace(",", "")
    x = re.sub(r"[^0-9.\-]", "", x)
    try:
        return float(x) if x else None
    except:
        return None


def postprocess_pairs(df):
    """
    Pair ingredient rows with following numeric rows.
    Fixes the 'ingredient + blank row + amount' misalignment.
    """
    df = df.copy().reset_index(drop=True)
    fixed_rows = []
    i = 0
    while i < len(df):
        ing = str(df.at[i, "Ingredient"]).strip()
        amt = extract_number(df.at[i, "Amount"])

        # If this row has no numeric amount, check the next row
        if amt is None and i + 1 < len(df):
            next_ing = str(df.at[i + 1, "Ingredient"]).strip()
            next_amt = extract_number(df.at[i + 1, "Amount"])
            if next_amt is not None and (next_ing == "" or next_ing.lower() in ["none", "nan"]):
                amt = next_amt
                i += 1  # skip the amount-only row

        if ing != "" and ing.lower() not in ["none", "nan"]:
            fixed_rows.append({"Ingredient": ing, "Amount": amt})
        i += 1

    return pd.DataFrame(fixed_rows)


def count_pdf_pages(path):
    """Count the number of pages in a PDF."""
    try:
        tables = camelot.read_pdf(path, flavor="lattice", pages="1")
        # Use PyPDF2 to get page count if available, otherwise estimate
        import PyPDF2
        with open(path, 'rb') as f:
            pdf = PyPDF2.PdfReader(f)
            return len(pdf.pages)
    except:
        # Fallback: try to read all pages and count
        try:
            tables = camelot.read_pdf(path, flavor="lattice", pages="all")
            if tables.n > 0:
                # Estimate pages from table distribution
                return max([t.page for t in tables])
            return 1
        except:
            return 1


def extract_ingredients_from_pdf(path, pages="all"):
    """Extract Ingredient + Amount columns from PDF using Camelot, then clean + fix misalignment."""
      # --- Read PDF tables ---
    tables = camelot.read_pdf(path, flavor="stream", pages=pages, edge_tol=50)
    if tables.n == 0:
        tables = camelot.read_pdf(path, flavor="lattice", pages=pages)
    if tables.n == 0:
        return pd.DataFrame(columns=["Ingredient", "Amount"])

    df_list = [t.df for t in tables]
    raw = pd.concat(df_list, ignore_index=True)

    # --- Find header row ---
    header_row = None
    for i, row in raw.iterrows():
        row_lower = [str(cell).lower() for cell in row]
        if any("ingredient" in c or "name" in c or "code" in c for c in row_lower) and \
           any("amount" in c or "lb" in c or "ton" in c for c in row_lower):
            header_row = i
            break
    if header_row is None:
        return pd.DataFrame(columns=["Ingredient", "Amount"])

    raw.columns = raw.iloc[header_row]
    raw = raw.drop(index=list(range(0, header_row + 1)))
    raw = raw.rename(columns=lambda x: str(x).strip())

    # --- Header normalization helper ---
    def normalize_header(s):
        if s is None:
            return ""
        s = str(s)
        s = re.sub(r"<[^>]*>", " ", s)     # remove <None> tokens
        s = s.replace("\xa0", " ")
        s = re.sub(r"[^A-Za-z0-9\s/]", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    # --- Ingredient column detection ---
    ing_col = None
    for c in raw.columns:
        if "name" in c.lower():
            ing_col = c
            break
    if not ing_col:
        for c in raw.columns:
            if "ingredient" in c.lower() or "code" in c.lower():
                ing_col = c
                break

    # --- Amount column detection (robust multi-tier logic) ---
    amt_col = None
    amt_col_idx = None
    dollar = "$"

    # 1️⃣ Direct "amount" match (for simpler PDFs)
    for idx, c in enumerate(raw.columns):
        if "amount" in str(c).lower() and dollar not in str(c):
            amt_col = c
            amt_col_idx = idx
            break

    # 2️⃣ Prefer column containing "lb/ton" and "af" (anywhere, any order)
    if not amt_col:
        for idx, c in enumerate(raw.columns):
            c_norm = normalize_header(c).lower()
            if "lb/ton" in c_norm and "af" in c_norm and dollar not in c_norm:
                amt_col = c
                amt_col_idx = idx
                break

    # 3️⃣ Broader: header contains "lb/ton" at all
    if not amt_col:
        for idx, c in enumerate(raw.columns):
            c_norm = normalize_header(c).lower()
            if "lb/ton" in c_norm and dollar not in c_norm:
                amt_col = c
                amt_col_idx = idx
                break

    # 4️⃣ Header contains both "lb" and "af" (handles merged headers like Seagull Bay)
    if not amt_col:
        for idx, c in enumerate(raw.columns):
            c_norm = normalize_header(c).lower()
            if "lb" in c_norm and "af" in c_norm and dollar not in c_norm:
                amt_col = c
                amt_col_idx = idx
                break

    # 5️⃣ Final fallback: pick the most numeric column (ignoring cost/%)
    if not amt_col:
        best_idx = None
        best_ratio = 0
        for idx, c in enumerate(raw.columns):
            cname = str(c).lower()
            if "$" in cname or "cost" in cname or "%" in cname:
                continue
            col = raw[c].dropna().astype(str)
            numeric_ratio = col.str.match(r"^\s*-?\d+(\.\d+)?\s*$").mean()
            if numeric_ratio > best_ratio:
                best_ratio = numeric_ratio
                best_idx = idx
        if best_idx is not None and best_ratio > 0.5:
            amt_col = raw.columns[best_idx]
            amt_col_idx = best_idx
            st.write(f"⚠️ Used numeric fallback column: {amt_col} (ratio={best_ratio:.2f})")

    # --- Debug info if no column found ---
    if not ing_col or not amt_col:
        st.write("DEBUG: Could not identify Ingredient or Amount column.")
        for i, c in enumerate(raw.columns):
            st.write(i, repr(c))
        return pd.DataFrame(columns=["Ingredient", "Amount"])

    # --- Keep Ingredient + Amount ---
    result = raw[[ing_col, amt_col]].copy()
    result.columns = ["Ingredient", "Amount"]
    result = result[result["Ingredient"].notna()]

    # --- Numeric cleaning ---
    def extract_number(x):
        if pd.isna(x):
            return None
        s = str(x).strip()
        s = re.sub(r"[^0-9.\-]", "", s)
        try:
            return float(s) if s else None
        except:
            return None

    result["Amount"] = result["Amount"].apply(extract_number)

    # --- Fix misalignment ---
    result = postprocess_pairs(result)

    # --- Stop at totals or nutrient sections ---
    stop_keywords = ["total", "nutrient analysis", "moisture %", "protein %",
                     "chloride %", "costs", "ingredient cost", "dcad balance"]
    final_rows = []
    for _, row in result.iterrows():
        ingredient = str(row["Ingredient"]).strip().lower()
        if any(k in ingredient for k in stop_keywords):
            break
        if ingredient in ["feeding rate ingredient", "ingredient name", "ingredient detail", ""]:
            continue
        final_rows.append(row)

    if not final_rows:
        return pd.DataFrame(columns=["Ingredient", "Amount"])

    return pd.DataFrame(final_rows).reset_index(drop=True)



# -------------------------------
# AI: JSON-first + robust fallback parsing
# -------------------------------

def ai_match_formulas(df1, df2, tolerance=0.05):
    """
    Ask the model to return a JSON array of comparison objects.
    Returns: (parsed_list_or_None, raw_text_from_model)
    """
    formula1_list = df1.to_dict(orient="records")
    formula2_list = df2.to_dict(orient="records")

    prompt = f"""
You are comparing two feed formulas.
Formula 1 (user-submitted): {formula1_list}
Formula 2 (software-entered): {formula2_list}

Rules:
- Identify the best match in Formula 2 for each Formula 1 ingredient.
- Consider synonyms, abbreviations, ingredient codes, and common feed naming conventions.
- Compare ingredient *amounts* numerically when both are available.
  • The tolerance is ±{int(tolerance*100)}%.
  • If Formula 2's amount is within this range of Formula 1, that is a **Match ✅**.
  • If it differs by more than this tolerance, it is a **Mismatch ❌**.
  • If an amount is missing (null), still attempt a name match but mark the amount comparison as "No Data".
- Do not mark amounts as mismatched if the difference is within tolerance.
- Include a short note in "Amount Notes" if the difference is outside tolerance, e.g. "Different moisture basis" or "Rounded value".
- Return results as described below.

OUTPUT REQUIREMENTS (IMPORTANT):
Return ONLY a JSON array, nothing else. Do NOT include explanations, markdown, or code fences.
Each array element must be an object with these exact keys:
  "Ingredient Formula1" (string),
  "Amount Formula1" (number or null),
  "Ingredient Formula2" (string or null),
  "Amount Formula2" (number or null),
  "Match Status" (string; e.g., "Match ✅", "Mismatch ❌", "No Match ❔"),
  "Ingredient Notes" (string; may be empty)
  "Amount Notes" (string; may be empty)

Example output (must match this shape exactly):
[
  {{
    "Ingredient Formula1": "Calcium Carbonate",
    "Amount Formula1": 845.3,
    "Ingredient Formula2": "CALCIUM CARBONATE 3",
    "Amount Formula2": 843.3203,
    "Match Status": "Match",
    "Notes": ""
  }}
]
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an expert in animal feed formulation comparisons."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    raw = response.choices[0].message.content.strip()

    # Clean code fences if present
    raw_clean = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.I)
    raw_clean = re.sub(r"\s*```$", "", raw_clean, flags=re.I).strip()

    # Try parse JSON
    try:
        data = json.loads(raw_clean)
        if isinstance(data, list) and all(isinstance(item, dict) for item in data):
            return data, raw_clean
    except Exception:
        pass

    # If JSON parse failed, return None + raw for fallback parsing
    return None, raw_clean


def fallback_parse_csv_text(csv_text):
    """
    Heuristic parsing of messy CSV lines. Returns list of dicts with the target keys.
    Tries regex anchored on numeric amounts. If regex fails for a line, falls back to
    splitting with max 5 splits.
    """
    rows = []
    lines = [ln.strip() for ln in csv_text.splitlines() if ln.strip()]

    pattern = re.compile(
        r'^\s*(?P<ing1>.+?)(?=,\s*[-]?\d+(?:\.\d+)?)\s*,\s*(?P<amt1>[-]?\d+(?:\.\d+)?|N/A|null)?\s*,\s*'
        r'(?P<ing2>.+?)(?=,\s*[-]?\d+(?:\.\d+)?)\s*,\s*(?P<amt2>[-]?\d+(?:\.\d+)?|N/A|null)?\s*,\s*'
        r'(?P<status>[^,]+?)\s*,\s*(?P<notes>.*)\s*$',
        flags=re.I
    )

    def to_num(x):
        if x is None:
            return None
        s = str(x).strip()
        if s.lower() in ("n/a", "na", "null", ""):
            return None
        try:
            return float(s)
        except:
            return None

    for ln in lines:
        m = pattern.match(ln)
        if m:
            g = m.groupdict()
            rows.append({
                "Ingredient Formula1": g["ing1"].strip(),
                "Amount Formula1": to_num(g.get("amt1")),
                "Ingredient Formula2": g["ing2"].strip(),
                "Amount Formula2": to_num(g.get("amt2")),
                "Match Status": (g.get("status") or "").strip(),
                "Notes": (g.get("notes") or "").strip()
            })
        else:
            parts = [p.strip() for p in ln.split(",", 5)]
            while len(parts) < 6:
                parts.append("")
            rows.append({
                "Ingredient Formula1": parts[0],
                "Amount Formula1": to_num(parts[1]),
                "Ingredient Formula2": parts[2],
                "Amount Formula2": to_num(parts[3]),
                "Match Status": parts[4],
                "Notes": parts[5]
            })

    return rows


# -------------------------------
# Streamlit UI
# -------------------------------
st.title("AI-Powered Feed Formula Comparison")

file1 = st.file_uploader("Upload Formula 1 (user PDF)", type=["pdf"])
file2 = st.file_uploader("Upload Formula 2 (software PDF)", type=["pdf"])

if st.button("🖥️ Extract Formulas"):
    if file1 is None or file2 is None:
        st.error("Please upload both PDF files before extracting.")
    else:
        with st.spinner("📄 Extracting ingredients from PDFs..."):
            # Save files to temp paths
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as t1:
                t1.write(file1.read())
                path1 = t1.name
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as t2:
                t2.write(file2.read())
                path2 = t2.name

            # Store paths for later use
            st.session_state.file1_path = path1
            st.session_state.file2_path = path2

            # Check if File1 has multiple pages
            num_pages = count_pdf_pages(path1)
            
            if num_pages > 1:
                st.session_state.file1_pages = num_pages
                st.info(f"📄 Formula 1 has {num_pages} pages. Please select which page to use below.")
            else:
                # Single page - extract immediately
                df1 = extract_ingredients_from_pdf(path1)
                df2 = extract_ingredients_from_pdf(path2)

                if df1.empty or df2.empty:
                    st.error("❌ Couldn't extract ingredients from one or both PDFs.")
                else:
                    st.session_state.df1 = df1
                    st.session_state.df2 = df2
                    st.session_state.file1_pages = None
                    st.success("✅ Formulas extracted successfully!")
                    
                # Clean up temp files for single-page case
                import os
                try:
                    os.unlink(path1)
                    os.unlink(path2)
                    st.session_state.file1_path = None
                    st.session_state.file2_path = None
                except:
                    pass

# Page selector for multi-page File1
if st.session_state.file1_pages is not None and st.session_state.file1_pages > 1:
    selected_page = st.selectbox(
        "Select page from Formula 1:",
        options=list(range(1, st.session_state.file1_pages + 1)),
        format_func=lambda x: f"Page {x}"
    )
    
    if st.button("📄 Extract from Selected Page"):
        with st.spinner(f"📄 Extracting from page {selected_page}..."):
            df1 = extract_ingredients_from_pdf(st.session_state.file1_path, pages=str(selected_page))
            df2 = extract_ingredients_from_pdf(st.session_state.file2_path)

            if df1.empty or df2.empty:
                st.error("❌ Couldn't extract ingredients from one or both PDFs.")
            else:
                st.session_state.df1 = df1
                st.session_state.df2 = df2
                st.session_state.file1_pages = None  # Clear the page selector
                st.success("✅ Formulas extracted successfully!")
                
                # Clean up temp files after extraction
                import os
                try:
                    os.unlink(st.session_state.file1_path)
                    os.unlink(st.session_state.file2_path)
                    st.session_state.file1_path = None
                    st.session_state.file2_path = None
                except:
                    pass
                
                st.rerun()

# Display extracted formulas if they exist in session state
if st.session_state.df1 is not None and st.session_state.df2 is not None:
    st.subheader("📄 Extracted Formula 1")
    st.dataframe(st.session_state.df1, use_container_width=True)
    st.subheader("📄 Extracted Formula 2")
    st.dataframe(st.session_state.df2, use_container_width=True)

    # Now the comparison button is always available after extraction
    if st.button("🚀 Run AI Comparison"):
        with st.spinner("🤖 Running AI-powered comparison..."):
            parsed_json, raw_text = ai_match_formulas(st.session_state.df1, st.session_state.df2)

        # If we got valid JSON, use it; otherwise attempt robust CSV fallback
        if parsed_json is not None:
            ai_rows = parsed_json
            st.success("Parsed JSON result from AI.")
        else:
            st.warning("AI did not return valid JSON — attempting robust CSV parsing fallback.")
            ai_rows = fallback_parse_csv_text(raw_text)

        try:
            ai_df = pd.DataFrame(ai_rows)
            if "Amount Formula1" in ai_df.columns:
                ai_df["Amount Formula1"] = pd.to_numeric(ai_df["Amount Formula1"], errors="coerce")
            if "Amount Formula2" in ai_df.columns:
                ai_df["Amount Formula2"] = pd.to_numeric(ai_df["Amount Formula2"], errors="coerce")

            st.subheader("✅ AI Comparison (parsed)")
            st.dataframe(ai_df, use_container_width=True)

            csv_blob = ai_df.to_csv(index=False)
            st.download_button("💾 Download Parsed Comparison (CSV)", data=csv_blob,
                               file_name="ai_comparison_parsed.csv", mime="text/csv")
            st.download_button("📄 Download Raw AI Output (text)", data=raw_text,
                               file_name="ai_comparison_raw.txt", mime="text/plain")
        except Exception as e:
            st.error(f"Failed to render AI results: {e}")
            st.subheader("Raw AI output")
            st.text(raw_text)