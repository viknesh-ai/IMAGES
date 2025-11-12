# main.py
"""
Streamlit single-file app: AI-assisted column rule generation + scalable validation.

Install requirements:
pip install streamlit pandas numpy requests python-dotenv rapidfuzz scikit-learn openpyxl sqlalchemy psycopg2-binary

Usage:
  set environment variables in .env (SOGPT_HOST, SOGPT_KEY_NAME, SOGPT_KEY_VALUE,
  SGC_CLIENT_ID, SGC_CLIENT_SECRET, SGC_TOKEN_URL). Then:
  streamlit run main.py
"""

import os
import io
import json
import re
import time
import math
import textwrap
import tempfile
from typing import List, Dict, Any, Optional, Tuple

import pandas as pd
import numpy as np
import requests
import streamlit as st
from rapidfuzz import fuzz
from sklearn.ensemble import IsolationForest

# optional: DB support
from sqlalchemy import create_engine

# load env
from dotenv import load_dotenv
load_dotenv()

# ----------------------
# Environment & SOGPT
# ----------------------
APP_NAME = os.getenv("SOGPT_APP_NAME", "DQLineageApp")
SOGPT_HOST = os.getenv("SOGPT_HOST", "").rstrip("/")
SOGPT_KEY_NAME = os.getenv("SOGPT_KEY_NAME", "")
SOGPT_KEY_VALUE = os.getenv("SOGPT_KEY_VALUE", "")
SGC_CLIENT_ID = os.getenv("SGC_CLIENT_ID", "")
SGC_CLIENT_SECRET = os.getenv("SGC_CLIENT_SECRET", "")
SGC_TOKEN_URL = os.getenv("SGC_TOKEN_URL", "")
MTLS_CERT = os.getenv("MTLS_CERT_PATH")
MTLS_KEY = os.getenv("MTLS_KEY_PATH")
CA_BUNDLE = os.getenv("CA_BUNDLE")

CERT_PAIR = (MTLS_CERT, MTLS_KEY) if (MTLS_CERT and MTLS_KEY) else None

REQUIRED = {
    "SOGPT_HOST": SOGPT_HOST,
    "SOGPT_APP_NAME": APP_NAME,
    "SOGPT_KEY_NAME": SOGPT_KEY_NAME,
    "SOGPT_KEY_VALUE": SOGPT_KEY_VALUE,
    "SGC_CLIENT_ID": SGC_CLIENT_ID,
    "SGC_CLIENT_SECRET": SGC_CLIENT_SECRET,
    "SGC_TOKEN_URL": SGC_TOKEN_URL,
}

# ----------------------
# Utilities
# ----------------------
def safe_json_parse(text: str, fallback=None):
    if text is None:
        return fallback or {}
    try:
        return json.loads(text)
    except Exception:
        # attempt to extract first JSON object
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return fallback or {}
    return fallback or {}

def redact_value(v: Any) -> str:
    if v is None:
        return "NULL"
    s = str(v)
    # simple redaction for numbers and long ids
    s = re.sub(r"\d{4,}", "xxxx", s)
    # redact emails partially
    s = re.sub(r"([^@]{1}).+@(.+)", r"\1***@\2", s)
    return s

# ----------------------
# SOGPT (so you can reuse corporate flow)
# ----------------------
def get_sg_token() -> str:
    data = {"grant_type": "client_credentials", "scope": "mail profile openid api.group-06608.v1"}
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    r = requests.post(
        SGC_TOKEN_URL, data=data, headers=headers,
        auth=(SGC_CLIENT_ID, SGC_CLIENT_SECRET),
        timeout=30, cert=CERT_PAIR, verify=CA_BUNDLE if CA_BUNDLE else True
    )
    r.raise_for_status()
    tok = r.json().get("access_token")
    if not tok:
        raise RuntimeError("SGConnect did not return access_token")
    return tok

def sogpt_headers(tok: str) -> Dict[str,str]:
    return {
        "Content-Type": "application/json",
        "X-Application": APP_NAME,
        "X-Key-Name": SOGPT_KEY_NAME,
        "X-Key-Value": SOGPT_KEY_VALUE,
        "X-SGC-Access-Token": tok,
        "Authorization": f"Bearer {tok}"
    }

def sogpt_complete(messages: List[Dict[str,str]], model: str="gpt-4o", max_new_tokens:int=1200, retries:int=2) -> Tuple[int,str,dict]:
    """
    Calls SoGPT messages completion endpoint. Returns (status_code, text, raw_json)
    Adjust endpoint path according to your SOGPT API version.
    """
    try:
        tok = get_sg_token()
    except Exception as e:
        return 500, f"SGConnect token error: {e}", {}

    url = f"{SOGPT_HOST}/api/v2/messages/completions"
    body = {
        "model": model,
        "temperature": 0.0,
        "max_new_tokens": max_new_tokens,
        "streaming": False,
        "sampling": False,
        "messages": messages,
        "response_format": {"type": "json_object"}
    }
    last_text = ""
    last_js = {}
    last_status = 0
    for _ in range(retries+1):
        try:
            r = requests.post(url, headers=sogpt_headers(tok), data=json.dumps(body),
                              timeout=180, cert=CERT_PAIR, verify=CA_BUNDLE if CA_BUNDLE else True)
            last_status = r.status_code
            try:
                last_js = r.json()
            except Exception:
                last_js = {}
            # try to extract textual completion
            last_text = last_js.get("completion") or ""
            if not last_text and isinstance(last_js.get("choices"), list) and last_js["choices"]:
                last_text = last_js["choices"][0].get("message", {}).get("content", "")
            if last_text:
                break
        except Exception as e:
            last_text = f"Error calling SOGPT: {e}"
        time.sleep(0.5)
    return last_status, last_text, last_js

# ----------------------
# Profiling & sampling
# ----------------------
def representative_10_row_sample(df: pd.DataFrame, column: str) -> List[Dict[str,Any]]:
    n = len(df)
    if n == 0:
        return []
    # If small, return head
    if n <= 50:
        return df.head(10).to_dict(orient="records")
    rows = []
    # head/tail
    rows.extend(df.head(2).to_dict(orient="records"))
    rows.extend(df.tail(2).to_dict(orient="records"))
    # middle
    mid = max(0, (n//2)-1)
    rows.extend(df.iloc[mid:mid+2].to_dict(orient="records"))
    # heavy hitters (distinct values)
    try:
        top_vals = df[column].value_counts(dropna=False).head(3).index.tolist()
    except Exception:
        top_vals = []
    for v in top_vals:
        sub = df[df[column]==v]
        if len(sub):
            rows.append(sub.iloc[0].to_dict())
    # rare examples
    try:
        tail_vals = df[column].value_counts(dropna=False).tail(2).index.tolist()
    except Exception:
        tail_vals = []
    for v in tail_vals:
        sub = df[df[column]==v]
        if len(sub):
            rows.append(sub.iloc[0].to_dict())
    # ensure unique and exactly 10
    out = []
    seen = set()
    for r in rows:
        key = json.dumps(r, sort_keys=True, default=str)
        if key not in seen:
            out.append(r); seen.add(key)
        if len(out) >= 10:
            break
    # fill if <10 with random sample
    if len(out) < 10:
        for _, r in df.sample(10-len(out), random_state=42).to_dict(orient="records"):
            k = json.dumps(r, sort_keys=True, default=str)
            if k not in seen:
                out.append(r); seen.add(k)
            if len(out) >= 10:
                break
    return out

def compute_compact_profile_for_column(s: pd.Series, sample_size:int=1000, top_k:int=100) -> Dict[str,Any]:
    n = int(len(s))
    s_obj = s.copy()
    # basic stats
    null_pct = float(s_obj.isna().mean()*100) if n>0 else 0.0
    as_str = s_obj.astype(str)
    blank_pct = float(((as_str.str.strip()== "") | as_str.str.lower().eq("null")).mean()*100) if n>0 else 0.0

    # sample for distinct estimation
    if n == 0:
        sample = s_obj
    else:
        sample = s_obj.sample(min(n, max(sample_size, top_k*5)), random_state=42)
    distinct_approx = int(sample.nunique(dropna=False)) if len(sample) else 0

    # top_k heavy hitters
    try:
        topk = sample.value_counts(dropna=False).head(top_k)
        topk_list = [{"value": (None if pd.isna(v) else str(v)), "count": int(c)} for v,c in topk.items()]
    except Exception:
        topk_list = []

    # length stats (for strings)
    lengths = None
    try:
        lengths = as_str.dropna().apply(len)
        length_stats = {
            "min": int(lengths.min()) if len(lengths) else None,
            "max": int(lengths.max()) if len(lengths) else None,
            "mean": float(round(lengths.mean(),2)) if len(lengths) else None
        }
    except Exception:
        length_stats = {"min": None, "max": None, "mean": None}

    # basic numeric detection
    numeric_pct = 0.0
    try:
        numeric = pd.to_numeric(s_obj, errors="coerce")
        numeric_pct = float((~numeric.isna()).mean()*100)
    except Exception:
        numeric_pct = 0.0

    # quick pattern detection
    pattern = None
    simple = [str(x) for x in sample.dropna().astype(str).head(200)]
    if simple:
        if sum(1 for x in simple if re.match(r"^[A-Z]{3}$", x)) / len(simple) > 0.6:
            pattern = r"^[A-Z]{3}$"
        elif sum(1 for x in simple if re.match(r"^\d{4}-\d{2}-\d{2}", x)) / len(simple) > 0.6:
            pattern = r"^\d{4}-\d{2}-\d{2}"
        elif numeric_pct > 70:
            pattern = "numeric"

    # add a small reservoir preview
    preview = [None if pd.isna(x) else str(x) for x in sample.head(50).tolist()]

    # anomaly detection quick hint (isolation forest on length or numeric)
    anomaly_hint = {}
    try:
        if numeric_pct < 50 and len(lengths) > 50:
            iso = IsolationForest(n_estimators=50, random_state=42, contamination=0.01)
            arr = lengths.fillna(lengths.mean()).values.reshape(-1,1)
            iso.fit(arr)
            scores = iso.decision_function(arr)
            # show 3 worst
            worst = np.argsort(scores)[:3]
            anomaly_hint["text_length_outlier_examples"] = [simple[i] for i in worst if i < len(simple)]
    except Exception:
        anomaly_hint = {}

    profile = {
        "count": n,
        "null_pct": round(null_pct,3),
        "blank_pct": round(blank_pct,3),
        "distinct_approx": distinct_approx,
        "topk": topk_list,
        "length_stats": length_stats,
        "numeric_pct": round(numeric_pct,2),
        "pattern_hint": pattern,
        "preview_sample": preview[:50],
        "anomaly_hint": anomaly_hint
    }
    return profile

# ----------------------
# Clustering / grouping tokens (fuzzy)
# ----------------------
def cluster_tokens_fuzzy(tokens: List[str], threshold:int=78) -> List[Dict[str,Any]]:
    tokens = [t for t in tokens if t and str(t).strip()]
    clusters: List[List[str]] = []
    for t in tokens:
        placed = False
        for c in clusters:
            # compare to representative
            if fuzz.ratio(t.lower(), c[0].lower()) >= threshold:
                c.append(t)
                placed = True
                break
        if not placed:
            clusters.append([t])
    out = []
    # sort by size
    clusters.sort(key=lambda x: -len(x))
    for c in clusters:
        out.append({"repr": c[0], "members": c, "size": len(c)})
    return out

# small curated reference for currencies (extend in prod)
ISO_CURRENCIES = {
    "USD": ["usd","dollar","us dollar","us$","$"],
    "EUR": ["eur","euro","€"],
    "INR": ["inr","rupee","rs","₹"],
    "GBP": ["gbp","pound","£"]
}
def match_cluster_to_reference(members: List[str], reference: Dict[str,List[str]]=ISO_CURRENCIES) -> Optional[str]:
    lower = {m.lower() for m in members}
    for code, aliases in reference.items():
        aliases_lower = {a.lower() for a in aliases}
        if code.lower() in lower or len(lower & aliases_lower) > 0:
            return code
    return None

# ----------------------
# LLM prompt builder for rule generation
# ----------------------
def build_rule_generation_prompt(dataset_name: str, column: str, sample_rows: List[Dict[str,Any]], profile: Dict[str,Any], clusters: List[Dict[str,Any]], user_hint: str="") -> List[Dict[str,str]]:
    system = "You are a senior Data Quality Engineer. Use evidence to propose actionable DQ rules (no fluff). Return JSON only."
    user = textwrap.dedent(f"""
    DATASET: "{dataset_name}"
    COLUMN: "{column}"
    USER_HINT: "{user_hint}"

    EVIDENCE SUMMARY:
    {json.dumps(profile, indent=2) if profile else "{}"}

    TOP CLUSTERS: {json.dumps(clusters[:20], indent=2)}

    REPRESENTATIVE ROWS (10):
    {json.dumps([{k: redact_value(v) for k,v in r.items()} for r in sample_rows], indent=2)}

    TASK:
    - Generate up to 8 rules across completeness, validity, consistency, accuracy, timeliness.
    - For each rule return:
      * dimension: one of [Completeness, Validity, Consistency, Accuracy, Timeliness]
      * nl_rule: human-readable short rule
      * json_rule: structured machine-friendly rule using types:
           NOT_NULL, VALID_VALUES (valid_values list), REGEX, MAP_VALUES (mappings), NUM_RANGE (min/max), CROSS_FIELD (expr)
      * confidence (0-1)
    - If you recommend mapping cluster members to canonical values, include mappings: [{"cluster_repr":"X","map_to":"CANON","members":[...]}]
    - Avoid enumerating huge lists; if distincts are large, prefer REGEX or suggest reference table.

    OUTPUT (STRICT JSON):
    {{
      "column": "{column}",
      "rules": [
        {{
          "dimension": "Validity",
          "nl_rule": "...",
          "json_rule": {{}},
          "confidence": 0.9
        }}
      ],
      "mappings": [{{"cluster_repr":"...","map_to":"...","members":[...]}}]
    }}
    """)
    return [{"role":"system","content":system},{"role":"user","content":user}]

# ----------------------
# Run rule (json_rule executor) - efficient, vectorized
# ----------------------
def run_rule_on_df(df: pd.DataFrame, column: str, json_rule: Dict[str,Any]) -> Tuple[pd.Series, str]:
    """
    Returns (violation_mask boolean Series, label)
    json_rule examples:
      {"type":"NOT_NULL"}
      {"type":"VALID_VALUES","valid_values":["USD","EUR"]}
      {"type":"REGEX","pattern":"^[A-Z]{3}$"}
      {"type":"MAP_VALUES","mappings":{"DOLLAR":"USD","usd":"USD"}}
      {"type":"NUM_RANGE","min":0,"max":1000}
      {"type":"CROSS_FIELD","expr":"value_date <= settlement_date"}
    """
    col = column
    jr = {k:str(v) if isinstance(v,str) else v for k,v in (json_rule or {}).items()}
    rtype = str(jr.get("type","")).upper()
    if col not in df.columns:
        return pd.Series([True]*len(df), index=df.index), f"MISSING_COL({col})"

    s = df[col]

    def is_blank(ser: pd.Series) -> pd.Series:
        return ser.isna() | ser.astype(str).str.strip().eq("")

    # NOT_NULL
    if rtype == "NOT_NULL":
        mask = is_blank(s)
        return mask, "NOT_NULL"

    # VALID_VALUES
    if rtype in ("VALID_VALUES","IN_SET","IN"):
        allowed = jr.get("valid_values") or jr.get("values") or []
        allowed_set = set([str(x) for x in allowed])
        sv = s.astype(str)
        null_mask = is_blank(s)
        mask = (~sv.isin(allowed_set)) & (~null_mask)
        return mask, f"VALID_VALUES({len(allowed_set)})"

    # MAP_VALUES: mappings applied then considered valid if mapped result is in mapped set
    if rtype == "MAP_VALUES":
        mappings = jr.get("mappings") or {}
        sv = s.astype(object).astype(str)
        mapped = sv.map(lambda v: mappings.get(v, v))
        null_mask = is_blank(s)
        mask = is_blank(mapped) & ~null_mask  # mapping to blank -> violation
        return mask, f"MAP_VALUES(len_map={len(mappings)})"

    # REGEX
    if rtype in ("REGEX","PATTERN"):
        pat = str(jr.get("pattern") or jr.get("regex") or ".*")
        try:
            rx = re.compile(pat)
            sv = s.astype(str)
            null_mask = is_blank(s)
            ok = sv.apply(lambda v: bool(rx.match(v)) if (v is not None and str(v).strip()!="") else False)
            mask = (~ok) & (~null_mask)
            return mask, f"REGEX({pat})"
        except re.error:
            return pd.Series([True]*len(df), index=df.index), "INVALID_REGEX"

    # NUM_RANGE
    if rtype in ("NUM_RANGE","NUM_BETWEEN","RANGE"):
        x = pd.to_numeric(s, errors="coerce")
        lo = jr.get("min", None)
        hi = jr.get("max", None)
        mask = pd.Series(False, index=df.index)
        if lo is not None:
            try:
                mask = mask | (x < float(lo))
            except Exception:
                pass
        if hi is not None:
            try:
                mask = mask | (x > float(hi))
            except Exception:
                pass
        mask = mask & (~x.isna())
        return mask, "NUM_RANGE"

    # LENGTH checks
    if rtype == "LENGTH":
        eq = jr.get("equals", None)
        lo = jr.get("min", None)
        hi = jr.get("max", None)
        ln = s.astype(str).apply(len)
        mask = pd.Series(False, index=df.index)
        if eq is not None:
            mask = ln != int(eq)
        if lo is not None:
            mask = mask | (ln < int(lo))
        if hi is not None:
            mask = mask | (ln > int(hi))
        return mask, "LENGTH"

    # CROSS_FIELD - naive safe eval of simple comparisons using pandas columns
    if rtype == "CROSS_FIELD":
        expr = jr.get("expr") or jr.get("predicate") or ""
        # very limited parsing for <=, >=, <, >
        try:
            # build boolean series evaluating expr with df columns in local namespace
            # avoid eval of arbitrary code by restricting to comparison operators
            allowed_ops = ["<=", ">=", "==", "!=", "<", ">"]
            if not any(op in expr for op in allowed_ops):
                return pd.Series([True]*len(df), index=df.index), "CROSS_FIELD_UNSUPPORTED"
            # replace column names with df["col"] access
            safe_expr = expr
            for c in df.columns:
                safe_expr = re.sub(rf"\b{re.escape(c)}\b", f"df['{c}']", safe_expr)
            mask = ~eval(safe_expr)
            mask = mask.fillna(False)
            return mask.astype(bool), "CROSS_FIELD"
        except Exception:
            return pd.Series([True]*len(df), index=df.index), "CROSS_FIELD_ERROR"

    # fallback - unknown rule => no violations
    return pd.Series([False]*len(df), index=df.index), "UNKNOWN_RULE_TYPE"

# ----------------------
# Chunked validation
# ----------------------
def validate_file_chunked(filelike: io.BytesIO, rules: List[Dict[str,Any]], column_map: Dict[str,str]=None, chunksize:int=50000, preview_limit:int=200):
    # we will read excel into CSV in temp file for chunked reading (Excel cannot be chunked directly)
    tmp_csv = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    try:
        df_full = pd.read_excel(filelike, dtype=str)
        df_full.to_csv(tmp_csv.name, index=False)
    except Exception as e:
        return {"error": f"Failed to convert excel to csv: {e}"}
    total_rows = 0
    rows_with_violations = set()
    anomalies_preview = []
    per_rule_counts = {r.get("id", f"r{i}"):0 for i,r in enumerate(rules)}
    for chunk in pd.read_csv(tmp_csv.name, chunksize=chunksize, dtype=str):
        total_rows += len(chunk)
        for i, r in enumerate(rules):
            rid = r.get("id", f"r{i}")
            col = r.get("column")
            jr = r.get("json_rule") or {}
            try:
                mask, label = run_rule_on_df(chunk, col, jr)
                cnt = int(mask.sum())
            except Exception as e:
                cnt = 0
            per_rule_counts[rid] += cnt
            if cnt > 0 and len(anomalies_preview) < preview_limit:
                # add up to (preview_limit - current) rows
                to_add = chunk[mask].head(preview_limit - len(anomalies_preview)).to_dict(orient="records")
                anomalies_preview.extend(to_add)
            # optional: record row indices globally - omitted for memory
    if total_rows == 0:
        dq_score = 100.0
    else:
        total_violation_rows = sum(per_rule_counts.values())
        dq_score = round(max(0.0, 100.0 * (1.0 - (total_violation_rows / max(1,total_rows)))),2)
    return {
        "rows": total_rows,
        "per_rule_counts": per_rule_counts,
        "anomalies_preview": anomalies_preview,
        "dq_score": dq_score
    }

# ----------------------
# Streamlit UI
# ----------------------
st.set_page_config(page_title="AI Data Lineage & DQ (POC)", layout="wide")
st.title("🧠 AI-assisted Data Quality — Column Rule Builder (POC)")

# Env check
missing = [k for k,v in REQUIRED.items() if not v]
if missing:
    st.error(f"Missing env vars for SOGPT/SGConnect: {missing}. Fill .env and restart.")
    st.stop()

# Session store for rules
if "saved_rules" not in st.session_state:
    st.session_state["saved_rules"] = {}  # dataset -> list of rules

# Left panel: data source selection
st.sidebar.header("Data Source")
src_type = st.sidebar.selectbox("Source Type", ["Upload Excel (file)", "Postgres (connection)"])
uploaded_file = None
engine = None
if src_type == "Upload Excel (file)":
    uploaded_file = st.sidebar.file_uploader("Upload Excel file (.xlsx/.xls)", type=["xlsx","xls"])
else:
    conn_str = st.sidebar.text_input("Postgres connection string (SQLAlchemy)", placeholder="postgresql://user:pwd@host:5432/dbname")
    table_name = st.sidebar.text_input("Table name or SQL query (select ...)", placeholder="public.my_table or SELECT ...")
    if conn_str and table_name:
        try:
            engine = create_engine(conn_str)
            # do not read yet - read on demand
        except Exception as e:
            st.sidebar.error(f"Invalid connection string: {e}")

st.sidebar.markdown("---")
st.sidebar.markdown("Model (SoGPT) options")
model_name = st.sidebar.text_input("Model id", value="gpt-4o")
st.sidebar.markdown("Note: only small compact evidence is sent to the LLM (sample + profile).")

# Main UI: Step 1 - Select column
st.header("Step 1 — Select dataset and column")
dataset_name = st.text_input("Logical dataset name (for saving rules)", value="MyDataset")
col_to_profile = None
df_preview = None

if src_type == "Upload Excel (file)":
    if uploaded_file is None:
        st.info("Upload an Excel file in the left panel to begin.")
        st.stop()
    # read small preview
    try:
        df_preview = pd.read_excel(uploaded_file, nrows=200, dtype=str)
    except Exception as e:
        st.error(f"Failed to read uploaded excel: {e}")
        st.stop()
    st.markdown(f"**File preview ({df_preview.shape[0]} rows × {df_preview.shape[1]} columns)**")
    st.dataframe(df_preview.head(8))
    col_to_profile = st.selectbox("Pick column to generate rules for", df_preview.columns.tolist())
else:
    if engine is None or not table_name:
        st.info("Provide Postgres connection string and table name in the left panel.")
        st.stop()
    # read small preview from DB
    try:
        query = table_name.strip()
        # if looks like table name, convert to SELECT *
        if not query.lower().startswith("select"):
            query = f"SELECT * FROM {table_name} LIMIT 200"
        df_preview = pd.read_sql(query, con=engine)
    except Exception as e:
        st.error(f"Failed to read from Postgres: {e}")
        st.stop()
    st.markdown(f"**DB preview ({df_preview.shape[0]} rows × {df_preview.shape[1]} columns)**")
    st.dataframe(df_preview.head(8))
    col_to_profile = st.selectbox("Pick column to generate rules for", df_preview.columns.tolist())

st.markdown("---")
st.header("Step 2 — Profile column & prepare evidence (local, no LLM yet)")

if st.button("Run local profile & sampling"):
    with st.spinner("Profiling column (this runs locally)..."):
        try:
            # load full small preview for profiling if upload; else sample small data from DB
            if src_type == "Upload Excel (file)":
                # read full into df (careful, for big file we avoid reading entire file just use sampled methods)
                df_full = pd.read_excel(uploaded_file, dtype=str)
            else:
                # pull a larger sample (100k rows could be heavy - we limit to 10000)
                query = table_name.strip()
                if not query.lower().startswith("select"):
                    query = f"SELECT * FROM {table_name} LIMIT 10000"
                else:
                    query = f"SELECT * FROM ({table_name}) as sub LIMIT 10000"
                df_full = pd.read_sql(query, con=engine)
        except Exception as e:
            st.error(f"Failed to load data for profiling: {e}")
            st.stop()

        # compute profile
        col_series = df_full[col_to_profile].astype(object)
        profile = compute_compact_profile_for_column(col_series, sample_size=2000, top_k=200)
        sample_rows = representative_10_row_sample(df_full, col_to_profile)
        # clusters (based on topk)
        top_tokens = [r["value"] for r in profile.get("topk", []) if r.get("value") is not None]
        clusters = cluster_tokens_fuzzy(top_tokens, threshold=78) if top_tokens else []
        # reference matches for clusters
        for c in clusters:
            c["reference_match"] = match_cluster_to_reference(c.get("members", []))
        # store in session
        st.session_state["last_profile"] = {
            "dataset": dataset_name,
            "column": col_to_profile,
            "profile": profile,
            "sample_rows": sample_rows,
            "clusters": clusters
        }
        st.success("Profile complete — evidence prepared.")
        st.json({"profile": profile, "clusters": clusters, "sample_rows_count": len(sample_rows)})

# show last profile if present
if "last_profile" in st.session_state:
    lp = st.session_state["last_profile"]
    st.markdown("**Last profile summary (local)**")
    st.write(f"Dataset: {lp['dataset']}, Column: {lp['column']}")
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Rows (sample used)", lp["profile"]["count"])
        st.metric("Null %", f"{lp['profile']['null_pct']}%")
        st.metric("Distinct approx", lp["profile"]["distinct_approx"])
    with c2:
        st.metric("Numeric %", f"{lp['profile']['numeric_pct']}%")
        st.metric("Pattern hint", lp['profile'].get("pattern_hint"))
        st.write("Top tokens (sample):")
        st.write([t["value"] for t in lp['profile'].get("topk",[])][:20])
    st.write("Representative 10-row sample (redacted):")
    st.dataframe([{k: redact_value(v) for k,v in r.items()} for r in lp["sample_rows"]])

    # show clusters
    st.write("Clusters from top tokens (fuzzy):")
    st.write(lp["clusters"][:20])

st.markdown("---")
st.header("Step 3 — Generate rules with LLM (only compact evidence is sent)")

user_hint = st.text_area("Prompt / hint for rule generation (optional)", placeholder="e.g., prefer ISO currency codes, treat currency aliases as map candidates", height=80)

if st.button("Generate rules via LLM (SoGPT)"):
    if "last_profile" not in st.session_state:
        st.error("Run profiling first.")
    else:
        lp = st.session_state["last_profile"]
        msgs = build_rule_generation_prompt(lp["dataset"], lp["column"], lp["sample_rows"], lp["profile"], lp["clusters"], user_hint=user_hint)
        with st.spinner("Calling SoGPT to generate rules (small payload only)..."):
            status, text, raw = sogpt_complete(msgs, model=model_name, max_new_tokens=1200, retries=2)
        if status >= 400 or not text:
            st.error(f"SOGPT failed ({status}): {text or raw}")
        else:
            parsed = safe_json_parse(text, {"column": lp["column"], "rules": [], "mappings": []})
            # normalize rules: ensure id, column
            rules = []
            for i,r in enumerate(parsed.get("rules", [])):
                rr = dict(r)
                rr.setdefault("id", f"{lp['column']}_R{i+1}")
                rr.setdefault("column", lp["column"])
                rr.setdefault("dimension", rr.get("dimension","Validity"))
                rr.setdefault("json_rule", rr.get("json_rule") or {})
                rr.setdefault("nl_rule", rr.get("nl_rule",""))
                rr.setdefault("confidence", float(rr.get("confidence", 0.5)))
                rules.append(rr)
            st.session_state["generated_rules"] = rules
            st.session_state["generated_mappings"] = parsed.get("mappings", [])
            st.success(f"Received {len(rules)} rules from SoGPT (please review & dry-run).")
            st.json({"rules": rules, "mappings": st.session_state["generated_mappings"]})

# display generated rules (editable)
if "generated_rules" in st.session_state:
    st.subheader("Review & Edit Generated Rules")
    edit_box = st.text_area("Editable JSON rules", value=json.dumps(st.session_state["generated_rules"], indent=2), height=240)
    if st.button("Update rules from JSON editor"):
        try:
            data = json.loads(edit_box)
            st.session_state["generated_rules"] = data
            st.success("Rules updated locally.")
        except Exception as e:
            st.error(f"Invalid JSON: {e}")

    st.write("Generated mappings (LLM suggestions):")
    st.write(st.session_state.get("generated_mappings", []))

st.markdown("---")
st.header("Step 4 — Dry-run rules on a larger local sample (1k) before saving")

dryrun_count = st.number_input("Sample size for dry-run (recommended 1000)", value=1000, min_value=100, max_value=10000, step=100)
if st.button("Run dry-run on local sample"):
    if "generated_rules" not in st.session_state:
        st.error("Generate rules first.")
    else:
        try:
            # load a modest sample from data source (prefer deterministic)
            if src_type == "Upload Excel (file)":
                df_full = pd.read_excel(uploaded_file, dtype=str)
            else:
                # read up to 2000 rows
                q = table_name.strip()
                if not q.lower().startswith("select"):
                    q = f"SELECT * FROM {table_name} LIMIT {min(2000,dryrun_count)}"
                else:
                    q = f"SELECT * FROM ({table_name}) as sub LIMIT {min(2000,dryrun_count)}"
                df_full = pd.read_sql(q, con=engine)
            sample_df = df_full.sample(min(len(df_full), dryrun_count), random_state=42) if len(df_full)>dryrun_count else df_full
            results = []
            for r in st.session_state["generated_rules"]:
                mask,label = run_rule_on_df(sample_df, r["column"], r.get("json_rule", {}))
                cnt = int(mask.sum())
                pct = round(100.0 * cnt / max(1, len(sample_df)), 3)
                results.append({"id": r.get("id"), "nl_rule": r.get("nl_rule"), "violations": cnt, "violation_pct": pct, "label": label})
            st.table(pd.DataFrame(results))
            st.session_state["last_dryrun"] = {"sample_size": len(sample_df), "results": results}
            st.success("Dry-run completed. Inspect violation rates and adjust rules if needed.")
        except Exception as e:
            st.error(f"Dry-run error: {e}")

# Save approved rules
st.markdown("---")
st.header("Step 5 — Save approved rules (persist in session for now)")
if st.button("Save generated rules to dataset"):
    if "generated_rules" not in st.session_state:
        st.error("No generated rules to save.")
    else:
        st.session_state["saved_rules"].setdefault(dataset_name, [])
        st.session_state["saved_rules"][dataset_name].extend(st.session_state["generated_rules"])
        st.success(f"Saved {len(st.session_state['generated_rules'])} rules to dataset {dataset_name}.")
        # clear generated rules
        st.session_state.pop("generated_rules", None)
        st.session_state.pop("generated_mappings", None)

if dataset_name in st.session_state["saved_rules"]:
    st.write(f"Rules saved for dataset: {dataset_name}")
    st.write(st.session_state["saved_rules"][dataset_name])

st.markdown("---")
st.header("Step 6 — Validate full dataset using saved rules (chunked)")

validate_ds = st.text_input("Dataset name to validate (must match saved dataset)", value=dataset_name)
validate_btn = st.button("Run full validation (chunked - may take time)")

if validate_btn:
    if validate_ds not in st.session_state["saved_rules"]:
        st.error("No saved rules for that dataset. Save rules first.")
    else:
        rules_for_ds = st.session_state["saved_rules"][validate_ds]
        with st.spinner("Running validation across full dataset (chunked CSV) ..."):
            try:
                # if upload - re-open uploaded file; else read DB table fully (careful)
                if src_type == "Upload Excel (file)":
                    uploaded_file.seek(0)
                    res = validate_file_chunked(uploaded_file, rules_for_ds, chunksize=20000)
                else:
                    # if DB - we will pull entire table in chunks via SQL LIMIT/OFFSET (naive but ok for POC)
                    # convert table query into csv by fetching in pieces - here we will fetch up to 100k rows for POC
                    qbase = table_name.strip()
                    if not qbase.lower().startswith("select"):
                        qbase = f"SELECT * FROM {table_name}"
                    # fallback: fetch limited batch to avoid memory issues
                    q = f"SELECT * FROM ({qbase}) as sub LIMIT 200000"
                    df_full = pd.read_sql(q, con=engine)
                    # run rules on df_full directly (may be heavy)
                    # We'll use chunk-like processing locally
                    buf = io.BytesIO()
                    df_full.to_excel(buf, index=False)
                    buf.seek(0)
                    res = validate_file_chunked(buf, rules_for_ds, chunksize=50000)
                if "error" in res:
                    st.error(res["error"])
                else:
                    st.subheader("Validation summary")
                    st.write(f"Rows scanned: {res['rows']}")
                    st.write(f"DQ Score: {res['dq_score']}%")
                    st.write("Per-rule violation counts:")
                    st.write(res["per_rule_counts"])
                    st.write("Anomalies (preview):")
                    st.dataframe(res["anomalies_preview"][:200])
            except Exception as e:
                st.error(f"Validation failed: {e}")

st.markdown("---")
st.header("Notes & next steps")
st.markdown("""
- This POC sends only **compact evidence** (10-row sample + compact profile + clusters) to the LLM — *never the full dataset*.
- Rules are stored as structured `json_rule` and executed locally (no LLM calls during full validation).
- For production: persist rules in DB, use SQL pushdown for validation, expand reference tables (currency/country/product), and add audit logs of prompts/samples.
- You can extend `json_rule` with more types as needed.
""")
