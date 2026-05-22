# ============================================================
# PBSL Blue Neon Sales Analytics & Forecasting Dashboard
# Run: streamlit run pbsl_blue_neon_top5_forecast.py
# Files in same folder as this app:
#   ensemble_forecast_dashboard.csv  -> date, ensemble
#   branch_forecast_dashboard.csv    -> date, branch, ensemble
#   mba_rules_dashboard.csv          -> antecedent, consequent, support, confidence, lift, optional branch
# Main sales CSV is uploaded from sidebar.
# ============================================================

import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from statsmodels.tsa.holtwinters import ExponentialSmoothing

st.set_page_config(
    page_title="PBSL Sales Analytics & Forecasting Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- Blue Neon Styling ----------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800;900&display=swap');
*{font-family:'Inter',sans-serif;}
.stApp{
  background:
    radial-gradient(circle at 12% 5%,rgba(35,136,255,.34),transparent 28%),
    radial-gradient(circle at 90% 14%,rgba(33,230,255,.18),transparent 26%),
    linear-gradient(135deg,#020617 0%,#04102a 48%,#020617 100%);
  color:#eaf6ff;
}
.block-container{padding-top:1.1rem;max-width:1600px;}
[data-testid="stSidebar"]{
  background:linear-gradient(180deg,#071a3d 0%,#020617 100%);
  border-right:1px solid rgba(33,230,255,.28);
  box-shadow:10px 0 32px rgba(0,0,0,.35);
}
[data-testid="stSidebar"] *{color:#dbeeff;}
.logo-main{font-size:42px;font-weight:900;color:white;text-shadow:0 0 24px rgba(33,230,255,.65);}
.logo-sub{font-size:12px;color:#9fc7ff;font-weight:900;letter-spacing:2.4px;text-transform:uppercase;line-height:1.6;}
.hero-title{font-size:36px;font-weight:900;color:white;text-shadow:0 0 18px rgba(35,136,255,.45);letter-spacing:-.8px;}
.hero-sub{color:#a9c8ff;font-size:15px;margin-top:4px;}
.neon-badge{display:inline-block;background:rgba(33,230,255,.12);border:1px solid rgba(33,230,255,.52);padding:8px 14px;border-radius:999px;font-weight:900;color:#eaf6ff;box-shadow:0 0 22px rgba(33,230,255,.14);}
.metric-card{
  background:linear-gradient(145deg,rgba(8,38,88,.96),rgba(3,13,34,.98));
  border:1px solid rgba(33,230,255,.38);border-radius:22px;padding:18px;min-height:142px;
  box-shadow:0 18px 44px rgba(0,0,0,.34), inset 0 1px 0 rgba(255,255,255,.06), 0 0 25px rgba(35,136,255,.11);
}
.metric-card:hover{border-color:rgba(33,230,255,.78);box-shadow:0 22px 55px rgba(0,0,0,.45),0 0 30px rgba(33,230,255,.22);transform:translateY(-2px);transition:.2s ease;}
.metric-icon{height:46px;width:62px;display:flex;align-items:center;justify-content:center;border-radius:15px;font-size:13px;font-weight:900;letter-spacing:.7px;background:linear-gradient(135deg,rgba(35,136,255,.40),rgba(33,230,255,.16));border:1px solid rgba(33,230,255,.65);margin-bottom:10px;box-shadow:0 0 18px rgba(33,230,255,.20);}
.metric-title{font-size:12px;color:#9fc7ff;font-weight:900;text-transform:uppercase;letter-spacing:.55px;}
.metric-value{font-size:clamp(18px,1.55vw,25px);font-weight:900;color:white;margin-top:6px;white-space:normal;overflow-wrap:anywhere;line-height:1.15;}
.metric-sub{font-size:12px;color:#28ff8f;margin-top:8px;font-weight:800;}
.chart-card{
  background:linear-gradient(145deg,rgba(7,26,61,.96),rgba(3,12,31,.98));
  border:1px solid rgba(33,230,255,.30);border-radius:22px;padding:22px;margin-bottom:26px;
  box-shadow:0 18px 42px rgba(0,0,0,.32), inset 0 1px 0 rgba(255,255,255,.04);
}
.chart-title{font-size:22px;font-weight:900;color:white;margin-bottom:14px;text-shadow:0 0 14px rgba(35,136,255,.35);}
.small-muted{font-size:12px;color:#a9c8ff;}
.rec-card{display:flex;align-items:center;gap:14px;background:linear-gradient(135deg,rgba(10,38,87,.82),rgba(5,18,45,.92));border:1px solid rgba(33,230,255,.32);border-radius:16px;padding:14px;margin-bottom:12px;box-shadow:0 0 18px rgba(35,136,255,.08);}
.rec-rank{height:36px;width:36px;border-radius:12px;display:flex;align-items:center;justify-content:center;font-weight:900;background:linear-gradient(135deg,#0077ff,#21e6ff);box-shadow:0 0 16px rgba(33,230,255,.30);color:white;}
.rec-name{font-weight:900;color:white;font-size:15px;line-height:1.25;}.rec-meta{font-size:12px;color:#a9c8ff;margin-top:4px;}
.conf-pill{margin-left:auto;background:rgba(40,255,143,.15);border:1px solid rgba(40,255,143,.50);color:#78ffb4;border-radius:12px;padding:7px 10px;font-weight:900;}
.stButton>button{background:linear-gradient(90deg,#0077ff,#21e6ff);color:white;border:0;border-radius:14px;font-weight:900;box-shadow:0 0 18px rgba(33,230,255,.18);}
h1,h2,h3{color:white;}
hr{border-color:rgba(33,230,255,.20);}
</style>
""", unsafe_allow_html=True)

# ---------------- Helpers ----------------
def clean_columns(df):
    df = df.copy()
    df.columns = df.columns.astype(str).str.strip().str.lower().str.replace(" ", "_", regex=False)
    return df

def money(x):
    try:
        x = float(x)
        if pd.isna(x): return "Nu. 0"
    except Exception:
        return "Nu. 0"
    if abs(x) >= 1_000_000: return f"Nu. {x/1_000_000:.2f}M"
    if abs(x) >= 1_000: return f"Nu. {x/1_000:.1f}K"
    return f"Nu. {x:,.0f}"

def number(x):
    try:
        x = float(x)
        if pd.isna(x): return "0"
    except Exception:
        return "0"
    if abs(x) >= 1_000_000: return f"{x/1_000_000:.2f}M"
    if abs(x) >= 1_000: return f"{x/1_000:.2f}K"
    return f"{x:,.0f}"

def make_abc_ved_summary(df):
    """Create ABC-VED product summary for dashboard.
    Uses existing ABC/VED/ABC_VED columns if available; otherwise derives
    ABC from revenue and approximates VED from construction-product keywords.
    """
    if df.empty or "product" not in df.columns or "net_amount" not in df.columns:
        return pd.DataFrame()

    work = df.copy()
    work["product"] = work["product"].astype(str).str.strip()
    work["net_amount"] = pd.to_numeric(work["net_amount"], errors="coerce").fillna(0)
    if "quantity" in work.columns:
        work["quantity"] = pd.to_numeric(work["quantity"], errors="coerce").fillna(0)
    else:
        work["quantity"] = 0

    prod = (
        work.groupby("product", as_index=False)
        .agg(total_sales=("net_amount", "sum"), quantity_sold=("quantity", "sum"), transactions=("invoice_id", "nunique"))
        .sort_values("total_sales", ascending=False)
    )
    if prod.empty or prod["total_sales"].sum() <= 0:
        return pd.DataFrame()

    # ABC from cumulative sales value
    prod["cum_pct"] = prod["total_sales"].cumsum() / prod["total_sales"].sum()
    prod["ABC"] = np.select(
        [prod["cum_pct"] <= 0.80, prod["cum_pct"] <= 0.95],
        ["A", "B"],
        default="C"
    )

    # Prefer existing VED/ABC_VED columns from the uploaded file if available
    lower_cols = {c.lower(): c for c in work.columns}
    ved_col = next((lower_cols[c] for c in ["ved", "ved_class", "ved_category"] if c in lower_cols), None)
    abcved_col = next((lower_cols[c] for c in ["abc_ved", "abc-ved", "abcved"] if c in lower_cols), None)

    if abcved_col:
        existing = work[["product", abcved_col]].dropna().copy()
        existing[abcved_col] = existing[abcved_col].astype(str).str.upper().str.replace("-", "", regex=False).str.strip()
        existing = existing[existing[abcved_col].str.len() >= 2]
        mode_map = existing.groupby("product")[abcved_col].agg(lambda x: x.mode().iat[0] if not x.mode().empty else x.iloc[0])
        prod["ABC_VED"] = prod["product"].map(mode_map)
        prod["VED"] = prod["ABC_VED"].astype(str).str[-1]
        prod["ABC"] = prod["ABC_VED"].astype(str).str[0]
    else:
        if ved_col:
            existing = work[["product", ved_col]].dropna().copy()
            existing[ved_col] = existing[ved_col].astype(str).str.upper().str[0]
            existing = existing[existing[ved_col].isin(["V", "E", "D"])]
            mode_map = existing.groupby("product")[ved_col].agg(lambda x: x.mode().iat[0] if not x.mode().empty else x.iloc[0])
            prod["VED"] = prod["product"].map(mode_map)
        else:
            vital_keywords = r"cement|rebar|tmt|steel|pipe|ms |gi |wire|nail|screw|rod|angle|channel|sheet|plywood|welding|electrode|sand|aggregate"
            essential_keywords = r"paint|primer|brush|fitting|valve|tap|socket|switch|cable|glue|adhesive|sealant|hinge|lock|bolt|nut|washer|cpvc|pvc"
            name = prod["product"].str.lower()
            prod["VED"] = np.select(
                [name.str.contains(vital_keywords, na=False, regex=True), name.str.contains(essential_keywords, na=False, regex=True)],
                ["V", "E"],
                default="D"
            )
        prod["VED"] = prod["VED"].fillna("D")
        prod["ABC_VED"] = prod["ABC"] + prod["VED"]

    prod["Sales Value"] = prod["total_sales"].round(2)
    prod["Quantity Sold"] = prod["quantity_sold"].round(2)
    prod["Transactions"] = prod["transactions"].astype(int)
    return prod

def chart_theme(fig, height=540):
    fig.update_layout(
        template="plotly_dark",
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=25, r=25, t=25, b=55),
        font=dict(color="#dbe7ff", size=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        legend_title_text="",
        hovermode="closest",
    )
    fig.update_xaxes(
        showgrid=False,
        title_font=dict(color="#eaf6ff", size=13),
        tickfont=dict(color="#bcd7ff", size=11),
        automargin=True
    )
    fig.update_yaxes(
        gridcolor="rgba(169,200,255,.13)",
        title_font=dict(color="#eaf6ff", size=13),
        tickfont=dict(color="#bcd7ff", size=11),
        automargin=True
    )
    for tr in fig.data:
        tr.name = "" if tr.name is None else str(tr.name)
        # Hide Plotly's secondary hover label box that was showing undefined,
        # while keeping pointer hover details active.
        if getattr(tr, "hovertemplate", None) in [None, ""]:
            if str(getattr(tr, "type", "")).lower() == "pie":
                tr.hovertemplate = "%{label}<br>Value: Nu. %{value:,.0f}<br>Share: %{percent}<extra></extra>"
            elif str(getattr(tr, "type", "")).lower() == "bar":
                tr.hovertemplate = "%{y}<br>Value: %{x:,.2f}<extra></extra>" if getattr(tr, "orientation", None) == "h" else "%{x}<br>Value: %{y:,.2f}<extra></extra>"
            else:
                tr.hovertemplate = "%{x}<br>Value: %{y:,.2f}<extra></extra>"
    return fig

# Interactive Plotly controls: hover pointer, zoom, pan, reset, select/lasso,
# drawing tools, and chart download as PNG.
CHART_CONFIG = {
    "displaylogo": False,
    "displayModeBar": True,
    "scrollZoom": True,
    "responsive": True,
    "modeBarButtonsToAdd": [
        "drawline",
        "drawopenpath",
        "drawrect",
        "eraseshape"
    ],
    "toImageButtonOptions": {
        "format": "png",
        "filename": "pbsl_chart",
        "height": 720,
        "width": 1280,
        "scale": 2
    }
}

@st.cache_data
def load_data(uploaded_file):
    """Clean the uploaded sales file using the same notebook workflow.

    Returns three datasets:
    1. data          -> cleaned full sales data for KPIs, totals, charts and ABC analysis
    2. data_mba      -> notebook-style cleaned MBA data for product count and recommendations
    3. data_forecast -> notebook-style forecasting data for sales/branch forecasting summaries
    """

    # ============================================================
    # Read uploaded file
    # ============================================================
    df = pd.read_csv(uploaded_file, encoding="latin1", low_memory=False)

    # Dashboard compatibility: standardize column names.
    # The remaining steps follow the notebook cleaning workflow.
    df = clean_columns(df)

    # ============================================================
    # Notebook EDA cleaning: drop unwanted columns
    # ============================================================
    cols_to_drop = [
        "sl_no",
        "entry_date"
    ]

    df = df.drop(columns=cols_to_drop, errors="ignore")
    data = df.copy()

    # ============================================================
    # Required columns check
    # ============================================================
    required = [
        "date",
        "branch",
        "product",
        "quantity",
        "net_amount",
        "invoice_id"
    ]

    missing = [col for col in required if col not in data.columns]

    if missing:
        st.error(f"Sales CSV is missing required columns: {missing}")
        st.stop()

    # ============================================================
    # Notebook EDA cleaning: convert data types
    # ============================================================
    data["date"] = pd.to_datetime(data["date"], errors="coerce")

    numeric_cols = ["selling_price", "quantity", "net_amount"]
    for col in numeric_cols:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")

    text_cols = ["invoice_id", "company", "branch", "customer", "product", "uom"]
    for col in text_cols:
        if col in data.columns:
            data[col] = data[col].astype("string").str.strip()
            data[col] = data[col].replace("", pd.NA)

    # Keep invoice_id and customer as string exactly as in the notebook
    if "invoice_id" in data.columns:
        data["invoice_id"] = data["invoice_id"].astype("string")
    if "customer" in data.columns:
        data["customer"] = data["customer"].astype("string")

    # Convert business descriptors to category exactly as in the notebook
    category_cols = ["company", "branch", "product", "uom"]
    for col in category_cols:
        if col in data.columns:
            data[col] = data[col].astype("category")

    # ============================================================
    # FULL CLEANED SALES DATASET FOR KPI TOTALS
    # ============================================================
    # Important: total sales, quantity, transactions and sales plots must use
    # the full cleaned sales data. We only drop rows missing the columns needed
    # for sales analytics. We do NOT drop rows because optional columns like
    # customer/company/uom are missing, and we do NOT deduplicate invoice-product
    # rows here. This keeps total sales aligned with the notebook total.
    data_sales = data.copy()
    data_sales = data_sales.dropna(subset=["date", "branch", "product", "quantity", "net_amount", "invoice_id"]).copy()

    # Convert categories back to string for Streamlit filters and Plotly
    for col in ["company", "branch", "product", "uom", "invoice_id", "customer"]:
        if col in data_sales.columns:
            data_sales[col] = data_sales[col].astype("string").str.strip()

    # ============================================================
    # Notebook dataset for sales forecasting
    # ============================================================
    forecast_cols = ["branch", "date", "net_amount", "selling_price", "quantity", "product"]
    available_forecast_cols = [col for col in forecast_cols if col in data_sales.columns]

    data_forecast = data_sales[available_forecast_cols].copy()
    data_forecast["weekday"] = data_forecast["date"].dt.day_name()
    data_forecast["month"] = data_forecast["date"].dt.month_name()
    data_forecast["day_of_week"] = data_forecast["date"].dt.dayofweek
    data_forecast["month_num"] = data_forecast["date"].dt.month

    # ============================================================
    # Notebook MBA dataset and product cleaning
    # ============================================================
    required_mba_cols = ["invoice_id", "product", "quantity", "uom", "net_amount", "branch", "date"]
    missing_mba_cols = [col for col in required_mba_cols if col not in data.columns]

    if missing_mba_cols:
        st.error(f"Sales CSV is missing required MBA columns: {missing_mba_cols}")
        st.stop()

    # MBA/recommendation uses notebook-style product cleaning separately.
    data_mba = data_sales[required_mba_cols].copy()

    data_mba["invoice_id"] = data_mba["invoice_id"].astype("string").str.strip()
    data_mba["product"] = data_mba["product"].astype("string").str.strip()
    data_mba["uom"] = data_mba["uom"].astype("string").str.strip()
    data_mba["branch"] = data_mba["branch"].astype("string").str.strip()
    data_mba["date"] = pd.to_datetime(data_mba["date"], errors="coerce")
    data_mba["quantity"] = pd.to_numeric(data_mba["quantity"], errors="coerce")
    data_mba["net_amount"] = pd.to_numeric(data_mba["net_amount"], errors="coerce")

    # Drop rows with missing key fields
    data_mba = data_mba.dropna(subset=["invoice_id", "product"])

    # Basic text cleaning
    data_mba["product"] = data_mba["product"].str.lower().str.strip()

    # Normalize spaces
    data_mba["product"] = data_mba["product"].str.replace(r"\s+", " ", regex=True)

    # Remove text inside brackets first
    data_mba["product"] = data_mba["product"].str.replace(r"\(.*?\)", "", regex=True)

    # Standardize quotes and sizes
    data_mba["product"] = data_mba["product"].str.replace('"', ' inch', regex=False)
    data_mba["product"] = data_mba["product"].str.replace(r"(\d+)mm\b", r"\1 mm", regex=True)
    data_mba["product"] = data_mba["product"].str.replace(r"(\d+)cm\b", r"\1 cm", regex=True)
    data_mba["product"] = data_mba["product"].str.replace(r"(\d+)inch\b", r"\1 inch", regex=True)
    data_mba["product"] = data_mba["product"].str.replace(r"(\d+)\s*in\b", r"\1 inch", regex=True)

    data_mba["product"] = data_mba["product"].str.replace(r"\s*-\s*duplicate item.*$", "", regex=True)
    data_mba["product"] = data_mba["product"].str.replace(r"\s*-\s*damaged.*$", "", regex=True)
    data_mba["product"] = data_mba["product"].str.replace(r"\s*-\s*sample.*$", "", regex=True)

    # Fix common spelling / spacing issues
    fix_dict = {
        "ha mmer": "hammer",
        "ti mmer": "trimmer",
        "fastner": "fastener",
        "co mmercial": "commercial",
        "llyod": "lloyd",
        "adron": "adorn",
        "ply wood": "plywood",
        "screw driver": "screwdriver"
    }

    for wrong, correct in fix_dict.items():
        data_mba["product"] = data_mba["product"].str.replace(wrong, correct, regex=False)

    # General fixes for broken spaces inside words
    data_mba["product"] = data_mba["product"].str.replace(r"\bha\s+mmer\b", "hammer", regex=True)
    data_mba["product"] = data_mba["product"].str.replace(r"\bti\s+mmer\b", "trimmer", regex=True)
    data_mba["product"] = data_mba["product"].str.replace(r"\bco\s+mmercial\b", "commercial", regex=True)

    # Remove unwanted entries
    unwanted_patterns = [
        r"duplicate item",
        r"^nan$",
        r"^none$",
        r"^null$",
        r"^\s*$"
    ]

    for pat in unwanted_patterns:
        data_mba = data_mba[~data_mba["product"].str.contains(pat, case=False, na=False, regex=True)]

    # Final cleanup
    data_mba["product"] = data_mba["product"].str.replace(r"[^\w\s./-]", "", regex=True)
    data_mba["product"] = data_mba["product"].str.replace(r"\s+", " ", regex=True).str.strip()

    # Remove rows that became empty after cleaning
    data_mba = data_mba[data_mba["product"].notna()]
    data_mba = data_mba[data_mba["product"] != ""]

    # Remove duplicate invoice-product rows for binary basket analysis
    data_mba = data_mba.drop_duplicates(subset=["invoice_id", "product"])

    # Convert to category at the end, exactly as in the notebook
    data_mba["product"] = data_mba["product"].astype("category")
    data_mba["uom"] = data_mba["uom"].astype("category")

    # Convert back for Streamlit display/filtering
    data_mba["product"] = data_mba["product"].astype("string")
    data_mba["uom"] = data_mba["uom"].astype("string")

    return data_sales.copy(), data_mba.copy(), data_forecast.copy()

def make_forecast(actual_weekly, steps=4):
    y = actual_weekly.asfreq("W-SUN").fillna(0)
    if len(y) < 8:
        idx = pd.date_range(y.index.max() + pd.Timedelta(days=7), periods=steps, freq="W-SUN") if len(y) else pd.date_range(pd.Timestamp.today(), periods=steps, freq="W-SUN")
        return pd.Series([y.mean() if len(y) else 0] * steps, index=idx, name="forecast")
    try:
        fit = ExponentialSmoothing(y, trend="add", seasonal=None, initialization_method="estimated").fit(optimized=True)
        pred = fit.forecast(steps)
    except Exception:
        pred = pd.Series([y.tail(4).mean()] * steps, index=pd.date_range(y.index.max()+pd.Timedelta(days=7), periods=steps, freq="W-SUN"))
    pred[pred < 0] = 0
    pred.name = "forecast"
    return pred

def read_forecast(path="ensemble_forecast_dashboard.csv", branches=None):
    if not os.path.exists(path):
        return pd.Series(dtype=float), pd.DataFrame(), False
    fc = clean_columns(pd.read_csv(path))
    if "date" not in fc.columns or "ensemble" not in fc.columns:
        return pd.Series(dtype=float), pd.DataFrame(), False
    fc["date"] = pd.to_datetime(fc["date"], errors="coerce")
    fc["ensemble"] = pd.to_numeric(fc["ensemble"], errors="coerce")
    fc = fc.dropna(subset=["date", "ensemble"])
    if branches and "branch" in fc.columns:
        fc = fc[fc["branch"].isin(branches)]
    s = fc.groupby("date")["ensemble"].sum().sort_index()
    s.name = "forecast"
    return s, fc, True


def read_branch_forecast(path="branch_forecast_dashboard.csv", branches=None):
    """Read branch-wise forecast exported from notebook.
    Expected columns: date, branch, ensemble
    """
    if not os.path.exists(path):
        return pd.DataFrame(), False
    fc = clean_columns(pd.read_csv(path))
    required = ["date", "branch", "ensemble"]
    if any(c not in fc.columns for c in required):
        return pd.DataFrame(), False
    fc["date"] = pd.to_datetime(fc["date"], errors="coerce")
    fc["ensemble"] = pd.to_numeric(fc["ensemble"], errors="coerce")
    fc["branch"] = fc["branch"].astype(str).str.strip()
    fc = fc.dropna(subset=required)
    if branches:
        fc = fc[fc["branch"].isin(branches)]
    return fc.sort_values(["branch", "date"]), True


def get_forecast_series(forecast_raw, selected_branch=None, fallback_actual=None, steps=4, branch_forecast_raw=None):
    """Return forecast series either for one branch or for the selected overall scope.
    Uses exported notebook forecast when available; otherwise uses dashboard baseline.
    """
    # For a specific branch, prefer branch_forecast_dashboard.csv
    if selected_branch and selected_branch != "Overall selected branches":
        if branch_forecast_raw is not None and not branch_forecast_raw.empty:
            fc_branch = branch_forecast_raw.copy()
            fc_branch = fc_branch[fc_branch["branch"].astype(str) == str(selected_branch)]
            if not fc_branch.empty and {"date", "ensemble"}.issubset(fc_branch.columns):
                out = fc_branch.groupby("date")["ensemble"].sum().sort_index()
                out.name = "forecast"
                return out, "Direct branch ensemble forecast"

    # For overall view, use ensemble_forecast_dashboard.csv
    if forecast_raw is not None and not forecast_raw.empty:
        fc = forecast_raw.copy()
        if selected_branch and selected_branch != "Overall selected branches" and "branch" in fc.columns:
            fc = fc[fc["branch"].astype(str) == str(selected_branch)]
        if not fc.empty and {"date", "ensemble"}.issubset(fc.columns):
            out = fc.groupby("date")["ensemble"].sum().sort_index()
            out.name = "forecast"
            return out, "Direct overall ensemble forecast"

    if fallback_actual is not None and len(fallback_actual):
        return make_forecast(fallback_actual, steps), "Baseline forecast"
    return pd.Series(dtype=float), "No forecast available"

def load_rules(path="mba_rules_dashboard.csv"):
    if not os.path.exists(path):
        return pd.DataFrame(), False
    rules = clean_columns(pd.read_csv(path))
    rules = rules.rename(columns={
        "antecedents":"antecedent",
        "consequents":"consequent",
        "selected_product":"antecedent",
        "recommended_product":"consequent"
    })
    req = ["antecedent", "consequent", "support", "confidence", "lift"]
    if any(c not in rules.columns for c in req):
        return pd.DataFrame(), False
    keep = req + (["branch"] if "branch" in rules.columns else [])
    rules = rules[keep].copy()
    rules["antecedent"] = rules["antecedent"].astype(str).str.strip()
    rules["consequent"] = rules["consequent"].astype(str).str.strip()
    for c in ["support", "confidence", "lift"]:
        rules[c] = pd.to_numeric(rules[c], errors="coerce")
    return rules.dropna(subset=req), True

def direct_recommendations(rules, selected_product, top_n=6, branch_filter=None):
    if rules.empty or not selected_product:
        return pd.DataFrame()
    df = rules.copy()
    if branch_filter and "branch" in df.columns:
        df = df[df["branch"].isin(branch_filter)]
    linked = df[(df["antecedent"] == selected_product) | (df["consequent"] == selected_product)].copy()
    if linked.empty:
        return linked
    linked["Recommended Product"] = np.where(linked["antecedent"] == selected_product, linked["consequent"], linked["antecedent"])
    linked["Rule"] = linked["antecedent"] + " → " + linked["consequent"]
    return linked.drop_duplicates("Recommended Product").sort_values(["lift", "confidence", "support"], ascending=False).head(top_n)

def live_mba(df, selected_product, top_n=6):
    if df.empty or not selected_product:
        return pd.DataFrame()
    baskets = df.groupby("invoice_id")["product"].apply(lambda x: set(x.dropna().astype(str))).tolist()
    total = len(baskets)
    selected_baskets = [b for b in baskets if selected_product in b]
    selected_count = len(selected_baskets)
    if total == 0 or selected_count == 0:
        return pd.DataFrame()
    product_counts, pair_counts = {}, {}
    for b in baskets:
        for item in b:
            product_counts[item] = product_counts.get(item, 0) + 1
        if selected_product in b:
            for item in b:
                if item != selected_product:
                    pair_counts[item] = pair_counts.get(item, 0) + 1
    rows=[]
    for item, both in pair_counts.items():
        support = both / total
        confidence = both / selected_count
        cons_support = product_counts.get(item, 0) / total
        lift = confidence / cons_support if cons_support else 0
        rows.append({"Recommended Product": item, "support": support, "confidence": confidence, "lift": lift, "Rule": f"{selected_product} → {item}"})
    return pd.DataFrame(rows).sort_values(["lift", "confidence", "support"], ascending=False).head(top_n) if rows else pd.DataFrame()

def popular_fallback(df, selected_product, top_n=6):
    pop = (
        df[df["product"].astype(str) != str(selected_product)]
        .groupby("product")
        .agg(revenue=("net_amount", "sum"), invoices=("invoice_id", "nunique"))
        .reset_index()
        .sort_values(["revenue", "invoices"], ascending=False)
        .head(top_n)
    )
    if pop.empty:
        return pd.DataFrame()
    total_inv = max(df["invoice_id"].nunique(), 1)
    selected_inv = max(df[df["product"].astype(str) == str(selected_product)]["invoice_id"].nunique(), 1)
    return pd.DataFrame({
        "Recommended Product": pop["product"],
        "support": pop["invoices"] / total_inv,
        "confidence": np.minimum(pop["invoices"] / selected_inv, 1),
        "lift": 1.0,
        "Rule": f"{selected_product} → top-selling fallback"
    })

def section_start(title):
    st.markdown(f"<div class='chart-card'><div class='chart-title'>{title}</div>", unsafe_allow_html=True)

def section_end():
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- Sidebar ----------------
st.sidebar.markdown("""
<div class='logo-main'>PBSL</div>
<div class='logo-sub'>Sales Analytics &<br>Forecasting Dashboard</div>
<hr>
""", unsafe_allow_html=True)

page = st.sidebar.radio(
    "Navigation",
    ["Overview", "Sales Analytics", "Forecasting", "Product Recommendations", "Branches & Products"],
    index=0
)
uploaded = st.sidebar.file_uploader("Upload Sales CSV", type=["csv"])
st.sidebar.caption("Keep forecast and MBA rule CSV files in the same folder as this app.")

if uploaded is None:
    st.markdown("<div class='hero-title'>PBSL Sales Analytics & Forecasting Dashboard</div>", unsafe_allow_html=True)
    st.markdown("<div class='hero-sub'>Upload your Sales CSV from the sidebar to open the dashboard.</div>", unsafe_allow_html=True)
    st.info("Required columns: date, branch, product, quantity, net_amount, invoice_id")
    st.stop()

data, data_mba, data_forecast = load_data(uploaded)

# Historical actual sales coverage only. This prevents future forecast dates from being displayed as actual data coverage.
actual_data = data.copy()
if "type" in actual_data.columns:
    actual_data = actual_data[actual_data["type"].astype(str).str.lower().str.strip() != "forecast"]
# Project historical sales cutoff
historical_cutoff = pd.Timestamp("2025-11-30")
actual_data = actual_data[actual_data["date"] <= historical_cutoff]
if actual_data.empty:
    actual_data = data.copy()
min_date, max_date = actual_data["date"].min().date(), actual_data["date"].max().date()
date_range = st.sidebar.date_input("Date Range", [min_date, max_date], min_value=min_date, max_value=max_date)
branches = sorted(data["branch"].dropna().unique())
selected_branches = st.sidebar.multiselect("Branch", branches, default=branches)
products = sorted(data["product"].dropna().unique())
selected_products = st.sidebar.multiselect("Product Filter", products, default=[])

if len(date_range) == 2:
    start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
else:
    start_date, end_date = data["date"].min(), data["date"].max()

filtered = data[(data["date"] >= start_date) & (data["date"] <= end_date)]
filtered = filtered[filtered["branch"].isin(selected_branches)]
if selected_products:
    filtered = filtered[filtered["product"].isin(selected_products)]

if filtered.empty:
    st.warning("No data available for the selected filter. Please adjust the sidebar filters.")
    st.stop()

# Notebook-style MBA data filtered using the same date and branch selections.
filtered_mba = data_mba[(data_mba["date"] >= start_date) & (data_mba["date"] <= end_date)].copy()
filtered_mba = filtered_mba[filtered_mba["branch"].isin(selected_branches)]
if selected_products:
    selected_products_cleaned = [str(p).lower().strip() for p in selected_products]
    filtered_mba = filtered_mba[filtered_mba["product"].astype(str).isin(selected_products_cleaned)]

# ---------------- Common calculations ----------------
total_sales = filtered["net_amount"].sum()
total_qty = filtered["quantity"].sum()
total_transactions = filtered["invoice_id"].nunique()
avg_order = filtered.groupby("invoice_id")["net_amount"].sum().mean() if total_transactions else 0
total_customers = filtered["customer"].nunique() if "customer" in filtered.columns else total_transactions
# Notebook-style unique product count from MBA-cleaned product names
# Unique products should follow the notebook MBA-cleaned product list.
total_unique_products = (
    data_mba["product"].astype(str).str.strip().nunique()
    if "product" in data_mba.columns
    else data["product"].astype(str).str.strip().nunique()
)
branch_sales = filtered.groupby("branch")["net_amount"].sum().sort_values(ascending=False)
best_branch = branch_sales.index[0] if len(branch_sales) else "N/A"
best_value = branch_sales.iloc[0] if len(branch_sales) else 0
weekly_actual = filtered.groupby(pd.Grouper(key="date", freq="W-SUN"))["net_amount"].sum().sort_index()
top_5_branches = branch_sales.head(5).index.tolist()
forecast, forecast_raw, direct_forecast_found = read_forecast()
forecast_label = "Overall sales forecast" if direct_forecast_found and len(forecast) else "Baseline forecast"
if len(forecast) == 0:
    forecast = make_forecast(weekly_actual, 4)
forecast_total = forecast.sum() if len(forecast) else 0
branch_forecast_df, branch_forecast_found = read_branch_forecast(branches=top_5_branches)

# ---------------- Header and KPI Cards ----------------
left, right = st.columns([2.3, 1])
with left:
    st.markdown(f"<div class='neon-badge'>{page}</div>", unsafe_allow_html=True)
    st.markdown("<div class='hero-title'>PBSL Sales Analytics Dashboard</div>", unsafe_allow_html=True)
    st.markdown("<div class='hero-sub'>Sales performance, forecasting, branch insights, ABC analysis and product recommendation</div>", unsafe_allow_html=True)
with right:
    st.markdown(f"""
    <div class='chart-card'>
      <b>Selected Period</b><br><span class='small-muted'>{start_date.date()} to {end_date.date()}</span><br><br>
      <b>Branch Filter</b><br><span class='small-muted'>{len(selected_branches)} branch(es) selected</span>
    </div>
    """, unsafe_allow_html=True)

cols = st.columns(6)
metrics = [
    ("💎", "Total Sales", money(total_sales), "Revenue generated"),
    ("📦", "Quantity Sold", number(total_qty), "Products sold"),
    ("🧾", "Transactions", number(total_transactions), "Invoices processed"),
    ("🛒", "Average Basket", money(avg_order), "Per invoice"),
    ("🧱", "Unique Products", number(total_unique_products), "Notebook MBA cleaned"),
    ("🚀", "Forecast Sales", money(forecast_total), forecast_label),
]
for col, (icon, title, value, sub) in zip(cols, metrics):
    col.markdown(f"""
    <div class='metric-card'>
      <div class='metric-icon'>{icon}</div>
      <div class='metric-title'>{title}</div>
      <div class='metric-value'>{value}</div>
      <div class='metric-sub'>{sub}</div>
    </div>
    """, unsafe_allow_html=True)

# ---------------- Pages ----------------
if page == "Overview":
    c1, c2 = st.columns([1.8, 1], gap="large")
    with c1:
        section_start("Sales Trend with Forecast")

        trend_options = ["Overall selected branches"] + top_5_branches
        trend_choice = st.selectbox(
            "View sales trend",
            trend_options,
            index=0,
            key="overview_trend_choice",
            help="Overall shows total actual sales with total forecast. Branch options show top-5 branch forecast."
        )

        if trend_choice == "Overall selected branches":
            actual_for_view = weekly_actual
            trend_forecast, trend_forecast_label = get_forecast_series(
                forecast_raw,
                selected_branch="Overall selected branches",
                fallback_actual=weekly_actual,
                steps=4,
                branch_forecast_raw=branch_forecast_df
            )
            actual_name = "Overall Actual Sales"
            forecast_name = "Overall Sales Forecast"
        else:
            actual_for_view = (
                filtered[filtered["branch"] == trend_choice]
                .groupby(pd.Grouper(key="date", freq="W-SUN"))["net_amount"]
                .sum()
                .sort_index()
            )
            trend_forecast, trend_forecast_label = get_forecast_series(
                forecast_raw,
                selected_branch=trend_choice,
                fallback_actual=actual_for_view,
                steps=4,
                branch_forecast_raw=branch_forecast_df
            )
            actual_name = f"Actual — {trend_choice}"
            forecast_name = f"Forecast — {trend_choice}"

        fig = go.Figure()
        if len(actual_for_view):
            fig.add_trace(go.Scatter(
                x=actual_for_view.index,
                y=actual_for_view.values,
                mode="lines+markers",
                name=actual_name,
                line=dict(width=4, color="#2f80ff"),
                marker=dict(size=8),
                fill="tozeroy",
                fillcolor="rgba(47,128,255,.16)",
                hovertemplate="Date: %{x|%b %d, %Y}<br>Actual Sales: Nu. %{y:,.0f}<extra></extra>"
            ))
        if len(trend_forecast):
            fig.add_trace(go.Scatter(
                x=trend_forecast.index,
                y=trend_forecast.values,
                mode="lines+markers",
                name=forecast_name,
                line=dict(width=4, dash="dash", color="#21e6ff"),
                marker=dict(size=8),
                hovertemplate="Date: %{x|%b %d, %Y}<br>Forecast Sales: Nu. %{y:,.0f}<extra></extra>"
            ))
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Sales / Forecast (Nu.)",
            hovermode="x unified"
        )
        fig.update_yaxes(tickprefix="Nu. ")
        st.caption(f"Forecast source: {trend_forecast_label}")
        st.plotly_chart(chart_theme(fig, 590), use_container_width=True, config=CHART_CONFIG)
        section_end()
    with c2:
        section_start("Sales by Top 5 Branches")
        pie_df = branch_sales.head(5).reset_index()
        pie_df.columns = ["branch", "sales"]
        fig = go.Figure(data=[go.Pie(
            labels=pie_df["branch"], values=pie_df["sales"], hole=.55,
            textinfo="label+percent", sort=False,
            marker=dict(line=dict(color="#06122f", width=2)),
            hovertemplate="%{label}<br>Sales: Nu. %{value:,.0f}<br>Share: %{percent}<extra></extra>"
        )])
        fig.add_annotation(text=f"TOP 5<br><b>{money(pie_df['sales'].sum()).replace('Nu. ', '')}</b>", x=.5, y=.5, showarrow=False, font=dict(size=18, color="white"))
        st.plotly_chart(chart_theme(fig, 590), use_container_width=True, config=CHART_CONFIG)
        section_end()

elif page == "Sales Analytics":
    section_start("Seasonality of Sales Across Weekdays")
    # Exclude Sunday because PBSL weekday seasonality is shown for working/business days only
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
    weekday = (
        filtered.assign(weekday=filtered["date"].dt.day_name())
        .query("weekday != 'Sunday'")
        .groupby("weekday")["net_amount"].sum()
        .reindex(order)
        .dropna()
        .reset_index()
    )
    weekday["sales_million"] = weekday["net_amount"] / 1_000_000
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=weekday["weekday"], y=weekday["sales_million"], mode="lines+markers",
        name="Weekday Sales", line=dict(width=4, color="#21e6ff"),
        marker=dict(size=11, color="#2f80ff")
    ))
    fig.update_layout(xaxis_title="Weekday", yaxis_title="Sales (Million Nu.)")
    st.plotly_chart(chart_theme(fig, 520), use_container_width=True, config=CHART_CONFIG)
    section_end()

    c1, c2 = st.columns(2, gap="large")
    with c1:
        section_start("Top 5 Branches Contribution")
        top5 = branch_sales.head(5).sort_values().reset_index()
        top5.columns = ["branch", "sales"]
        top5["sales_million"] = top5["sales"] / 1_000_000
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=top5["sales_million"], y=top5["branch"], orientation="h",
            text=[f"{v:.2f}M" for v in top5["sales_million"]], textposition="inside",
            marker=dict(color=top5["sales_million"], colorscale="Blues", line=dict(color="#21e6ff", width=1)),
            name="Branch Revenue"
        ))
        fig.update_layout(xaxis_title="Total Sales (Million Nu.)", yaxis_title="Branch")
        st.plotly_chart(chart_theme(fig, 540), use_container_width=True, config=CHART_CONFIG)
        section_end()
    with c2:
        section_start("Top 10 Products by Revenue")
        top_prod = filtered.groupby("product")["net_amount"].sum().sort_values(ascending=False).head(10).sort_values().reset_index()
        top_prod["revenue_million"] = top_prod["net_amount"] / 1_000_000
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=top_prod["revenue_million"], y=top_prod["product"], orientation="h",
            text=[f"{v:.2f}M" for v in top_prod["revenue_million"]], textposition="inside",
            marker=dict(color=top_prod["revenue_million"], colorscale="Blues", line=dict(color="#21e6ff", width=1)),
            name="Product Revenue"
        ))
        fig.update_layout(xaxis_title="Revenue (Million Nu.)", yaxis_title="Product")
        fig.update_yaxes(tickfont=dict(size=10))
        st.plotly_chart(chart_theme(fig, 600), use_container_width=True, config=CHART_CONFIG)
        section_end()

elif page == "Forecasting":
    section_start("Final Sales Forecast")

    top5_branch_options = branch_sales.head(5).index.tolist()
    forecast_view_options = ["Overall selected branches"] + top5_branch_options
    forecast_view = st.selectbox(
        "Forecast view",
        forecast_view_options,
        index=0,
        key="forecast_view_selector"
    )

    if forecast_view == "Overall selected branches":
        actual_for_view = weekly_actual
    else:
        actual_for_view = (
            filtered[filtered["branch"] == forecast_view]
            .groupby(pd.Grouper(key="date", freq="W-SUN"))["net_amount"]
            .sum()
            .sort_index()
        )

    view_forecast, view_forecast_label = get_forecast_series(
        forecast_raw,
        selected_branch=forecast_view,
        fallback_actual=actual_for_view,
        steps=4,
        branch_forecast_raw=branch_forecast_df
    )

    fig = go.Figure()
    if len(actual_for_view):
        fig.add_trace(go.Scatter(
            x=actual_for_view.index,
            y=actual_for_view.values,
            mode="lines+markers",
            name="Historical Weekly Sales",
            line=dict(color="#2f80ff", width=4),
            marker=dict(size=8),
            hovertemplate="Date: %{x|%b %d, %Y}<br>Actual Sales: Nu. %{y:,.0f}<extra></extra>"
        ))
    if len(view_forecast):
        fig.add_trace(go.Scatter(
            x=view_forecast.index,
            y=view_forecast.values,
            mode="lines+markers",
            name="Forecast",
            line=dict(color="#21e6ff", width=4, dash="dash"),
            marker=dict(size=9),
            hovertemplate="Date: %{x|%b %d, %Y}<br>Forecast Sales: Nu. %{y:,.0f}<extra></extra>"
        ))
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Sales / Forecast (Nu.)",
        hovermode="x unified"
    )
    fig.update_yaxes(tickprefix="Nu. ")
    st.caption(f"Forecast source: {view_forecast_label}")
    st.plotly_chart(chart_theme(fig, 620), use_container_width=True, config=CHART_CONFIG)

    if len(view_forecast):
        display = view_forecast.reset_index()
        display.columns = ["Date", "Forecast Sales"]
        display["Forecast Sales"] = display["Forecast Sales"].round(0).astype(int)
        if forecast_view != "Overall selected branches":
            display.insert(1, "Branch", forecast_view)
        st.dataframe(display, hide_index=True, use_container_width=True)
    section_end()

elif page == "Product Recommendations":
    section_start("Frequently Purchased Together — Product Recommendation")
    rules, rules_found = load_rules()
    available = sorted(filtered_mba["product"].dropna().astype(str).unique())
    if not available:
        st.warning("No products available after notebook-style MBA cleaning for the selected filters.")
        section_end()
        st.stop()
    c1, c2 = st.columns([3, 1])
    with c1:
        selected_prod = st.selectbox("Select product", available)
    with c2:
        top_n = st.slider("Top N", 2, 10, 5)

    rec = direct_recommendations(rules, selected_prod, top_n, selected_branches) if rules_found else pd.DataFrame()
    source = "precomputed FP-Growth rules"
    if rec.empty:
        rec = live_mba(filtered_mba, selected_prod, top_n)
        source = "live MBA from uploaded sales data"
    if rec.empty:
        rec = popular_fallback(filtered_mba, selected_prod, top_n)
        source = "top-selling fallback because no co-purchase pair exists"
    st.caption(f"Recommendation source: {source}")

    if rec.empty:
        st.warning("No recommendation can be created because the filtered data is too small.")
    else:
        for i, row in rec.reset_index(drop=True).iterrows():
            st.markdown(f"""
            <div class='rec-card'>
              <div class='rec-rank'>{i+1}</div>
              <div>
                <div class='rec-name'>{row['Recommended Product']}</div>
                <div class='rec-meta'>{row.get('Rule','')} | Support: {row['support']:.3f} | Lift: {row['lift']:.2f}</div>
              </div>
              <div class='conf-pill'>{row['confidence']:.0%}</div>
            </div>
            """, unsafe_allow_html=True)

        # Simple recommendation bar chart, cleaner than congested network graph
        rec_plot = rec.reset_index(drop=True).copy()
        rec_plot["confidence_pct"] = rec_plot["confidence"] * 100
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=rec_plot["confidence_pct"],
            y=rec_plot["Recommended Product"],
            orientation="h",
            text=[f"{v:.0f}%" for v in rec_plot["confidence_pct"]],
            textposition="inside",
            marker=dict(color=rec_plot["confidence_pct"], colorscale="Blues", line=dict(color="#21e6ff", width=1)),
            name="Confidence"
        ))
        fig.update_layout(xaxis_title="Confidence (%)", yaxis_title="Recommended Product")
        st.plotly_chart(chart_theme(fig, 500), use_container_width=True, config=CHART_CONFIG)
    section_end()

elif page == "Branches & Products":
    c1, c2 = st.columns(2, gap="large")
    with c1:
        section_start("Branch Performance Ranking")
        rank = branch_sales.reset_index()
        rank.columns = ["Branch", "Sales"]
        for i, r in rank.head(10).iterrows():
            st.markdown(f"""
            <div class='rec-card'>
              <div class='rec-rank'>{i+1}</div>
              <div><div class='rec-name'>{r['Branch']}</div><div class='rec-meta'>{money(r['Sales'])}</div></div>
            </div>
            """, unsafe_allow_html=True)
        section_end()
    with c2:
        section_start("ABC Revenue Contribution")

        abc = (
            filtered
            .groupby("product", as_index=False)["net_amount"]
            .sum()
            .sort_values("net_amount", ascending=False)
        )

        if not abc.empty and abc["net_amount"].sum() > 0:
            # ABC classification based on cumulative revenue contribution
            abc["cum_pct"] = abc["net_amount"].cumsum() / abc["net_amount"].sum()
            abc["class"] = np.select(
                [abc["cum_pct"] <= 0.80, abc["cum_pct"] <= 0.95],
                ["A", "B"],
                default="C"
            )

            # Summary for donut chart
            summ = (
                abc
                .groupby("class", as_index=False)["net_amount"]
                .sum()
                .sort_values("class")
            )

            class_color_map = {"A": "#2f80ff", "B": "#23e46d", "C": "#ffa31a"}
            chart_colors = [class_color_map.get(c, "#21e6ff") for c in summ["class"]]

            fig = go.Figure(data=[go.Pie(
                labels=summ["class"],
                values=summ["net_amount"],
                hole=.55,
                marker=dict(
                    colors=chart_colors,
                    line=dict(color="#06122f", width=2)
                ),
                textinfo="label+percent",
                hovertemplate="<b>Class %{label}</b><br>Revenue: Nu. %{value:,.0f}<br>Share: %{percent}<extra></extra>"
            )])

            fig.add_annotation(
                text="ABC<br><b>CLASS</b>",
                x=.5,
                y=.5,
                showarrow=False,
                font=dict(size=16, color="white")
            )

            st.plotly_chart(
                chart_theme(fig, 430),
                use_container_width=True,
                config=CHART_CONFIG
            )

            # Interactive class selector
            selected_class = st.radio(
                "Select ABC Class to View Products",
                ["A", "B", "C"],
                horizontal=True,
                key="abc_class_selector"
            )

            class_products = abc[abc["class"] == selected_class].copy()
            class_products["Revenue (Nu.)"] = class_products["net_amount"].round(2)
            class_products["Cumulative %"] = (class_products["cum_pct"] * 100).round(2)

            st.markdown(
                f"<div class='small-muted'>Showing <b>{len(class_products):,}</b> products in Class {selected_class}</div>",
                unsafe_allow_html=True
            )

            st.dataframe(
                class_products[["product", "Revenue (Nu.)", "Cumulative %", "class"]],
                hide_index=True,
                use_container_width=True
            )
        else:
            st.info("ABC analysis is not available for the selected filters.")

        section_end()


    section_start("ABC-VED Product Classification")

    abc_ved_df = make_abc_ved_summary(filtered)

    if abc_ved_df.empty:
        st.info("ABC-VED analysis is not available for the selected filters.")
    else:
        abc_ved_counts = abc_ved_df["ABC_VED"].value_counts().reset_index()
        abc_ved_counts.columns = ["ABC_VED", "count"]
        abc_ved_counts = abc_ved_counts.sort_values("ABC_VED")

        custom_colors = [
            "#264653", "#2A9D8F", "#E9C46A", "#F4A261", "#E76F51",
            "#8AB17D", "#6D597A", "#B56576", "#457B9D"
        ]

        fig = go.Figure(data=[go.Pie(
            labels=abc_ved_counts["ABC_VED"],
            values=abc_ved_counts["count"],
            hole=0.55,
            textinfo="label+percent",
            sort=False,
            marker=dict(colors=custom_colors[:len(abc_ved_counts)], line=dict(color="white", width=2)),
            hovertemplate="<b>%{label}</b><br>Products: %{value:,}<br>Share: %{percent}<extra></extra>"
        )])
        fig.add_annotation(
            text="ABC-VED<br><b>PRODUCTS</b>",
            x=.5, y=.5, showarrow=False,
            font=dict(size=17, color="white")
        )
        fig.update_layout(title=dict(text="Distribution of Products Across ABC-VED Categories", x=0.5))
        st.plotly_chart(chart_theme(fig, 520), use_container_width=True, config=CHART_CONFIG)

        selected_abc_ved = st.selectbox(
            "Select ABC-VED class to view products",
            sorted(abc_ved_df["ABC_VED"].dropna().unique()),
            key="abc_ved_class_selector"
        )

        class_products = (
            abc_ved_df[abc_ved_df["ABC_VED"] == selected_abc_ved]
            .sort_values("total_sales", ascending=False)
            .copy()
        )

        st.markdown(
            f"<div class='small-muted'>Showing <b>{len(class_products):,}</b> products under class <b>{selected_abc_ved}</b></div>",
            unsafe_allow_html=True
        )

        st.dataframe(
            class_products[["product", "ABC", "VED", "ABC_VED", "Sales Value", "Quantity Sold", "Transactions"]],
            hide_index=True,
            use_container_width=True
        )

    section_end()

# ---------------- Footer ----------------
f1, f2, f3, f4 = st.columns(4)
footer = [
    ("Best Performing Branch", best_branch, money(best_value)),
    ("Forecast Source", forecast_label, money(forecast_total)),
    ("Total Unique Products", number(total_unique_products), "Notebook MBA cleaned products"),
    ("Data Last Updated", pd.Timestamp.now().strftime("%b %d, %Y"), "Live dashboard sync"),
]
for col, (title, main, sub) in zip([f1, f2, f3, f4], footer):
    col.markdown(f"""
    <div class='chart-card'>
      <b>{title}</b><br>
      <span style='font-size:18px;font-weight:850;color:white'>{main}</span><br>
      <span class='small-muted'>{sub}</span>
    </div>
    """, unsafe_allow_html=True)
