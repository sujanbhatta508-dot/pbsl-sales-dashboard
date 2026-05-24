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
import hashlib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    XGBRegressor = None
from statsmodels.tsa.arima.model import ARIMA


if not XGBOOST_AVAILABLE:
    print("XGBoost not installed — dashboard will use Random Forest + ARIMA ensemble.")

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

.overview-hero{
  background:
    linear-gradient(135deg,rgba(0,119,255,.28),rgba(33,230,255,.10)),
    radial-gradient(circle at 15% 20%,rgba(33,230,255,.22),transparent 28%),
    radial-gradient(circle at 90% 10%,rgba(40,255,143,.12),transparent 24%);
  border:1px solid rgba(33,230,255,.42);
  border-radius:28px;
  padding:26px;
  margin-bottom:24px;
  box-shadow:0 24px 60px rgba(0,0,0,.36),0 0 36px rgba(33,230,255,.12);
}
.overview-hero-title{
  font-size:38px;
  font-weight:950;
  color:white;
  letter-spacing:-1px;
  text-shadow:0 0 20px rgba(33,230,255,.38);
}
.overview-hero-sub{
  margin-top:8px;
  color:#b8d8ff;
  font-size:15px;
  line-height:1.55;
}
.kpi-grid-card{
  background:linear-gradient(145deg,rgba(7,30,70,.96),rgba(2,10,28,.98));
  border:1px solid rgba(33,230,255,.36);
  border-radius:22px;
  padding:18px;
  min-height:150px;
  box-shadow:0 18px 42px rgba(0,0,0,.34), inset 0 1px 0 rgba(255,255,255,.06);
  position:relative;
  overflow:hidden;
}
.kpi-grid-card:before{
  content:"";
  position:absolute;
  right:-34px;
  top:-34px;
  width:92px;
  height:92px;
  border-radius:999px;
  background:rgba(33,230,255,.12);
}
.kpi-grid-card:hover{
  transform:translateY(-3px);
  transition:.22s ease;
  border-color:rgba(33,230,255,.80);
  box-shadow:0 22px 56px rgba(0,0,0,.45),0 0 30px rgba(33,230,255,.18);
}
.kpi-icon{
  height:44px;
  width:44px;
  border-radius:15px;
  display:flex;
  align-items:center;
  justify-content:center;
  font-size:22px;
  background:linear-gradient(135deg,#0077ff,#21e6ff);
  box-shadow:0 0 22px rgba(33,230,255,.30);
  margin-bottom:13px;
}
.kpi-label{
  font-size:11px;
  color:#9fc7ff;
  text-transform:uppercase;
  letter-spacing:.9px;
  font-weight:900;
}
.kpi-number{
  font-size:clamp(20px,1.7vw,30px);
  color:white;
  font-weight:950;
  margin-top:8px;
  line-height:1.08;
  overflow-wrap:anywhere;
}
.highlight-card{
  background:
    linear-gradient(145deg,rgba(8,38,88,.98),rgba(3,13,34,.98));
  border:1px solid rgba(40,255,143,.35);
  border-radius:24px;
  padding:24px;
  min-height:180px;
  text-align:center;
  box-shadow:0 20px 50px rgba(0,0,0,.34),0 0 28px rgba(40,255,143,.08);
}
.highlight-label{
  font-size:12px;
  color:#9fc7ff;
  font-weight:900;
  letter-spacing:1px;
  text-transform:uppercase;
}
.highlight-main{
  font-size:clamp(26px,2.2vw,38px);
  font-weight:950;
  color:white;
  margin-top:18px;
  text-shadow:0 0 16px rgba(33,230,255,.22);
  overflow-wrap:anywhere;
}
.highlight-sub{
  font-size:18px;
  color:#21e6ff;
  font-weight:900;
  margin-top:10px;
}


.info-strip{
  display:grid;
  grid-template-columns:repeat(2,1fr);
  gap:20px;
  margin-top:4px;
  margin-bottom:24px;
}
.info-tile{
  background:linear-gradient(135deg,rgba(8,38,88,.90),rgba(3,13,34,.98));
  border:1px solid rgba(33,230,255,.34);
  border-radius:22px;
  padding:22px;
  box-shadow:0 16px 40px rgba(0,0,0,.30);
}
.info-tile-label{
  color:#9fc7ff;
  font-size:12px;
  font-weight:900;
  letter-spacing:.9px;
  text-transform:uppercase;
}
.info-tile-value{
  color:white;
  font-size:24px;
  font-weight:950;
  margin-top:10px;
  line-height:1.25;
}

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

def load_data(uploaded_file):
    """Clean the uploaded sales file using the same notebook workflow.

    Returns three datasets:
    1. data          -> notebook-cleaned full sales data for KPIs, charts and ABC analysis
    2. data_mba      -> notebook-style cleaned MBA data for product count and recommendations
    3. data_forecast -> notebook-style forecasting data for sales/branch forecasting summaries
    """

    # ============================================================
    # Read uploaded file freshly every time
    # ============================================================
    uploaded_file.seek(0)
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
    # Robust date parsing: keeps the dashboard consistent even if the same CSV is renamed.
    data["date"] = pd.to_datetime(data["date"], errors="coerce", dayfirst=False)

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
    # Notebook-cleaned sales dataset for KPI totals and charts
    # ============================================================
    # This follows the notebook logic:
    # data = data.dropna()
    # data_forecast = data[["branch", "date", "net_amount", "selling_price", "quantity", "product"]].copy()
    # total_sales = data_forecast["net_amount"].sum()
    data_sales = data.copy()
    data_sales = data_sales.dropna().copy()

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


def build_dashboard_forecast(data_forecast, selected_branches=None, steps=4):
    """Build forecast directly inside dashboard from uploaded sales data.
    This removes dependency on old forecast CSV files.

    Output columns:
    date, branch, ensemble
    """
    if data_forecast is None or data_forecast.empty:
        return pd.DataFrame()

    work = data_forecast.copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    work["net_amount"] = pd.to_numeric(work["net_amount"], errors="coerce")
    work["branch"] = work["branch"].astype(str).str.strip()
    work = work.dropna(subset=["date", "branch", "net_amount"])

    if selected_branches:
        work = work[work["branch"].isin(selected_branches)]

    rows = []
    for br, br_df in work.groupby("branch"):
        weekly = (
            br_df.groupby(pd.Grouper(key="date", freq="W-SUN"))["net_amount"]
            .sum()
            .sort_index()
        )

        fc = make_forecast(weekly, steps=steps)

        for dt, val in fc.items():
            rows.append({
                "date": dt,
                "branch": br,
                "ensemble": float(val)
            })

    return pd.DataFrame(rows)


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


def read_branch_forecast(path="branch_forecast_dashboard.csv", branches=None, uploaded_file=None):
    """Read branch-wise forecast exported from notebook.
    Expected columns: date, branch, ensemble
    """
    if uploaded_file is not None:
        uploaded_file.seek(0)
        fc = clean_columns(pd.read_csv(uploaded_file))
    else:
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





@st.cache_data(show_spinner=False)
def build_notebook_style_forecast(data_forecast, forecast_year=None):
    """Dynamic notebook-style December forecast.

    This follows the notebook structure:
    1. Aggregate daily sales by branch.
    2. Convert to weekly W-SUN branch sales.
    3. Select Top 5 branches.
    4. Build lag/rolling features.
    5. Validate XGBoost, Random Forest and ARIMA on September.
    6. Compute OOS weights using September MAPE.
    7. Retrain on full Jan-Nov and recursively forecast December weeks.

    It recalculates from the uploaded CSV, so new data will produce new forecasts.
    """
    work = data_forecast.copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    work["net_amount"] = pd.to_numeric(work["net_amount"], errors="coerce")
    work["branch"] = work["branch"].astype(str).str.strip()
    work = work.dropna(subset=["branch", "date", "net_amount"])

    if work.empty:
        return pd.DataFrame()

    # Forecast the December of the latest year available in the uploaded data.
    # This keeps the app reusable if the CSV year changes later.
    if forecast_year is None:
        forecast_year = int(work["date"].dt.year.max())

    # Direct daily sales per branch
    daily_branch_sales = (
        work.groupby(["branch", "date"], as_index=False)["net_amount"]
        .sum()
        .rename(columns={"net_amount": "sales"})
    )

    # Weekly branch sales exactly like notebook
    weekly_branch_sales = (
        daily_branch_sales
        .groupby(["branch", pd.Grouper(key="date", freq="W-SUN")])["sales"]
        .sum()
        .reset_index()
    )
    weekly_branch_sales["week"] = weekly_branch_sales["date"].dt.isocalendar().week.astype(int)
    weekly_branch_sales["month"] = weekly_branch_sales["date"].dt.strftime("%b")

    # Top 5 branches by total revenue
    branch_revenue = (
        work.groupby("branch")["net_amount"]
        .sum()
        .sort_values(ascending=False)
    )
    top_5 = branch_revenue.head(5).index.tolist()

    if not top_5:
        return pd.DataFrame()

    # Bhutan holiday dates used in the notebook, adjusted to the forecast year.
    # These are used as a simple holiday-proximity feature for the models.
    bhutan_holiday_month_days = [
        "01-02", "01-30", "02-21", "02-22", "02-23", "02-28",
        "03-01", "05-02", "05-07", "06-11", "07-28", "09-23",
        "10-02", "10-28", "11-01", "11-11"
    ]
    bhutan_holidays = pd.DataFrame({
        "ds": pd.to_datetime([f"{int(forecast_year)}-{md}" for md in bhutan_holiday_month_days], errors="coerce")
    }).dropna()

    def is_inactive_branch(series, threshold=0.85):
        col = "y" if "y" in series.columns else "sales"
        return (series[col] == 0).mean() > threshold if len(series) else True

    def build_features(df, branches):
        df_xgb = df.copy()
        df_xgb["date"] = pd.to_datetime(df_xgb["date"])
        df_xgb = df_xgb[df_xgb["branch"].isin(branches)]
        df_xgb = df_xgb.sort_values(["branch", "date"]).reset_index(drop=True)

        for lag in [1, 2, 4, 8, 12]:
            df_xgb[f"lag_{lag}"] = df_xgb.groupby("branch")["sales"].shift(lag)
        df_xgb["lag_13"] = df_xgb.groupby("branch")["sales"].shift(13)

        df_xgb["is_october"] = (df_xgb["date"].dt.month == 10).astype(int)
        df_xgb["is_fiscal_end"] = (df_xgb["date"].dt.month.isin([9, 10, 11])).astype(int)

        for window in [4, 8, 12]:
            df_xgb[f"roll_mean_{window}"] = (
                df_xgb.groupby("branch")["sales"]
                .transform(lambda x: x.shift(1).rolling(window).mean())
            )
        for window in [4, 8]:
            df_xgb[f"roll_std_{window}"] = (
                df_xgb.groupby("branch")["sales"]
                .transform(lambda x: x.shift(1).rolling(window).std())
            )
            df_xgb[f"roll_max_{window}"] = (
                df_xgb.groupby("branch")["sales"]
                .transform(lambda x: x.shift(1).rolling(window).max())
            )
            df_xgb[f"roll_min_{window}"] = (
                df_xgb.groupby("branch")["sales"]
                .transform(lambda x: x.shift(1).rolling(window).min())
            )

        df_xgb["ewm_4"] = (
            df_xgb.groupby("branch")["sales"]
            .transform(lambda x: x.shift(1).ewm(span=4).mean())
        )
        df_xgb["week_of_year"] = df_xgb["date"].dt.isocalendar().week.astype(int)
        df_xgb["month_num"] = df_xgb["date"].dt.month
        df_xgb["quarter"] = df_xgb["date"].dt.quarter
        df_xgb["is_month_start"] = (df_xgb["date"].dt.day <= 7).astype(int)
        df_xgb["is_month_end"] = (df_xgb["date"].dt.day >= 24).astype(int)
        df_xgb["trend"] = df_xgb.groupby("branch").cumcount()
        df_xgb["sales_vs_avg"] = (
            df_xgb.groupby("branch")["sales"]
            .transform(lambda x: x / (x.shift(1).rolling(8).mean() + 1))
        )

        branch_map = {b: i for i, b in enumerate(branches)}
        df_xgb["branch_enc"] = df_xgb["branch"].map(branch_map)

        holiday_dates = bhutan_holidays["ds"].tolist()
        df_xgb["is_holiday"] = df_xgb["date"].apply(
            lambda d: int(any(abs((pd.Timestamp(d) - h).days) <= 6 for h in holiday_dates))
        )

        df_xgb = df_xgb.dropna().copy()
        return df_xgb

    forecast_months = [9, 10, 11]
    calibration_month = 9

    df_xgb = build_features(weekly_branch_sales, top_5)
    if df_xgb.empty:
        return pd.DataFrame()

    string_cols = df_xgb.select_dtypes(include=["object", "string"]).columns.tolist()
    safe_exclude = ["date", "branch", "sales"] + string_cols
    feature_cols = [c for c in df_xgb.columns if c not in safe_exclude]

    # Light but stable model parameters for dashboard use
    xgb_params = dict(
        n_estimators=300,
        learning_rate=0.04,
        max_depth=3,
        min_child_weight=2,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.2,
        reg_lambda=2.0,
        objective="reg:squarederror",
        random_state=42,
        tree_method="hist",
    )
    rf_params = dict(
        n_estimators=300,
        max_depth=6,
        min_samples_split=4,
        min_samples_leaf=2,
        max_features="sqrt",
        bootstrap=True,
        random_state=42,
        n_jobs=-1,
    )

    def fit_arima_order(train_series):
        # Notebook searched over a grid; dashboard uses a compact stable choice
        # to keep app responsive while retaining ARIMA component.
        return (1, 1, 1)

    def mape(actual, pred):
        actual = np.array(actual, dtype=float)
        pred = np.array(pred, dtype=float)
        mask = actual != 0
        if mask.sum() == 0:
            return np.nan
        return np.mean(np.abs((actual[mask] - pred[mask]) / actual[mask])) * 100

    model_results = {"xgb": {}, "random_forest": {}, "arima": {}}

    # Walk-forward validation for Sep, Oct, Nov, mainly to compute September weights.
    for branch in top_5:
        br_features = df_xgb[df_xgb["branch"] == branch].copy()
        br_weekly = (
            weekly_branch_sales[weekly_branch_sales["branch"] == branch]
            .set_index("date")["sales"]
            .asfreq("W-SUN")
            .astype(float)
            .fillna(0)
        )

        if br_features.empty or is_inactive_branch(weekly_branch_sales[weekly_branch_sales["branch"] == branch]):
            continue

        for month in forecast_months:
            train = br_features[br_features["date"].dt.month < month].copy()
            test = br_features[br_features["date"].dt.month == month].copy()

            if len(train) >= 8 and len(test) > 0 and train["sales"].sum() > 0:
                X_train = train[feature_cols]
                y_train = np.log1p(train["sales"])
                X_test = test[feature_cols]
                actuals = test["sales"].values.astype(float)

                if XGBOOST_AVAILABLE:
                    try:
                        xgb_m = XGBRegressor(**xgb_params)
                        xgb_m.fit(X_train, y_train, verbose=False)
                        pred = np.clip(np.expm1(xgb_m.predict(X_test)), 0, None)
                        model_results["xgb"].setdefault(branch, {})[month] = {
                            "dates": test["date"].tolist(),
                            "actuals": actuals,
                            "predictions": pred,
                            "mape": mape(actuals, pred),
                        }
                    except Exception:
                        pass

                try:
                    rf_m = RandomForestRegressor(**rf_params)
                    rf_m.fit(X_train, y_train)
                    pred = np.clip(np.expm1(rf_m.predict(X_test)), 0, None)
                    model_results["random_forest"].setdefault(branch, {})[month] = {
                        "dates": test["date"].tolist(),
                        "actuals": actuals,
                        "predictions": pred,
                        "mape": mape(actuals, pred),
                    }
                except Exception:
                    pass

            train_ts = br_weekly[br_weekly.index.month < month]
            test_ts = br_weekly[br_weekly.index.month == month]

            if len(train_ts) >= 16 and len(test_ts) > 0 and train_ts.sum() > 0:
                try:
                    arima_model = ARIMA(np.log1p(train_ts.astype(float)), order=fit_arima_order(train_ts)).fit()
                    pred = np.clip(np.expm1(arima_model.forecast(steps=len(test_ts)).values), 0, None)
                    model_results["arima"].setdefault(branch, {})[month] = {
                        "dates": test_ts.index.tolist(),
                        "actuals": test_ts.values.astype(float),
                        "predictions": pred,
                        "mape": mape(test_ts.values, pred),
                    }
                except Exception:
                    pass

    # OOS weights using September exactly like notebook idea
    branch_weights = {}
    for branch in top_5:
        model_mapes = {}
        for model_name, result_dict in model_results.items():
            if branch in result_dict and calibration_month in result_dict[branch]:
                val = result_dict[branch][calibration_month].get("mape", np.nan)
                if not pd.isna(val):
                    model_mapes[model_name] = val

        if not model_mapes:
            branch_weights[branch] = {"xgb": 0.34, "random_forest": 0.33, "arima": 0.33}
        else:
            models = list(model_mapes.keys())
            mapes = np.array([model_mapes[m] for m in models], dtype=float)
            if np.nanmean(mapes) == 0 or np.isnan(np.nanmean(mapes)):
                branch_weights[branch] = {m: 1 / len(models) for m in models}
            else:
                scores = np.exp(-mapes / np.nanmean(mapes))
                norm = scores / scores.sum()
                branch_weights[branch] = {m: float(w) for m, w in zip(models, norm)}
                for m in ["xgb", "random_forest", "arima"]:
                    branch_weights[branch].setdefault(m, 0.0)

    forecast_year = int(forecast_year)
    dec_start = pd.Timestamp(f"{forecast_year}-12-01")
    dec_weeks = sorted([
        d for d in pd.date_range(f"{forecast_year}-11-30", f"{forecast_year}-12-31", freq="W-SUN")
        if d.month == 12
    ])

    weekly_branch_sales_clean = (
        weekly_branch_sales[
            (weekly_branch_sales["date"] < dec_start) &
            (weekly_branch_sales["branch"].isin(top_5))
        ]
        .drop_duplicates(subset=["branch", "date"])
        .sort_values(["branch", "date"])
        .reset_index(drop=True)
        .copy()
    )

    if "week" not in weekly_branch_sales_clean.columns:
        weekly_branch_sales_clean["week"] = weekly_branch_sales_clean["date"].dt.isocalendar().week.astype(int)
    if "month" not in weekly_branch_sales_clean.columns:
        weekly_branch_sales_clean["month"] = weekly_branch_sales_clean["date"].dt.strftime("%b")

    dec_placeholder = pd.DataFrame({
        "branch": top_5 * len(dec_weeks),
        "date": [d for d in dec_weeks for _ in top_5],
        "sales": 0.0,
        "week": [int(d.isocalendar()[1]) for d in dec_weeks for _ in top_5],
        "month": [d.strftime("%b") for d in dec_weeks for _ in top_5],
    })

    forecast_wbs = (
        pd.concat([weekly_branch_sales_clean, dec_placeholder], ignore_index=True)
        .drop_duplicates(subset=["branch", "date"])
        .sort_values(["branch", "date"])
        .reset_index(drop=True)
    )
    df_ext = build_features(forecast_wbs, top_5)

    def make_arima_ts(branch, cutoff_date):
        ts = (
            weekly_branch_sales_clean[
                (weekly_branch_sales_clean["branch"] == branch) &
                (weekly_branch_sales_clean["date"] <= cutoff_date)
            ]
            .drop_duplicates(subset=["date"])
            .set_index("date")["sales"]
            .sort_index()
            .astype(float)
        )
        if ts.empty:
            return ts
        full_range = pd.date_range(ts.index.min(), ts.index.max(), freq="W-SUN")
        return ts.reindex(full_range).fillna(0)

    def blend_step(xgb_m, rf_m, arima_pred, pred_row, weights, fallback):
        available = {}

        if xgb_m is not None and not pred_row.empty:
            try:
                available["xgb"] = float(np.clip(np.expm1(xgb_m.predict(pred_row[feature_cols])), 0, None)[0])
            except Exception:
                available["xgb"] = fallback

        if rf_m is not None and not pred_row.empty:
            try:
                available["random_forest"] = float(np.clip(np.expm1(rf_m.predict(pred_row[feature_cols])), 0, None)[0])
            except Exception:
                available["random_forest"] = fallback

        if arima_pred is not None:
            available["arima"] = float(arima_pred)

        if not available:
            return max(0.0, fallback)

        total_w = sum(weights.get(m, 0) for m in available if weights.get(m, 0) > 0)
        if total_w == 0:
            blended = float(np.mean(list(available.values())))
        else:
            blended = sum(weights.get(m, 0) * v for m, v in available.items() if weights.get(m, 0) > 0) / total_w
        return max(0.0, blended)

    rows = []
    for branch in top_5:
        weights = branch_weights.get(branch, {"xgb": 0.34, "random_forest": 0.33, "arima": 0.33})

        final_train_df = df_ext[
            (df_ext["branch"] == branch) &
            (df_ext["date"] < dec_start)
        ].copy()

        if final_train_df.empty:
            continue

        xgb_f = rf_f = arima_f = None
        if XGBOOST_AVAILABLE:
            try:
                xgb_f = XGBRegressor(**xgb_params)
                xgb_f.fit(final_train_df[feature_cols], np.log1p(final_train_df["sales"]), verbose=False)
            except Exception:
                xgb_f = None

        try:
            rf_f = RandomForestRegressor(**rf_params)
            rf_f.fit(final_train_df[feature_cols], np.log1p(final_train_df["sales"]))
        except Exception:
            rf_f = None

        final_ts = make_arima_ts(branch, pd.Timestamp(f"{forecast_year}-11-30"))
        if len(final_ts) >= 16 and final_ts.sum() > 0:
            try:
                arima_f = ARIMA(np.log1p(final_ts.astype(float)), order=(1, 1, 1)).fit()
            except Exception:
                arima_f = None

        branch_hist = (
            weekly_branch_sales_clean[weekly_branch_sales_clean["branch"] == branch]
            .drop_duplicates(subset=["date"])
            .sort_values("date")
            .copy()
        )
        if branch_hist.empty:
            continue

        fallback_val = float(branch_hist["sales"].iloc[-1])

        if arima_f is not None:
            try:
                arima_dec_fc = np.clip(np.expm1(arima_f.forecast(steps=len(dec_weeks)).values), 0, None).tolist()
            except Exception:
                arima_dec_fc = [fallback_val] * len(dec_weeks)
        else:
            arima_dec_fc = [fallback_val] * len(dec_weeks)

        running_dec = branch_hist.copy()

        for step_i, week_date in enumerate(dec_weeks):
            fallback = float(running_dec["sales"].iloc[-1]) if len(running_dec) > 0 else 0.0

            temp_wbs = pd.concat([
                running_dec,
                pd.DataFrame({
                    "branch": [branch],
                    "date": [week_date],
                    "sales": [0.0],
                    "week": [int(week_date.isocalendar()[1])],
                    "month": [week_date.strftime("%b")],
                })
            ], ignore_index=True).drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)

            df_temp = build_features(temp_wbs, [branch])
            pred_row = df_temp[df_temp["date"] == week_date]

            blend = blend_step(
                xgb_f, rf_f, arima_dec_fc[step_i],
                pred_row, weights, fallback
            )

            rows.append({
                "date": week_date,
                "branch": branch,
                "ensemble": float(blend),
            })

            running_dec = pd.concat([
                running_dec,
                pd.DataFrame({
                    "branch": [branch],
                    "date": [week_date],
                    "sales": [blend],
                    "week": [int(week_date.isocalendar()[1])],
                    "month": [week_date.strftime("%b")],
                })
            ], ignore_index=True).drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)

    return pd.DataFrame(rows)


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
                return out, "Notebook forecast CSV"

    # For overall view, use ensemble_forecast_dashboard.csv
    if forecast_raw is not None and not forecast_raw.empty:
        fc = forecast_raw.copy()
        if selected_branch and selected_branch != "Overall selected branches" and "branch" in fc.columns:
            fc = fc[fc["branch"].astype(str) == str(selected_branch)]
        if not fc.empty and {"date", "ensemble"}.issubset(fc.columns):
            out = fc.groupby("date")["ensemble"].sum().sort_index()
            out.name = "forecast"
            return out, "Notebook forecast CSV"

    if fallback_actual is not None and len(fallback_actual):
        return pd.Series(dtype=float, name="forecast"), "No notebook forecast CSV"
    return pd.Series(dtype=float, name="forecast"), "No notebook forecast CSV"


def add_actual_forecast_connection(fig, actual_series, forecast_series, line_color="#21e6ff"):
    """Add a visible dotted line from the last actual point to the first forecast point."""
    if actual_series is None or forecast_series is None:
        return fig

    actual_clean = actual_series.dropna().sort_index()
    forecast_clean = forecast_series.dropna().sort_index()

    if actual_clean.empty or forecast_clean.empty:
        return fig

    last_actual_date = actual_clean.index.max()
    first_forecast_date = forecast_clean.index.min()
    last_actual_value = actual_clean.loc[last_actual_date]
    first_forecast_value = forecast_clean.loc[first_forecast_date]

    fig.add_trace(go.Scatter(
        x=[last_actual_date, first_forecast_date],
        y=[last_actual_value, first_forecast_value],
        mode="lines",
        name="Actual–Forecast Connection",
        line=dict(color=line_color, width=3, dash="dot"),
        hovertemplate=(
            "Connection<br>"
            "Date: %{x|%b %d, %Y}<br>"
            "Sales: Nu. %{y:,.0f}<extra></extra>"
        ),
        showlegend=True
    ))
    return fig

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
    ["Overview", "Sales Analytics", "Branch", "Forecast", "Product Bundles"],
    index=0
)
uploaded = st.sidebar.file_uploader("Upload Sales CSV", type=["csv"])
forecast_uploaded = st.sidebar.file_uploader("Upload Notebook Forecast CSV", type=["csv"])



if uploaded is None:
    st.markdown("<div class='hero-title'>PBSL Sales Analytics & Forecasting Dashboard</div>", unsafe_allow_html=True)
    st.markdown("<div class='hero-sub'>Upload your Sales CSV from the sidebar to open the dashboard.</div>", unsafe_allow_html=True)
    st.info("Required columns: date, branch, product, quantity, net_amount, invoice_id")
    st.stop()

# Create a unique file key from actual file content.
# This prevents Streamlit from reusing old filter values when another CSV is uploaded.
uploaded_bytes = uploaded.getvalue()
uploaded_hash = hashlib.md5(uploaded_bytes).hexdigest()[:10]
uploaded.seek(0)

data, data_mba, data_forecast = load_data(uploaded)

# Historical actual sales coverage only. This prevents future forecast dates from being displayed as actual data coverage.
actual_data = data.copy()
if "type" in actual_data.columns:
    actual_data = actual_data[actual_data["type"].astype(str).str.lower().str.strip() != "forecast"]
# Use notebook-cleaned sales data for actual sales.
# No manual cutoff is applied here because the notebook total uses the full cleaned dataset.
actual_data = actual_data.copy()

min_date, max_date = actual_data["date"].min().date(), actual_data["date"].max().date()

# Make filter widgets reset correctly when a different file is uploaded.
# This prevents old date/branch/product selections from carrying over to a new CSV.
file_key = f"data_{uploaded_hash}"

date_range = st.sidebar.date_input(
    "Date Range",
    [min_date, max_date],
    min_value=min_date,
    max_value=max_date,
    key=f"date_range_{file_key}"
)

branches = sorted(actual_data["branch"].dropna().unique())
selected_branches = st.sidebar.multiselect(
    "Branch",
    branches,
    default=branches,
    key=f"branch_filter_{file_key}"
)

products = sorted(actual_data["product"].dropna().unique())
selected_products = st.sidebar.multiselect(
    "Product Filter",
    products,
    default=[],
    key=f"product_filter_{file_key}"
)

if len(date_range) == 2:
    start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
else:
    start_date, end_date = data["date"].min(), data["date"].max()

filtered = actual_data[(actual_data["date"] >= start_date) & (actual_data["date"] <= end_date)].copy()

# Apply branch filter only to rows with available branch values.
# If all branches are selected, keep rows with missing branch too so Total Sales matches notebook total.
if len(selected_branches) < len(branches):
    filtered = filtered[filtered["branch"].isin(selected_branches)]

# Apply product filter only when user selects specific products.
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
# KPI values follow the notebook logic:
# data = data.dropna()
# data_forecast = data[["branch", "date", "net_amount", "selling_price", "quantity", "product"]].copy()
# total_sales = data_forecast["net_amount"].sum()

filtered_forecast = data_forecast[
    (data_forecast["date"] >= start_date) &
    (data_forecast["date"] <= end_date)
].copy()

if len(selected_branches) < len(branches):
    filtered_forecast = filtered_forecast[filtered_forecast["branch"].isin(selected_branches)]

if selected_products:
    filtered_forecast = filtered_forecast[filtered_forecast["product"].isin(selected_products)]

# Forecast KPI and forecast graph should NOT change when Product Filter changes.
# This copy uses only Date Range and Branch filters, not Product Filter.
forecast_actual_source = data_forecast[
    (data_forecast["date"] >= start_date) &
    (data_forecast["date"] <= end_date)
].copy()

if len(selected_branches) < len(branches):
    forecast_actual_source = forecast_actual_source[forecast_actual_source["branch"].isin(selected_branches)]

total_sales = pd.to_numeric(filtered_forecast["net_amount"], errors="coerce").sum()
total_qty = pd.to_numeric(filtered_forecast["quantity"], errors="coerce").sum() if "quantity" in filtered_forecast.columns else 0

# Transactions and unique products follow notebook MBA-cleaned dataset
total_transactions = filtered_mba["invoice_id"].dropna().nunique() if "invoice_id" in filtered_mba.columns else 0
total_unique_products = (
    filtered_mba["product"].astype(str).str.strip().nunique()
    if "product" in filtered_mba.columns
    else filtered_forecast["product"].astype(str).str.strip().nunique()
)

avg_order = (
    filtered.dropna(subset=["invoice_id"])
    .groupby("invoice_id")["net_amount"]
    .sum()
    .mean()
    if total_transactions else 0
)
total_customers = filtered["customer"].dropna().nunique() if "customer" in filtered.columns else total_transactions

branch_sales_source = filtered_forecast.dropna(subset=["branch"]).copy()
branch_sales_source["net_amount"] = pd.to_numeric(branch_sales_source["net_amount"], errors="coerce")
branch_sales = branch_sales_source.groupby("branch")["net_amount"].sum().sort_values(ascending=False)
best_branch = branch_sales.index[0] if len(branch_sales) else "N/A"
best_value = branch_sales.iloc[0] if len(branch_sales) else 0

forecast_branch_sales_source = forecast_actual_source.dropna(subset=["branch"]).copy()
forecast_branch_sales_source["net_amount"] = pd.to_numeric(forecast_branch_sales_source["net_amount"], errors="coerce")
forecast_branch_sales = forecast_branch_sales_source.groupby("branch")["net_amount"].sum().sort_values(ascending=False)

weekly_actual = forecast_actual_source.groupby(pd.Grouper(key="date", freq="W-SUN"))["net_amount"].sum().sort_index()
top_5_branches = branch_sales.head(5).index.tolist()
top_5_forecast_branches = forecast_branch_sales.head(5).index.tolist()
# Forecast is read only from notebook-exported CSV.
# No model is trained or forecast is generated inside this dashboard.
branch_forecast_df, branch_forecast_found = read_branch_forecast(uploaded_file=forecast_uploaded)

if branch_forecast_found:
    forecast_raw = branch_forecast_df.copy()
    if selected_branches:
        forecast_raw = forecast_raw[forecast_raw["branch"].isin(selected_branches)]
    forecast = forecast_raw.groupby("date")["ensemble"].sum().sort_index()
    forecast.name = "forecast"
else:
    forecast_raw = pd.DataFrame()
    forecast = pd.Series(dtype=float, name="forecast")

# This total comes only from the notebook forecast CSV and branch selection.
# It does not change when Product Filter is changed.
forecast_total = forecast.sum() if len(forecast) else 0

# ---------------- Header ----------------
left, right = st.columns([2.3, 1])
with left:
    st.markdown(f"<div class='neon-badge'>{page}</div>", unsafe_allow_html=True)
    st.markdown("<div class='hero-title'>PBSL Interactive Sales Dashboard</div>", unsafe_allow_html=True)
    st.markdown("<div class='hero-sub'>Sales overview, trends, contribution analysis, branch ABC-VED, forecasting and product bundles</div>", unsafe_allow_html=True)
with right:
    st.markdown(f"""
    <div class='chart-card'>
      <b>Selected Period</b><br><span class='small-muted'>{start_date.date()} to {end_date.date()}</span><br><br>
      <b>Branch Filter</b><br><span class='small-muted'>{len(selected_branches)} branch(es) selected</span>
    </div>
    """, unsafe_allow_html=True)

# ---------------- Pages ----------------
if page == "Overview":
    data_last_updated = filtered["date"].max().strftime("%b %d, %Y") if not filtered.empty else "N/A"

    st.markdown(f"""
    <div class='overview-hero'>
      <div class='overview-hero-title'>PBSL Sales Performance Overview</div>
      <div class='overview-hero-sub'>
        Interactive summary of sales, products, branch performance and forecast movement for the selected period.
      </div>
    </div>
    """, unsafe_allow_html=True)

    overview_metrics = [
        ("💎", "Total Sales", money(total_sales)),
        ("📦", "Quantity Sold", number(total_qty)),
        ("🧾", "Transactions", number(total_transactions)),
        ("🧱", "Unique Products", number(total_unique_products)),
        ("🚀", "Notebook Forecast Total", money(forecast_total)),
    ]

    cols = st.columns(5)
    for i, (icon, title, value) in enumerate(overview_metrics):
        cols[i].markdown(f"""
        <div class='kpi-grid-card'>
          <div class='kpi-icon'>{icon}</div>
          <div class='kpi-label'>{title}</div>
          <div class='kpi-number'>{value}</div>
        </div>
        """, unsafe_allow_html=True)

    with st.expander("KPI check"):
        st.write({
            "Dataset content ID": uploaded_hash,
            "KPI source": "Notebook logic: data.dropna(), data_forecast net_amount sum",
            "Rows used for KPI after current filters": len(filtered),
            "Total sales used": money(total_sales),
            "Date range used": f"{start_date.date()} to {end_date.date()}",
            "Branches selected": len(selected_branches),
            "Product filter applied": bool(selected_products),
            "Forecast note": "Forecast KPI and forecast graph ignore Product Filter and use notebook forecast CSV only.",
            "Best branch used": str(best_branch),
            "Best branch value": money(best_value)
        })

    st.markdown("<br>", unsafe_allow_html=True)

    b1, b2 = st.columns([2, 1], gap="large")
    with b1:
        st.markdown(f"""
        <div class='highlight-card'>
            <div class='highlight-label'>Best Performing Branch</div>
            <div class='highlight-main'>{best_branch}</div>
            <div class='highlight-sub'>{money(best_value)}</div>
        </div>
        """, unsafe_allow_html=True)

    with b2:
        st.markdown(f"""
        <div class='highlight-card'>
            <div class='highlight-label'>Data Last Updated</div>
            <div class='highlight-main'>{data_last_updated}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    oc1, oc2 = st.columns([1.7, 1], gap="large")
    with oc1:
        section_start("Actual Sales Connected with Forecast")
        overview_forecast, overview_forecast_label = get_forecast_series(
            forecast_raw,
            selected_branch="Overall selected branches",
            fallback_actual=weekly_actual,
            steps=4,
            branch_forecast_raw=branch_forecast_df
        )
        fig = go.Figure()
        if len(weekly_actual):
            fig.add_trace(go.Scatter(
                x=weekly_actual.index,
                y=weekly_actual.values,
                mode="lines+markers",
                name="Actual weekly sales",
                line=dict(color="#2f80ff", width=4),
                marker=dict(size=8),
                fill="tozeroy",
                fillcolor="rgba(47,128,255,.12)",
                hovertemplate="Date: %{x|%b %d, %Y}<br>Actual Sales: Nu. %{y:,.0f}<extra></extra>"
            ))
        if len(overview_forecast):
            fig.add_trace(go.Scatter(
                x=overview_forecast.index,
                y=overview_forecast.values,
                mode="lines+markers",
                name="Forecast sales",
                line=dict(color="#21e6ff", width=4, dash="dash"),
                marker=dict(size=9),
                hovertemplate="Date: %{x|%b %d, %Y}<br>Forecast Sales: Nu. %{y:,.0f}<extra></extra>"
            ))
        fig = add_actual_forecast_connection(fig, weekly_actual, overview_forecast)
        fig.update_layout(xaxis_title="Date", yaxis_title="Sales / Forecast (Nu.)", hovermode="x unified")
        fig.update_yaxes(tickprefix="Nu. ")
        st.plotly_chart(chart_theme(fig, 560), use_container_width=True, config=CHART_CONFIG)
        section_end()

    with oc2:
        section_start("Top 5 Branch Sales Share")
        pie_df = branch_sales.head(5).reset_index()
        pie_df.columns = ["branch", "sales"]
        fig = go.Figure(data=[go.Pie(
            labels=pie_df["branch"],
            values=pie_df["sales"],
            hole=.58,
            textinfo="label+percent",
            sort=False,
            marker=dict(
                colors=["#21e6ff", "#2f80ff", "#28ff8f", "#f4c542", "#ff7a59"],
                line=dict(color="#06122f", width=2)
            ),
            hovertemplate="%{label}<br>Sales: Nu. %{value:,.0f}<br>Share: %{percent}<extra></extra>"
        )])
        fig.add_annotation(
            text=f"TOP 5<br><b>{money(pie_df['sales'].sum()).replace('Nu. ', '')}</b>",
            x=.5, y=.5, showarrow=False, font=dict(size=17, color="white")
        )
        st.plotly_chart(chart_theme(fig, 560), use_container_width=True, config=CHART_CONFIG)
        section_end()

    st.markdown(f"""
    <div class='info-strip'>
      <div class='info-tile'>
        <div class='info-tile-label'>Selected Branches</div>
        <div class='info-tile-value'>{len(selected_branches)} Branch(es)</div>
      </div>
      <div class='info-tile'>
        <div class='info-tile-label'>Selected Period</div>
        <div class='info-tile-value'>{start_date.date()} to {end_date.date()}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

elif page == "Sales Analytics":
    section_start("Sales Analytics")
    sales_view = st.selectbox(
        "Select sales chart",
        ["Overall weekly trends", "Weekly trends for top 5 branches", "Seasonality of weekdays"],
        key="sales_analytics_dropdown"
    )

    if sales_view == "Overall weekly trends":
        weekly = filtered.groupby(pd.Grouper(key="date", freq="W-SUN"))["net_amount"].sum().sort_index()
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=weekly.index, y=weekly.values, mode="lines+markers", name="Overall weekly sales",
            line=dict(width=4, color="#21e6ff"), marker=dict(size=8),
            hovertemplate="Week: %{x|%b %d, %Y}<br>Sales: Nu. %{y:,.0f}<extra></extra>"
        ))
        fig.update_layout(xaxis_title="Week", yaxis_title="Sales (Nu.)", hovermode="x unified")
        fig.update_yaxes(tickprefix="Nu. ")
        st.plotly_chart(chart_theme(fig, 560), use_container_width=True, config=CHART_CONFIG)

    elif sales_view == "Weekly trends for top 5 branches":
        fig = go.Figure()
        for br in top_5_branches:
            wk = filtered[filtered["branch"] == br].groupby(pd.Grouper(key="date", freq="W-SUN"))["net_amount"].sum().sort_index()
            fig.add_trace(go.Scatter(
                x=wk.index, y=wk.values, mode="lines+markers", name=str(br),
                hovertemplate=f"Branch: {br}<br>Week: %{{x|%b %d, %Y}}<br>Sales: Nu. %{{y:,.0f}}<extra></extra>"
            ))
        fig.update_layout(xaxis_title="Week", yaxis_title="Sales (Nu.)", hovermode="x unified")
        fig.update_yaxes(tickprefix="Nu. ")
        st.plotly_chart(chart_theme(fig, 590), use_container_width=True, config=CHART_CONFIG)

    else:
        # Weekday order without Sunday
        order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]

        weekday = (
            filtered.assign(weekday=filtered["date"].dt.day_name())
            .groupby("weekday")["net_amount"]
            .sum()
            .reindex(order)
            .dropna()
            .reset_index()
        )

        # Convert to million Nu.
        weekday["million_sales"] = weekday["net_amount"] / 1_000_000

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=weekday["weekday"],
            y=weekday["million_sales"],
            mode="lines+markers",
            line=dict(color="#b22222", width=3),
            marker=dict(size=7, color="#b22222"),
            hovertemplate="%{x}<br>Sales: %{y:.2f} Million Nu.<extra></extra>"
        ))

        fig.update_layout(
            xaxis_title="Weekday",
            yaxis_title="Value (Million Nu.)",
            hovermode="x unified"
        )
        st.plotly_chart(
            chart_theme(fig, 540),
            use_container_width=True,
            config=CHART_CONFIG
        )
    section_end()

    section_start("Contributions")
    contribution_view = st.selectbox(
        "Select contribution chart",
        ["Top 10 products by revenue", "Top 10 products by quantity", "Least 10 products by revenue", "Least 10 products by quantity"],
        key="contribution_dropdown"
    )
    value_col = "net_amount" if "revenue" in contribution_view else "quantity"
    label = "Revenue (Nu.)" if value_col == "net_amount" else "Quantity Sold"
    ascending = contribution_view.startswith("Least")
    prod_contrib = (filtered.groupby("product", as_index=False)[value_col].sum()
                    .sort_values(value_col, ascending=ascending).head(10).sort_values(value_col))
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=prod_contrib[value_col], y=prod_contrib["product"], orientation="h",
        text=[money(v) if value_col == "net_amount" else number(v) for v in prod_contrib[value_col]],
        textposition="inside",
        marker=dict(color=prod_contrib[value_col], colorscale="Blues", line=dict(color="#21e6ff", width=1)),
        hovertemplate="%{y}<br>" + label + ": %{x:,.2f}<extra></extra>"
    ))
    fig.update_layout(xaxis_title=label, yaxis_title="Product")
    if value_col == "net_amount":
        fig.update_xaxes(tickprefix="Nu. ")
    fig.update_yaxes(tickfont=dict(size=10))
    st.plotly_chart(chart_theme(fig, 620), use_container_width=True, config=CHART_CONFIG)
    section_end()

elif page == "Branch":
    branch_options = top_5_branches
    branch_choice = st.selectbox(
        "Select top 5 branch for ABC-VED analysis",
        branch_options,
        key="abc_ved_branch_dropdown"
    )
    branch_df = filtered[filtered["branch"] == branch_choice].copy()
    branch_abc_ved = make_abc_ved_summary(branch_df)

    c1, c2 = st.columns([1.2, 1], gap="large")
    with c1:
        section_start("Top 5 Branch Revenue")
        top5 = branch_sales.head(5).sort_values().reset_index()
        top5.columns = ["branch", "sales"]
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=top5["sales"], y=top5["branch"], orientation="h",
            text=[money(v) for v in top5["sales"]], textposition="inside",
            marker=dict(color=top5["sales"], colorscale="Blues", line=dict(color="#21e6ff", width=1)),
            hovertemplate="%{y}<br>Revenue: Nu. %{x:,.0f}<extra></extra>"
        ))
        fig.update_layout(xaxis_title="Revenue (Nu.)", yaxis_title="Branch")
        fig.update_xaxes(tickprefix="Nu. ")
        st.plotly_chart(chart_theme(fig, 520), use_container_width=True, config=CHART_CONFIG)
        section_end()

    with c2:
        section_start(f"Revenue Contribution by ABC-VED — {branch_choice}")
        if branch_abc_ved.empty:
            st.info("ABC-VED analysis is not available for the selected branch.")
        else:
            abc_rev = branch_abc_ved.groupby("ABC_VED", as_index=False)["Sales Value"].sum().sort_values("ABC_VED")
            fig = go.Figure(data=[go.Pie(
                labels=abc_rev["ABC_VED"], values=abc_rev["Sales Value"], hole=.55,
                textinfo="label+percent", sort=False,
                marker=dict(line=dict(color="#06122f", width=2)),
                hovertemplate="Class %{label}<br>Revenue: Nu. %{value:,.0f}<br>Share: %{percent}<extra></extra>"
            )])
            fig.add_annotation(text="ABC-VED<br><b>REVENUE</b>", x=.5, y=.5, showarrow=False, font=dict(size=16, color="white"))
        st.plotly_chart(chart_theme(fig, 520), use_container_width=True, config=CHART_CONFIG)
        section_end()

    section_start(f"ABC-VED Product List — {branch_choice}")
    if branch_abc_ved.empty:
        st.info("No ABC-VED products available for the selected branch.")
    else:
        class_choice = st.selectbox("Select ABC-VED class", sorted(branch_abc_ved["ABC_VED"].dropna().unique()), key="branch_abc_ved_class")
        show_df = branch_abc_ved[branch_abc_ved["ABC_VED"] == class_choice].sort_values("Sales Value", ascending=False)
        st.markdown(f"<div class='small-muted'>Showing <b>{len(show_df):,}</b> products in <b>{branch_choice}</b> under ABC-VED class <b>{class_choice}</b>.</div>", unsafe_allow_html=True)
        st.dataframe(show_df[["product", "ABC", "VED", "ABC_VED", "Sales Value", "Quantity Sold", "Transactions"]], hide_index=True, use_container_width=True)
    section_end()

elif page == "Forecast":
    section_start("Forecast")
    top5_branch_options = branch_sales.head(5).index.tolist()
    forecast_view_options = ["Overall selected branches"] + top5_branch_options
    forecast_view = st.selectbox("Select forecast view", forecast_view_options, index=0, key="forecast_view_selector")

    if forecast_view == "Overall selected branches":
        actual_for_view = weekly_actual
    else:
        actual_for_view = (filtered[filtered["branch"] == forecast_view]
                           .groupby(pd.Grouper(key="date", freq="W-SUN"))["net_amount"].sum().sort_index())

    view_forecast, view_forecast_label = get_forecast_series(
        forecast_raw, selected_branch=forecast_view, fallback_actual=actual_for_view, steps=4, branch_forecast_raw=branch_forecast_df
    )

    fig = go.Figure()
    if len(actual_for_view):
        fig.add_trace(go.Scatter(
            x=actual_for_view.index, y=actual_for_view.values, mode="lines+markers", name="Historical weekly sales",
            line=dict(color="#2f80ff", width=4), marker=dict(size=8),
            hovertemplate="Date: %{x|%b %d, %Y}<br>Actual Sales: Nu. %{y:,.0f}<extra></extra>"
        ))
    if len(view_forecast):
        fig.add_trace(go.Scatter(
            x=view_forecast.index, y=view_forecast.values, mode="lines+markers", name="Forecast",
            line=dict(color="#21e6ff", width=4, dash="dash"), marker=dict(size=9),
            hovertemplate="Date: %{x|%b %d, %Y}<br>Forecast Sales: Nu. %{y:,.0f}<extra></extra>"
        ))

    # Connect the last actual point to the first forecast point in the dashboard
    fig = add_actual_forecast_connection(fig, actual_for_view, view_forecast)

    fig.update_layout(xaxis_title="Date", yaxis_title="Sales / Forecast (Nu.)", hovermode="x unified")
    fig.update_yaxes(tickprefix="Nu. ")
    
    st.plotly_chart(chart_theme(fig, 620), use_container_width=True, config=CHART_CONFIG)

    if len(view_forecast):
        display = view_forecast.reset_index()
        display.columns = ["Date", "Forecast Sales"]
        display["Forecast Sales"] = display["Forecast Sales"].round(0).astype(int)
        if forecast_view != "Overall selected branches":
            display.insert(1, "Branch", forecast_view)
        st.dataframe(display, hide_index=True, use_container_width=True)
    section_end()

elif page == "Product Bundles":
    section_start("Product Bundles / Frequently Purchased Together")
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
        top_n = st.slider("Bundle size", 2, 5, 3)

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
        st.warning("No product bundle can be created because the filtered data is too small.")
    else:
        for i, row in rec.reset_index(drop=True).iterrows():
            st.markdown(f"""
            <div class='rec-card'>
              <div class='rec-rank'>{i+1}</div>
              <div>
                <div class='rec-name'>{selected_prod} + {row['Recommended Product']}</div>
                <div class='rec-meta'>{row.get('Rule','')} | Support: {row['support']:.3f} | Lift: {row['lift']:.2f}</div>
              </div>
              <div class='conf-pill'>{row['confidence']:.0%}</div>
            </div>
            """, unsafe_allow_html=True)

        rec_plot = rec.reset_index(drop=True).copy()
        rec_plot["confidence_pct"] = rec_plot["confidence"] * 100
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=rec_plot["confidence_pct"], y=rec_plot["Recommended Product"], orientation="h",
            text=[f"{v:.0f}%" for v in rec_plot["confidence_pct"]], textposition="inside",
            marker=dict(color=rec_plot["confidence_pct"], colorscale="Blues", line=dict(color="#21e6ff", width=1)),
            name="Confidence"
        ))
        fig.update_layout(xaxis_title="Bundle Confidence (%)", yaxis_title="Recommended Product")
        st.plotly_chart(chart_theme(fig, 500), use_container_width=True, config=CHART_CONFIG)
    section_end()
