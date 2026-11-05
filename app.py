"""Flight Delay MLOps — interactive demo of an end-to-end ML pipeline with
data versioning, a feature store, MLflow tracking/registry and a prediction
service.  Run:  streamlit run app.py
"""
from __future__ import annotations

import os

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src import versioning
from src.security import MAX_RUNS, check_admin, cooldown_ok, sanitize_text
from src.config import (BASE_CATEGORICAL, BASE_NUMERIC, CHAMPION_ALIAS, mlflow_db_path, new_mlflow_db,
                        REGISTERED_MODEL, STORE_FEATURES, TARGET, TEAM)
from src.data_generation import AIRPORTS, CARRIERS, NEW_CARRIER
from src.feature_store import FEATURE_VIEWS, FeatureStore

st.set_page_config(page_title="Flight Delay MLOps", page_icon="✈️", layout="wide")

# ---- Themes: dark, neon and light ------------------------------------------
_D = dict(mode="dark", lesson="#FDE9B0", good="#C8F5E2", bad="#FBD0D0", on_acc="#0B0F14", glow=False)
_L = dict(mode="light", lesson="#7A5300", good="#065F46", bad="#9F1239", on_acc="#FFFFFF", glow=False)
THEMES = {
    "Deep Ocean":      {**_D, **dict(bg="#06181D", side="#081F25", panel="#0C2830", panel2="#0A222A", border="#1B4550",
                           acc="#2DD4BF", acc2="#7DD3FC", acc_rgb="45,212,191", h1="#0C2A33", h2="#124350",
                           text="#E6F2F2", head="#F4FBFB", sub="#99F6E4", muted="#8FB3B8")},
    "Neon Glow":       {**_D, **dict(bg="#070B14", side="#0A0F1C", panel="#0E1424", panel2="#0B1120", border="#1E2A44",
                           acc="#00F0FF", acc2="#FF2BD6", acc_rgb="0,240,255", h1="#1A0F2E", h2="#0B2A3A",
                           text="#E6F1FF", head="#FFFFFF", sub="#7DF9FF", muted="#8892B0", glow=True)},
    "Midnight Navy":   {**_D, **dict(bg="#0B1220", side="#0E172A", panel="#111A2E", panel2="#0F1829", border="#22304D",
                           acc="#38BDF8", acc2="#A5B4FC", acc_rgb="56,189,248", h1="#13203A", h2="#1C2F52",
                           text="#E8EEF7", head="#F1F5FB", sub="#BFDBFE", muted="#94A3B8")},
    "Royal Plum":      {**_D, **dict(bg="#130B1D", side="#170E25", panel="#1D1430", panel2="#1A1129", border="#3A2858",
                           acc="#C084FC", acc2="#F0ABFC", acc_rgb="192,132,252", h1="#22153A", h2="#352058",
                           text="#F1EAFB", head="#FAF5FF", sub="#E9D5FF", muted="#A99BC0")},
    "Graphite & Gold": {**_D, **dict(bg="#111214", side="#15161A", panel="#1B1D22", panel2="#18191D", border="#30333B",
                           acc="#E2B55A", acc2="#F5D58A", acc_rgb="226,181,90", h1="#1D1F25", h2="#2C2F37",
                           text="#ECEAE4", head="#FAF8F2", sub="#F5DFA8", muted="#A3A09A")},
    "Emerald":         {**_D, **dict(bg="#06150E", side="#081B12", panel="#0D2419", panel2="#0B2016", border="#1D4430",
                           acc="#34D399", acc2="#A7F3D0", acc_rgb="52,211,153", h1="#0D2A1C", h2="#14432D",
                           text="#E6F4EC", head="#F3FBF6", sub="#BBF7D0", muted="#8DB09C")},
    "Light · Paper":   {**_L, **dict(bg="#FAFAF7", side="#F2F1EC", panel="#FFFFFF", panel2="#FBFBF8", border="#D6D3CC",
                           acc="#3F3F46", acc2="#18181B", acc_rgb="63,63,70", h1="#FFFFFF", h2="#F1EFEA",
                           text="#1C1917", head="#0C0A09", sub="#27272A", muted="#71717A")},
    "Light · Classic": {**_L, **dict(bg="#F5F7FB", side="#EEF2F8", panel="#FFFFFF", panel2="#F8FAFD", border="#D9E1EC",
                           acc="#1D4ED8", acc2="#1E40AF", acc_rgb="29,78,216", h1="#FFFFFF", h2="#E8EFFC",
                           text="#1E293B", head="#0F172A", sub="#1E3A8A", muted="#64748B")},
    "Light · Ivory":   {**_L, **dict(bg="#FAF7F0", side="#F3EEE2", panel="#FFFDF8", panel2="#FBF8F1", border="#E4DCC8",
                           acc="#9A6B12", acc2="#1E3A5F", acc_rgb="154,107,18", h1="#FFFDF8", h2="#F4EBD6",
                           text="#2B2A26", head="#1C1B18", sub="#1E3A5F", muted="#7A7466")},
    "Light · Sky":     {**_L, **dict(bg="#F1F8FC", side="#E6F2F9", panel="#FFFFFF", panel2="#F6FBFE", border="#CFE3EE",
                           acc="#0369A1", acc2="#0E7490", acc_rgb="3,105,161", h1="#FFFFFF", h2="#DCEFF9",
                           text="#0F2A3A", head="#0B1F2C", sub="#075985", muted="#5B7486")},
}
st.session_state.setdefault("theme", "Deep Ocean")
THEME_NAME = st.session_state["theme"]
T = THEMES.get(THEME_NAME, THEMES["Deep Ocean"])
LIGHT = T["mode"] == "light"

NAVY, TEAL = T["text"], T["acc"]
AMBER, RED, GREEN, GREY = ("#D97706", "#DC2626", "#059669", "#7C3AED") if LIGHT else ("#FBBF24", "#F87171", "#34D399", "#A78BFA")
if THEME_NAME == "Neon Glow":
    AMBER, RED, GREEN, GREY = "#FFC300", "#FF3B6B", "#39FF88", "#B26BFF"
if THEME_NAME == "Light · Paper":   # muted, print-friendly data colours
    AMBER, RED, GREEN, GREY = "#B45309", "#B91C1C", "#3F6212", "#A8A29E"
MAGENTA = "#9A3412" if THEME_NAME == "Light · Paper" else "#C026D3" if LIGHT else ("#FF2BD6" if T["glow"] else "#C084FC")
BG, PANEL, BORDER, MUTED = T["bg"], T["panel"], T["border"], T["muted"]
PLOT = dict(template="plotly_white" if LIGHT else "plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor=T["panel2"], font=dict(family="Inter, Segoe UI, sans-serif", color=NAVY, size=13),
            title_font=dict(family="Source Serif 4, Georgia, serif", size=17),
            margin=dict(l=10, r=10, t=48, b=10))

SH = ("0 1px 2px rgba(15,23,42,.06), 0 10px 24px -12px rgba(15,23,42,.18)" if LIGHT else
      "0 1px 0 rgba(255,255,255,.04) inset, 0 10px 22px -10px rgba(0,0,0,.7), 0 2px 4px rgba(0,0,0,.3)")
SH_HOVER = ("0 18px 30px -14px rgba(15,23,42,.28)" if LIGHT else "0 22px 34px -14px rgba(0,0,0,.8)")
_css = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Source+Serif+4:opsz,wght@8..60,500;8..60,600;8..60,700&family=JetBrains+Mono:wght@400;500&display=swap');
:root {--bg:%(bg)s; --side:%(side)s; --panel:%(panel)s; --panel2:%(panel2)s; --border:%(border)s; --acc:%(acc)s;
       --acc2:%(acc2)s; --acc-rgb:%(acc_rgb)s; --h1:%(h1)s; --h2:%(h2)s; --text:%(text)s; --head:%(head)s;
       --sub:%(sub)s; --muted:%(muted)s; --lesson:%(lesson)s; --good:%(good)s; --bad:%(bad)s; --on-acc:%(on_acc)s;}
html, body, [class*="css"], .stMarkdown, p, label, li, input, textarea, button {font-family:'Inter', 'Segoe UI', sans-serif;}
.stApp {background: radial-gradient(1100px 520px at 0%% -10%%, rgba(var(--acc-rgb),.08), transparent 60%%),
                    radial-gradient(900px 480px at 100%% 0%%, rgba(var(--acc-rgb),.05), transparent 55%%), var(--bg);
        color:var(--text);}
.stApp p, .stApp li, .stApp label, .stApp [data-testid="stMarkdownContainer"], .stApp [data-testid="stCaptionContainer"],
.stApp [data-baseweb="select"] div, .stApp [data-baseweb="select"] span, .stApp input, .stApp [data-testid="stSliderTickBarMin"],
.stApp [data-testid="stSliderTickBarMax"], .stApp [data-testid="stSliderThumbValue"],
.stApp [data-testid="stExpander"] summary, .stApp [data-testid="stWidgetLabel"] {color:var(--text);}
.stApp [data-testid="stCaptionContainer"] {color:var(--muted);}
header[data-testid="stHeader"] {background:transparent;}
.block-container {padding-top: 1.6rem; max-width: 1360px;}
h1, h2, h3 {font-family:'Source Serif 4', Georgia, serif !important; font-weight:600 !important; letter-spacing:0;}
h2 {color:var(--head) !important; font-size:1.75rem !important;}
h3 {color:var(--sub) !important; font-size:1.3rem !important;}
section[data-testid="stSidebar"] {background:var(--side); border-right:1px solid var(--border);}
section[data-testid="stSidebar"] h3 {font-size:1.15rem !important;}
.stApp div[data-baseweb="select"] div, section[data-testid="stSidebar"] div[data-baseweb="select"] div,
div[data-baseweb="select"] > div, div[data-baseweb="input"] > div, .stNumberInput input, .stTextInput input,
div[data-baseweb="base-input"], div[data-baseweb="popover"] ul, div[data-baseweb="popover"] li,
div[data-baseweb="calendar"] {background:var(--panel) !important; border-color:var(--border) !important; color:var(--text);}
.stApp [data-testid="stSelectbox"] div[role], .stApp [data-testid="stMultiSelect"] div[role],
.stApp [data-testid="stDateInput"] div[data-rac], .stApp [data-testid="stNumberInput"] div[data-rac],
.stApp [data-testid="stTextInput"] div[data-rac], .stApp [data-testid="stNumberInput"] input,
.stApp [data-testid="stTextInput"] input, .stApp [data-testid="stDateInput"] input {
       background-color:var(--panel) !important; border-color:var(--border) !important; color:var(--text) !important;}
.stApp [data-testid="stDateInput"] div, .stApp [data-testid="stNumberInput"] div,
.stApp [data-testid="stNumberInput"] button, .stApp [data-testid="stTextInput"] div {
       background-color:var(--panel) !important; color:var(--text) !important; border-color:var(--border) !important;}
.stApp [data-testid="stSelectbox"] svg, .stApp [data-testid="stDateInput"] svg {color:var(--muted); fill:currentColor;}
[role="listbox"], [role="option"], [role="dialog"], div[data-rac][role="grid"] {background-color:var(--panel) !important;
       color:var(--text) !important;}
[role="option"]:hover, [role="option"][data-focused], [role="option"][aria-selected="true"] {
       background-color:rgba(var(--acc-rgb),.14) !important;}
/* ---- 3D depth ---- */
.hero {padding:28px 34px; border-radius:18px; margin-bottom:24px;
       background: linear-gradient(120deg, var(--h1) 0%%, var(--h2) 100%%);
       border:1px solid var(--border); border-top:3px solid var(--acc);
       box-shadow: %(sh)s; transform: perspective(1400px) rotateX(1.2deg); transform-origin: top center;}
.hero h1 {font-family:'Source Serif 4', Georgia, serif; font-weight:700; font-size:2.1rem; margin:0 0 8px 0; color:var(--head);}
.hero p {color:var(--text); opacity:.88; margin:0; font-size:1.03rem; line-height:1.55;}
.pill {display:inline-block; background:rgba(var(--acc-rgb),.10); border:1px solid rgba(var(--acc-rgb),.45);
       color:var(--sub); padding:3px 12px; border-radius:20px; font-size:.8rem; font-weight:500; margin:14px 6px 0 0;}
.pill.m, .pill.g, .pill.y {background:rgba(var(--acc-rgb),.04); border-color:var(--border); color:var(--text);}
.card, .kpi {transform-style: preserve-3d; transition: transform .25s ease, box-shadow .25s ease, border-color .25s ease;
       box-shadow: %(sh)s;}
.card {background:linear-gradient(180deg, var(--panel) 0%%, var(--panel2) 100%%); border:1px solid var(--border);
       border-radius:14px; padding:16px 18px; height:100%%;}
.card:hover, .kpi:hover {transform: perspective(900px) translateY(-4px) rotateX(3deg);
       border-color:var(--acc); box-shadow: %(shh)s, 0 0 0 1px rgba(var(--acc-rgb),.25);}
.card h4 {font-family:'Source Serif 4', Georgia, serif; font-weight:600; margin:0 0 6px 0; color:var(--acc2); font-size:1.05rem;}
.card p {margin:0; color:var(--text); opacity:.85; font-size:.92rem; line-height:1.5;}
.kpi {background:linear-gradient(180deg, var(--panel) 0%%, var(--panel2) 100%%); border:1px solid var(--border);
      border-radius:14px; padding:14px 18px; border-left:3px solid var(--acc);}
.kpi .v {font-family:'Inter', sans-serif; font-size:1.6rem; font-weight:700; color:var(--head); line-height:1.15;
         font-variant-numeric: tabular-nums;}
.kpi .l {font-size:.74rem; color:var(--muted); text-transform:uppercase; letter-spacing:.07em; margin-top:4px; font-weight:500;}
.owner {display:inline-block; font-size:.78rem; color:var(--sub); font-weight:600; letter-spacing:.04em;
        background:rgba(var(--acc-rgb),.10); border:1px solid rgba(var(--acc-rgb),.4); padding:3px 12px; border-radius:20px;
        margin-bottom:6px;}
.lesson {background:rgba(251,191,36,.08); border:1px solid rgba(217,119,6,.30); border-left:4px solid #F59E0B;
         padding:12px 16px; border-radius:10px; color:var(--lesson); font-size:.94rem; line-height:1.55; margin:8px 0;}
.good {background:rgba(16,185,129,.08); border:1px solid rgba(16,185,129,.30); border-left:4px solid #10B981;
       padding:12px 16px; border-radius:10px; color:var(--good); font-size:.94rem; line-height:1.55; margin:8px 0;}
.bad {background:rgba(239,68,68,.08); border:1px solid rgba(239,68,68,.30); border-left:4px solid #EF4444;
      padding:12px 16px; border-radius:10px; color:var(--bad); font-size:.94rem; line-height:1.55; margin:8px 0;}
.mono {font-family:'JetBrains Mono', ui-monospace, Menlo, monospace; font-size:.84rem; color:var(--acc2);}
.tid {color:var(--acc); font-size:.8rem;} .tpart {color:var(--muted); font-size:.85rem;}
div[data-testid="stTabs"] button p {font-family:'Inter', sans-serif; font-size:.95rem; font-weight:600;}
div[data-testid="stTabs"] button[aria-selected="true"] p {color:var(--acc);}
div[data-baseweb="tab-highlight"] {background:var(--acc) !important; height:3px;}
.stButton > button, .stDownloadButton > button {border-radius:10px; border:1px solid var(--acc); color:var(--acc);
       background:rgba(var(--acc-rgb),.07); font-weight:600; box-shadow:0 3px 0 rgba(0,0,0,.25);
       transition: transform .15s ease, box-shadow .15s ease, background .2s ease;}
.stButton > button p, .stDownloadButton > button p {color:inherit;}
.stButton > button:hover, .stDownloadButton > button:hover {background:rgba(var(--acc-rgb),.16);
       transform: translateY(-1px); box-shadow:0 5px 0 rgba(0,0,0,.25);}
.stButton > button:active {transform: translateY(2px); box-shadow:0 1px 0 rgba(0,0,0,.25);}
.stButton > button[kind="primary"] {background:linear-gradient(180deg, rgba(var(--acc-rgb),.98), rgba(var(--acc-rgb),.80));
       color:var(--on-acc); border:1px solid var(--acc);}
div[data-testid="stDataFrame"] {border:1px solid var(--border); border-radius:10px; box-shadow: %(sh)s;}
div[data-testid="stExpander"] {border:1px solid var(--border); border-radius:10px; background:var(--panel2);}
div[data-testid="stPlotlyChart"], div[data-testid="stGraphVizChart"] {background:var(--panel2); border:1px solid var(--border);
       border-radius:14px; padding:6px; box-shadow: %(sh)s;}
</style>""" % {**T, "sh": SH, "shh": SH_HOVER}

if LIGHT:   # tables are canvas-drawn in the app's dark base theme -> invert them for light pages
    _css += """<style>
div[data-testid="stDataFrame"] > div {filter: invert(1) hue-rotate(180deg) contrast(.92);}
div[data-testid="stCode"] pre {border:1px solid var(--border);}
</style>"""
if T["glow"]:   # neon: glows, animated rainbow edge, glowing numbers and cards
    _css += """<style>
.hero {background: linear-gradient(115deg, rgba(255,43,214,.20), rgba(157,78,221,.16) 45%, rgba(0,240,255,.20));
       border:1px solid rgba(0,240,255,.45); box-shadow: 0 0 28px rgba(0,240,255,.25), inset 0 0 40px rgba(255,43,214,.10);
       position:relative; overflow:hidden; border-top:1px solid rgba(0,240,255,.45);}
.hero:before {content:""; position:absolute; left:0; right:0; top:0; height:3px;
       background:linear-gradient(90deg,#FF2BD6,#00F0FF,#39FF88,#FF2BD6); background-size:300% 100%;
       animation:flow 8s linear infinite;}
@keyframes flow {0%{background-position:0% 0} 100%{background-position:300% 0}}
.hero h1 {text-shadow:0 0 10px #00F0FF, 0 0 26px rgba(255,43,214,.55);}
h2 {text-shadow:0 0 18px rgba(0,240,255,.35);} h3 {text-shadow:0 0 12px rgba(0,240,255,.4);}
.pill {box-shadow:0 0 10px rgba(0,240,255,.35); color:#7DF9FF;}
.pill.m {border-color:#FF2BD6; color:#FF7BE5; box-shadow:0 0 10px rgba(255,43,214,.35);}
.pill.g {border-color:#39FF88; color:#8CFFC0; box-shadow:0 0 10px rgba(57,255,136,.35);}
.pill.y {border-color:#FFC300; color:#FFE07A; box-shadow:0 0 10px rgba(255,195,0,.35);}
.kpi {border-top:2px solid #00F0FF; border-left:1px solid var(--border); box-shadow:0 0 16px rgba(0,240,255,.15);}
.kpi:nth-child(odd) {border-top-color:#FF2BD6;}
.kpi .v {text-shadow:0 0 12px rgba(0,240,255,.6);}
.card h4 {text-shadow:0 0 8px rgba(255,43,214,.5);}
.card:hover, .kpi:hover {border-color:#FF2BD6; box-shadow:0 0 24px rgba(255,43,214,.4);}
.owner {box-shadow:0 0 12px rgba(0,240,255,.4);}
div[data-baseweb="tab-highlight"] {background:linear-gradient(90deg,#FF2BD6,#00F0FF) !important; box-shadow:0 0 12px #00F0FF;}
div[data-testid="stTabs"] button[aria-selected="true"] p {text-shadow:0 0 10px rgba(0,240,255,.7);}
.stButton > button[kind="primary"] {background:linear-gradient(90deg,#FF2BD6,#9D4EDD 50%,#00F0FF); color:#fff;
       border:none; box-shadow:0 0 18px rgba(255,43,214,.45);}
.stButton > button:hover {box-shadow:0 0 18px rgba(0,240,255,.55);}
div[data-testid="stPlotlyChart"], div[data-testid="stGraphVizChart"] {box-shadow:0 0 18px rgba(0,240,255,.12);}
</style>"""
st.markdown(_css, unsafe_allow_html=True)

from streamlit.delta_generator import DeltaGenerator  # noqa: E402

if not getattr(DeltaGenerator.plotly_chart, "_themed", False):
    _orig_plotly_chart = DeltaGenerator.plotly_chart

    def _themed_plotly_chart(self, fig, *args, **kwargs):
        """Use our own Plotly template instead of Streamlit's (which follows the app's dark base)."""
        kwargs.setdefault("theme", None)
        return _orig_plotly_chart(self, fig, *args, **kwargs)

    _themed_plotly_chart._themed = True
    DeltaGenerator.plotly_chart = _themed_plotly_chart

# `st.plotly_chart` is bound to the main container at import time, so patch it too
st.plotly_chart = lambda fig, *a, **k: DeltaGenerator.plotly_chart(st._main, fig, *a, **k)


def gv(dot: str, **kw):
    """Graphviz with theme colors swapped in."""
    for a, b in {"#111A2E": T["panel"], "#2B3D63": T["border"], "#1C2340": T["h2"], "#38BDF8": T["acc"],
                 "#E8EEF7": T["text"], "#7DD3FC": T["acc2"], "#B6C2D6": T["muted"], "#64748B": T["muted"],
                 "#C084FC": MAGENTA, "#34D399": GREEN, "#FBBF24": AMBER}.items():
        dot = dot.replace(a, b)
    # natural size: a 'stretch' graph inside a hidden tab is measured at 0 px and renders blank
    kw.pop("width", None)
    st.graphviz_chart(dot, width="content")


# ---------------------------------------------------------------- bootstrap
@st.cache_resource(show_spinner=False)
def bootstrap() -> bool:
    """First launch (e.g. on Streamlit Cloud): build everything from scratch."""
    from src.pipeline import run_all
    from src.train import registry_versions
    try:
        from src.predict import _load_cached, champion_info
        from src.train import setup_mlflow
        ready = bool(mlflow_db_path().exists() and versioning.list_versions()
                     and FeatureStore().offline_metadata())
        if ready:   # champion must actually load (artifacts present)
            _load_cached(setup_mlflow(), str(champion_info()["version"]))
    except Exception:
        ready = False
    if not ready:
        new_mlflow_db()   # fresh tracking DB (an old engine may still point at a deleted file)
        run_all()
    return True


with st.spinner("First launch: versioning data, building the feature store and training models (~40 s)…"):
    bootstrap()

from src.predict import batch_predict, champion_info, predict, risk_grid  # noqa: E402
from src.data_generation import AIRPORTS as _AP  # noqa: E402
from src.train import (MODEL_TYPES, build_dataset, evaluate, feature_columns, build_pipeline,  # noqa: E402
                       leakage_experiment, promote, register_run, registry_versions, runs_dataframe,
                       time_split, train_and_log)

store = FeatureStore()


@st.cache_data(show_spinner=False)
def load_version(tag: str) -> pd.DataFrame:
    return versioning.checkout(tag)


def kpi(col, value, label):
    col.markdown(f'<div class="kpi"><div class="v">{value}</div><div class="l">{label}</div></div>',
                 unsafe_allow_html=True)


def owner(i: int):
    m = TEAM[i]
    who = m["name"] + (f" · {m['id']}" if m["id"] else "")
    st.markdown(f'<div class="owner">{m["member"]} · {who}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.selectbox("Theme", list(THEMES), key="theme",
                 help="Dark, Neon Glow, or Light backgrounds (Light · Classic is best for projectors)")
    st.markdown("### Flight Delay MLOps")
    st.caption("DSE ZG529 · Data Management for ML\nExperiential Learning Assignment")
    st.markdown("**Team**")
    for m in TEAM:
        st.markdown(f"**{m['member']}** — {m['name']}  \n<span class='tid'>{m['id']}</span> · <span class='tpart'>{m['part']}</span>",
                    unsafe_allow_html=True)
    st.divider()
    try:
        ch = champion_info()
        st.markdown("**Live pipeline status**")
        st.markdown(f"Data versions: **{len(versioning.list_versions())}**")
        st.markdown(f"Feature store built from: **{store.offline_metadata().get('data_version')}**")
        st.markdown(f"Champion: **v{ch['version']}** · {ch['model_type']} · AUC **{ch['roc_auc']:.3f}**")
    except Exception as e:
        st.warning(f"Pipeline not ready: {e}")
    st.divider()
    _admin_pw = None
    try:
        _admin_pw = st.secrets.get("ADMIN_PASSWORD")
    except Exception:
        _admin_pw = os.environ.get("ADMIN_PASSWORD")
    with st.expander("Admin"):
        if not _admin_pw:
            st.caption("Destructive actions are disabled on the public app. Set an `ADMIN_PASSWORD` secret to enable.")
        else:
            entered = st.text_input("Admin password", type="password", max_chars=128, key="admin_pw")
            if st.button("Rebuild entire pipeline", width="stretch",
                         help="Deletes versions, feature store and MLflow DB, then reruns every stage"):
                if not check_admin(entered, _admin_pw):
                    st.error("Incorrect admin password.")
                else:
                    ok, left = cooldown_ok(st.session_state, "rebuild", 60)
                    if not ok:
                        st.warning(f"Please wait {left:.0f}s before rebuilding again.")
                    else:
                        import shutil
                        from src.config import DATA_DIR, MLFLOW_ARTIFACTS
                        for p in [DATA_DIR / ".versions", DATA_DIR / "versions", DATA_DIR / "feature_store",
                                  MLFLOW_ARTIFACTS]:
                            shutil.rmtree(p, ignore_errors=True)
                        new_mlflow_db()   # fresh tracking DB; never delete a DB an engine still has open
                        st.cache_resource.clear(); st.cache_data.clear()
                        st.rerun()

st.markdown("""
<div class="hero">
  <h1>Flight Delay Prediction — an End-to-End MLOps Pipeline</h1>
  <p>Predicting whether an Indian domestic flight departs &gt;15 min late, with every stage of the
  data &amp; model lifecycle made reproducible, traceable and servable.</p>
  <span class="pill">DVC-style data versioning</span><span class="pill m">Feature store · point-in-time joins</span>
  <span class="pill g">MLflow tracking &amp; registry</span><span class="pill y">Prediction service · lineage</span>
</div>""", unsafe_allow_html=True)

tabs = st.tabs(["Overview", "① Data Versioning", "② Feature Store",
                "③ Experiments & Registry", "④ Prediction Service"])

# ================================================================ OVERVIEW
with tabs[0]:
    versions = versioning.list_versions()
    runs = runs_dataframe()
    c = st.columns(4)
    kpi(c[0], len(versions), "dataset versions")
    kpi(c[1], len(STORE_FEATURES), "store features · 3 views")
    kpi(c[2], len(runs), "MLflow runs logged")
    kpi(c[3], f"{champion_info()['roc_auc']:.3f}", "champion ROC-AUC")
    st.write("")
    left, right = st.columns([1.35, 1])
    with left:
        st.subheader("Architecture")
        gv("""
        digraph G { rankdir=LR; bgcolor="transparent"; node [shape=box style="rounded,filled" fontname="Inter" fontsize=11 color="#2B3D63" fillcolor="#111A2E" fontcolor="#E8EEF7"];
          edge [color="#64748B" fontcolor="#B6C2D6"];
          src [label="Source systems\\n(flight ops extracts)" fillcolor="#1C2340" color="#C084FC"];
          subgraph cluster1 {label="① Data Versioning" style="rounded,dashed" color="#38BDF8" fontcolor="#38BDF8" fontname="Inter";
            dvc [label="Content-addressed cache\\nMD5 → bytes"]; meta [label=".dvc pointers in Git\\nv1.0 → v1.1 → v2.0 → v3.0"]; }
          subgraph cluster2 {label="② Feature Store" style="rounded,dashed" color="#34D399" fontcolor="#34D399" fontname="Inter";
            off [label="Offline store (Parquet)\\nfull feature history"]; on [label="Online store (SQLite)\\nlatest value per entity"]; }
          subgraph cluster3 {label="③ MLflow" style="rounded,dashed" color="#FBBF24" fontcolor="#FBBF24" fontname="Inter";
            exp [label="Experiment tracking\\nparams · metrics · artifacts"]; reg [label="Model registry\\n@champion alias"]; }
          subgraph cluster4 {label="④ Serving" style="rounded,dashed" color="#C084FC" fontcolor="#C084FC" fontname="Inter";
            api [label="Prediction service\\nFastAPI / Streamlit"]; }
          user [label="Client / ops dashboard" fillcolor="#1C2340" color="#C084FC"];
          src -> dvc -> meta; dvc -> off [label=" point-in-time\\n training set" fontsize=9];
          off -> on [label=" materialize" fontsize=9]; off -> exp -> reg; reg -> api; on -> api; api -> user;
        }""", width="stretch")
    with right:
        st.subheader("Who does what")
        for i, m in enumerate(TEAM):
            st.markdown(f'<div class="card" style="margin-bottom:10px"><h4>{["①","②","③","④"][i]} {m["part"]}</h4>'
                        f'<p>{m["member"]} — {m["name"]} · <span class="mono">{m["id"]}</span></p></div>', unsafe_allow_html=True)
    st.subheader("Route network — 3D globe")
    g1, g2 = st.columns([1.6, 1])
    with g1:
        rs = store.online_table("route_stats")
        fig = go.Figure()
        vmax = max(rs["route_delay_rate_30d"].max(), 1e-6)
        for r in rs.itertuples():
            o, d = r.route.split("-")
            rate = r.route_delay_rate_30d
            col = RED if rate >= .42 else (AMBER if rate >= .32 else T["acc"])
            fig.add_trace(go.Scattergeo(lon=[_AP[o][1], _AP[d][1]], lat=[_AP[o][0], _AP[d][0]], mode="lines",
                                        line=dict(width=0.6 + 3.2 * rate / vmax, color=col), opacity=.75,
                                        hoverinfo="text", text=f"{r.route}: {rate:.0%} delayed (30d)", showlegend=False))
        fig.add_trace(go.Scattergeo(lon=[v[1] for v in _AP.values()], lat=[v[0] for v in _AP.values()],
                                    text=list(_AP), mode="markers+text", textposition="top center",
                                    marker=dict(size=9, color=T["head"], line=dict(width=2, color=T["acc"])),
                                    textfont=dict(color=T["head"], size=12), showlegend=False, hoverinfo="text"))
        fig.update_geos(projection_type="orthographic", projection_rotation=dict(lon=79, lat=18),
                        projection_scale=2.1, showland=True, landcolor=T["h2"], showocean=True,
                        oceancolor=T["bg"], showcountries=True, countrycolor=T["border"], coastlinecolor=T["muted"],
                        showlakes=False, bgcolor="rgba(0,0,0,0)", showframe=False)
        fig.update_layout(**{**PLOT, "margin": dict(l=0, r=0, t=10, b=0)}, height=430, dragmode="pan")
        st.plotly_chart(fig, width="stretch")
    with g2:
        st.markdown('<div class="card"><h4>Reading the globe</h4><p>Each arc is a route served by the pipeline. '
                    'Thickness and colour show its <b>30-day delay rate</b> straight from the <b>online feature '
                    'store</b> — teal &lt; 32%, amber 32–42%, red &gt; 42%. Drag to rotate, scroll to zoom.</p></div>',
                    unsafe_allow_html=True)
        top = rs.sort_values("route_delay_rate_30d", ascending=False).head(5)
        st.write("")
        top = top.assign(pct=(top["route_delay_rate_30d"] * 100).round(1))
        st.markdown("**Most delay-prone routes**")
        st.dataframe(top[["route", "pct"]], hide_index=True, width="stretch",
                     column_config={"pct": st.column_config.ProgressColumn("30-day delay rate", min_value=0,
                                                                           max_value=100, format="%.0f%%")})
    st.subheader("Why this pipeline matters")
    a, b, c3 = st.columns(3)
    a.markdown('<div class="card"><h4>Reproducibility</h4><p>"Which data trained the model in production?" '
               'Every run is pinned to a dataset hash, so any result can be rebuilt bit-for-bit.</p></div>',
               unsafe_allow_html=True)
    b.markdown('<div class="card"><h4>No training/serving skew</h4><p>The same feature definitions feed '
               'training (offline, point-in-time) and inference (online), eliminating silent leakage.</p></div>',
               unsafe_allow_html=True)
    c3.markdown('<div class="card"><h4>Governed promotion</h4><p>Models move to production through a '
                'registry alias, carrying their run, metrics and data lineage with them.</p></div>',
                unsafe_allow_html=True)
    with st.expander("Syllabus mapping (DSE ZG529 handout)"):
        st.markdown("""
| Pipeline stage | Contact sessions | Handout lab |
|---|---|---|
| Data versioning, lineage, reproducibility | CS 9 *Analytics Engineering* · CS 6 *ML lifecycle* | Exp 4 |
| Feature engineering, feature store, leakage | CS 11 *Feature Preparations* · CS 8 *Data Leakage* | Exp 4 |
| Experiment tracking, metadata, registry | CS 12 *ML Experimentation & Metadata* | Exp 3 |
| Prediction service, lineage, monitoring hooks | CS 5 *Data Serving* · CS 16 *Observability* | Exp 4 |
""")

# ================================================================ 1. DATA VERSIONING
with tabs[1]:
    owner(0)
    st.header("① Data Versioning — every dataset is immutable and addressable")
    st.markdown('<div class="lesson"><b>Problem:</b> data files get overwritten in place ("final_v2_new.csv"). '
                'Months later nobody can say which data trained the model in production, so results cannot be '
                'reproduced or audited. <b>Fix:</b> hash the content, store each version once, and track a tiny '
                'pointer file in Git — exactly how DVC works.</div>', unsafe_allow_html=True)
    versions = versioning.list_versions()
    vdf = pd.DataFrame([{"tag": m["tag"], "md5": m["outs"][0]["md5"][:10] + "…", "rows": m["rows"],
                         "columns": len(m["columns"]), "size (KB)": round(m["outs"][0]["size"] / 1024),
                         "from": m["date_range"][0][:10], "to": m["date_range"][1][:10],
                         "parent": m["parent"] or "—", "commit message": m["message"]} for m in versions])
    l, r = st.columns([2.2, 1])
    with l:
        st.subheader("Version history")
        st.dataframe(vdf, hide_index=True, width="stretch")
    with r:
        st.subheader("Lineage")
        gv('digraph{rankdir=TB;bgcolor=transparent;node[shape=box style="rounded,filled" fillcolor="#111A2E" color="#38BDF8" fontcolor="#E8EEF7" fontname=Inter fontsize=11];'
                          + "".join(f'"{m["tag"]}"[label="{m["tag"]}\\nmd5 {m["outs"][0]["md5"][:8]}\\n{m["rows"]:,} rows"];' for m in versions)
                          + 'edge[color="#C084FC"];' + "".join(f'"{m["parent"]}"->"{m["tag"]}";' for m in versions if m["parent"]) + "}")

    st.subheader("Checkout & compare two versions")
    tags = [m["tag"] for m in versions]
    c1, c2 = st.columns(2)
    ta = c1.selectbox("Version A", tags, index=0)
    tb = c2.selectbox("Version B", tags, index=len(tags) - 1)
    d = versioning.diff(ta, tb)
    k = st.columns(6)
    kpi(k[0], f"{d['rows_b'] - d['rows_a']:+,}", "row change")
    kpi(k[1], f"{d['dup_a']}→{d['dup_b']}", "duplicate rows")
    kpi(k[2], f"{d['null_a']}→{d['null_b']}", "null cells")
    kpi(k[3], ", ".join(d["added_columns"]) or "none", "columns added")
    kpi(k[4], ", ".join(d["carriers_added"]) or "none", "new carriers")
    kpi(k[5], f"{d['delay_rate_a']:.1%}→{d['delay_rate_b']:.1%}", "delay rate")
    A, B = load_version(ta), load_version(tb)
    g1, g2 = st.columns(2)
    ma = A.groupby(A["scheduled_departure"].dt.month)["is_delayed"].mean().rename(ta)
    mb = B.groupby(B["scheduled_departure"].dt.month)["is_delayed"].mean().rename(tb)
    mm = pd.concat([ma, mb], axis=1).reset_index(names="month").melt("month", var_name="version", value_name="delay rate")
    fig = px.line(mm, x="month", y="delay rate", color="version", markers=True,
                  color_discrete_sequence=[GREY, TEAL], title="Monthly delay rate by version")
    fig.update_layout(**PLOT, yaxis_tickformat=".0%"); g1.plotly_chart(fig, width="stretch")
    wx = pd.concat([A[["weather_index"]].assign(version=ta), B[["weather_index"]].assign(version=tb)])
    fig = px.histogram(wx, x="weather_index", color="version", barmode="overlay", nbins=50, histnorm="probability",
                       color_discrete_sequence=[GREY, TEAL], title="Weather index distribution")
    fig.update_layout(**PLOT); g2.plotly_chart(fig, width="stretch")
    with st.expander(f"Preview checked-out data · {tb}"):
        st.dataframe(B.head(200), hide_index=True, width="stretch")

    st.subheader("Experiment: what breaks without versioning?")
    st.caption("Train the same model twice. Between the two runs, someone 'fixes' a few rows in place.")
    e1, e2 = st.columns([1, 2])
    with e1:
        rep_tag = st.selectbox("Dataset version", tags, index=tags.index("v1.1"), key="rep")
        n_edit = st.slider("Rows silently edited in place", 0, 2000, 400, step=100)
        go_rep = st.button("Run reproducibility test", type="primary")
    if go_rep and not cooldown_ok(st.session_state, "repro", 5)[0]:
        st.warning("Please wait a moment before re-running the test.")
        go_rep = False
    if go_rep:
        base = load_version(rep_tag).drop_duplicates("flight_id").copy()
        base["weather_index"] = base["weather_index"].fillna(base["weather_index"].median())

        def quick_auc(df):
            cat, num = BASE_CATEGORICAL, BASE_NUMERIC
            tr, te = time_split(df)
            m = build_pipeline("Logistic Regression", {}, False).fit(tr[cat + num], tr[TARGET])
            return evaluate(m, te[cat + num], te[TARGET])["roc_auc"]

        rng = np.random.default_rng(0)
        edited = base.copy()
        idx = rng.choice(len(edited), n_edit, replace=False)
        edited.loc[edited.index[idx], "is_delayed"] = 1 - edited.loc[edited.index[idx], "is_delayed"]
        h0, h1, h2 = versioning.md5_of(base), versioning.md5_of(base), versioning.md5_of(edited)
        r0, r1, r2 = quick_auc(base), quick_auc(base), quick_auc(edited)
        with e2:
            res = pd.DataFrame({"run": ["Run 1 (checkout " + rep_tag + ")", "Run 2 (checkout " + rep_tag + ")",
                                        "Run 2 (file edited in place)"],
                                "dataset md5": [h0[:12], h1[:12], h2[:12]], "ROC-AUC": [r0, r1, r2]})
            st.dataframe(res, hide_index=True, width="stretch")
            if n_edit:
                st.markdown(f'<div class="bad">Without versioning, run 2 reports AUC <b>{r2:.4f}</b> instead of '
                            f'<b>{r0:.4f}</b> and nobody can tell why. With hashing, the changed md5 '
                            f'(<span class="mono">{h2[:12]}</span>) flags it instantly.</div>', unsafe_allow_html=True)
            st.markdown('<div class="good">Same version ⇒ same md5 ⇒ identical metric. Reproducibility is '
                        'guaranteed by content addressing, not by discipline.</div>', unsafe_allow_html=True)
    with st.expander("Equivalent DVC CLI workflow (dvc.yaml is in the repo)"):
        st.code("""dvc init
dvc add data/raw/flights.csv            # hash + cache, writes flights.csv.dvc
git add data/raw/flights.csv.dvc .gitignore && git commit -m "data v1.0"
git tag data-v1.0
dvc remote add -d storage s3://bucket/dvc   # or gdrive / azure blob
dvc push                                 # upload data, Git keeps only the pointer
git checkout data-v1.0 && dvc checkout   # time-travel to an exact dataset
dvc repro                                # rerun data → features → train stages
dvc diff data-v1.0 data-v2.0""", language="bash")

# ================================================================ 2. FEATURE STORE
with tabs[2]:
    owner(1)
    st.header("② Feature Store — one definition, two stores, zero leakage")
    st.markdown('<div class="lesson"><b>Problem:</b> features are re-implemented in training notebooks and again '
                'in the serving code, and training joins often use information that would not exist at prediction '
                'time. Models look great offline and fail in production. <b>Fix:</b> define features once, keep an '
                'offline history for point-in-time training sets and an online store for low-latency serving.</div>',
                unsafe_allow_html=True)
    cols = st.columns(3)
    for col, (name, fv) in zip(cols, FEATURE_VIEWS.items()):
        col.markdown(f'<div class="card"><h4>{name}</h4><p><b>entity:</b> {fv.entity} · <b>TTL:</b> '
                     f'{fv.ttl_days} d<br>{fv.description}<br><span class="mono">{"<br>".join(fv.features)}</span>'
                     f'</p></div>', unsafe_allow_html=True)
    st.write("")
    st.subheader("Offline store — feature history")
    o1, o2, o3 = st.columns([1, 1, 2])
    view = o1.selectbox("Feature view", list(FEATURE_VIEWS))
    fv = FEATURE_VIEWS[view]
    hist = store.read_offline(view)
    ents = sorted(hist[fv.entity].unique())
    chosen = o2.multiselect(f"{fv.entity}(s)", ents, default=ents[:3])
    feat = o3.selectbox("Feature", fv.features)
    hs = hist[hist[fv.entity].isin(chosen)]
    fig = px.line(hs, x="event_timestamp", y=feat, color=fv.entity, title=f"{feat} over time",
                  color_discrete_sequence=[TEAL, MAGENTA, GREEN, AMBER, GREY, RED])
    fig.update_layout(**PLOT); st.plotly_chart(fig, width="stretch")

    st.subheader("Point-in-time join vs. naive join")
    st.caption("For one flight, which feature value is the model allowed to see?")
    pit_df, leaky_df = build_dataset("v2.0"), build_dataset("v2.0", leaky=True)
    p1, p2 = st.columns([1, 2])
    with p1:
        ap = st.selectbox("Origin airport", list(AIRPORTS), index=0, key="pit_ap")
        day = st.date_input("Flight date", pd.Timestamp("2025-08-14"), min_value=pd.Timestamp("2025-01-10"),
                            max_value=pd.Timestamp("2025-09-30"))
        sub = pit_df[(pit_df["origin"] == ap) & (pit_df["scheduled_departure"].dt.date == day)]
        if sub.empty:
            st.info("No flights from this airport on that date — pick another.")
        else:
            fid = st.selectbox("Flight", sub["flight_id"], format_func=lambda f: f"{f} · "
                               + sub.set_index("flight_id").loc[f, "scheduled_departure"].strftime("%H:%M"))
    if not sub.empty:
        row_p = pit_df.set_index("flight_id").loc[fid]
        row_l = leaky_df.set_index("flight_id").loc[fid]
        with p2:
            comp = pd.DataFrame({"feature": STORE_FEATURES,
                                 "point-in-time (as of 00:00 that day)": [row_p[f] for f in STORE_FEATURES],
                                 "naive join (includes same day)": [row_l[f] for f in STORE_FEATURES]})
            st.dataframe(comp.style.format(precision=3), hide_index=True, width="stretch")
            same_day = leaky_df[(leaky_df["origin"] == ap) & (leaky_df["scheduled_departure"].dt.date == day)]
            st.markdown(f'Flight departs <b>{row_p["scheduled_departure"]:%H:%M}</b> and was actually '
                        f'<b>{"DELAYED" if row_p[TARGET] else "ON TIME"}</b>. The naive <span class="mono">'
                        f'origin_delay_rate_1d</span> = <b>{row_l["origin_delay_rate_1d"]:.2f}</b> is the share of '
                        f'all {len(same_day)} flights from {ap} that day that were delayed — including flights later '
                        f'that evening <i>and this flight\'s own label</i>.', unsafe_allow_html=True)
            tl = same_day.sort_values("scheduled_departure")
            fig = go.Figure()
            fig.add_vrect(x0=pd.Timestamp(day) - pd.Timedelta(hours=2), x1=pd.Timestamp(day),
                          fillcolor=GREEN, opacity=.12, line_width=0, annotation_text="features available")
            fig.add_trace(go.Scatter(x=tl["scheduled_departure"], y=tl[TARGET], mode="markers",
                                     marker=dict(size=11, color=[RED if v else GREEN for v in tl[TARGET]]),
                                     name="same-day outcomes", hovertext=tl["flight_id"]))
            fig.add_vline(x=row_p["scheduled_departure"], line_dash="dash", line_color=NAVY)
            fig.add_annotation(x=row_p["scheduled_departure"], y=1.15, text="prediction time", showarrow=False)
            fig.update_layout(**PLOT, height=230, yaxis=dict(tickvals=[0, 1], ticktext=["on time", "delayed"]),
                              title="Outcomes the naive join leaks (red = delayed)", showlegend=False)
            st.plotly_chart(fig, width="stretch")

    st.subheader("Experiment: the leakage trap")
    if st.button("Train with naive join vs. point-in-time join", type="primary"):
        ok, left = cooldown_ok(st.session_state, "leak", 15)
        if not ok:
            st.warning(f"Please wait {left:.0f}s before running the experiment again.")
        elif len(runs_dataframe()) >= MAX_RUNS:
            st.warning(f"Run limit reached ({MAX_RUNS}). Ask an admin to rebuild the pipeline.")
        else:
            with st.spinner("Training both models…"):
                st.session_state["leak"] = leakage_experiment("v2.0")
    if "leak" in st.session_state:
        res = st.session_state["leak"]
        bars = pd.DataFrame({"scenario": ["Naive join · offline test", "Naive join · in production",
                                          "Point-in-time · offline test"],
                             "ROC-AUC": [res["leaky_offline"]["roc_auc"], res["leaky_in_production"]["roc_auc"],
                                         res["pit_offline"]["roc_auc"]]})
        lc, rc = st.columns([1.4, 1])
        fig = px.bar(bars, x="scenario", y="ROC-AUC", text="ROC-AUC", color="scenario",
                     color_discrete_sequence=[AMBER, RED, GREEN], title="Offline promise vs. production reality")
        fig.update_layout(**PLOT, showlegend=False, yaxis_range=[0.5, max(bars["ROC-AUC"]) + .05])
        lc.plotly_chart(fig, width="stretch")
        drop = res["leaky_offline"]["roc_auc"] - res["leaky_in_production"]["roc_auc"]
        rc.markdown(f'<div class="bad"><b>Naive model:</b> looks like AUC {res["leaky_offline"]["roc_auc"]:.3f} '
                    f'offline, but in production (where same-day outcomes don\'t exist yet) it drops by '
                    f'<b>{drop:.3f}</b> to {res["leaky_in_production"]["roc_auc"]:.3f}.</div>'
                    f'<div class="good"><b>Point-in-time model:</b> {res["pit_offline"]["roc_auc"]:.3f} offline — '
                    f'an honest estimate, and better in production than the leaky one.</div>'
                    '<div class="lesson">The run was logged to MLflow tagged <span class="mono">leakage_safe=false, '
                    'stage=rejected</span> so it can never be promoted by mistake.</div>', unsafe_allow_html=True)

    st.subheader("Online store — materialize & look up")
    m1, m2 = st.columns([1, 2])
    with m1:
        as_of = st.date_input("Materialize as of", pd.Timestamp("2025-10-01"),
                              min_value=pd.Timestamp("2025-01-05"), max_value=pd.Timestamp("2025-10-01"))
        if st.button("Materialize to online store"):
            ok, left = cooldown_ok(st.session_state, "materialize")
            if not ok:
                st.warning(f"Please wait {left:.0f}s.")
            else:
                counts = store.materialize(pd.Timestamp(as_of))
                st.success(f"Pushed latest values: {counts}")
        st.markdown("**Lookup**")
        lo = st.selectbox("Origin", list(AIRPORTS), key="lo")
        ld = st.selectbox("Destination", [a for a in AIRPORTS if a != lo], key="ld")
        lcar = st.selectbox("Carrier", list(CARRIERS) + [NEW_CARRIER[0]], key="lc")
    with m2:
        import time as _t
        t0 = _t.perf_counter()
        feats = store.get_online_features(lo, ld, lcar)
        ms = (_t.perf_counter() - t0) * 1000
        st.dataframe(pd.DataFrame([{"feature": k, "value": round(float(v), 4) if v is not None else None} for k, v in feats.items() if not k.startswith("_")]),
                     hide_index=True, width="stretch")
        st.caption(f"Lookup latency {ms:.1f} ms · values as of "
                   f"{', '.join(f'{k[1:-3]}={v}' for k, v in feats.items() if k.startswith('_'))}")
        st.markdown('<div class="good">Training (offline, point-in-time) and serving (online) use the <b>same '
                    'feature definitions</b>, so there is no training/serving skew.</div>', unsafe_allow_html=True)

# ================================================================ 3. EXPERIMENTS
with tabs[3]:
    owner(2)
    st.header("③ Experiment Tracking & Model Registry — MLflow")
    st.markdown('<div class="lesson"><b>Problem:</b> "the model from last Tuesday was better" — but which code, '
                'which hyper-parameters, which data? <b>Fix:</b> every run logs parameters, metrics, artifacts, the '
                'model itself and the exact dataset hash; promotion to production happens through a registry '
                'alias, never by copying files.</div>', unsafe_allow_html=True)
    st.subheader("Launch a tracked experiment")
    f1, f2, f3 = st.columns(3)
    with f1:
        t_tag = st.selectbox("Data version", [m["tag"] for m in versioning.list_versions()], index=2)
        t_model = st.selectbox("Model", list(MODEL_TYPES), index=2)
        t_store = st.toggle("Use feature-store features", value=True)
    with f2:
        params = {}
        if t_model == "Logistic Regression":
            params["C"] = st.select_slider("C (inverse regularization)", [0.01, 0.1, 0.5, 1.0, 5.0, 10.0], 1.0)
        elif t_model == "Random Forest":
            params["n_estimators"] = st.slider("n_estimators", 50, 400, 150, 50)
            params["max_depth"] = st.slider("max_depth", 3, 25, 12)
            params["min_samples_leaf"] = st.slider("min_samples_leaf", 1, 50, 5)
        else:
            params["learning_rate"] = st.select_slider("learning_rate", [0.01, 0.03, 0.05, 0.08, 0.1, 0.2, 0.3], 0.08)
            params["max_depth"] = st.slider("max_depth", 2, 12, 6)
            params["max_iter"] = st.slider("max_iter", 50, 500, 200, 50)
    with f3:
        t_author = st.selectbox("Logged by", [m["name"] for m in TEAM])
        t_name = st.text_input("Run name", f"{t_model} | {t_tag}", max_chars=60,
                               help="Letters, numbers and _ . , : | ( ) - only")
        if st.button("Train & log to MLflow", type="primary", width="stretch"):
            ok, left = cooldown_ok(st.session_state, "train", 10)
            if not ok:
                st.warning(f"Please wait {left:.0f}s before training again.")
            elif len(runs_dataframe()) >= MAX_RUNS:
                st.warning(f"Run limit reached ({MAX_RUNS}). Ask an admin to rebuild the pipeline.")
            else:
                with st.spinner("Training…"):
                    st.session_state["last_run"] = train_and_log(t_tag, t_model, params, t_store,
                                                                 sanitize_text(t_name), t_author)
    if "last_run" in st.session_state:
        r = st.session_state["last_run"]
        k = st.columns(6)
        for col, m in zip(k, ["roc_auc", "f1", "precision", "recall", "accuracy", "train_time_s"]):
            kpi(col, r[m], m)
        st.caption(f"run_id `{r['run_id']}` · model `{r['model_uri']}`")

    runs = runs_dataframe()
    if not runs.empty:
        st.subheader("Run history")
        show = pd.DataFrame({
            "run": runs["tags.mlflow.runName"], "data": runs["params.data_version"],
            "model": runs["params.model_type"], "feature store": runs.get("params.use_feature_store"),
            "ROC-AUC": runs["metrics.roc_auc"], "F1": runs.get("metrics.f1"), "recall": runs.get("metrics.recall"),
            "leakage safe": runs.get("tags.leakage_safe"), "author": runs.get("tags.author"),
            "started": runs["start_time"].dt.strftime("%H:%M:%S"), "run_id": runs["run_id"]})
        st.dataframe(show, hide_index=True, width="stretch",
                     column_config={"ROC-AUC": st.column_config.ProgressColumn(min_value=0.5, max_value=0.85, format="%.4f")})
        cc1, cc2 = st.columns(2)
        fig = px.scatter(show.dropna(subset=["F1"]), x="ROC-AUC", y="F1", color="model", symbol="data",
                         hover_name="run", size_max=14, title="Run comparison", color_discrete_sequence=[TEAL, AMBER, RED])
        fig.update_traces(marker=dict(size=13, line=dict(width=1, color="white")))
        fig.update_layout(**PLOT); cc1.plotly_chart(fig, width="stretch")
        same = show[show["data"] == "v2.0"].dropna(subset=["F1"])
        if same.empty:
            cc2.info("No runs on v2.0 yet.")
        else:
            fig = px.bar(same, x="run", y="ROC-AUC", color="model", title="Like-for-like: runs on v2.0 only",
                         color_discrete_sequence=[TEAL, MAGENTA, GREEN], text_auto=".3f")
            fig.update_layout(**PLOT, yaxis_range=[0.5, same["ROC-AUC"].max() + .05], xaxis_title="")
            cc2.plotly_chart(fig, width="stretch")
        st.markdown('<div class="lesson">Metrics are only comparable on the <b>same data version</b> (same '
                    'holdout window) — another reason the data version is logged with every run.</div>',
                    unsafe_allow_html=True)

        st.subheader("Model registry")
        rg1, rg2 = st.columns([1.6, 1])
        reg = registry_versions()
        with rg1:
            if not reg.empty:
                st.dataframe(reg, hide_index=True, width="stretch")
        with rg2:
            safe = show[show["leakage safe"] == "true"]
            if safe.empty:
                st.info("No leakage-safe runs to register yet.")
            else:
                pick = st.selectbox("Register a run", safe["run_id"],
                                    format_func=lambda x: f"{safe.set_index('run_id').loc[x, 'run']} · "
                                    f"AUC {safe.set_index('run_id').loc[x, 'ROC-AUC']:.3f}")
                promote_now = st.checkbox(f"Promote to @{CHAMPION_ALIAS}", value=False)
                if st.button("Register model version", width="stretch"):
                    ok, left = cooldown_ok(st.session_state, "register")
                    if not ok:
                        st.warning(f"Please wait {left:.0f}s.")
                    else:
                        out = register_run(pick, promote=promote_now)
                        st.success(f"Registered {REGISTERED_MODEL} v{out['version']}"
                                   + (f" and promoted to @{CHAMPION_ALIAS}" if promote_now else ""))
                        st.rerun()
            if not reg.empty:
                pv = st.selectbox("Or move @champion to version", reg["version"].tolist())
                if st.button("Promote", width="stretch"):
                    ok, left = cooldown_ok(st.session_state, "promote")
                    if not ok:
                        st.warning(f"Please wait {left:.0f}s.")
                    else:
                        promote(pv); st.success(f"@{CHAMPION_ALIAS} → v{pv}"); st.rerun()
    st.caption("Local MLflow UI:  `mlflow ui --backend-store-uri sqlite:///mlflow.db`")

# ================================================================ 4. SERVING
with tabs[4]:
    owner(3)
    st.header("④ Prediction Service — champion model + online features + lineage")
    st.markdown('<div class="lesson"><b>Problem:</b> a prediction nobody can explain or trace is a liability. '
                '<b>Fix:</b> the service resolves the model through the registry alias, pulls features from the '
                'online store, and returns the full lineage with every prediction.</div>', unsafe_allow_html=True)
    ch = champion_info()
    k = st.columns(5)
    kpi(k[0], f"v{ch['version']}", "champion version")
    kpi(k[1], ch["model_type"], "model")
    kpi(k[2], f"{ch['roc_auc']:.3f}", "holdout ROC-AUC")
    kpi(k[3], ch["data_version"], "trained on data")
    kpi(k[4], ch["data_md5"][:8], "data md5")
    st.write("")
    i1, i2 = st.columns([1, 1.5])
    with i1:
        st.subheader("Score a flight")
        car = st.selectbox("Carrier", list(CARRIERS) + [NEW_CARRIER[0]], index=3)
        org = st.selectbox("Origin", list(AIRPORTS), index=0)
        dst = st.selectbox("Destination", [a for a in AIRPORTS if a != org], index=0)
        dd = st.date_input("Departure date", pd.Timestamp("2025-10-01"), key="dd")
        hr = st.slider("Departure hour", 5, 23, 19)
        wx = st.slider("Forecast weather index (0 clear → 3 severe)", 0.0, 3.0, 0.8, 0.1)
    dep = pd.Timestamp(dd) + pd.Timedelta(hours=hr)
    res = predict(car, org, dst, dep, wx, store)
    with i2:
        p = res["delay_probability"]
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=p * 100, number={"suffix": "%", "font": {"size": 46}},
            title={"text": f"{car} {org}→{dst} · {dep:%d %b %H:%M}<br><b>{res['prediction']}</b>"},
            gauge={"axis": {"range": [0, 100]}, "bar": {"color": RED if p >= .5 else GREEN},
                   "bgcolor": T["panel2"], "bordercolor": T["border"],
                   "steps": [{"range": [0, 35], "color": "rgba(57,255,136,.12)"}, {"range": [35, 50], "color": "rgba(255,195,0,.12)"},
                             {"range": [50, 100], "color": "rgba(255,59,107,.14)"}],
                   "threshold": {"line": {"color": NAVY, "width": 3}, "value": 50}}))
        fig.update_layout(**PLOT, height=300); st.plotly_chart(fig, width="stretch")
        hours = list(range(5, 24))
        sweep = [predict(car, org, dst, pd.Timestamp(dd) + pd.Timedelta(hours=h), wx, store)["delay_probability"]
                 for h in hours]
        fig = px.area(x=hours, y=sweep, labels={"x": "departure hour", "y": "P(delay)"},
                      title="What-if: delay risk across the day", color_discrete_sequence=[TEAL])
        fig.add_hline(y=.5, line_dash="dot", line_color=RED)
        fig.update_layout(**PLOT, height=250, yaxis_tickformat=".0%"); st.plotly_chart(fig, width="stretch")

    st.subheader("3D risk surface — departure hour × weather")
    hrs = list(range(5, 24)); wxs = [round(x, 2) for x in np.linspace(0, 3, 16)]
    Z = risk_grid(car, org, dst, pd.Timestamp(dd), hrs, wxs, store)
    s1, s2 = st.columns([1.7, 1])
    with s1:
        fig = go.Figure(go.Surface(x=hrs, y=wxs, z=Z, colorscale=[[0, T["acc"]], [.5, AMBER], [1, RED]],
                                   cmin=0, cmax=1, opacity=.97, contours=dict(z=dict(show=True, usecolormap=True,
                                   project_z=True, highlightcolor=T["head"])),
                                   colorbar=dict(title="P(delay)", tickformat=".0%", len=.7)))
        fig.add_trace(go.Scatter3d(x=[hr], y=[wx], z=[res["delay_probability"]], mode="markers",
                                   marker=dict(size=7, color=T["head"], line=dict(color=T["acc"], width=3)),
                                   name="this flight"))
        zlo, zhi = max(0.0, float(Z.min()) - .05), min(1.0, float(Z.max()) + .05)
        # fixed pixel size: Plotly's WebGL scene initialises at 0 px inside a hidden tab otherwise
        fig.update_layout(**{**PLOT, "margin": dict(l=0, r=0, t=10, b=0)}, width=780, height=500, showlegend=False,
                          scene=dict(xaxis_title="departure hour", yaxis_title="weather index",
                                     zaxis=dict(title="P(delay)", range=[zlo, zhi], tickformat=".0%",
                                                gridcolor=T["border"], backgroundcolor=T["panel2"], showbackground=True),
                                     xaxis=dict(gridcolor=T["border"], backgroundcolor=T["panel2"], showbackground=True),
                                     yaxis=dict(gridcolor=T["border"], backgroundcolor=T["panel2"], showbackground=True),
                                     aspectmode="manual", aspectratio=dict(x=1.5, y=1.1, z=.75),
                                     bgcolor="rgba(0,0,0,0)", camera=dict(eye=dict(x=1.55, y=-1.55, z=.85))))
        st.plotly_chart(fig, width="content")
    with s2:
        hi = np.unravel_index(Z.argmax(), Z.shape); lo = np.unravel_index(Z.argmin(), Z.shape)
        st.markdown(f'<div class="card"><h4>What the surface shows</h4><p>{len(hrs) * len(wxs)} what-if scenarios '
                    f'for <b>{car} {org}→{dst}</b>, scored in one batched call to the champion model. Drag to rotate.'
                    f'<br><br>Lowest risk: <b>{hrs[lo[1]]}:00</b> in clear weather (<b>{Z.min():.0%}</b>).<br>'
                    f'Highest risk: <b>{hrs[hi[1]]}:00</b> at weather {wxs[hi[0]]} (<b>{Z.max():.0%}</b>).<br><br>'
                    f'The white marker is the flight scored above.</p></div>', unsafe_allow_html=True)

    l1, l2 = st.columns([1, 1.3])
    with l1:
        st.subheader("Feature vector sent to the model")
        fv_df = pd.DataFrame([{"feature": k2, "value": f"{v:.4g}" if isinstance(v, float) else str(v),
                               "source": "online store" if k2 in STORE_FEATURES else "request"}
                              for k2, v in res["features"].items()])
        st.dataframe(fv_df, hide_index=True, width="stretch", height=420)
    with l2:
        st.subheader("Prediction lineage")
        L = res["lineage"]
        gv(f"""digraph{{rankdir=TB;bgcolor=transparent;node[shape=box style="rounded,filled" fontname=Inter fontsize=10 fillcolor="#111A2E" color="#2B3D63" fontcolor="#E8EEF7"]; edge[color="#64748B" fontcolor="#B6C2D6"];
          d[label="Dataset {L['data_version']}\\nmd5 {L['data_md5'][:12]}" color="#38BDF8"];
          f[label="Feature store\\nvalues as of {list(res['feature_timestamps'].values())[0]}" color="#34D399"];
          r[label="MLflow run\\n{L['run_id'][:12]}\\n{L['model_type']}" color="#FBBF24"];
          m[label="{REGISTERED_MODEL}\\nv{L['version']} @{L['alias']}" color="#FBBF24"];
          p[label="Prediction\\nP(delay)={res['delay_probability']:.3f}\\n{res['latency_ms']} ms" color="#C084FC" penwidth=2];
          d->r; d->f; f->r[style=dashed label=" training set" fontsize=9]; r->m; m->p; f->p[label=" online lookup" fontsize=9]; }}""",
                          width="stretch")
        st.code("""# REST API (FastAPI) — run: uvicorn serving.api:app --port 8000
curl -X POST localhost:8000/predict -H "Content-Type: application/json" \\
  -d '{"carrier":"%s","origin":"%s","dest":"%s","departure":"%s","weather_index":%s}'""" %
                (car, org, dst, dep.isoformat(), wx), language="bash")

    st.subheader("Batch scoring")
    b1, b2 = st.columns([1, 2.2])
    with b1:
        n = st.slider("Flights in tomorrow's schedule", 10, 150, 60, 10)
        seed = st.number_input("Schedule seed", 0, 999, 7)
        run_batch = st.button("Score schedule", type="primary")
    if run_batch and not cooldown_ok(st.session_state, "batch", 3)[0]:
        st.warning("Please wait a moment before scoring again.")
        run_batch = False
    if run_batch:
        rng = np.random.default_rng(int(seed))
        aps = list(AIRPORTS)
        rows = []
        for _ in range(n):
            o = rng.choice(aps)
            rows.append({"carrier": rng.choice(list(CARRIERS)), "origin": o, "dest": rng.choice([a for a in aps if a != o]),
                         "departure": pd.Timestamp("2025-10-01") + pd.Timedelta(hours=int(rng.integers(5, 24))),
                         "weather_index": round(float(rng.gamma(1.4, .45)), 2)})
        st.session_state["batch"] = batch_predict(pd.DataFrame(rows), store)
    if "batch" in st.session_state:
        bt = st.session_state["batch"]
        with b2:
            fig = px.histogram(bt, x="delay_probability", nbins=20, color="prediction",
                               color_discrete_map={"ON TIME": GREEN, "DELAYED (>15 min)": RED},
                               title=f"{(bt['prediction'] != 'ON TIME').sum()} of {len(bt)} flights flagged high-risk")
            fig.update_layout(**PLOT, height=280); st.plotly_chart(fig, width="stretch")
        st.dataframe(bt.sort_values("delay_probability", ascending=False), hide_index=True, width="stretch",
                     column_config={"delay_probability": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.2f")})
        st.download_button("Download predictions (CSV)", bt.to_csv(index=False), "predictions.csv", "text/csv")

st.markdown("<br><center style='color:var(--muted);font-size:.8rem;letter-spacing:.05em'>DSE ZG529 · Data Management for Machine Learning · "
            "BITS Pilani WILP · Experiential Learning Assignment</center>", unsafe_allow_html=True)
