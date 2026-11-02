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
from src.config import (BASE_CATEGORICAL, BASE_NUMERIC, CHAMPION_ALIAS, mlflow_db_path, new_mlflow_db,
                        REGISTERED_MODEL, STORE_FEATURES, TARGET, TEAM)
from src.data_generation import AIRPORTS, CARRIERS, NEW_CARRIER
from src.feature_store import FEATURE_VIEWS, FeatureStore

st.set_page_config(page_title="Flight Delay MLOps", page_icon="✈️", layout="wide")

# ---- Refined palette (dark navy, calm accents) ------------------------------
NAVY, TEAL, AMBER, RED, GREEN, GREY = "#E8EEF7", "#38BDF8", "#FBBF24", "#F87171", "#34D399", "#A78BFA"
MAGENTA, BG, PANEL, BORDER, MUTED = "#C084FC", "#0B1220", "#111A2E", "#22304D", "#94A3B8"
PLOT = dict(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(17,26,46,.6)",
            font=dict(family="Inter, Segoe UI, sans-serif", color=NAVY, size=13),
            title_font=dict(family="Source Serif 4, Georgia, serif", size=17),
            margin=dict(l=10, r=10, t=48, b=10))

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Source+Serif+4:opsz,wght@8..60,500;8..60,600;8..60,700&family=JetBrains+Mono:wght@400;500&display=swap');
html, body, [class*="css"], .stMarkdown, p, label, li, input, textarea, button {font-family:'Inter', 'Segoe UI', sans-serif;}
.stApp {background: radial-gradient(1100px 520px at 0% -10%, rgba(56,189,248,.07), transparent 60%),
                    radial-gradient(900px 480px at 100% 0%, rgba(167,139,250,.07), transparent 55%), #0B1220;}
.block-container {padding-top: 1.6rem; max-width: 1360px;}
h1, h2, h3 {font-family:'Source Serif 4', Georgia, serif !important; font-weight:600 !important; letter-spacing:0;}
h2 {color:#F1F5FB !important; font-size:1.75rem !important;}
h3 {color:#BFDBFE !important; font-size:1.3rem !important;}
section[data-testid="stSidebar"] {background:#0E172A; border-right:1px solid #22304D;}
section[data-testid="stSidebar"] h3 {font-size:1.15rem !important;}
.hero {padding:28px 34px; border-radius:16px; margin-bottom:22px;
       background: linear-gradient(120deg, #13203A 0%, #182A4A 55%, #1C2F52 100%);
       border:1px solid #2B3D63; border-top:3px solid #38BDF8;
       box-shadow: 0 10px 30px rgba(0,0,0,.35);}
.hero h1 {font-family:'Source Serif 4', Georgia, serif; font-weight:700; font-size:2.1rem; margin:0 0 8px 0; color:#F8FAFC;}
.hero p {color:#CBD5E1; margin:0; font-size:1.03rem; line-height:1.55;}
.pill {display:inline-block; background:rgba(56,189,248,.10); border:1px solid rgba(56,189,248,.45); color:#BAE6FD;
       padding:3px 12px; border-radius:20px; font-size:.8rem; font-weight:500; margin:14px 6px 0 0;}
.pill.m {background:rgba(192,132,252,.10); border-color:rgba(192,132,252,.45); color:#E9D5FF;}
.pill.g {background:rgba(52,211,153,.10); border-color:rgba(52,211,153,.45); color:#A7F3D0;}
.pill.y {background:rgba(251,191,36,.10); border-color:rgba(251,191,36,.45); color:#FDE68A;}
.card {background:#111A2E; border:1px solid #22304D; border-radius:14px; padding:16px 18px; height:100%;
       transition: border-color .2s ease, transform .2s ease;}
.card:hover {border-color:#38BDF8; transform:translateY(-1px);}
.card h4 {font-family:'Source Serif 4', Georgia, serif; font-weight:600; margin:0 0 6px 0; color:#7DD3FC; font-size:1.05rem;}
.card p {margin:0; color:#B6C2D6; font-size:.92rem; line-height:1.5;}
.kpi {background:#111A2E; border:1px solid #22304D; border-radius:14px; padding:14px 18px; border-left:3px solid #38BDF8;}
.kpi .v {font-family:'Inter', sans-serif; font-size:1.6rem; font-weight:700; color:#F8FAFC; line-height:1.15;
         font-variant-numeric: tabular-nums;}
.kpi .l {font-size:.74rem; color:#94A3B8; text-transform:uppercase; letter-spacing:.07em; margin-top:4px; font-weight:500;}
.owner {display:inline-block; font-size:.78rem; color:#A7F3D0; font-weight:600; letter-spacing:.04em;
        background:rgba(52,211,153,.10); border:1px solid rgba(52,211,153,.4); padding:3px 12px; border-radius:20px;
        margin-bottom:6px;}
.lesson {background:rgba(251,191,36,.06); border:1px solid rgba(251,191,36,.28); border-left:4px solid #FBBF24;
         padding:12px 16px; border-radius:10px; color:#FDE9B0; font-size:.94rem; line-height:1.55; margin:8px 0;}
.good {background:rgba(52,211,153,.06); border:1px solid rgba(52,211,153,.28); border-left:4px solid #34D399;
       padding:12px 16px; border-radius:10px; color:#C8F5E2; font-size:.94rem; line-height:1.55; margin:8px 0;}
.bad {background:rgba(248,113,113,.07); border:1px solid rgba(248,113,113,.3); border-left:4px solid #F87171;
      padding:12px 16px; border-radius:10px; color:#FBD0D0; font-size:.94rem; line-height:1.55; margin:8px 0;}
.mono {font-family:'JetBrains Mono', ui-monospace, Menlo, monospace; font-size:.84rem; color:#7DD3FC;}
div[data-testid="stTabs"] button p {font-family:'Inter', sans-serif; font-size:.95rem; font-weight:600;}
div[data-testid="stTabs"] button[aria-selected="true"] p {color:#7DD3FC;}
div[data-baseweb="tab-highlight"] {background:#38BDF8 !important; height:3px;}
.stButton > button, .stDownloadButton > button {border-radius:10px; border:1px solid #38BDF8; color:#BAE6FD;
       background:rgba(56,189,248,.06); font-weight:600; transition:background .2s ease;}
.stButton > button:hover, .stDownloadButton > button:hover {background:rgba(56,189,248,.16); color:#fff; border-color:#7DD3FC;}
.stButton > button[kind="primary"] {background:#2563EB; color:#fff; border:1px solid #3B82F6;}
.stButton > button[kind="primary"]:hover {background:#1D4ED8;}
div[data-testid="stDataFrame"] {border:1px solid #22304D; border-radius:10px;}
div[data-testid="stExpander"] {border:1px solid #22304D; border-radius:10px; background:#0F1829;}
</style>""", unsafe_allow_html=True)


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

from src.predict import batch_predict, champion_info, predict  # noqa: E402
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
    st.markdown("### ✈️ Flight Delay MLOps")
    st.caption("DSE ZG529 · Data Management for ML\nExperiential Learning Assignment")
    st.markdown("**Team**")
    for m in TEAM:
        st.markdown(f"**{m['member']}** — {m['name']}  \n<span style='color:#38BDF8;font-size:.8rem'>{m['id']}</span> · <span style='color:#94A3B8;font-size:.85rem'>{m['part']}</span>",
                    unsafe_allow_html=True)
    st.divider()
    try:
        ch = champion_info()
        st.markdown("**Live pipeline status**")
        st.markdown(f"🗂️ Data versions: **{len(versioning.list_versions())}**")
        st.markdown(f"🧮 Feature store built from: **{store.offline_metadata().get('data_version')}**")
        st.markdown(f"🏆 Champion: **v{ch['version']}** · {ch['model_type']} · AUC **{ch['roc_auc']:.3f}**")
    except Exception as e:
        st.warning(f"Pipeline not ready: {e}")
    st.divider()
    if st.button("♻️ Rebuild entire pipeline", width="stretch",
                 help="Deletes versions, feature store and MLflow DB, then reruns every stage"):
        import shutil
        from src.config import DATA_DIR, MLFLOW_ARTIFACTS
        for p in [DATA_DIR / ".versions", DATA_DIR / "versions", DATA_DIR / "feature_store", MLFLOW_ARTIFACTS]:
            shutil.rmtree(p, ignore_errors=True)
        new_mlflow_db()   # fresh tracking DB; never delete a DB an engine still has open
        st.cache_resource.clear(); st.cache_data.clear()
        st.rerun()

st.markdown("""
<div class="hero">
  <h1>✈️ Flight Delay Prediction — an End-to-End MLOps Pipeline</h1>
  <p>Predicting whether an Indian domestic flight departs &gt;15 min late, with every stage of the
  data &amp; model lifecycle made reproducible, traceable and servable.</p>
  <span class="pill">DVC-style data versioning</span><span class="pill m">Feature store · point-in-time joins</span>
  <span class="pill g">MLflow tracking &amp; registry</span><span class="pill y">Prediction service · lineage</span>
</div>""", unsafe_allow_html=True)

tabs = st.tabs(["🏠 Overview", "① Data Versioning", "② Feature Store",
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
        st.graphviz_chart("""
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
        st.graphviz_chart("digraph{rankdir=TB;bgcolor=transparent;node[shape=box style='rounded,filled' fillcolor='#111A2E' color='#38BDF8' fontcolor='#E8EEF7' fontname=Inter fontsize=11];"
                          + "".join(f'"{m["tag"]}"[label="{m["tag"]}\\nmd5 {m["outs"][0]["md5"][:8]}\\n{m["rows"]:,} rows"];' for m in versions)
                          + "edge[color='#C084FC'];" + "".join(f'"{m["parent"]}"->"{m["tag"]}";' for m in versions if m["parent"]) + "}")

    st.subheader("🔍 Checkout & compare two versions")
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

    st.subheader("🧪 Experiment: what breaks without versioning?")
    st.caption("Train the same model twice. Between the two runs, someone 'fixes' a few rows in place.")
    e1, e2 = st.columns([1, 2])
    with e1:
        rep_tag = st.selectbox("Dataset version", tags, index=tags.index("v1.1"), key="rep")
        n_edit = st.slider("Rows silently edited in place", 0, 2000, 400, step=100)
        go_rep = st.button("Run reproducibility test", type="primary")
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
        col.markdown(f'<div class="card"><h4>🧩 {name}</h4><p><b>entity:</b> {fv.entity} · <b>TTL:</b> '
                     f'{fv.ttl_days} d<br>{fv.description}<br><span class="mono">{"<br>".join(fv.features)}</span>'
                     f'</p></div>', unsafe_allow_html=True)
    st.write("")
    st.subheader("📚 Offline store — feature history")
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

    st.subheader("⏱️ Point-in-time join vs. naive join")
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

    st.subheader("🧪 Experiment: the leakage trap")
    if st.button("Train with naive join vs. point-in-time join", type="primary"):
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

    st.subheader("⚡ Online store — materialize & look up")
    m1, m2 = st.columns([1, 2])
    with m1:
        as_of = st.date_input("Materialize as of", pd.Timestamp("2025-10-01"),
                              min_value=pd.Timestamp("2025-01-05"), max_value=pd.Timestamp("2025-10-01"))
        if st.button("Materialize to online store"):
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
    st.subheader("🚀 Launch a tracked experiment")
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
        t_name = st.text_input("Run name", f"{t_model} | {t_tag}")
        if st.button("Train & log to MLflow", type="primary", width="stretch"):
            with st.spinner("Training…"):
                st.session_state["last_run"] = train_and_log(t_tag, t_model, params, t_store, t_name, t_author)
    if "last_run" in st.session_state:
        r = st.session_state["last_run"]
        k = st.columns(6)
        for col, m in zip(k, ["roc_auc", "f1", "precision", "recall", "accuracy", "train_time_s"]):
            kpi(col, r[m], m)
        st.caption(f"run_id `{r['run_id']}` · model `{r['model_uri']}`")

    runs = runs_dataframe()
    if not runs.empty:
        st.subheader("📒 Run history")
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
        fig = px.bar(same, x="run", y="ROC-AUC", color="model", title="Like-for-like: runs on v2.0 only",
                     color_discrete_sequence=[TEAL, MAGENTA, GREEN], text_auto=".3f")
        fig.update_layout(**PLOT, yaxis_range=[0.5, same["ROC-AUC"].max() + .05], xaxis_title="")
        cc2.plotly_chart(fig, width="stretch")
        st.markdown('<div class="lesson">Metrics are only comparable on the <b>same data version</b> (same '
                    'holdout window) — another reason the data version is logged with every run.</div>',
                    unsafe_allow_html=True)

        st.subheader("🏛️ Model registry")
        rg1, rg2 = st.columns([1.6, 1])
        reg = registry_versions()
        with rg1:
            if not reg.empty:
                st.dataframe(reg, hide_index=True, width="stretch")
        with rg2:
            safe = show[show["leakage safe"] == "true"]
            pick = st.selectbox("Register a run", safe["run_id"],
                                format_func=lambda x: f"{safe.set_index('run_id').loc[x, 'run']} · "
                                f"AUC {safe.set_index('run_id').loc[x, 'ROC-AUC']:.3f}")
            promote_now = st.checkbox(f"Promote to @{CHAMPION_ALIAS}", value=False)
            if st.button("Register model version", width="stretch"):
                out = register_run(pick, promote=promote_now)
                st.success(f"Registered {REGISTERED_MODEL} v{out['version']}"
                           + (f" and promoted to @{CHAMPION_ALIAS}" if promote_now else ""))
                st.rerun()
            if not reg.empty:
                pv = st.selectbox("Or move @champion to version", reg["version"].tolist())
                if st.button("Promote", width="stretch"):
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
        st.subheader("✈️ Score a flight")
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
                   "bgcolor": "rgba(14,20,36,.6)", "bordercolor": "#1E2A44",
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
        st.graphviz_chart(f"""digraph{{rankdir=TB;bgcolor=transparent;node[shape=box style="rounded,filled" fontname=Inter fontsize=10 fillcolor="#111A2E" color="#2B3D63" fontcolor="#E8EEF7"]; edge[color="#64748B" fontcolor="#B6C2D6"];
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

    st.subheader("📦 Batch scoring")
    b1, b2 = st.columns([1, 2.2])
    with b1:
        n = st.slider("Flights in tomorrow's schedule", 10, 150, 60, 10)
        seed = st.number_input("Schedule seed", 0, 999, 7)
        run_batch = st.button("Score schedule", type="primary")
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

st.markdown("<br><center style='color:#94A3B8;font-size:.8rem;letter-spacing:.05em'>DSE ZG529 · Data Management for Machine Learning · "
            "BITS Pilani WILP · Experiential Learning Assignment</center>", unsafe_allow_html=True)
