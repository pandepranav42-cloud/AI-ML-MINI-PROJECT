"""Student Success & Placement Predictor — Streamlit application.

Run:  streamlit run app.py
"""
from __future__ import annotations

import json

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.predict import InvalidInputError, StudentPredictor
from src.preprocessing import clean_dataset, load_raw
from src.utils import (ACADEMIC_LABELS, CATEGORY_COLORS, FEATURE_LABELS, FIG_DIR,
                       GENDERS, METRICS_DIR, MODELS_DIR, PALETTE,
                       READINESS_LABELS, impact_label, validate_input)

st.set_page_config(page_title="Student Success & Placement Predictor",
                   page_icon="🎓", layout="wide")

# --------------------------------------------------------------------------
# Styling
# --------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Public+Sans:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], .stMarkdown, .stButton, input, label, p, li {
  font-family: 'Public Sans', 'Segoe UI', system-ui, sans-serif;
}
h1, h2, h3 { font-family: 'Public Sans', 'Segoe UI', system-ui, sans-serif; letter-spacing: -0.01em; }
h1 { font-weight: 800 !important; }
.block-container { padding-top: 2rem; max-width: 1180px; }
[data-testid="stSidebar"] { background: #1D2B36; }
[data-testid="stSidebar"] * { color: #E9EFEC !important; }
[data-testid="stSidebar"] .note { color: #A9B8B3 !important; font-size: 0.82rem; line-height: 1.45; }
[data-testid="stMetric"] {
  background: #FFFFFF; border: 1px solid #D5DDD8; border-radius: 6px; padding: 14px 18px;
}
[data-testid="stMetricValue"] { font-weight: 700; color: #1F5C63; font-size: 1.85rem; }
.intro { color: #4A5A64; max-width: 72ch; margin-top: -0.4rem; margin-bottom: 1.2rem; }

/* Report card: the one bold element in the interface */
.report {
  background: #FFFFFF; border: 1px solid #C9D3CE; border-top: 6px solid #1F5C63;
  border-radius: 4px; padding: 22px 26px 18px; margin: 8px 0 18px;
}
.report-head { display: flex; justify-content: space-between; align-items: baseline;
  border-bottom: 1px dashed #C9D3CE; padding-bottom: 10px; margin-bottom: 16px; }
.report-head .t { font-weight: 700; font-size: 1.05rem; }
.report-head .s { color: #6B7A82; font-size: 0.85rem; }
.report-grid { display: grid; grid-template-columns: 1fr 1fr 1.25fr; gap: 26px; }
@media (max-width: 800px) { .report-grid { grid-template-columns: 1fr; } }
.cell .k { color: #52636C; font-size: 0.88rem; font-weight: 600; }
.cell .v { font-size: 1.9rem; font-weight: 800; line-height: 1.15; margin: 4px 0 2px; }
.cell .c { color: #6B7A82; font-size: 0.82rem; }
.chip { display: inline-block; width: 12px; height: 12px; border-radius: 2px; margin-right: 8px; vertical-align: 2px; }
.pbar { height: 12px; background: #E4EAE6; border-radius: 6px; overflow: hidden; margin: 8px 0 4px; }
.pbar > div { height: 100%; border-radius: 6px; }
.scale { display: flex; justify-content: space-between; color: #8A979E; font-size: 0.72rem; }
.caveat { color: #52636C; font-size: 0.84rem; margin-top: 14px; border-top: 1px solid #E4EAE6; padding-top: 10px; }
.factor { display: flex; justify-content: space-between; padding: 7px 0; border-bottom: 1px solid #E4EAE6; font-size: 0.93rem; }
.factor .imp { color: #6B7A82; font-size: 0.8rem; }
.pos { color: #1F5C63; font-weight: 700; } .neg { color: #B3524B; font-weight: 700; }
div.stFormSubmitButton > button, div.stButton > button[kind="primary"] {
  background: #1F5C63; color: #fff; font-weight: 800; letter-spacing: 0.08em;
  padding: 0.7rem 2.6rem; border-radius: 4px; border: none; font-size: 1.05rem;
}
div.stFormSubmitButton > button:hover { background: #17474D; color: #fff; }
div.stFormSubmitButton > button:focus-visible { outline: 3px solid #D9A441; outline-offset: 2px; }
</style>
""", unsafe_allow_html=True)

PLOT_FONT = dict(family="Public Sans, Segoe UI, sans-serif", color=PALETTE["ink"], size=13)


def style_fig(fig, height=380):
    fig.update_layout(font=PLOT_FONT, height=height, margin=dict(l=10, r=10, t=56, b=10),
                      paper_bgcolor="white", plot_bgcolor="white",
                      title_font=dict(size=16), legend_title_font=dict(size=12))
    fig.update_xaxes(gridcolor="#E4EAE6", zeroline=False)
    fig.update_yaxes(gridcolor="#E4EAE6", zeroline=False)
    return fig


# --------------------------------------------------------------------------
# Cached resources
# --------------------------------------------------------------------------
@st.cache_data
def load_data() -> pd.DataFrame:
    df, _ = clean_dataset(load_raw(), verbose=False)
    df["academic_performance"] = pd.Categorical(df["academic_performance"], ACADEMIC_LABELS, ordered=True)
    df["placement_readiness"] = pd.Categorical(df["placement_readiness"], READINESS_LABELS, ordered=True)
    return df


@st.cache_resource
def load_predictor() -> StudentPredictor:
    return StudentPredictor()


@st.cache_data
def load_metric_tables():
    t = {name: pd.read_csv(METRICS_DIR / f"{name}_model_comparison.csv")
         for name in ("academic", "placement", "probability")}
    imp = {name: pd.read_csv(METRICS_DIR / f"feature_importance_{name}.csv")
           for name in ("academic", "placement", "probability")}
    meta = json.loads((MODELS_DIR / "model_metadata.json").read_text())
    return t, imp, meta


try:
    data = load_data()
    predictor = load_predictor()
    tables, importances, meta = load_metric_tables()
except FileNotFoundError as exc:
    st.error(f"Required file not found: {exc.filename}. Run `python -m src.generate_data` "
             "and `python -m src.train` first (see README).")
    st.stop()

# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown("<div style='font-weight:800;font-size:1.15rem;line-height:1.3;margin-bottom:0.8rem'>🎓 Student Success &amp; Placement Predictor</div>", unsafe_allow_html=True)
    page = st.radio("Go to", ["Dashboard", "Student Prediction", "Analytics", "About"],
                    label_visibility="collapsed")
    st.markdown("---")
    st.markdown(
        f"<div class='note'>Models trained on {meta['rows_after_cleaning']:,} <b>synthetic</b> "
        f"student records. Predictions are statistical estimates for learning purposes, "
        f"not placement decisions.</div>", unsafe_allow_html=True)


# ==========================================================================
# Dashboard
# ==========================================================================
def page_dashboard():
    st.title("Dashboard")
    st.markdown("<p class='intro'>A snapshot of the synthetic cohort the models learned from.</p>",
                unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Students in dataset", f"{len(data):,}")
    c2.metric("Average CGPA", f"{data['cgpa'].mean():.2f} / 10")
    c3.metric("Average attendance", f"{data['attendance_pct'].mean():.1f}%")
    c4.metric("Average coding score", f"{data['coding_score'].mean():.1f} / 100")

    left, right = st.columns(2)
    with left:
        counts = data["placement_readiness"].value_counts().reindex(READINESS_LABELS)
        fig = go.Figure(go.Bar(
            x=counts.index, y=counts.values,
            marker_color=[CATEGORY_COLORS[c] for c in counts.index],
            text=[f"{v}<br>({v / counts.sum():.0%})" for v in counts.values], textposition="outside"))
        fig.update_layout(title="Placement-readiness distribution",
                          xaxis_title="Readiness category", yaxis_title="Number of students")
        fig.update_yaxes(range=[0, counts.max() * 1.25])
        st.plotly_chart(style_fig(fig), width="stretch")
    with right:
        counts = data["academic_performance"].value_counts().reindex(ACADEMIC_LABELS)
        fig = go.Figure(go.Bar(
            x=counts.index, y=counts.values,
            marker_color=[CATEGORY_COLORS[c] for c in counts.index],
            text=[f"{v}<br>({v / counts.sum():.0%})" for v in counts.values], textposition="outside"))
        fig.update_layout(title="Academic-performance distribution",
                          xaxis_title="Performance category", yaxis_title="Number of students")
        fig.update_yaxes(range=[0, counts.max() * 1.25])
        st.plotly_chart(style_fig(fig), width="stretch")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("With internship", f"{(data['internship'] == 'Yes').mean():.0%}")
    c2.metric("Joined a hackathon", f"{(data['hackathon'] == 'Yes').mean():.0%}")
    c3.metric("With 1+ backlog", f"{(data['backlogs'] > 0).mean():.0%}")
    c4.metric("Avg. placement probability", f"{data['placement_probability'].mean():.0f}%")

    tm = meta["test_metrics"]
    st.subheader("Selected models")
    st.dataframe(pd.DataFrame([
        {"Prediction": "Academic performance", "Model": meta["selected_models"]["academic"],
         "Test result": f"accuracy {tm['academic']['accuracy']:.1%}, macro-F1 {tm['academic']['f1']:.3f}"},
        {"Prediction": "Placement readiness", "Model": meta["selected_models"]["placement"],
         "Test result": f"accuracy {tm['placement']['accuracy']:.1%}, macro-F1 {tm['placement']['f1']:.3f}"},
        {"Prediction": "Placement probability", "Model": meta["selected_models"]["probability"],
         "Test result": f"MAE {tm['probability']['mae']:.1f} pts, R² {tm['probability']['r2']:.3f}"},
    ]), hide_index=True, width="stretch")


# ==========================================================================
# Prediction
# ==========================================================================
PRESETS = {
    "Blank form (typical values)": dict(
        age=21, gender="Male", cgpa=7.0, attendance_pct=80.0, backlogs=0, coding_score=57.0,
        aptitude_score=59.0, communication_score=64.0, projects=2, certifications=1,
        internship="No", hackathon="No", extracurricular_score=58.0, technical_skills_score=46.0,
        programming_languages=2, study_hours_per_week=14.0),
    "Sample: high performer": dict(
        age=21, gender="Female", cgpa=9.1, attendance_pct=93.0, backlogs=0, coding_score=86.0,
        aptitude_score=81.0, communication_score=76.0, projects=4, certifications=3,
        internship="Yes", hackathon="Yes", extracurricular_score=72.0, technical_skills_score=78.0,
        programming_languages=5, study_hours_per_week=22.0),
    "Sample: average student": dict(
        age=20, gender="Male", cgpa=7.1, attendance_pct=79.0, backlogs=0, coding_score=58.0,
        aptitude_score=60.0, communication_score=63.0, projects=2, certifications=1,
        internship="No", hackathon="No", extracurricular_score=55.0, technical_skills_score=47.0,
        programming_languages=2, study_hours_per_week=13.0),
    "Sample: student at risk": dict(
        age=22, gender="Male", cgpa=5.4, attendance_pct=58.0, backlogs=4, coding_score=34.0,
        aptitude_score=41.0, communication_score=52.0, projects=0, certifications=0,
        internship="No", hackathon="No", extracurricular_score=40.0, technical_skills_score=24.0,
        programming_languages=1, study_hours_per_week=5.0),
}


def apply_preset():
    for k, v in PRESETS[st.session_state["preset"]].items():
        st.session_state[f"in_{k}"] = v


def page_prediction():
    st.title("Student Prediction")
    st.markdown("<p class='intro'>Enter a student's academic and skills profile, then press PREDICT. "
                "Every field is checked before the models run.</p>", unsafe_allow_html=True)
    if "in_cgpa" not in st.session_state:
        st.session_state["preset"] = "Blank form (typical values)"
        apply_preset()
    st.selectbox("Start from", list(PRESETS), key="preset", on_change=apply_preset)

    num = lambda label, key, step, fmt, help=None: st.number_input(  # noqa: E731
        label, key=f"in_{key}", step=step, format=fmt, help=help)

    with st.form("student_form", border=True):
        a, b, c = st.columns(3)
        with a:
            st.markdown("**Academics**")
            num("CGPA (0–10)", "cgpa", 0.1, "%.2f")
            num("Attendance % (0–100)", "attendance_pct", 1.0, "%.1f")
            num("Backlogs", "backlogs", 1, "%d", "Number of currently uncleared subjects")
            num("Study hours per week", "study_hours_per_week", 1.0, "%.1f")
            num("Aptitude score (0–100)", "aptitude_score", 1.0, "%.0f")
        with b:
            st.markdown("**Skills**")
            num("Coding / DSA score (0–100)", "coding_score", 1.0, "%.0f")
            num("Technical skills score (0–100)", "technical_skills_score", 1.0, "%.0f")
            num("Communication score (0–100)", "communication_score", 1.0, "%.0f")
            num("Programming languages known", "programming_languages", 1, "%d")
            num("Extracurricular score (0–100)", "extracurricular_score", 1.0, "%.0f")
        with c:
            st.markdown("**Experience & profile**")
            num("Projects completed", "projects", 1, "%d")
            num("Certifications", "certifications", 1, "%d")
            st.radio("Internship experience", ["Yes", "No"], key="in_internship", horizontal=True)
            st.radio("Hackathon participation", ["Yes", "No"], key="in_hackathon", horizontal=True)
            num("Age", "age", 1, "%d")
            st.selectbox("Gender", GENDERS, key="in_gender",
                         help="Recorded for fairness auditing only. Gender and age are not model inputs.")
        submitted = st.form_submit_button("PREDICT")

    if not submitted:
        return
    record = {k[3:]: v for k, v in st.session_state.items() if k.startswith("in_")}
    errors = validate_input(record)
    if errors:
        st.error("**Please fix the following before predicting:**\n\n" + "\n".join(f"- {e}" for e in errors))
        return
    try:
        res = predictor.predict(record)
    except InvalidInputError as exc:  # defensive: validate_input already ran
        st.error("\n".join(exc.errors))
        return
    render_result(res)


def render_result(res):
    acc_color = CATEGORY_COLORS[res.academic_performance]
    rdy_color = CATEGORY_COLORS[res.placement_readiness]
    p = res.placement_probability
    bar_color = PALETTE["brick"] if p < 35 else PALETTE["mustard"] if p < 65 else PALETTE["teal"]
    st.markdown(f"""
<div class="report">
  <div class="report-head"><span class="t">Prediction report</span>
  <span class="s">Model estimate from synthetic training data</span></div>
  <div class="report-grid">
    <div class="cell"><div class="k">Academic Performance</div>
      <div class="v"><span class="chip" style="background:{acc_color}"></span>{res.academic_performance}</div>
      <div class="c">Model confidence {pct(res.academic_confidence)}</div></div>
    <div class="cell"><div class="k">Placement Readiness</div>
      <div class="v"><span class="chip" style="background:{rdy_color}"></span>{res.placement_readiness}</div>
      <div class="c">Model confidence {pct(res.readiness_confidence)}</div></div>
    <div class="cell"><div class="k">Estimated Placement Probability</div>
      <div class="v">{p:.0f}%</div>
      <div class="pbar"><div style="width:{p:.1f}%; background:{bar_color}"></div></div>
      <div class="scale"><span>0%</span><span>Cohort average {res.baseline_probability:.0f}%</span><span>100%</span></div>
    </div>
  </div>
  <div class="caveat">This is a statistical estimate from a model trained on synthetic data. It does not
  guarantee, or rule out, a job offer and must not be used to make decisions about real students.</div>
</div>""", unsafe_allow_html=True)

    st.subheader("What drove this estimate")
    top = res.top_factors(8)
    total = sum(abs(f.contribution) for f in res.factors) or 1.0
    left, right = st.columns([1.25, 1])
    with left:
        d = pd.DataFrame({"label": [f"{f.label} = {fmt_value(f.value)}" for f in top],
                          "pts": [f.contribution for f in top]}).iloc[::-1]
        fig = go.Figure(go.Bar(
            x=d["pts"], y=d["label"], orientation="h",
            marker_color=[PALETTE["teal"] if v > 0 else PALETTE["brick"] for v in d["pts"]],
            text=[f"{v:+.1f}" for v in d["pts"]], textposition="outside"))
        fig.update_layout(title=f"Contribution to probability (percentage points vs. {res.baseline_probability:.0f}% average)",
                          xaxis_title="Percentage points", yaxis_title=None, title_font=dict(size=14))
        lim = max(abs(d["pts"]).max() * 1.3, 1)
        fig.update_xaxes(range=[-lim, lim])
        st.plotly_chart(style_fig(fig, 390), width="stretch")
        st.caption(f"Method: {res.explanation_method}. Teal raises the estimate, red lowers it.")
    with right:
        html = ""
        for f in top[:6]:
            tier = impact_label(abs(f.contribution) / total)
            cls = "pos" if f.contribution > 0 else "neg"
            html += (f"<div class='factor'><span><b>{f.label}</b> ({fmt_value(f.value)})<br>"
                     f"<span class='imp'>{tier} impact</span></span>"
                     f"<span class='{cls}'>{f.direction} {abs(f.contribution):.1f} pts</span></div>")
        st.markdown(html, unsafe_allow_html=True)
        strengths = [f for f in sorted(res.factors, key=lambda f: -f.contribution) if f.contribution > 1][:3]
        gaps = [f for f in sorted(res.factors, key=lambda f: f.contribution) if f.contribution < -1][:3]
        st.markdown("")
        if strengths:
            st.markdown("**Strengths:** " + ", ".join(f"{f.label} ({fmt_value(f.value)})" for f in strengths))
        if gaps:
            st.markdown("**Areas to work on:** " + ", ".join(f"{f.label} ({fmt_value(f.value)})" for f in gaps))

    with st.expander("Class probabilities from each classifier"):
        c1, c2 = st.columns(2)
        for col, title, probs in ((c1, "Academic performance", res.academic_probabilities),
                                  (c2, "Placement readiness", res.readiness_probabilities)):
            fig = go.Figure(go.Bar(x=list(probs), y=[v * 100 for v in probs.values()],
                                   marker_color=[CATEGORY_COLORS[k] for k in probs],
                                   text=[f"{v:.0%}" for v in probs.values()], textposition="outside"))
            fig.update_layout(title=title, yaxis_title="Probability (%)", xaxis_title="Category")
            fig.update_yaxes(range=[0, 115])
            col.plotly_chart(style_fig(fig, 300), width="stretch")


def share_by_band(df, col, cap, title, xlabel, cmap):
    """100%-stacked bar of readiness share for each value of a count feature (last band = cap+)."""
    band = df[col].clip(upper=cap).astype(int).map(lambda v: f"{cap}+" if v >= cap else str(v))
    order = [str(i) for i in range(cap)] + [f"{cap}+"]
    ct = pd.crosstab(band, df["placement_readiness"], normalize="index").reindex(order) * 100
    n = band.value_counts().reindex(order)
    ct.index = [f"{o}<br>(n={c})" for o, c in zip(order, n)]
    fig = px.bar(ct, barmode="stack", color_discrete_map=cmap, title=title,
                 labels={"index": xlabel, "value": "Share of students (%)", "placement_readiness": "Readiness"})
    fig.update_xaxes(type="category", title=xlabel)
    return fig


def pct(p):
    return ">99%" if p >= 0.995 else f"{p:.0%}"


def fmt_value(v):
    if isinstance(v, str):
        return v
    return f"{v:g}" if float(v).is_integer() else f"{v:.1f}"


# ==========================================================================
# Analytics
# ==========================================================================
def page_analytics():
    st.title("Analytics")
    st.markdown("<p class='intro'>Explore relationships in the dataset and how the models perform. "
                "Hover over any chart for exact values.</p>", unsafe_allow_html=True)
    tab1, tab2, tab3 = st.tabs(["Dataset relationships", "Feature importance", "Model evaluation"])
    rmap = {c: CATEGORY_COLORS[c] for c in READINESS_LABELS}
    amap = {c: CATEGORY_COLORS[c] for c in ACADEMIC_LABELS}

    with tab1:
        l, r = st.columns(2)
        fig = px.histogram(data, x="cgpa", nbins=30, color_discrete_sequence=[PALETTE["teal"]],
                           title="CGPA distribution", labels={"cgpa": "CGPA"})
        fig.add_vline(data["cgpa"].mean(), line_dash="dash", line_color=PALETTE["brick"],
                      annotation_text=f"mean {data['cgpa'].mean():.2f}")
        fig.update_layout(yaxis_title="Number of students", bargap=0.05)
        l.plotly_chart(style_fig(fig), width="stretch")

        fig = px.scatter(data, x="attendance_pct", y="cgpa", color="academic_performance",
                         color_discrete_map=amap, opacity=0.7, category_orders={"academic_performance": ACADEMIC_LABELS},
                         title="Attendance vs CGPA", labels={"attendance_pct": "Attendance %", "cgpa": "CGPA",
                                                             "academic_performance": "Academic performance"})
        r.plotly_chart(style_fig(fig), width="stretch")

        l, r = st.columns(2)
        fig = px.box(data, x="placement_readiness", y="coding_score", color="placement_readiness",
                     color_discrete_map=rmap, category_orders={"placement_readiness": READINESS_LABELS},
                     title="Coding score vs placement readiness",
                     labels={"placement_readiness": "Placement readiness", "coding_score": "Coding / DSA score"})
        fig.update_layout(showlegend=False)
        l.plotly_chart(style_fig(fig), width="stretch")

        fig = share_by_band(data, "projects", 5, "Projects vs placement readiness", "Projects completed", rmap)
        r.plotly_chart(style_fig(fig), width="stretch")

        l, r = st.columns(2)
        ct = pd.crosstab(data["internship"], data["placement_readiness"], normalize="index").reindex(["No", "Yes"]) * 100
        fig = px.bar(ct, barmode="stack", color_discrete_map=rmap,
                     title="Internship vs placement readiness",
                     labels={"internship": "Internship experience", "value": "Share of students (%)",
                             "placement_readiness": "Readiness"})
        l.plotly_chart(style_fig(fig), width="stretch")

        fig = share_by_band(data.dropna(subset=["certifications"]), "certifications", 4,
                            "Certifications vs placement readiness", "Certifications earned", rmap)
        r.plotly_chart(style_fig(fig), width="stretch")

    with tab2:
        choice = st.radio("Model", ["Placement readiness", "Placement probability", "Academic performance"],
                          horizontal=True)
        key = {"Placement readiness": "placement", "Placement probability": "probability",
               "Academic performance": "academic"}[choice]
        imp = importances[key].sort_values("share")
        tier_colors = {"High": PALETTE["teal"], "Medium-high": PALETTE["sage"],
                       "Medium": PALETTE["mustard"], "Low": "#B9C2C0"}
        fig = px.bar(imp, x=imp["share"] * 100, y="label", color="impact", orientation="h",
                     color_discrete_map=tier_colors,
                     category_orders={"impact": list(tier_colors)},
                     title=f"Feature importance: {choice.lower()} ({meta['selected_models'][key]})",
                     labels={"x": "Share of permutation importance (%)", "label": "", "impact": "Impact"})
        fig.update_yaxes(categoryorder="array", categoryarray=list(imp["label"]))
        st.plotly_chart(style_fig(fig, 520), width="stretch")
        st.caption("Permutation importance: how much test-set performance drops when one feature's values are "
                   "shuffled. It shows what the model relies on, which is not the same as what causes placement.")
        shap_png = FIG_DIR / f"shap_summary_{key}.png"
        if shap_png.exists():
            with st.expander("SHAP summary plot (per-student impact of each feature)"):
                st.image(str(shap_png), width="stretch")

    with tab3:
        cls_cols = {"model": "Model", "cv_f1": "CV macro-F1", "test_accuracy": "Accuracy",
                    "test_precision": "Precision", "test_recall": "Recall", "test_f1": "Macro-F1",
                    "test_roc_auc": "ROC-AUC"}
        reg_cols = {"model": "Model", "cv_rmse": "CV RMSE", "test_mae": "MAE", "test_rmse": "RMSE", "test_r2": "R²"}
        for key, title, cols in (("academic", "Academic performance", cls_cols),
                                 ("placement", "Placement readiness", cls_cols),
                                 ("probability", "Placement probability", reg_cols)):
            st.markdown(f"**{title}** — selected: {meta['selected_models'][key]}")
            t = tables[key][list(cols)].rename(columns=cols)
            st.dataframe(t.style.format({c: "{:.3f}" for c in t.columns if c != "Model"}),
                         hide_index=True, width="stretch")
        st.caption("Models are ranked by 5-fold cross-validation on the training set (CV columns). "
                   "The other columns are measured once on the held-out test set.")
        l, r = st.columns(2)
        l.image(str(FIG_DIR / "confusion_matrix_academic.png"), caption="Academic performance, test set",
                width="stretch")
        r.image(str(FIG_DIR / "confusion_matrix_placement.png"), caption="Placement readiness, test set",
                width="stretch")
        st.image(str(FIG_DIR / "probability_regression.png"), caption="Placement probability, test set",
                 width="stretch")


# ==========================================================================
# About
# ==========================================================================
def page_about():
    st.title("About this project")
    sm = meta["selected_models"]
    st.markdown(f"""
### Objective
Estimate, from a student's academic and extracurricular profile, three things: an **academic-performance
category**, a **placement-readiness category** and an **estimated placement probability** — and explain
which inputs pushed each estimate up or down.

### Dataset
The dataset is **synthetic**: {meta['rows_after_cleaning']:,} simulated student records (after removing
duplicates) produced by `src/generate_data.py`. Relationships such as "more projects tend to improve readiness"
were written into the generator as assumptions, with random noise added. The generator also injects missing
values, duplicate rows and impossible entries so the cleaning pipeline has real work to do.
**Results describe this simulated data only and are not evidence about real placement outcomes.**

### Machine-learning approach
- Preprocessing (scikit-learn `Pipeline` + `ColumnTransformer`): median imputation and standard scaling for
  numeric fields, most-frequent imputation and one-hot encoding for Yes/No fields, variance-threshold
  feature selection. It is fitted inside each training fold, so no test information leaks in.
- Classifiers compared: Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, XGBoost.
- Regressors compared for probability: Ridge, Decision Tree, Random Forest, Gradient Boosting, XGBoost.
- Selection: highest 5-fold cross-validated macro-F1 (classification) and lowest CV RMSE (regression).
  Selected: **{sm['academic']}** (academic), **{sm['placement']}** (readiness), **{sm['probability']}** (probability).
- Explainability: permutation importance (global) and SHAP values (per student).

### Technologies
Python · pandas · NumPy · scikit-learn · XGBoost · SHAP · Matplotlib · Seaborn · Plotly · Streamlit · Jupyter

### Limitations
- Synthetic data: the model can only rediscover the rules written into the generator.
- The three models are trained separately, so occasionally the readiness category and the probability
  can look slightly inconsistent for borderline students.
- Scores such as "communication score" are simplifications of skills that are hard to measure.
- Real placement also depends on the job market, company requirements, interviews and chance.

### Ethical considerations
- Age and gender are collected in the form but **not used by the models**; a fairness audit
  (`outputs/metrics/fairness_by_gender.csv`) checks outcomes across gender groups.
- Predictions must never be used to rank, filter or deny opportunities to real students.
- A low estimate is a prompt for support (mentoring, practice, projects), not a verdict.
- Real student data would require informed consent, anonymisation and secure storage.
""")


{"Dashboard": page_dashboard, "Student Prediction": page_prediction,
 "Analytics": page_analytics, "About": page_about}[page]()
