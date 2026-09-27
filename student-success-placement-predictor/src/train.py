"""Train, compare, select and export all models.

Run from the project root:

    python -m src.train

Outputs
-------
models/   academic_model.pkl, placement_model.pkl, probability_model.pkl,
          preprocessing_pipeline.pkl, shap_background.pkl, model_metadata.json
outputs/metrics/  comparison tables, test metrics, feature importance, fairness audit
outputs/figures/  confusion matrices, importance charts, SHAP summaries, regression plots
outputs/reports/  data-cleaning report, classification reports, training summary
"""
from __future__ import annotations

import json
import warnings
from datetime import datetime, timezone

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.ensemble import (GradientBoostingClassifier,
                              GradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, mean_absolute_error,
                             mean_squared_error, precision_score, r2_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import (KFold, StratifiedKFold, cross_validate,
                                     train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from src.preprocessing import (build_preprocessor, clean_dataset,
                               detect_outliers_iqr, load_raw, readable_name,
                               transformed_feature_names)
from src.utils import (ACADEMIC_LABELS, FIG_DIR, METRICS_DIR, MODEL_FEATURES,
                       MODELS_DIR, PALETTE, RANDOM_SEED, READINESS_LABELS,
                       REPORTS_DIR, TARGET_ACADEMIC, TARGET_PROBABILITY,
                       TARGET_READINESS, TEST_SIZE, impact_label)

warnings.filterwarnings("ignore", category=UserWarning)

try:
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGB = True
except Exception:  # pragma: no cover - xgboost is optional
    HAS_XGB = False

try:
    import shap
    HAS_SHAP = True
except Exception:  # pragma: no cover - shap is optional
    HAS_SHAP = False

CV_FOLDS = 5
sns.set_theme(style="whitegrid", font_scale=0.95)


# --------------------------------------------------------------------------
# Model zoo
# --------------------------------------------------------------------------
def classifier_candidates() -> dict:
    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000, C=1.0, random_state=RANDOM_SEED),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, min_samples_leaf=10, random_state=RANDOM_SEED),
        "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=3,
                                                random_state=RANDOM_SEED, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, learning_rate=0.05,
                                                        max_depth=3, random_state=RANDOM_SEED),
    }
    if HAS_XGB:
        models["XGBoost"] = XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=3,
                                          subsample=0.9, colsample_bytree=0.9,
                                          eval_metric="mlogloss", random_state=RANDOM_SEED,
                                          n_jobs=2, verbosity=0)
    return models


def regressor_candidates() -> dict:
    models = {
        "Ridge Regression": Ridge(alpha=1.0),
        "Decision Tree": DecisionTreeRegressor(max_depth=6, min_samples_leaf=10, random_state=RANDOM_SEED),
        "Random Forest": RandomForestRegressor(n_estimators=300, min_samples_leaf=3,
                                               random_state=RANDOM_SEED, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=300, learning_rate=0.05,
                                                       max_depth=3, random_state=RANDOM_SEED),
    }
    if HAS_XGB:
        models["XGBoost"] = XGBRegressor(n_estimators=400, learning_rate=0.05, max_depth=3,
                                         subsample=0.9, colsample_bytree=0.9,
                                         random_state=RANDOM_SEED, n_jobs=2, verbosity=0)
    return models


def make_pipeline(estimator) -> Pipeline:
    return Pipeline([("preprocess", build_preprocessor()), ("model", estimator)])


# --------------------------------------------------------------------------
# Classification
# --------------------------------------------------------------------------
def run_classification(task: str, labels: list[str], X_train, y_train, X_test, y_test):
    """Cross-validate all candidates, pick the best by CV macro-F1, test it."""
    print(f"\n=== {task} ===")
    enc = {lab: i for i, lab in enumerate(labels)}
    ytr = y_train.map(enc).to_numpy()
    yte = y_test.map(enc).to_numpy()
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    scoring = {"accuracy": "accuracy", "precision": "precision_macro",
               "recall": "recall_macro", "f1": "f1_macro", "roc_auc": "roc_auc_ovr"}

    rows, fitted, cms = [], {}, {}
    for name, est in classifier_candidates().items():
        pipe = make_pipeline(clone(est))
        cvr = cross_validate(pipe, X_train, ytr, cv=cv, scoring=scoring, n_jobs=1)
        pipe.fit(X_train, ytr)
        pred = pipe.predict(X_test)
        proba = pipe.predict_proba(X_test)
        fitted[name] = pipe
        cms[name] = confusion_matrix(yte, pred, labels=range(len(labels)))
        row = {"model": name}
        for m in scoring:
            row[f"cv_{m}"] = cvr[f"test_{m}"].mean()
            row[f"cv_{m}_std"] = cvr[f"test_{m}"].std()
        row.update({
            "test_accuracy": accuracy_score(yte, pred),
            "test_precision": precision_score(yte, pred, average="macro", zero_division=0),
            "test_recall": recall_score(yte, pred, average="macro", zero_division=0),
            "test_f1": f1_score(yte, pred, average="macro"),
            "test_roc_auc": roc_auc_score(yte, proba, multi_class="ovr", average="macro"),
        })
        rows.append(row)
        print(f"  {name:20s} CV F1={row['cv_f1']:.3f}±{row['cv_f1_std']:.3f}  test F1={row['test_f1']:.3f}")

    table = pd.DataFrame(rows).sort_values("cv_f1", ascending=False).reset_index(drop=True)
    best = table.loc[0, "model"]
    print(f"  -> selected: {best} (highest mean cross-validated macro-F1)")

    best_pipe = fitted[best]
    pred = best_pipe.predict(X_test)
    report = classification_report(yte, pred, target_names=labels, digits=3)
    return table, best, best_pipe, cms, report


def plot_confusion_grid(cms: dict, labels: list[str], title: str, path, selected: str):
    n = len(cms)
    fig, axes = plt.subplots(1, n, figsize=(4.1 * n, 4.0))
    for ax, (name, cm) in zip(np.atleast_1d(axes), cms.items()):
        sns.heatmap(cm, annot=True, fmt="d", cbar=False, cmap="BuGn",
                    xticklabels=labels, yticklabels=labels, ax=ax,
                    annot_kws={"fontsize": 9})
        acc = np.trace(cm) / cm.sum()
        mark = "  (selected)" if name == selected else ""
        ax.set_title(f"{name}{mark}\naccuracy {acc:.1%}", fontsize=10)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.tick_params(axis="x", rotation=35, labelsize=8)
        ax.tick_params(axis="y", rotation=0, labelsize=8)
    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_single_confusion(cm, labels, title, path):
    fig, ax = plt.subplots(figsize=(6, 5))
    pct = cm / cm.sum(axis=1, keepdims=True)
    annot = np.array([[f"{c}\n({p:.0%})" for c, p in zip(r1, r2)] for r1, r2 in zip(cm, pct)])
    sns.heatmap(cm, annot=annot, fmt="", cmap="BuGn", cbar_kws={"label": "Number of students"},
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Predicted category")
    ax.set_ylabel("Actual category")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------
# Regression
# --------------------------------------------------------------------------
def run_regression(X_train, y_train, X_test, y_test):
    print("\n=== Placement probability (regression) ===")
    cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    scoring = {"mae": "neg_mean_absolute_error", "rmse": "neg_root_mean_squared_error", "r2": "r2"}
    rows, fitted = [], {}
    for name, est in regressor_candidates().items():
        pipe = make_pipeline(clone(est))
        cvr = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring, n_jobs=1)
        pipe.fit(X_train, y_train)
        pred = np.clip(pipe.predict(X_test), 0, 100)
        fitted[name] = pipe
        row = {"model": name,
               "cv_mae": -cvr["test_mae"].mean(), "cv_rmse": -cvr["test_rmse"].mean(),
               "cv_rmse_std": cvr["test_rmse"].std(), "cv_r2": cvr["test_r2"].mean(),
               "test_mae": mean_absolute_error(y_test, pred),
               "test_rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
               "test_r2": r2_score(y_test, pred)}
        rows.append(row)
        print(f"  {name:20s} CV RMSE={row['cv_rmse']:.2f}  test RMSE={row['test_rmse']:.2f}  test R²={row['test_r2']:.3f}")
    table = pd.DataFrame(rows).sort_values("cv_rmse").reset_index(drop=True)
    best = table.loc[0, "model"]
    print(f"  -> selected: {best} (lowest mean cross-validated RMSE)")
    return table, best, fitted[best]


def plot_regression(y_true, y_pred, name, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    ax = axes[0]
    ax.scatter(y_true, y_pred, s=14, alpha=0.55, color=PALETTE["teal"], label="Test students")
    ax.plot([0, 100], [0, 100], ls="--", color=PALETTE["brick"], label="Perfect prediction")
    ax.set_xlabel("Actual placement probability (%)")
    ax.set_ylabel("Predicted placement probability (%)")
    ax.set_title(f"{name}: predicted vs actual")
    ax.legend()
    resid = y_pred - y_true
    ax = axes[1]
    sns.histplot(resid, bins=35, color=PALETTE["sage"], ax=ax, label="Residuals")
    ax.axvline(0, color=PALETTE["brick"], ls="--", label="Zero error")
    ax.set_xlabel("Prediction error (percentage points)")
    ax.set_ylabel("Number of students")
    ax.set_title(f"Residual distribution (MAE {np.abs(resid).mean():.1f} pts)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------
# Explainability
# --------------------------------------------------------------------------
def permutation_table(pipe, X_test, y_test, scoring) -> pd.DataFrame:
    res = permutation_importance(pipe, X_test, y_test, scoring=scoring, n_repeats=10,
                                 random_state=RANDOM_SEED, n_jobs=1)
    df = pd.DataFrame({"feature": X_test.columns, "importance": res.importances_mean,
                       "std": res.importances_std})
    df["importance"] = df["importance"].clip(lower=0)
    df["share"] = df["importance"] / df["importance"].sum()
    df["impact"] = df["share"].apply(impact_label)
    df["label"] = df["feature"].map(readable_name)
    return df.sort_values("importance", ascending=False).reset_index(drop=True)


def plot_importance(tables: dict, path):
    fig, axes = plt.subplots(1, len(tables), figsize=(6.2 * len(tables), 5.6))
    colors = {"High": PALETTE["teal"], "Medium-high": PALETTE["sage"],
              "Medium": PALETTE["mustard"], "Low": "#B9C2C0"}
    for ax, (title, df) in zip(np.atleast_1d(axes), tables.items()):
        d = df.sort_values("share")
        ax.barh(d["label"], d["share"] * 100, xerr=d["std"] / df["importance"].sum() * 100,
                color=[colors[i] for i in d["impact"]], error_kw={"ecolor": "#555", "lw": 0.8})
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Share of permutation importance (%)")
        for y, (v, lab) in enumerate(zip(d["share"] * 100, d["impact"])):
            ax.text(v + 0.6, y, f"{v:.1f}%  {lab}", va="center", fontsize=8)
        ax.set_xlim(0, d["share"].max() * 100 * 1.35)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colors.values()]
    fig.legend(handles, [f"{k} impact" for k in colors], loc="lower center", ncol=4, frameon=False)
    fig.suptitle("Which inputs drive each model? (permutation importance on the test set)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def shap_summary(pipe, X_test, title, path, class_index=None):
    """Global SHAP beeswarm for a fitted pipeline. Returns True on success."""
    if not HAS_SHAP:
        return False
    pre, est = pipe.named_steps["preprocess"], pipe.named_steps["model"]
    Xt = pd.DataFrame(pre.transform(X_test), columns=[readable_name(n) for n in transformed_feature_names(pre)])
    try:
        explainer = shap.TreeExplainer(est)
        sv = explainer(Xt)
    except Exception:
        explainer = shap.Explainer(est, Xt)
        sv = explainer(Xt)
    if sv.values.ndim == 3:
        sv = sv[:, :, class_index if class_index is not None else -1]
    plt.figure()
    shap.plots.beeswarm(sv, max_display=14, show=False)
    plt.title(title, fontsize=11, fontweight="bold")
    plt.xlabel("SHAP value (impact on model output)")
    plt.tight_layout()
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close("all")
    return True


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    for d in (MODELS_DIR, FIG_DIR, METRICS_DIR, REPORTS_DIR):
        d.mkdir(parents=True, exist_ok=True)

    raw = load_raw()
    print(f"Loaded raw dataset: {raw.shape}")
    detect_outliers_iqr(raw).to_csv(METRICS_DIR / "iqr_outlier_report.csv", index=False)
    df, clean_report = clean_dataset(raw)
    (REPORTS_DIR / "data_cleaning_report.json").write_text(json.dumps(clean_report, indent=2))

    X = df[MODEL_FEATURES]
    y_acad, y_ready, y_prob = df[TARGET_ACADEMIC], df[TARGET_READINESS], df[TARGET_PROBABILITY]
    idx_train, idx_test = train_test_split(df.index, test_size=TEST_SIZE,
                                           random_state=RANDOM_SEED, stratify=y_ready)
    X_train, X_test = X.loc[idx_train], X.loc[idx_test]
    print(f"Train: {len(idx_train)} rows  |  Test: {len(idx_test)} rows (held out, untouched until final evaluation)")

    # ---- Academic performance
    acad_table, acad_best, acad_pipe, acad_cms, acad_report = run_classification(
        "Academic performance", ACADEMIC_LABELS, X_train, y_acad.loc[idx_train], X_test, y_acad.loc[idx_test])
    # ---- Placement readiness
    ready_table, ready_best, ready_pipe, ready_cms, ready_report = run_classification(
        "Placement readiness", READINESS_LABELS, X_train, y_ready.loc[idx_train], X_test, y_ready.loc[idx_test])
    # ---- Placement probability
    prob_table, prob_best, prob_pipe = run_regression(X_train, y_prob.loc[idx_train], X_test, y_prob.loc[idx_test])

    acad_table.round(4).to_csv(METRICS_DIR / "academic_model_comparison.csv", index=False)
    ready_table.round(4).to_csv(METRICS_DIR / "placement_model_comparison.csv", index=False)
    prob_table.round(4).to_csv(METRICS_DIR / "probability_model_comparison.csv", index=False)
    (REPORTS_DIR / "academic_classification_report.txt").write_text(f"Selected model: {acad_best}\n\n{acad_report}")
    (REPORTS_DIR / "placement_classification_report.txt").write_text(f"Selected model: {ready_best}\n\n{ready_report}")

    # ---- Figures: confusion matrices
    plot_confusion_grid(acad_cms, ACADEMIC_LABELS, "Academic performance: confusion matrices (test set)",
                        FIG_DIR / "confusion_matrices_academic_all.png", acad_best)
    plot_confusion_grid(ready_cms, READINESS_LABELS, "Placement readiness: confusion matrices (test set)",
                        FIG_DIR / "confusion_matrices_placement_all.png", ready_best)
    plot_single_confusion(acad_cms[acad_best], ACADEMIC_LABELS,
                          f"Academic performance: {acad_best}", FIG_DIR / "confusion_matrix_academic.png")
    plot_single_confusion(ready_cms[ready_best], READINESS_LABELS,
                          f"Placement readiness: {ready_best}", FIG_DIR / "confusion_matrix_placement.png")
    prob_pred = np.clip(prob_pipe.predict(X_test), 0, 100)
    plot_regression(y_prob.loc[idx_test].to_numpy(), prob_pred, prob_best, FIG_DIR / "probability_regression.png")

    # ---- Model comparison chart
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.3))
    for ax, (t, table, col, lab) in zip(axes, [
            ("Academic performance", acad_table, "cv_f1", "CV macro-F1"),
            ("Placement readiness", ready_table, "cv_f1", "CV macro-F1"),
            ("Placement probability", prob_table, "cv_rmse", "CV RMSE (lower is better)")]):
        d = table.sort_values(col, ascending=(col == "cv_rmse"))
        colors = [PALETTE["teal"] if i == 0 else "#B9C2C0" for i in range(len(d))]
        ax.barh(d["model"][::-1], d[col][::-1], color=colors[::-1],
                xerr=d[col + "_std"][::-1], error_kw={"lw": 0.8})
        for y, v in enumerate(d[col][::-1]):
            ax.text(v, y, f" {v:.3f}" if col != "cv_rmse" else f" {v:.2f}", va="center", fontsize=8)
        ax.set_title(t, fontweight="bold")
        ax.set_xlabel(lab)
        if col == "cv_f1":
            ax.set_xlim(max(0, d[col].min() - 0.15), 1.0)
    fig.suptitle("Model comparison (5-fold cross-validation on the training set; selected model in teal)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "model_comparison.png", dpi=130)
    plt.close(fig)

    # ---- Explainability
    ytest_acad = y_acad.loc[idx_test].map({l: i for i, l in enumerate(ACADEMIC_LABELS)})
    ytest_ready = y_ready.loc[idx_test].map({l: i for i, l in enumerate(READINESS_LABELS)})
    imp_acad = permutation_table(acad_pipe, X_test, ytest_acad, "f1_macro")
    imp_ready = permutation_table(ready_pipe, X_test, ytest_ready, "f1_macro")
    imp_prob = permutation_table(prob_pipe, X_test, y_prob.loc[idx_test], "neg_root_mean_squared_error")
    for name, t in [("academic", imp_acad), ("placement", imp_ready), ("probability", imp_prob)]:
        t.round(5).to_csv(METRICS_DIR / f"feature_importance_{name}.csv", index=False)
    plot_importance({"Academic performance": imp_acad, "Placement readiness": imp_ready,
                     "Placement probability": imp_prob}, FIG_DIR / "feature_importance.png")
    shap_ok = shap_summary(prob_pipe, X_test, "SHAP: placement probability model",
                           FIG_DIR / "shap_summary_probability.png")
    shap_summary(ready_pipe, X_test, "SHAP: placement readiness ('Highly Ready' class)",
                 FIG_DIR / "shap_summary_placement.png", class_index=READINESS_LABELS.index("Highly Ready"))
    shap_summary(acad_pipe, X_test, "SHAP: academic performance ('Excellent' class)",
                 FIG_DIR / "shap_summary_academic.png", class_index=ACADEMIC_LABELS.index("Excellent"))

    # ---- Fairness audit (gender is NOT a model input; we check outcomes by group)
    test_df = df.loc[idx_test].copy()
    test_df["pred_readiness"] = [READINESS_LABELS[i] for i in ready_pipe.predict(X_test)]
    test_df["pred_probability"] = prob_pred
    fair = test_df.groupby("gender").apply(lambda g: pd.Series({
        "students": len(g),
        "readiness_accuracy": (g["pred_readiness"] == g[TARGET_READINESS]).mean(),
        "mean_actual_probability": g[TARGET_PROBABILITY].mean(),
        "mean_predicted_probability": g["pred_probability"].mean(),
        "pct_predicted_ready_or_above": g["pred_readiness"].isin(["Ready", "Highly Ready"]).mean(),
    }), include_groups=False).round(3)
    fair.to_csv(METRICS_DIR / "fairness_by_gender.csv")

    # ---- Final test metrics of the selected models
    final = {
        "academic": acad_table.loc[0].to_dict(),
        "placement": ready_table.loc[0].to_dict(),
        "probability": prob_table.loc[0].to_dict(),
    }
    (METRICS_DIR / "final_test_metrics.json").write_text(json.dumps(final, indent=2, default=float))

    # ---- Save artifacts
    joblib.dump(acad_pipe, MODELS_DIR / "academic_model.pkl")
    joblib.dump(ready_pipe, MODELS_DIR / "placement_model.pkl")
    joblib.dump(prob_pipe, MODELS_DIR / "probability_model.pkl")
    preprocessor = clone(build_preprocessor()).fit(X_train)
    joblib.dump(preprocessor, MODELS_DIR / "preprocessing_pipeline.pkl")
    background = X_train.sample(100, random_state=RANDOM_SEED)
    joblib.dump(background, MODELS_DIR / "shap_background.pkl")
    meta = {
        "trained_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
        "random_seed": RANDOM_SEED, "test_size": TEST_SIZE, "cv_folds": CV_FOLDS,
        "rows_after_cleaning": int(len(df)), "train_rows": int(len(idx_train)), "test_rows": int(len(idx_test)),
        "model_features": MODEL_FEATURES,
        "transformed_features": transformed_feature_names(preprocessor),
        "academic_labels": ACADEMIC_LABELS, "readiness_labels": READINESS_LABELS,
        "selected_models": {"academic": acad_best, "placement": ready_best, "probability": prob_best},
        "selection_rule": {"classification": "highest mean 5-fold CV macro-F1 on training set",
                           "regression": "lowest mean 5-fold CV RMSE on training set"},
        "shap_available": bool(shap_ok),
        "xgboost_available": HAS_XGB,
        "test_metrics": {
            "academic": {k: round(float(final["academic"][f"test_{k}"]), 4)
                         for k in ["accuracy", "precision", "recall", "f1", "roc_auc"]},
            "placement": {k: round(float(final["placement"][f"test_{k}"]), 4)
                          for k in ["accuracy", "precision", "recall", "f1", "roc_auc"]},
            "probability": {k: round(float(final["probability"][f"test_{k}"]), 4)
                            for k in ["mae", "rmse", "r2"]},
        },
        "synthetic_data": True,
    }
    (MODELS_DIR / "model_metadata.json").write_text(json.dumps(meta, indent=2))
    write_summary(meta, acad_table, ready_table, prob_table, imp_ready, imp_prob, fair, clean_report)
    print("\nAll artifacts saved. Selected models:", meta["selected_models"])
    return meta


def _md_table(df: pd.DataFrame, cols: dict, fmt="{:.3f}") -> str:
    head = "| " + " | ".join(cols.values()) + " |\n|" + "---|" * len(cols) + "\n"
    body = ""
    for _, r in df.iterrows():
        cells = [str(r[c]) if isinstance(r[c], str) else fmt.format(r[c]) for c in cols]
        body += "| " + " | ".join(cells) + " |\n"
    return head + body


def write_summary(meta, acad, ready, prob, imp_ready, imp_prob, fair, clean):
    cls_cols = {"model": "Model", "cv_f1": "CV macro-F1", "cv_accuracy": "CV accuracy",
                "test_accuracy": "Test accuracy", "test_precision": "Test precision",
                "test_recall": "Test recall", "test_f1": "Test macro-F1", "test_roc_auc": "Test ROC-AUC"}
    reg_cols = {"model": "Model", "cv_rmse": "CV RMSE", "cv_mae": "CV MAE", "cv_r2": "CV R²",
                "test_mae": "Test MAE", "test_rmse": "Test RMSE", "test_r2": "Test R²"}
    imp_cols = {"label": "Feature", "share": "Share", "impact": "Impact"}
    txt = f"""# Training summary

Generated {meta['trained_at_utc']} UTC. **All data is synthetic** — see README.

## Data
- Rows after cleaning: {meta['rows_after_cleaning']} (train {meta['train_rows']}, test {meta['test_rows']}, stratified by placement readiness, seed {meta['random_seed']})
- Duplicates removed: {clean['duplicates_removed']}
- Impossible values replaced with NaN (then median-imputed inside the pipeline): {clean['impossible_values_set_to_nan']}
- Model inputs: {len(meta['model_features'])} academic/skill features. Age and gender are **excluded** from modelling.

## Selection rule
Classification: highest mean 5-fold cross-validated macro-F1 on the training set.
Regression: lowest mean 5-fold cross-validated RMSE. The test set is used only once, for final reporting.

## Academic performance — selected: **{meta['selected_models']['academic']}**
{_md_table(acad, cls_cols)}
## Placement readiness — selected: **{meta['selected_models']['placement']}**
{_md_table(ready, cls_cols)}
## Placement probability — selected: **{meta['selected_models']['probability']}**
{_md_table(prob, reg_cols, fmt="{:.3f}")}
## Top drivers of placement readiness
{_md_table(imp_ready.head(8), imp_cols)}
## Top drivers of placement probability
{_md_table(imp_prob.head(8), imp_cols)}
## Fairness audit by gender (test set; gender is not a model input)
{fair.to_markdown() if hasattr(fair, 'to_markdown') else fair.to_string()}
"""
    (REPORTS_DIR / "training_summary.md").write_text(txt)


if __name__ == "__main__":
    main()
