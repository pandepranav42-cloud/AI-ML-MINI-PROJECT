"""Load trained models and produce explained predictions for one student.

Usage from Python:

    from src.predict import StudentPredictor
    p = StudentPredictor()
    result = p.predict({...})

Usage from the command line (prints a demo prediction):

    python -m src.predict
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import joblib
import numpy as np
import pandas as pd

from src.preprocessing import base_feature, readable_name, transformed_feature_names
from src.utils import (ACADEMIC_LABELS, FEATURE_LABELS, MODEL_FEATURES,
                       MODELS_DIR, READINESS_LABELS, validate_input)

try:
    import shap
    HAS_SHAP = True
except Exception:  # pragma: no cover
    HAS_SHAP = False


class InvalidInputError(ValueError):
    def __init__(self, errors: list[str]):
        super().__init__("; ".join(errors))
        self.errors = errors


@dataclass
class Factor:
    feature: str          # raw column name
    label: str            # human-readable name
    value: object         # the student's value
    contribution: float   # signed contribution (percentage points for probability)

    @property
    def direction(self) -> str:
        return "raises" if self.contribution > 0 else "lowers"


@dataclass
class PredictionResult:
    academic_performance: str
    academic_confidence: float
    academic_probabilities: dict
    placement_readiness: str
    readiness_confidence: float
    readiness_probabilities: dict
    placement_probability: float
    baseline_probability: float
    factors: list[Factor] = field(default_factory=list)
    readiness_factors: list[Factor] = field(default_factory=list)
    explanation_method: str = ""

    def top_factors(self, n: int = 6) -> list[Factor]:
        return sorted(self.factors, key=lambda f: abs(f.contribution), reverse=True)[:n]

    def summary(self) -> str:
        lines = [
            f"Academic Performance      : {self.academic_performance} ({self.academic_confidence:.0%} model confidence)",
            f"Placement Readiness       : {self.placement_readiness} ({self.readiness_confidence:.0%} model confidence)",
            f"Estimated Placement Prob. : {self.placement_probability:.0f}%",
            f"Main factors ({self.explanation_method}, vs. average student at {self.baseline_probability:.0f}%):",
        ]
        for f in self.top_factors():
            lines.append(f"  {f.label:24s} = {str(f.value):>6s}  {f.contribution:+6.1f} pts")
        return "\n".join(lines)


class StudentPredictor:
    def __init__(self, models_dir=MODELS_DIR):
        self.academic = joblib.load(models_dir / "academic_model.pkl")
        self.placement = joblib.load(models_dir / "placement_model.pkl")
        self.probability = joblib.load(models_dir / "probability_model.pkl")
        self.background = joblib.load(models_dir / "shap_background.pkl")
        self.meta = json.loads((models_dir / "model_metadata.json").read_text())
        self._explainers: dict = {}

    # ------------------------------------------------------------------
    @staticmethod
    def to_frame(record: dict) -> pd.DataFrame:
        row = {}
        for col in MODEL_FEATURES:
            v = record[col]
            row[col] = v if col in ("internship", "hackathon") else float(v)
        return pd.DataFrame([row], columns=MODEL_FEATURES)

    def _explainer(self, name: str, pipe):
        """Cache one SHAP explainer per model (tree or linear, else permutation)."""
        if name in self._explainers:
            return self._explainers[name]
        pre, est = pipe.named_steps["preprocess"], pipe.named_steps["model"]
        bg = pre.transform(self.background)
        explainer, method = None, None
        if HAS_SHAP:
            for build, label in (
                (lambda: shap.TreeExplainer(est), "SHAP (tree)"),
                (lambda: shap.LinearExplainer(est, bg), "SHAP (linear)"),
            ):
                try:
                    explainer = build()
                    method = label
                    break
                except Exception:
                    continue
        self._explainers[name] = (explainer, method, pre, est, bg)
        return self._explainers[name]

    def _contributions(self, name, pipe, X: pd.DataFrame, class_index=None):
        explainer, method, pre, est, bg = self._explainer(name, pipe)
        Xt = pre.transform(X)
        names = transformed_feature_names(pre)
        if explainer is not None:
            sv = explainer(Xt)
            values, base = np.asarray(sv.values), np.asarray(sv.base_values)
            if values.ndim == 3:
                values = values[:, :, class_index]
                base = base[:, class_index] if base.ndim == 2 else base[class_index]
            return values[0], float(np.ravel(base)[0]), names, method
        # Fallback without SHAP: occlusion — replace each feature with the
        # background mean and measure the change in prediction.
        def f(Z):
            return (est.predict_proba(Z)[:, class_index] if class_index is not None else est.predict(Z))
        ref = bg.mean(axis=0)
        full = f(Xt)[0]
        contrib = []
        for j in range(Xt.shape[1]):
            Z = Xt.copy()
            Z[0, j] = ref[j]
            contrib.append(full - f(Z)[0])
        return np.array(contrib), float(f(ref[None, :])[0]), names, "occlusion"

    @staticmethod
    def _factors(values, names, record) -> list[Factor]:
        out = []
        for v, n in zip(values, names):
            raw = base_feature(n)
            out.append(Factor(raw, readable_name(n), record.get(raw), float(v)))
        return out

    # ------------------------------------------------------------------
    def predict(self, record: dict, explain: bool = True) -> PredictionResult:
        errors = validate_input(record)
        if errors:
            raise InvalidInputError(errors)
        X = self.to_frame(record)

        acad_p = self.academic.predict_proba(X)[0]
        ready_p = self.placement.predict_proba(X)[0]
        # Clip to 1-99%: the model never has grounds for absolute certainty.
        prob = float(np.clip(self.probability.predict(X)[0], 1, 99))
        acad_i, ready_i = int(np.argmax(acad_p)), int(np.argmax(ready_p))

        result = PredictionResult(
            academic_performance=ACADEMIC_LABELS[acad_i],
            academic_confidence=float(acad_p[acad_i]),
            academic_probabilities={l: float(p) for l, p in zip(ACADEMIC_LABELS, acad_p)},
            placement_readiness=READINESS_LABELS[ready_i],
            readiness_confidence=float(ready_p[ready_i]),
            readiness_probabilities={l: float(p) for l, p in zip(READINESS_LABELS, ready_p)},
            placement_probability=prob,
            baseline_probability=prob,
        )
        if explain:
            vals, base, names, method = self._contributions("probability", self.probability, X)
            result.factors = self._factors(vals, names, record)
            result.baseline_probability = float(np.clip(base, 0, 100))
            result.explanation_method = method
            rvals, _, rnames, _ = self._contributions("placement", self.placement, X, class_index=ready_i)
            result.readiness_factors = self._factors(rvals, rnames, record)
        return result


EXAMPLE_STUDENT = {
    "age": 21, "gender": "Female", "cgpa": 8.4, "attendance_pct": 88, "backlogs": 0,
    "coding_score": 78, "aptitude_score": 72, "communication_score": 70, "projects": 3,
    "certifications": 2, "internship": "Yes", "hackathon": "Yes", "extracurricular_score": 65,
    "technical_skills_score": 68, "programming_languages": 4, "study_hours_per_week": 18,
}


if __name__ == "__main__":
    predictor = StudentPredictor()
    print(predictor.predict(EXAMPLE_STUDENT).summary())
