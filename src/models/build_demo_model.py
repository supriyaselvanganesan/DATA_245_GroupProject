"""Fit one set-D RBF SVM on gene-disjoint repeat 0, fold 0 for the live demo."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

def find_project_root():
    for candidate in (Path.cwd(), *Path.cwd().parents):
        if (candidate / "data/processed" / "fold_assignments.csv").is_file() and (candidate / "data/processed" / "clinvar_expanded_model_features.csv").is_file():
            return candidate
    raise FileNotFoundError("Could not find project root; run from the repository root or notebooks folder.")

PROJECT_ROOT = find_project_root()
DATA_DIR = PROJECT_ROOT / "data/processed"
RESULTS_DIR = PROJECT_ROOT / "results" / "rbf_svm"
OUTPUT_PATH = PROJECT_ROOT / "results" / "rbf_svm" / "rbf_svm_setD_gene_disjoint_r0_f0.joblib"
THRESHOLD = 0.5
SEED = 42

def make_model():
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("svm", SVC(kernel="rbf", class_weight="balanced", cache_size=500)),
    ])

def main():
    config = json.loads((DATA_DIR / "feature_config.json").read_text(encoding="utf-8"))
    feature_cols = config["feature_sets"]["D"]
    data = pd.read_csv(DATA_DIR / "clinvar_expanded_model_features.csv")
    folds = pd.read_csv(DATA_DIR / "fold_assignments.csv")
    outer = folds[(folds["validation"] == "gene_disjoint") & (folds["repeat"] == 0)]
    test_ids = set(outer.loc[outer["test_fold"] == 0, "VariationID"])
    train_ids = set(outer.loc[outer["test_fold"] != 0, "VariationID"])
    train = data[data["VariationID"].isin(train_ids)].copy()
    test = data[data["VariationID"].isin(test_ids)].copy()
    if train.empty or test.empty or train["GeneSymbol"].isin(test["GeneSymbol"]).any():
        raise ValueError("The selected outer split is empty or has gene overlap.")

    fold_results = pd.read_csv(RESULTS_DIR / "svm_fold_results.csv")
    selected = fold_results[(fold_results["scheme"] == "gene_disjoint") & (fold_results["set"] == "D") & (fold_results["repeat"] == 0) & (fold_results["fold"] == 0)]
    if len(selected) != 1:
        raise ValueError("Expected one saved tuning result for set D, gene-disjoint repeat 0 fold 0.")
    best = selected.iloc[0]
    X_train = train[feature_cols].to_numpy()
    y_train = train["label"].to_numpy()
    groups = train["GeneSymbol"].to_numpy()
    inner = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=SEED)
    inner_splits = list(inner.split(X_train, y_train, groups=groups))
    params = {"svm__C": float(best["best_C"]), "svm__gamma": float(best["best_gamma"])}

    # Calibrate only with out-of-fold scores from this outer training split.
    oof_scores = cross_val_predict(make_model().set_params(**params), X_train, y_train, cv=inner_splits, method="decision_function")
    platt = LogisticRegression(C=1e6).fit(oof_scores.reshape(-1, 1), y_train)
    model = make_model().set_params(**params).fit(X_train, y_train)
    artifact = {
        "model": model,
        "platt": platt,
        "feature_cols": feature_cols,
        "threshold": THRESHOLD,
        "validation": "gene_disjoint",
        "repeat": 0,
        "fold": 0,
        "best_C": params["svm__C"],
        "best_gamma": params["svm__gamma"],
        "heldout_ids": sorted(test_ids),
        "sklearn_version": sklearn.__version__,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, OUTPUT_PATH, compress=3)
    print(f"Saved demo model: {OUTPUT_PATH}")
    print(f"Compressed size: {OUTPUT_PATH.stat().st_size / (1024**2):.2f} MiB")
    print(f"Training rows: {len(train):,}; held-out rows: {len(test):,}; features: {len(feature_cols)}")
    print(f"Selected C={params['svm__C']}, gamma={params['svm__gamma']}")

if __name__ == "__main__":
    main()
