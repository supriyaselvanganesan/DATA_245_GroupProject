"""Python export of rbf_svm_demo.ipynb. Notebook outputs are omitted."""

# %% [markdown] cell 0
# # RBF SVM Live Demo
#
# This notebook loads a saved Set D RBF SVM trained on gene-disjoint repeat 0, fold 0. It predicts three selected variants from different genes in that fold's held-out test set.
#
# The trained model is already included in `results/rbf_svm/`. The demo loads it directly and does not rerun model training or cross-validation. If you want to rebuild it, run `src/models/build_demo_model.py` first. This is an educational demo, not a clinical decision tool.

# %% code cell 1
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

def find_project_root():
    for candidate in (Path.cwd(), *Path.cwd().parents):
        if (candidate / "data/processed" / "fold_assignments.csv").is_file() and (candidate / "data/processed" / "clinvar_expanded_model_features.csv").is_file():
            return candidate
    raise FileNotFoundError("Could not find project root. Run this notebook from the repository root or notebooks folder.")

PROJECT_ROOT = find_project_root()
DATA_DIR = PROJECT_ROOT / "data/processed"
MODEL_PATH = PROJECT_ROOT / "results" / "rbf_svm" / "rbf_svm_setD_gene_disjoint_r0_f0.joblib"
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Demo model not found at {MODEL_PATH}. Run build_demo_model.py once before presenting.")
artifact = joblib.load(MODEL_PATH)
model = artifact["model"]
platt = artifact["platt"]
feature_cols = artifact["feature_cols"]
threshold = artifact["threshold"]
print(f"Loaded set-D model from gene-disjoint fold {artifact['fold']} (scikit-learn {artifact['sklearn_version']}).")

# %% code cell 2
import pandas as pd
import numpy as np

# Three selected variants from different held-out genes
demo_ids = [1008673, 938741, 242459]

# Load only the columns needed for the demo
columns = ["VariationID", "GeneSymbol", "label"] + feature_cols

data = pd.read_csv(
    DATA_DIR / "clinvar_expanded_model_features.csv",
    usecols=columns
)

# Select the three variants in the chosen order
samples = data.set_index("VariationID").loc[demo_ids].reset_index()

# Confirm these variants belong to the saved model's test fold
assert set(demo_ids).issubset(set(artifact["heldout_ids"]))
assert samples["GeneSymbol"].nunique() == 3

# Predict only these three variants
X_demo = samples[feature_cols].to_numpy(dtype=float)

scores = model.decision_function(X_demo)

# Binary Platt calibration: sigmoid(coef * SVM score + intercept).
# Using the saved coefficients also avoids scikit-learn version-specific
# predict_proba internals when loading the model on another machine.
if not np.array_equal(platt.classes_, [0, 1]):
    raise ValueError("The saved calibrator does not use benign=0 and pathogenic=1.")
logits = scores * float(platt.coef_[0, 0]) + float(platt.intercept_[0])
probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -700, 700)))

predictions = (probabilities >= threshold).astype(int)

labels = {0: "Benign", 1: "Pathogenic"}

results = pd.DataFrame({
    "Variant ID": samples["VariationID"],
    "Gene": samples["GeneSymbol"],
    "Actual": samples["label"].map(labels),
    "Predicted": [labels[p] for p in predictions],
    "Pathogenic probability": probabilities.round(3)
})

results["Result"] = np.where(
    samples["label"].to_numpy() == predictions,
    "Correct",
    "Incorrect"
)

print("RBF SVM Demo: Predictions on Unseen Genes")
print(results.to_string(index=False))
