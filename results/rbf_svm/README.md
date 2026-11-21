# RBF SVM contribution

Rishitha Gogineni implemented the shared data preparation and the RBF SVM experiment. The data-preparation and RBF experiment notebooks remain in `notebooks/` with matching Python files in `src/` and `src/models/`. The live demo notebook and saved model are in `src/saved_models/`; its matching Python export remains in `src/models/`.

## Quick demo

From the repository root, open `src/saved_models/rbf_svm_demo.ipynb` and run its two code cells, or run `python src/models/rbf_svm_demo.py`. It loads `rbf_svm_setD_gene_disjoint_r0_f0.joblib` from `src/saved_models/` and scores three variants from genes held out in gene-disjoint repeat 0, fold 0. The saved model has already been fitted. The demo does not retrain it.

The demo needs `data/processed/clinvar_expanded_model_features.csv` and `data/processed/fold_assignments.csv`. It uses NumPy, pandas, scikit-learn, and joblib. The model was saved with scikit-learn 1.9.1; use a matching version if model loading reports a compatibility warning. Do not use its predictions for clinical decisions.

## Full experiment

`notebooks/rbf_svm_rishitha.ipynb` contains the 90-fold RBF SVM experiment and analysis. A complete fresh run is computationally expensive. This directory contains the small result tables and figures; the large per-variant prediction files and intermediate fold checkpoints are not committed.

`notebooks/data_preparation.ipynb` downloads the public source dataset from Zenodo, checks its MD5 checksum, and writes generated files under `data/processed/`. It places the downloaded source in `data/raw/`. That notebook also needs Biopython, Matplotlib, and the UCSC conservation-score utility for its full preparation workflow.

The PDF report figures are in `figures/report_figures/`; the individual RBF SVM evaluation plots are in `figures/evaluation_plots/`.
