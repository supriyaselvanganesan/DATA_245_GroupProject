# Predicting Missense Variant Pathogenicity

## Overview

This DATA 245 team project studies whether classical machine learning models can predict whether missense genetic variants are pathogenic or benign.

The project uses a cleaned ClinVar-based dataset containing approximately 46,000 variants from about 2,100 genes. Several models are evaluated using standard validation and gene-disjoint validation.

The main goal is to determine whether the models can generalize to variants from genes that were not included during training.

## Research Questions

1. How accurately can classical machine learning models predict missense variant pathogenicity?
2. How does performance change when testing on completely unseen genes?
3. Which biological feature groups contribute most to model performance?
4. How do generative and discriminative models compare?
5. Does standard validation provide overly optimistic performance estimates?

## Models

The project includes:

- Regularized Logistic Regression
- RBF-SVM
- Multi-Kernel SVM
- Generative Gaussian Mixture Classifier
- Majority-class baseline
- Conservation-only baseline
- Conformal prediction analysis
- Confidence calibration and abstention analysis

The Gaussian mixture classifier includes:

- Expectation-Maximization implementation
- Diagonal and tied covariance structures
- BIC-based model selection
- Multiple random starts
- Bayesian classification
- Stability and convergence analysis

## Dataset

The dataset is based on ClinVar variant annotations combined with biological feature information.

It contains approximately:

- 46,000 missense variants
- 2,100 genes
- 28 biological features

The main feature groups are:

- Conservation features
- Population-frequency features
- Amino-acid property features

The processed data files are located in:

    data/processed/

Important files include:

- `clinvar_expanded_model_features.csv`
- `clinvar_expanded_project_dataset.csv`
- `feature_config.json`
- `fold_assignments.csv`

## Feature Configurations

The following feature configurations are evaluated:

- **Set A:** Conservation features only
- **Set B:** Conservation and population-frequency features
- **Set D:** All available biological features

These configurations measure the effect of adding feature groups step by step.

## Evaluation Strategy

Each model is evaluated using two validation strategies.

### Standard Stratified Folds

The data is divided into folds while preserving class proportions. Variants from the same gene may appear in both the training and testing sets.

### Gene-Disjoint Folds

All variants from the same gene are assigned to the same fold. Therefore, the genes in the test fold are completely unseen during training.

The experiments use:

- Five folds
- Three repeated experiments
- Fifteen test folds in total

Preprocessing, imputation, scaling, feature selection, hyperparameter tuning, and model selection are performed using training data only.

## Evaluation Metrics

The project reports:

- ROC-AUC
- PR-AUC
- Matthews Correlation Coefficient
- Accuracy
- Balanced accuracy
- Precision for pathogenic variants
- Recall or sensitivity for pathogenic variants
- Specificity for benign variants
- F1-score for pathogenic variants
- Confusion-matrix counts

Additional analyses include:

- Standard versus gene-disjoint performance
- Feature-set comparisons
- BIC-based component selection
- Covariance-structure comparison
- Model convergence
- Stability across random starts
- Runtime analysis
- Large-gene sensitivity analysis
- Gene-bootstrap confidence intervals
- Corrected resampled t-tests
- Calibration and abstention analysis
- Conformal prediction analysis
- Final comparison across team models

## Hypotheses

### H1: Effect of Additional Feature Groups

Adding population-frequency and amino-acid features improves PR-AUC compared with using conservation features alone when evaluated on unseen genes.

### H2: Potential Label Circularity

The improvement from adding population-frequency features is smaller on unseen genes than on standard folds.

This examines whether part of the observed improvement may be related to information connected to the labeling process.

## Team Contributions

- **Rishitha:** RBF-SVM modeling, hyperparameter tuning, runtime analysis, and memory analysis
- **Sharan:** Regularized Logistic Regression, majority-class baseline, and conservation-based baseline
- **Supriya:** Gaussian Mixture Classifier, EM implementation, BIC model selection, Bayesian classification, stability analysis, unseen-gene evaluation, and H1/H2 analysis
- **Sanjana:** Multi-Kernel SVM, confidence calibration, and abstention analysis
- **Bavya:** Data-quality analysis, review-status sensitivity analysis, and conformal prediction

## Technologies

- Python
- NumPy
- pandas
- SciPy
- scikit-learn
- Matplotlib
- joblib
- threadpoolctl
- Jupyter Notebook

## Installation

Clone the repository:

    git clone https://github.com/supriyaselvanganesan/DATA_245_GroupProject.git
    cd DATA_245_GroupProject

Install the main dependencies:

    pip install numpy pandas scipy scikit-learn matplotlib joblib threadpoolctl jupyter

## Running the Project

Start Jupyter Notebook:

    jupyter notebook

The analysis notebooks are located in:

    notebooks/

The model implementations are located in:

    src/models/

The generated tables, metrics, predictions, and figures are stored in:

    results/

## Results

The final results compare the team models using:

- Standard validation
- Gene-disjoint validation
- Multiple feature configurations
- Multiple evaluation metrics
- Fold-level performance variation

The final comparison files are stored in:

    results/final_comparison/

Model-specific results are stored in:

    results/gaussian_mixture/
    results/logistic_regression/
    results/rbf_svm/
    results/MultiKernelSVM_Results/

## Project Goal

This project does not focus only on obtaining the highest classification score.

Instead, it evaluates whether model performance remains reliable when predicting variants from genes that were not included during training.

The analysis helps identify:

- Which biological feature groups are most useful
- How performance changes on unseen genes
- Whether standard validation may overestimate performance
- How generative and discriminative models compare
- Whether a Gaussian mixture classifier is useful for this task
- How calibration and abstention affect model reliability
- How classical machine learning methods can support missense variant analysis

## Limitations

The dataset is based on existing ClinVar annotations, so label quality depends on the available clinical evidence.

The models are evaluated for research and educational purposes. The results should not be interpreted as clinical validation.

Performance may vary across variant populations, genes, annotation sources, and feature availability.

## Reproducibility

The repository contains the notebooks, source files, processed data references, result tables, predictions, and generated figures used in the analysis.

The experiments use fixed validation folds and recorded random seeds where applicable. Intermediate results are saved so completed model fits can be inspected without rerunning the full analysis.

## Important Note

This project is intended for research and educational evaluation only.

The model predictions should not be used as a substitute for expert clinical, genetic, or medical interpretation.
