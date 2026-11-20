# Predicting Missense Variant Pathogenicity

## Overview

This project applies classical machine learning methods to predict whether missense genetic variants are pathogenic or benign.

The study uses a cleaned ClinVar-based dataset containing approximately 46,000 variants from around 2,100 genes. Multiple machine learning models are compared under standard validation and unseen-gene evaluation.

The main focus is determining whether these models can generalize to genes that were not included during training.

## Research Question

How accurately can classical machine learning models predict missense variant pathogenicity for previously unseen genes, and which biological feature groups contribute most to model performance?

## Models

This repository includes the following models:

- Regularized Logistic Regression
- RBF-SVM
- Multi-Kernel SVM
- Generative Gaussian Mixture Classifier
- Majority-class baseline
- Conservation-based baseline

## Dataset and Features

The dataset is based on ClinVar variant annotations and contains 28 biological features grouped into:

- Conservation features
- Population-frequency features
- Amino-acid property features

The following feature configurations are evaluated:

- **Set A:** Conservation features only
- **Set B:** Conservation and population-frequency features
- **Set D:** All available features

## Evaluation Strategy

Each model is evaluated using two validation strategies.

### Standard Stratified Folds

The data is divided into folds while preserving class proportions. Variants from the same gene may appear in both the training and testing sets.

### Gene-Disjoint Folds

All variants from the same gene are kept in the same fold. This ensures that the model is evaluated on genes that were completely unseen during training.

The experiments use five folds repeated three times. All preprocessing, feature scaling, hyperparameter tuning, and model selection are performed using training data only to prevent data leakage.

## Evaluation Metrics

The primary evaluation metrics are:

- ROC-AUC
- PR-AUC
- Matthews Correlation Coefficient (MCC)

Accuracy is reported as a secondary metric.

Additional analyses include:

- Performance drop on unseen genes
- BIC-based model selection
- Model convergence and stability
- Runtime and computational behavior
- Feature-set comparisons
- Fold-level performance variation

## Hypotheses

### H1: Effect of Additional Feature Groups

Adding population-frequency and amino-acid features improves PR-AUC compared with using conservation features alone when evaluated on unseen genes.

### H2: Potential Label Circularity

The improvement from adding population-frequency features is smaller on unseen genes than on standard folds. This may indicate that part of the improvement is related to circularity in the labeling process.

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
- scikit-learn
- SciPy
- Matplotlib
- Jupyter Notebook

## Installation

Clone the repository and install the required dependencies:

```bash
git clone <repository-url>
cd DATA_245_GroupProject
```

## Running the Project

The model implementations are located in the `src/models/` directory.

The analysis notebooks are located in the `notebooks/` directory.

Model outputs, evaluation metrics, tables, and figures are stored in the `results/` directory. Results from each model are organized in separate folders for easy comparison.

## Project Goal

This project does not focus only on achieving the highest classification score. Instead, it evaluates whether model performance remains reliable when predicting variants from genes that were not included during training.

The results help identify:

- Which biological feature groups are most useful
- How much performance decreases on unseen genes
- Whether standard validation may overestimate real-world performance
- Whether a generative model is a useful alternative to discriminative classifiers
- How reliably classical machine learning models can support missense variant analysis

## Important Note

This project is intended for research and educational evaluation. The model predictions should not be used as a substitute for expert clinical or genetic interpretation.
