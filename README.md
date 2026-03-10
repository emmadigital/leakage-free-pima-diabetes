
# Leakage-Free Diabetes Prediction with Reproducible Model Evaluation

## Overview

This repository contains the **MATLAB implementation of a leakage-free machine learning pipeline** for evaluating baseline and neural network models for diabetes prediction using the **Pima Indians Diabetes Dataset**.

The project accompanies the research study:

**“Leakage-Free and Reproducible Evaluation of Neural Networks for Diabetes Prediction: A Multi-Seed Cross-Validation Study.”**

The code implements a **reproducible evaluation framework** that prevents data leakage by enforcing:

* fold-contained preprocessing
* stratified multi-seed cross-validation
* pooled out-of-fold predictions
* calibration analysis
* statistical significance testing

The repository enables full replication of the experiments reported in the manuscript.

---

# Repository Structure

```
leakage-free-pima-diabetes
│
├── scripts          # Main experiment scripts
├── utils            # Helper functions
├── evaluation       # Statistical analysis and figure generation
├── optimization     # Optimization algorithms
├── docs             # Workflow diagrams and documentation
└── data             # Dataset instructions
```

---

# Models Implemented

## Baseline Models

* Logistic Regression
* Random Forest

These classical models serve as **reference benchmarks** for evaluating neural network performance.

---

## Neural Network Models

The following neural architectures are evaluated:

* Plain Feedforward Neural Network (FFNN)
* Backpropagation Neural Network (BPNN)
* Generalized Regression Neural Network (GRNN)
* Artificial Bee Colony optimized FFNN (ABC-FFNN)

All models are evaluated under the **same leakage-free experimental pipeline**.

---

# Experimental Framework

The evaluation pipeline follows a strict **leakage-free machine learning protocol**:

1. Fold-contained preprocessing
2. Stratified multi-seed cross-validation
3. Model training and prediction
4. Performance evaluation
5. Calibration assessment
6. Statistical significance testing

Key evaluation metrics include:

* Accuracy
* Sensitivity
* Specificity
* F1-score
* ROC-AUC
* Brier Score
* Calibration slope and intercept

Statistical comparisons include:

* Paired t-test
* Wilcoxon signed-rank test
* DeLong test for ROC-AUC
* McNemar’s test for classification performance

---

# Dataset

This study uses the **Pima Indians Diabetes Dataset**, publicly available from the UCI Machine Learning Repository:

[https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)

To run the experiments:

1. Download the dataset.
2. Place it inside the `data/` directory.
3. Run the data preparation script.

---

# Running the Experiments

Execute the scripts in the following order.

## Step 1 — Prepare Dataset

```
PREPARE_PIMA_DATASET.m
```

---

## Step 2 — Train Baseline Models

```
RUN_LOGREG_SEEDS_CV.m
RUN_RF_SEEDS_CV.m
```

---

## Step 3 — Train Neural Network Models

```
RUN_FFNN_SEEDS_CV.m
RUN_BPNN_SEEDS_CV.m
RUN_GRNN_SEEDS_CV.m
RUN_ABC_FFNN_SEEDS_CV.m
```

These scripts perform:

* leakage-free preprocessing
* stratified cross-validation
* multi-seed model evaluation

---

## Step 4 — Statistical Model Comparison

```
STAT_COMPARE_MODELS.m
```

This script computes statistical comparisons between models.

---

## Step 5 — Generate Figures

Run the scripts in the `evaluation` folder:

```
FIG_AUC_AND_CI_ROBUST.m
FIG_TRADEOFFS_SENS_SPEC_F1.m
FIG_CALIBRATION_PERFORMANCE.m
MAKE_6PANEL_CONFMAT.m
```

These scripts generate the figures used in the manuscript.

---

# Software Requirements

The project was implemented using:

```
MATLAB R2025b
```

Required toolboxes:

* Statistics and Machine Learning Toolbox
* Deep Learning Toolbox

All experiments were executed on CPU without GPU acceleration.

---

# Reproducibility

The repository provides a fully reproducible implementation including:

* fold-contained preprocessing
* multi-seed cross-validation
* pooled predictions across folds
* bootstrap confidence intervals
* calibration analysis
* statistical model comparison

Random seeds used in experiments are included in the scripts to ensure reproducibility.

---

# Workflow Diagram

The study workflow is illustrated below.

![Workflow Diagram](https://github.com/user-attachments/assets/64ceac8a-0fe8-43d4-907f-f3643c06d59a)

```
docs/Figure 3.svg
```

---

# Citation

If you use this repository in your research, please cite the associated article:

```
Author(s). Leakage-Free and Reproducible Evaluation of Neural Networks
for Diabetes Prediction: A Multi-Seed Cross-Validation Study.
```

---

# License

This repository is released under the **MIT License**.

---

# Contact

For questions regarding the code or experiments, please contact:

```
Author Name: Amresha Soomro
Institution: Altinbas University
Email: amreshasoomro@gmail.com
```
