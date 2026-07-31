# FOSSEE Screening Task 3: Surrogate Modeling of Binary Distillation Column

## Overview
This repository contains the dataset, simulation setup, and machine learning models developed to build a high-accuracy surrogate model for a Benzene-Toluene binary distillation column.

## Folder Structure
- `base_distillation.dwxmz`: DWSIM simulation file configured with Peng-Robinson EOS.
- `Dataset.csv`: 1,000-sample dataset generated across valid physical operating ranges.
- `Code/generate_dataset.py`: Python script for dataset generation.
- `Code/train_model.py`: Training script for Linear Regression, Random Forest, and XGBoost.
- `Results_Summary.txt`: Metric evaluation summary.

## Instructions to Run
1. Ensure Python 3.8+ is installed with dependencies:
   ```bash
   pip install pandas numpy scikit-learn xgboost matplotlib seaborn