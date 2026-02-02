# Occupancy Prediction from Environmental Sensor Data

> End-to-end machine learning project for room occupancy detection using environmental sensor data.

---

## 📌 Project Overview

This project focuses on predicting **room occupancy** based on environmental sensor measurements.
The objective is to evaluate how well different machine learning models can infer **human presence**
from physical signals such as temperature, humidity, light intensity, CO₂ concentration, and derived features.

The project is implemented as a **complete machine learning pipeline**, covering:

- data loading and preprocessing,
- baseline modeling,
- supervised model training,
- robust evaluation and comparison,
- and analysis of model behavior.

A strong emphasis is placed on **proper evaluation practices** to avoid misleading results caused by
data leakage or overly optimistic train/test splits.

---

## 🎯 Problem Definition

Given **time-ordered sensor measurements**, the task is to classify whether a room is:

- **occupied (1)**  
- **not occupied (0)**  

at a given time step.

This is a **binary classification problem with temporal structure**, meaning that
model evaluation must respect the **chronological order of the data**.

---

## 🧠 Why This Problem Matters

Accurate occupancy detection has practical, real-world applications such as:

- 🏢 smart building automation,
- ⚡ energy efficiency optimization,
- ❄️ HVAC system control,
- 🔒 privacy-preserving presence detection (no cameras involved).

The dataset used in this project is **widely referenced in academic literature**,
making it suitable both for learning purposes and realistic experimentation.

---

## 📁 Project Structure

```text
internship-ml-productivity-classifier/
│
├── data/
│   └── occupancy.csv
│       # Original time-ordered environmental sensor dataset
│
├── src/
│   ├── load_data.py
│   │   # Dataset loading utilities
│   │
│   ├── preprocess.py
│   │   # Feature selection and dataset preparation logic
│   │
│   ├── clean_data.py
│   │   # Data cleaning and optional time-based feature engineering
│   │
│   ├── metrics.py
│   │   # Centralized metric computation and formatted evaluation output
│   │
│   ├── train_dummy.py
│   │   # Baseline model (DummyClassifier – most frequent class)
│   │
│   ├── train_logistic.py
│   │   # Logistic Regression model with feature scaling
│   │
│   ├── train_random_forest.py
│   │   # Random Forest classifier
│   │
│   ├── cross_validation.py
│   │   # Cross-validation logic for robust model evaluation
│   │
│   ├── compare_models.py
│   │   # Unified model comparison and result aggregation
│   │
│   ├── ablation_plot.py
│   │   # Feature ablation analysis and visualization
│   │
│   ├── feature_importance.py
│   │   # Random Forest feature importance analysis
│   │
│   ├── realtime_simulation.py
│   │   # Sliding-window simulation to mimic online prediction behavior
│   │
│   └── run.py
│       # Main entry point (CLI) for training, evaluation, and comparison
│
├── results/
│   ├── model_comparison.csv
│   │   # Side-by-side performance comparison of all models
│   │
│   ├── metrics_cv.csv
│   │   # Cross-validation summary statistics
│   │
│   ├── metrics_cv_folds.csv
│   │   # Per-fold cross-validation metrics
│   │
│   ├── feature_importance.png
│   │   # Feature importance visualization
│   │
│   └── ablation_test.png
│       # Feature ablation accuracy comparison
│
├── notebooks/
│   # Optional exploratory notebooks
│
├── requirements.txt
│
├── .gitignore
│
└── README.md
```
---

## 🚀 How to Run
This project is designed to be executed via a single CLI entry point (run.py).
No notebooks are required to reproduce results.

### 1. Environment setup

```text
Python 3.10+
```

Create virtual environment (recommended)
```bash
python -m venv .venv
```

Activate it:
- Windows
```bash
.venv/Scripts/activate
```

- Linux/macOS
```bash
source .venv/bin/activate
```

Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Dataset

The dataset is expected at:
```bash
data/occupancy.csv
```

It is a time-ordered environmental sensor dataset with the following columns:
- Temperature
- Humidity
- Light
- CO2
- HumidityRatio
- Occupancy (target label: 0 or 1)
No manual preprocessing is required before running the pipeline.


### 3. Train individual models

All training scripts can be executed directly, but the recommended way is via ```run.py```.

Random Forest
```bash
python src/run.py train --model rf
```

Logistic Regression (with feature scaling)
```bash
python src/run.py train --model logreg
```

Baseline (DummyClassifier – most frequent class)
```bash
python src/run.py train --model dummy
```

Each command prints:
- confusion matrix,
- precision / recall / F1,
- overall accuracy.

