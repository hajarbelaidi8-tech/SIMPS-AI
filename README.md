# 🧠 SIMPS-AI: Intelligence Engine for Stress Prediction

**SIMPS-AI** (Système Intelligent de Prédiction du Stress) is a high-performance Machine Learning framework designed to classify individual stress levels with extreme precision. This repository contains the complete "Intelligence Engine"—from high-fidelity data synthesis to a production-ready API serving an optimized **XGBoost** model.

---

## 🎯 Project Objectives

The system solves a complex multi-class classification problem by predicting **11 distinct levels of stress (0–10)**. It evaluates 20 multi-domain features to identify patterns across:

* **Physical Health:** Blood pressure, Glycemia, and Cholesterol.
* **Behavioral Habits:** Sleep quality, caffeine intake, and physical activity.
* **Socio-Professional Factors:** Income, work hours, and professional status.

---

## 📈 Model Performance

The pipeline evaluates several candidate algorithms (Random Forest, SVM, KNN, Logistic Regression) before selecting and optimizing the final predictor.

* **Final Selected Model:** XGBoost Optimized.
* **Key Metric:** **99.07% F1-Score (Macro)**.

---

## 🏗 Project Architecture

```text
📦 stress_predictor
 ┣ 📂 api/                   # FastAPI implementation
 ┃ ┗ 📜 main.py              # API entry point & routes
 ┣ 📂 data/
 ┃ ┣ 📂 processed/           # Training/Testing sets & Priority rankings
 ┃ ┃ ┣ 📂 mappings/          # Comprehensive JSON mappings for all features
 ┃ ┃ ┣ 📜 feature_priority.json
 ┃ ┃ ┣ 📜 test_set.csv
 ┃ ┃ ┗ 📜 train_set.csv
 ┃ ┗ 📂 raw/                 # Final balanced dataset
 ┃   ┗ 📜 final_stress_dataset.csv
 ┣ 📂 etc/                   # Historical research & legacy data
 ┃ ┗ 📂 old data/
 ┃   ┣ 📂 datasets/          # Original 6 raw sources (Finance, Sleep, Tech, etc.)
 ┃   ┣ 📂 metadata/          # Descriptive summaries for each source
 ┃   ┣ 📂 processed_datasets/ # Legacy cleaned and merged CSVs
 ┃   ┗ 📂 scripts/           # Initial cleaning & transformation logic
 ┣ 📂 models/                # Production model assets
 ┃ ┣ 📂 decoders/            # .pkl files for label decoding
 ┃ ┣ 📂 encoders/            # .pkl files for feature encoding
 ┃ ┣ 📜 metadata.json        # Training history & F1-Score reports
 ┃ ┣ 📜 scaler.pkl           # Fitted StandardScaler
 ┃ ┗ 📜 stress_predictor_model.pkl # Final optimized model
 ┣ 📂 notebooks/             # [Pending] Data exploration & Model evaluation
 ┃ ┗ 📜 stress_analysis.ipynb
 ┣ 📂 src/                   # Core Python Logic
 ┃ ┣ 📜 dataset_info.py      # Statistical inspection
 ┃ ┣ 📜 features_priority.py # Correlation analysis
 ┃ ┣ 📜 generate_synthetic_data.py # 80/20 logic data generation
 ┃ ┣ 📜 heatmap.py           # Correlation visualizations
 ┃ ┣ 📜 model_logic.py       # Competitive training pipeline
 ┃ ┣ 📜 predictor.py         # Real-time inference class
 ┃ ┣ 📜 preprocess.py        # Stratified splitting logic
 ┃ ┣ 📜 workflow_manager.py  # End-to-end execution script
 ┃ ┗ 📜 __init__.py
 ┣ 📜 README.md
 ┗ 📜 requirements.txt

```

---

## 🧪 Intelligence Pipeline

### 1. Legacy Research (`etc/`)

This section contains the project's data foundations. It documents the initial phase where 6 separate datasets—ranging from *Personal Finance* to *Sleep Health*—were cleaned, transformed, and merged into the original research prototypes.

### 2. Data Synthesis (`src/generate_synthetic_data.py`)

To ensure a perfectly balanced training set, the system uses an **80/20 logic**. It generates high-fidelity data where 80% follows defined stress-level profiles while 20% introduces random noise to ensure the model remains robust against real-world anomalies.

### 3. Automated Training (`src/model_logic.py`)

The training engine executes an automated competition:

1. **Scaling:** Normalizes features using `StandardScaler`.
2. **Comparison:** Evaluates 5 different classifiers simultaneously.
3. **Optimization:** Automatically selects the winner and applies **XGBoost** optimization to secure the final **99.07% F1-score**.

---

## 🚀 Execution & Setup

### 1. Environment Setup

```bash
python -m venv venv
source venv/Scripts/activate
pip install -r requirements.txt

```

### 2. Running the Full Pipeline

To generate the data, perform stratified splitting, and train the model:

```bash
python src/workflow_manager.py

```

### 3. Launching the API

To serve the model for real-time stress prediction:

```bash
python -m api.main

```

---

## 🛠 Tech Stack

* **API Framework:** FastAPI
* **Data Science:** Pandas, NumPy, Scikit-Learn, XGBoost
* **Serialization:** Joblib (Models/Scalers) & JSON (Mappings/Metadata)
* **Visualization:** Seaborn, Matplotlib, and Rich (CLI)


