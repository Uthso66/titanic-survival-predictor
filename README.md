```markdown
# Titanic Survival Predictor 🚢

An end-to-end machine learning pipeline that predicts passenger survival on the Titanic using Logistic Regression.
Built with **clean code architecture**, **reproducibility**, and **industry-grade structure** to showcase production-ready ML skills.

---

## 📂 Project Structure

```

titanic-survival-predictor/
├── config/
│   └── config.yaml             \# Project configs: paths, model hyperparams
├── data/
│   ├── raw/                    \# Original dataset (Titanic-Dataset.csv)
│   ├── processed/              \# Train, val, test splits
│   └── interim/                \# (Optional cleaned data)
├── models/
│   └── titanic\_model.pkl       \# Trained model
├── outputs/
│   ├── confusion\_matrix.png    \# Confusion matrix plot
│   ├── roc\_curve.png           \# ROC curve plot
│   └── classification\_report.txt \# Text classification report
├── src/
│   ├── data/
│   │   └── preprocess.py
│   ├── features/
│   │   └── build\_features.py
│   ├── models/
│   │   ├── train\_model.py
│   │   └── evaluate\_test.py
│   └── utils/
│       └── helpers.py
├── requirements.txt
├── .gitignore
├── README.md
└── run.py                      \# (Optional runner script)

````

---

## 🚀 How to Run

1️⃣ Preprocess the data

```bash
python src/data/preprocess.py
````

2️⃣ Train + validate model

```bash
python src/models/train_model.py
```

3️⃣ Evaluate on test set

```bash
python src/models/evaluate_test.py
```

Or run all at once:

```bash
python run.py
```

-----

## 📊 Example Results

| Metric              | Value |
|---------------------|-------|
| Test Accuracy       | 0.79  |
| Test Precision (0)  | 0.80  |
| Test Recall (0)     | 0.86  |
| Test F1-score (0)   | 0.83  |
| Test Precision (1)  | 0.78  |
| Test Recall (1)     | 0.70  |
| Test F1-score (1)   | 0.74  |

✅ Confusion matrix + ROC curve saved in `/outputs/`
✅ Full metrics saved in `models/model_metrics.json`

-----

## 🛠 Requirements

```
pandas
scikit-learn
matplotlib
seaborn
pyyaml
joblib
```

```
```
