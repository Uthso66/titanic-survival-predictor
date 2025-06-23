## Titanic Survival Predictor 🚢

An end-to-end machine learning pipeline that predicts passenger survival on the Titanic using Logistic Regression.
Built with **clean code architecture**, **reproducibility**, and **industry-grade structure** to showcase production-ready ML skills.

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

![confusion_matrix](https://github.com/user-attachments/assets/3b37b408-a829-4fbb-a586-3763d6d38e5e)

![roc_curve](https://github.com/user-attachments/assets/706bac33-a4e0-4ea2-8730-e2f8f75cdd1d)

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
