# 🏠 California House Price Prediction with Gradient Boosting

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

A regression project that predicts the median house value of California districts using **Gradient Boosting**, compared with a **Linear Regression** baseline, with hyperparameter tuning via **GridSearchCV**.

This is an educational project I built to practice the standard supervised-learning workflow: exploring data, splitting it correctly, building a baseline, tuning a model with cross-validation, and interpreting the result with feature importance and a learning curve.

## 📑 Table of Contents

- [Overview](#-overview)
- [Dataset](#-dataset)
- [Pipeline](#-pipeline)
- [Results](#-results)
- [Getting Started](#-getting-started)
- [Project Structure](#-project-structure)
- [Limitations](#-limitations)
- [Possible Improvements](#-possible-improvements)
- [What I Learned](#-what-i-learned)
- [License](#-license)
- [Author](#-author)

## 🔎 Overview

| | |
|---|---|
| **Task** | Tabular regression |
| **Target** | Median house value (`MedHouseVal`, in units of $100,000) |
| **Models** | Linear Regression (baseline), Gradient Boosting (default and tuned) |
| **Tuning** | `GridSearchCV`, 3-fold CV, 27 parameter combinations |
| **Metrics** | RMSE, R² |
| **Tech stack** | Python, scikit-learn, pandas, Matplotlib |

## 📊 Dataset

The [California Housing dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html) (from the 1990 U.S. census) contains 20,640 districts and 8 numerical features. It is downloaded automatically by scikit-learn, so no manual download is needed. There are no missing values.

| Feature | Description |
|---|---|
| `MedInc` | Median income in the block group |
| `HouseAge` | Median house age |
| `AveRooms` | Average number of rooms per household |
| `AveBedrms` | Average number of bedrooms per household |
| `Population` | Block group population |
| `AveOccup` | Average number of household members |
| `Latitude` | Block group latitude |
| `Longitude` | Block group longitude |

## 🔧 Pipeline

1. **Exploration** – shape, summary statistics and missing-value check.
2. **Train/test split** – 80% / 20% with a fixed random seed.
3. **Baseline** – Linear Regression.
4. **Gradient Boosting** – default parameters.
5. **Hyperparameter tuning** – `GridSearchCV` over:

   | Parameter | Values |
   |---|---|
   | `n_estimators` | 100, 200, 300 |
   | `learning_rate` | 0.05, 0.1, 0.2 |
   | `max_depth` | 3, 4, 5 |

6. **Evaluation** – RMSE and R² on the untouched test set.
7. **Interpretation** – feature importance and a learning curve (train vs. test MSE as trees are added).

## 📈 Results

Run the script and copy the table it prints (also saved in `outputs/results.md`) into this section. Metrics are computed on the held-out test set (20% of the data).

| Model | RMSE | R² |
|---|---|---|
| Linear Regression | 0.9938 | 0.2612 |
| Gradient Boosting (default) | 0.5408 | 0.7812 |
| Gradient Boosting (tuned) | 0.5083 | 0.8068 |

**Best params:** `{'learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 200}`

The target is expressed in units of $100,000, so an RMSE of 0.5 corresponds to an average error of roughly $50,000.

<!--
After running the script and uploading the plots to images/, remove these comment markers to display them:

![Feature importance](images/feature_importance.png)
![Learning curve](images/learning_curve.png)
![Actual vs predicted](images/actual_vs_predicted.png)
-->

## 🚀 Getting Started

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/california-housing-gb.git
cd california-housing-gb

# 2. (Optional) create a virtual environment
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Train, tune and evaluate (the grid search takes a few minutes)
python housing_gradient_boosting.py
```

**Outputs**

- `images/feature_importance.png`
- `images/learning_curve.png`
- `images/actual_vs_predicted.png`
- `outputs/results.csv` and `outputs/results.md` – test-set metrics of all models (the latter is a ready-to-paste Markdown table)

## 📁 Project Structure

```
california-housing-gb/
├── images/                          # generated plots
├── housing_gradient_boosting.py     # full pipeline
├── requirements.txt                 # Python dependencies
├── LICENSE
└── README.md
```

## ⚠️ Limitations

- The dataset comes from the 1990 census, so it does not reflect modern prices.
- House values are capped at $500,000 in the data, which limits accuracy for expensive districts.
- Location is used only through raw latitude/longitude; no geographic feature engineering.
- The learning curve is plotted on the test set for diagnosis only; hyperparameters are selected with cross-validation on the training set.
- A single train/test split is used for the final evaluation.

## 💡 Possible Improvements

- Feature engineering (rooms per person, distance to major cities, geographic clusters)
- Compare with Random Forest, XGBoost, LightGBM and CatBoost
- Use `RandomizedSearchCV` or Optuna for a broader search
- Add early stopping to choose the number of trees automatically
- Explain predictions with SHAP values

## 🎓 What I Learned

- How to build a fair baseline before using a complex model
- How Gradient Boosting builds an ensemble of shallow trees sequentially
- How to tune hyperparameters with cross-validation without touching the test set
- How to read feature importance and a learning curve to understand a model
- Why train/test splitting must be done carefully (the order of the returned arrays matters)

## 📝 License

Released under the [MIT License](LICENSE).

## 👤 Author

**Your Name**
[GitHub](https://github.com/<your-username>) · [LinkedIn](https://www.linkedin.com/in/<your-profile>)
