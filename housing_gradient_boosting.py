"""House price prediction on the California Housing dataset.

The script:
    1. Loads the dataset (downloaded automatically by scikit-learn).
    2. Inspects it (shape, statistics, missing values).
    3. Splits it into train / test sets (80 / 20).
    4. Trains a Linear Regression baseline and a default Gradient Boosting model.
    5. Tunes Gradient Boosting with GridSearchCV (3-fold cross-validation).
    6. Evaluates all models on the test set and saves plots and a results table.

Usage:
    python housing_gradient_boosting.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_error
from sklearn.model_selection import GridSearchCV, train_test_split

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
SEED = 42
TEST_SIZE = 0.2
CV_FOLDS = 3
IMAGES_DIR = Path("images")
OUTPUT_DIR = Path("outputs")

PARAM_GRID = {
    "n_estimators": [100, 200, 300],
    "learning_rate": [0.05, 0.1, 0.2],
    "max_depth": [3, 4, 5],
}


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def load_data():
    """Load the California Housing data as a DataFrame and print an overview."""
    data = fetch_california_housing(as_frame=True)
    df = data.frame

    print("Columns:", df.columns.to_list())
    print("Shape:", df.shape)
    print("\nStatistics:\n", df.describe())
    print("\nMissing values:\n", df.isnull().sum())

    X = df.iloc[:, :-1]      # 8 features
    y = df.iloc[:, -1]       # MedHouseVal (in units of $100,000)
    return X, y


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------
def evaluate(model, name: str, X_test, y_test) -> dict:
    """Score a fitted model on the test set and print the result."""
    pred = model.predict(X_test)
    rmse = root_mean_squared_error(y_test, pred)
    r2 = r2_score(y_test, pred)
    print(f"{name:<24} RMSE = {rmse:.4f} | R² = {r2:.4f}")
    return {"Model": name, "RMSE": rmse, "R2": r2}


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------
def plot_feature_importance(model, columns, save_path: Path) -> None:
    importance = pd.Series(model.feature_importances_, index=columns)
    importance = importance.sort_values(ascending=False)
    print("\nFeature importance:\n", importance)

    importance.plot(kind="bar", figsize=(7, 4), title="Feature Importance")
    plt.ylabel("Importance")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_learning_curve(model, X_train, X_test, y_train, y_test,
                        save_path: Path) -> None:
    """Train / test MSE as a function of the number of trees."""
    train_err, test_err = [], []
    for p_tr, p_te in zip(model.staged_predict(X_train),
                          model.staged_predict(X_test)):
        train_err.append(mean_squared_error(y_train, p_tr))
        test_err.append(mean_squared_error(y_test, p_te))

    plt.figure(figsize=(7, 4))
    plt.plot(train_err, label="Train MSE")
    plt.plot(test_err, label="Test MSE")
    plt.xlabel("Number of trees")
    plt.ylabel("MSE")
    plt.title("Learning Curve (Tuned Gradient Boosting)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_actual_vs_predicted(y_test, pred, save_path: Path) -> None:
    plt.figure(figsize=(5.5, 5.5))
    plt.scatter(y_test, pred, s=6, alpha=0.3)
    lims = [0, max(y_test.max(), pred.max())]
    plt.plot(lims, lims, "r--", label="Perfect prediction")
    plt.xlabel("Actual (×$100k)")
    plt.ylabel("Predicted (×$100k)")
    plt.title("Actual vs Predicted House Value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> None:
    IMAGES_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)

    X, y = load_data()

    # NOTE: train_test_split returns (X_train, X_test, y_train, y_test) in
    # exactly this order.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED
    )
    print(f"\nTrain: {X_train.shape} | Test: {X_test.shape}\n")

    # 1. Baseline: linear regression
    lr = LinearRegression().fit(X_train, y_train)

    # 2. Gradient Boosting with default parameters
    gb = GradientBoostingRegressor(random_state=SEED).fit(X_train, y_train)

    # 3. Gradient Boosting tuned with cross-validated grid search
    print("Running grid search (this can take a few minutes)...")
    grid = GridSearchCV(
        GradientBoostingRegressor(random_state=SEED),
        PARAM_GRID,
        cv=CV_FOLDS,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )
    grid.fit(X_train, y_train)
    best = grid.best_estimator_
    print("Best params:", grid.best_params_)
    print(f"Best CV RMSE: {-grid.best_score_:.4f}\n")

    # 4. Test-set evaluation
    print("=" * 52)
    results = pd.DataFrame([
        evaluate(lr, "Linear Regression", X_test, y_test),
        evaluate(gb, "Gradient Boosting (default)", X_test, y_test),
        evaluate(best, "Gradient Boosting (tuned)", X_test, y_test),
    ])
    print("=" * 52)

    results = results.round(4)
    results.to_csv(OUTPUT_DIR / "results.csv", index=False)

    # Ready-to-paste Markdown table for the README
    lines = ["| Model | RMSE | R² |", "|---|---|---|"]
    for _, row in results.iterrows():
        lines.append(f"| {row['Model']} | {row['RMSE']:.4f} | {row['R2']:.4f} |")
    table = "\n".join(lines)
    best_params = ", ".join(f"`{k}={v}`" for k, v in grid.best_params_.items())
    snippet = f"{table}\n\nBest parameters: {best_params}\n"
    (OUTPUT_DIR / "results.md").write_text(snippet, encoding="utf-8")
    print("\nMarkdown table for the README:\n")
    print(snippet)

    # 5. Plots
    plot_feature_importance(best, X.columns, IMAGES_DIR / "feature_importance.png")
    plot_learning_curve(best, X_train, X_test, y_train, y_test,
                        IMAGES_DIR / "learning_curve.png")
    plot_actual_vs_predicted(y_test, best.predict(X_test),
                             IMAGES_DIR / "actual_vs_predicted.png")

    print(f"\nSaved plots to '{IMAGES_DIR}/' and results to '{OUTPUT_DIR}/'.")


if __name__ == "__main__":
    main()
