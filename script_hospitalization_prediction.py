# Load wastewatercounts and hospitalization data

import pandas as pd

df = pd.read_csv("wastewater_hospitalization.csv")

X = df[["WastewaterCounts"]]
y = df["Hospitalizations"]

# Random split (55% train / 45% test)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.45,
    random_state=42
)

# Check distribution (IMPORTANT)

print("Train WastewaterCounts summary:")
print(X_train.describe())

print("\nTest WastewaterCounts summary:")
print(X_test.describe())

# Linear Regression
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X_train, y_train)
pred_lr = lr.predict(X_test)

# Random Forest

from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(
    n_estimators=500,
    random_state=42
)

rf.fit(X_train, y_train)
pred_rf = rf.predict(X_test)

# XGBoost

from xgboost import XGBRegressor

xgb = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=4,
    random_state=42
)

xgb.fit(X_train, y_train)
pred_xgb = xgb.predict(X_test)

# Support Vector machine

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

svr = Pipeline([
    ("scaler", StandardScaler()),
    ("svr", SVR(kernel="rbf", C=100))
])

svr.fit(X_train, y_train)
pred_svr = svr.predict(X_test)

# Gradient Boosting
from sklearn.ensemble import GradientBoostingRegressor

gbr = GradientBoostingRegressor(
    n_estimators=500,
    learning_rate=0.05,
    random_state=42
)

gbr.fit(X_train, y_train)
pred_gbr = gbr.predict(X_test)

# Evaluation

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

def evaluate(y_true, y_pred):
    return {
        "R2": r2_score(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE": mean_absolute_error(y_true, y_pred)
    }

# Compare models

results = pd.DataFrame({
    "Linear Regression": evaluate(y_test, pred_lr),
    "Random Forest": evaluate(y_test, pred_rf),
    "XGBoost": evaluate(y_test, pred_xgb),
    "SVR": evaluate(y_test, pred_svr),
    "Gradient Boosting": evaluate(y_test, pred_gbr)
}).T

print(results.round(3))

# Combined Plot: Actual vs Predicted (All Models)

import matplotlib.pyplot as plt

plt.figure(figsize=(12,6))

# Actual values
plt.plot(
    y_test.values,
    label="Actual Hospitalizations",
    linewidth=3,
    color="black"
)

# Predictions from all models
plt.plot(pred_lr, label="Linear Regression", linestyle="--")
plt.plot(pred_rf, label="Random Forest", linestyle="--")
plt.plot(pred_xgb, label="XGBoost", linestyle="--")
plt.plot(pred_svr, label="SVR", linestyle="--")
plt.plot(pred_gbr, label="Gradient Boosting", linestyle="--")

plt.title("Test Data: Actual vs Predicted Hospitalizations (All Models)")
plt.xlabel("Test Sample Index")
plt.ylabel("Hospitalizations")
plt.legend()
plt.tight_layout()

plt.savefig("All_Models_Test_Predictions.png", dpi=600)
plt.show()

# SVR: Observed vs Predicted Hospitalizations
import matplotlib.pyplot as plt

plt.figure(figsize=(6,6))

# SVR predictions
plt.scatter(
    y_test,
    pred_svr,
    alpha=0.7
)

# 45-degree reference line (perfect prediction)
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    'k--',
    linewidth=2
)

plt.xlabel("Observed Hospitalizations")
plt.ylabel("Predicted Hospitalizations (SVR)")
plt.title("SVR Model: Observed vs Predicted")

plt.tight_layout()

plt.savefig(
    "SVR_Observed_vs_Predicted.png",
    dpi=600
)

plt.show()

