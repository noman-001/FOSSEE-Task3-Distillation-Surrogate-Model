import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb

df = pd.read_csv("Dataset.csv")

X_cols = ['Feed_Temp_C', 'Feed_Pressure_atm', 'Feed_x_Benzene', 'Num_Stages', 'Feed_Stage', 'Reflux_Ratio', 'Bottoms_Rate_kmol_h']
Y_cols = ['xD_Benzene', 'xB_Toluene', 'Qc_kW', 'Qr_kW']

X = df[X_cols]
Y = df[Y_cols]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "XGBoost": xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
}

results = []
predictions_dict = {}

print("=" * 80)
print(" FOSSEE TASK 3 - DETAILED EVALUATION METRICS (MAE, RMSE, R²)")
print("=" * 80)

for name, model in models.items():
    if name == "Linear Regression":
        model.fit(X_train_scaled, Y_train)
        preds = model.predict(X_test_scaled)
    else:
        model.fit(X_train, Y_train)
        preds = model.predict(X_test)
    
    predictions_dict[name] = preds
    
    for idx, col in enumerate(Y_cols):
        r2 = r2_score(Y_test.iloc[:, idx], preds[:, idx])
        rmse = np.sqrt(mean_squared_error(Y_test.iloc[:, idx], preds[:, idx]))
        mae = mean_absolute_error(Y_test.iloc[:, idx], preds[:, idx])
        
        results.append({
            "Model": name,
            "Target": col,
            "MAE": round(mae, 5),
            "RMSE": round(rmse, 5),
            "R2_Score": round(r2, 5)
        })

results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))
results_df.to_csv("Model_Comparison.csv", index=False)

best_preds = predictions_dict["XGBoost"]
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for idx, col in enumerate(Y_cols):
    axes[idx].scatter(Y_test.iloc[:, idx], best_preds[:, idx], alpha=0.6, color='blue')
    axes[idx].plot([Y_test.iloc[:, idx].min(), Y_test.iloc[:, idx].max()],
                   [Y_test.iloc[:, idx].min(), Y_test.iloc[:, idx].max()], 'r--', lw=2)
    axes[idx].set_title(f"Predicted vs Actual: {col}")
    axes[idx].set_xlabel("Actual Values")
    axes[idx].set_ylabel("Predicted Values")
    axes[idx].grid(True)

plt.tight_layout()
plt.savefig("predicted_vs_actual.png")
print("\nSaved plot to 'predicted_vs_actual.png'")

rf_model = models["Random Forest"]
importances = rf_model.feature_importances_
feat_imp = pd.Series(importances, index=X_cols).sort_values(ascending=True)

plt.figure(figsize=(8, 5))
feat_imp.plot(kind='barh', color='teal')
plt.title("Feature Importance Analysis (Random Forest)")
plt.xlabel("Relative Importance")
plt.tight_layout()
plt.savefig("feature_importance.png")
print("Saved plot to 'feature_importance.png'")