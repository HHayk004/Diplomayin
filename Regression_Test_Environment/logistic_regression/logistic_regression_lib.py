import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import sys

# =========================
# Load CLEANED CSV (no headers)
# =========================
try:
    df = pd.read_csv('defaultofcreditcardclients_cleaned.csv', header=None)
except Exception as e:
    print(f"Error loading dataset: {e}")
    sys.exit(1)

# =========================
# Assign column names
# Last column = target (Y)
# =========================
num_cols = df.shape[1]
feature_cols = [f'X{i}' for i in range(1, num_cols)]
target_col = 'Y'

df.columns = feature_cols + [target_col]

# =========================
# Ensure target is integer
# =========================
df[target_col] = df[target_col].astype(int)

# =========================
# Train / test split
# =========================
train_df = df.iloc[:25000]
test_df  = df.iloc[25000:30000]

X_train = train_df[feature_cols]
y_train = train_df[target_col]

X_test  = test_df[feature_cols]
y_test  = test_df[target_col]

# =========================
# Scale features
# =========================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# =========================
# Train logistic regression
# =========================
model = LogisticRegression(max_iter=5000, solver='lbfgs')
model.fit(X_train_scaled, y_train)

# =========================
# Test on last 5,000 rows
# =========================
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy on last 5,000 rows:", accuracy)

results = pd.DataFrame({
    'Y_true': y_test.values,
    'Y_pred': y_pred,
    'P_default': y_prob
})

results.to_csv('logistic_regression/results/logistic_regression_lib_results_5000.csv', index=False)

# =========================
# TEST 2: ALL 30,000 rows
# =========================
X_all = df[feature_cols]
y_all = df[target_col]

X_all_scaled = scaler.transform(X_all)

y_pred_all = model.predict(X_all_scaled)
y_prob_all = model.predict_proba(X_all_scaled)[:, 1]

acc_all = accuracy_score(y_all, y_pred_all)
print("Accuracy on all 30,000 rows:", acc_all)

pd.DataFrame({
    'Y_true': y_all.values,
    'Y_pred': y_pred_all,
    'P_default': y_prob_all
}).to_csv('logistic_regression/results/logistic_regression_lib_results_30000.csv', index=False)

# =========================
# Save weights & bias
# =========================
weights_df = pd.DataFrame({
    'feature': feature_cols,
    'weight': model.coef_[0]
})

bias_df = pd.DataFrame({
    'feature': ['BIAS'],
    'weight': [model.intercept_[0]]
})

pd.concat([weights_df, bias_df]).to_csv(
    'logistic_regression/results/logistic_regression_lib_weights_bias.csv', index=False
)

print("Results and model parameters saved successfully.")
