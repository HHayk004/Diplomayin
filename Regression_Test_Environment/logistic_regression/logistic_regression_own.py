import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

# =========================
# Logistic Regression Functions
# =========================
def sigmoid_function(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

# def cost_function(h, y):
#     h = np.clip(h, 1e-12, 1 - 1e-12)
#     return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()

def logistic_reg(alpha, X, y, max_iterations=5000):
    # Initialize theta: last element = bias
    theta = np.zeros(X.shape[1] + 1)

    for i in range(max_iterations):
        z = np.dot(X, theta[:-1]) + theta[-1]  # bias last
        h = sigmoid_function(z)
        gradient = np.dot(X.T, h - y) / y.size
        theta[:-1] -= alpha * gradient  # update weights
        theta[-1] -= alpha * (h - y).mean()  # update bias

        # if i % 1000 == 0:
        #     loss = cost_function(h, y)
        #     print(f"Iteration {i}, loss: {loss:.6f}")

    return theta

def predict_prob(X, theta):
    return sigmoid_function(np.dot(X, theta[:-1]) + theta[-1])

# =========================
# Load CLEANED CSV (no headers)
# =========================
df = pd.read_csv("defaultofcreditcardclients_cleaned.csv", header=None)

# =========================
# Assign column names
# Last column is target
# =========================
num_cols = df.shape[1]
feature_cols = [f"X{i}" for i in range(1, num_cols)]
target_col = "Y"
df.columns = feature_cols + [target_col]

# =========================
# Split X and y
# =========================
X = df[feature_cols].values
y = df[target_col].values.astype(int)

# =========================
# Train / Test split
# =========================
X_train = X[:25000]
y_train = y[:25000]

X_test = X[25000:30000]
y_test = y[25000:30000]

# =========================
# Train model
# =========================
alpha = 0.01  # small learning rate for stability
theta = logistic_reg(alpha, X_train, y_train, max_iterations=5000)

# =========================
# TEST 1: Last 5,000 rows
# =========================
y_prob_5k = predict_prob(X_test, theta)
y_pred_5k = (y_prob_5k >= 0.5).astype(int)

accuracy_5k = accuracy_score(y_test, y_pred_5k)
print("\nAccuracy on last 5,000 rows:", accuracy_5k)

results_5k = pd.DataFrame({
    "Actual_Y": y_test,
    "Predicted_Y": y_pred_5k,
    "P_value": y_prob_5k
})
results_5k.to_csv("logistic_regression/logistic_regression_own_5000.csv", index=False)

# =========================
# TEST 2: ALL 30,000 rows
# =========================
y_prob_all = predict_prob(X, theta)
y_pred_all = (y_prob_all >= 0.5).astype(int)

accuracy_all = accuracy_score(y, y_pred_all)
print("Accuracy on all 30,000 rows:", accuracy_all)

results_all = pd.DataFrame({
    "Actual_Y": y,
    "Predicted_Y": y_pred_all,
    "P_value": y_prob_all
})
results_all.to_csv("logistic_regression/logistic_regression_own_30000.csv", index=False)

# =========================
# Save model parameters (weights + bias last)
# =========================
model_params = pd.DataFrame({
    "feature": feature_cols + ["BIAS"],
    "weight": theta
})
model_params.to_csv("logistic_regression/logistic_regression_own_weights_bias.csv", index=False)

print("\nAll results and model parameters saved successfully.")
