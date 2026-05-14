import matplotlib.pyplot as plt

labels = [
    "Lib Model - Last 5k",
    "Lib Model - All 30k",
    "My Model  - Last 5k",
    "My Model  - All 30k"
]

accuracies = [0.819, 0.8106666666666666, 0.7884, 0.778]

plt.figure()
plt.bar(labels, accuracies)

# Zoom the Y-axis range to emphasize differences
plt.ylim(0.6, 0.83)

plt.ylabel("Accuracy")
plt.title("Logistic Regression Accuracy Comparison")
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()