import numpy as np
import matplotlib.pyplot as plt

labels = ['Last 5,000 rows', 'All 30,000 rows']
tanh_results = [0.4768, 0.4758]
sigmoid_results = [0.7884, 0.7788]

x = np.arange(len(labels))
width = 0.35

plt.figure()

# Black & white bars (use grayscale shades)
plt.bar(x - width/2, tanh_results, width, color='black', label='tanh')
plt.bar(x + width/2, sigmoid_results, width, color='white', edgecolor='black', label='sigmoid')

# Labels
plt.xticks(x, labels, fontsize=12)
plt.ylabel('Accuracy', fontsize=14)
plt.title('Tanh vs Sigmoid Accuracy Comparison', fontsize=16)

# Black grid
plt.grid(axis='y', linestyle='--', linewidth=0.5, color='black')

# Legend (black frame)
legend = plt.legend()
legend.get_frame().set_edgecolor('black')

# Make ticks black
plt.tick_params(colors='black')

plt.show()