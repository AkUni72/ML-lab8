import numpy as np
from A6 import X_normalized, y

# Add bias term
X_with_bias = np.hstack((np.ones((X_normalized.shape[0], 1)), X_normalized))

# Pseudo-inverse solution
W_pseudo = np.linalg.pinv(X_with_bias) @ y

# Compare with perceptron weights from A6
print("Pseudo-inverse weights:", W_pseudo)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Evaluate pseudo-inverse solution
correct = 0
for i in range(X_with_bias.shape[0]):
    prediction = 1 if sigmoid(np.dot(X_with_bias[i], W_pseudo)) >= 0.5 else 0
    if prediction == y[i]:
        correct += 1

print(f"Pseudo-inverse Accuracy: {correct / X_with_bias.shape[0] * 100:.2f}%")