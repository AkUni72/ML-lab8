import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Customer data
data = np.array([
    [20, 6, 2, 386, 1],
    [16, 3, 6, 289, 1],
    [27, 6, 2, 393, 1],
    [19, 1, 2, 110, 0],
    [24, 4, 2, 280, 1],
    [22, 1, 5, 167, 0],
    [15, 4, 2, 271, 1],
    [18, 4, 2, 274, 1],
    [21, 1, 4, 148, 0],
    [16, 2, 4, 198, 0]
])

# Split features and target
X = data[:, :-1]
y = data[:, -1]

# Normalize features
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)

# Add bias term
X_with_bias = np.hstack((np.ones((X.shape[0], 1)), X_normalized))

# Initialize weights
W = np.random.randn(X_with_bias.shape[1])

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def train_perceptron(X, y, W, alpha=0.01, max_epochs=1000, convergence_error=0.002):
    errors = []
    
    for epoch in range(max_epochs):
        total_error = 0
        for i in range(X.shape[0]):
            # Forward pass
            summation = np.dot(X[i], W)
            prediction = 1 if sigmoid(summation) >= 0.5 else 0
            error = y[i] - prediction
            
            # Update weights
            W += alpha * error * X[i]
            total_error += error ** 2
        
        mean_error = total_error / X.shape[0]
        errors.append(mean_error)
        
        if mean_error <= convergence_error:
            print(f"Converged at epoch {epoch+1}")
            break
    
    return W, epoch+1, errors

# Train the perceptron
W, epochs, errors = train_perceptron(X_with_bias, y, W)

# Plot training error
plt.plot(range(1, epochs+1), errors)
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error')
plt.title('Customer Data Classification Training')
plt.grid(True)
plt.show()

# Evaluate
correct = 0
for i in range(X_with_bias.shape[0]):
    summation = np.dot(X_with_bias[i], W)
    prediction = 1 if sigmoid(summation) >= 0.5 else 0
    if prediction == y[i]:
        correct += 1
    print(f"Customer {i+1}: Predicted {prediction}, Actual {y[i]}")

print(f"\nAccuracy: {correct / X.shape[0] * 100:.2f}%")
print(f"Final weights: {W}")