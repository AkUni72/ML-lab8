import numpy as np
import matplotlib.pyplot as plt

# AND Gate data
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 0, 0, 1])

def step_activation(x):
    return 1 if x >= 0 else 0

def perceptron_train(X, y, W, alpha, max_epochs=1000, convergence_error=0.002):
    errors = []
    bias = np.ones((X.shape[0], 1))
    X_with_bias = np.hstack((bias, X))
    
    for epoch in range(max_epochs):
        total_error = 0
        for i in range(X.shape[0]):
            summation = np.dot(X_with_bias[i], W)
            prediction = step_activation(summation)
            error = y[i] - prediction
            W += alpha * error * X_with_bias[i]
            total_error += error ** 2
        
        sum_squared_error = total_error / X.shape[0]
        errors.append(sum_squared_error)
        
        if sum_squared_error <= convergence_error:
            return epoch+1
    
    return max_epochs

# Test different learning rates
learning_rates = np.arange(0.1, 1.1, 0.1)
epochs_needed = []

for alpha in learning_rates:
    W = np.array([10, 0.2, -0.75])  # Reset weights
    epochs = perceptron_train(X, y, W, alpha)
    epochs_needed.append(epochs)
    print(f"Learning rate {alpha:.1f}: {epochs} epochs")

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(learning_rates, epochs_needed, 'bo-')
plt.xlabel('Learning Rate')
plt.ylabel('Epochs to Converge')
plt.title('Effect of Learning Rate on Convergence')
plt.grid(True)
plt.show()