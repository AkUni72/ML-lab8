import numpy as np
import matplotlib.pyplot as plt

# AND Gate data
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 0, 0, 1])

# Initial weights and learning rate
W = np.array([10, 0.2, -0.75])  # w0, w1, w2
alpha = 0.05

def step_activation(x):
    return 1 if x >= 0 else 0

def perceptron_train(X, y, W, alpha, max_epochs=1000, convergence_error=0.002):
    """Trains a perceptron model."""
    errors = []
    bias = np.ones((X.shape[0], 1))
    X_with_bias = np.hstack((bias, X))
    
    for epoch in range(max_epochs):
        total_error = 0
        for i in range(X.shape[0]):
            # Calculate prediction
            summation = np.dot(X_with_bias[i], W)
            prediction = step_activation(summation)
            
            # Update weights
            error = y[i] - prediction
            W += alpha * error * X_with_bias[i]
            
            # Accumulate squared error
            total_error += error ** 2
        
        # Calculate sum squared error for all samples
        sum_squared_error = total_error / X.shape[0]
        errors.append(sum_squared_error)
        
        # Check for convergence
        if sum_squared_error <= convergence_error:
            print(f"Converged at epoch {epoch+1} with error {sum_squared_error:.4f}")
            break
    
    return W, epoch+1, errors

# Train the perceptron
trained_W, epochs, errors = perceptron_train(X, y, W, alpha)

# Plotting
plt.plot(range(1, epochs+1), errors)
plt.xlabel('Epochs')
plt.ylabel('Sum-Squared Error')
plt.title('Perceptron Learning for AND Gate')
plt.grid(True)
plt.show()

print(f"Final weights: W0 = {trained_W[0]:.4f}, W1 = {trained_W[1]:.4f}, W2 = {trained_W[2]:.4f}")
print(f"Epochs to converge: {epochs}")