import numpy as np
import matplotlib.pyplot as plt

# XOR Gate data
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Network architecture
input_size = 2
hidden_size = 2
output_size = 1

# Initialize weights
W1 = np.random.randn(input_size, hidden_size)
W2 = np.random.randn(hidden_size, output_size)

# Training parameters
alpha = 0.05
max_epochs = 10000
convergence_error = 0.002

errors = []

for epoch in range(max_epochs):
    # Forward propagation
    hidden_input = np.dot(X, W1)
    hidden_output = sigmoid(hidden_input)
    
    final_input = np.dot(hidden_output, W2)
    final_output = sigmoid(final_input)
    
    # Calculate error
    error = y.reshape(-1, 1) - final_output
    sum_squared_error = np.mean(error ** 2)
    errors.append(sum_squared_error)
    
    if sum_squared_error <= convergence_error:
        print(f"Converged at epoch {epoch+1}")
        break
    
    # Backpropagation
    d_output = error * sigmoid_derivative(final_output)
    error_hidden = d_output.dot(W2.T)
    d_hidden = error_hidden * sigmoid_derivative(hidden_output)
    
    # Update weights
    W2 += hidden_output.T.dot(d_output) * alpha
    W1 += X.T.dot(d_hidden) * alpha

# Plot training error
plt.plot(range(1, epoch+2), errors)
plt.xlabel('Epochs')
plt.ylabel('Sum-Squared Error')
plt.title('XOR Gate Learning with MLP')
plt.grid(True)
plt.show()

# Test the trained network
print("\nXOR Gate Results:")
for i in range(X.shape[0]):
    hidden = sigmoid(np.dot(X[i], W1))
    output = sigmoid(np.dot(hidden, W2))
    print(f"Input: {X[i]}, Output: {output[0]:.4f} (Expected: {y[i]})")