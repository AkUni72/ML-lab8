import numpy as np
import matplotlib.pyplot as plt

# XOR Gate data
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])  # XOR output

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Network parameters
input_size = 2
hidden_size = 2  # Need at least 2 hidden units for XOR
output_size = 1
alpha = 0.05
max_epochs = 10000
convergence_error = 0.002

# Initialize weights with slightly larger range
W1 = np.random.uniform(-1, 1, (input_size, hidden_size))
W2 = np.random.uniform(-1, 1, (hidden_size, output_size))

errors = []

for epoch in range(max_epochs):
    # Forward pass
    hidden_input = np.dot(X, W1)
    hidden_output = sigmoid(hidden_input)
    
    final_input = np.dot(hidden_output, W2)
    final_output = sigmoid(final_input)
    
    # Error calculation
    error = y - final_output
    sum_squared_error = np.mean(error ** 2)
    errors.append(sum_squared_error)
    
    if sum_squared_error <= convergence_error:
        print(f"Converged at epoch {epoch+1}")
        break
    
    # Backward pass
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
plt.title('Backpropagation for XOR Gate')
plt.grid(True)
plt.show()

# Test the network
print("\nXOR Gate Results:")
for i in range(X.shape[0]):
    hidden = sigmoid(np.dot(X[i], W1))
    output = sigmoid(np.dot(hidden, W2))
    print(f"Input: {X[i]}, Output: {output[0]:.4f} (Expected: {y[i][0]})")

# Print final weights
print("\nFinal Hidden Layer Weights:")
print(W1)
print("\nFinal Output Layer Weights:")
print(W2)