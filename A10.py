import numpy as np
import matplotlib.pyplot as plt

# AND Gate data with two outputs
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[1, 0], [1, 0], [1, 0], [0, 1]])  # [1,0] for 0, [0,1] for 1

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Network parameters
input_size = 2
hidden_size = 2
output_size = 2
alpha = 0.05
max_epochs = 10000
convergence_error = 0.002

# Initialize weights
W1 = np.random.randn(input_size, hidden_size)
W2 = np.random.randn(hidden_size, output_size)

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
plt.title('Two Output Nodes for AND Gate')
plt.grid(True)
plt.show()

# Test the network
print("\nTwo Output Results:")
for i in range(X.shape[0]):
    hidden = sigmoid(np.dot(X[i], W1))
    output = sigmoid(np.dot(hidden, W2))
    predicted_class = np.argmax(output)
    true_class = np.argmax(y[i])
    print(f"Input: {X[i]}, Output: {output}, Predicted: {predicted_class}, Expected: {true_class}")