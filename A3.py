import numpy as np
import matplotlib.pyplot as plt

# AND Gate data
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 0, 0, 1])

# Initial weights and learning rate
W = np.array([10, 0.2, -0.75])  # w0, w1, w2
alpha = 0.05

def bipolar_step_activation(x):
    return 1 if x > 0 else (-1 if x < 0 else 0)

def sigmoid_activation(x):
    return 1 / (1 + np.exp(-x))

def relu_activation(x):
    return max(0, x)

def perceptron_train(X, y, W, alpha, activation_func, max_epochs=1000, convergence_error=0.002):
    """
    Generic perceptron training function with different activation functions
    """
    errors = []
    bias = np.ones((X.shape[0], 1))
    X_with_bias = np.hstack((bias, X))
    
    for epoch in range(max_epochs):
        total_error = 0
        for i in range(X.shape[0]):
            summation = np.dot(X_with_bias[i], W)
            
            if activation_func == 'bipolar_step':
                prediction = bipolar_step_activation(summation)
                # Convert target to -1/1 for bipolar step
                target = -1 if y[i] == 0 else 1
            else:
                if activation_func == 'sigmoid':
                    prediction = 1 if sigmoid_activation(summation) >= 0.5 else 0
                elif activation_func == 'relu':
                    prediction = 1 if relu_activation(summation) >= 0.5 else 0
                target = y[i]
            
            error = target - prediction
            W += alpha * error * X_with_bias[i]
            total_error += error ** 2
        
        sum_squared_error = total_error / X.shape[0]
        errors.append(sum_squared_error)
        
        if sum_squared_error <= convergence_error:
            print(f"{activation_func} converged at epoch {epoch+1}")
            break
    
    return epoch+1, errors

# Test different activation functions
activations = {
    'bipolar_step': bipolar_step_activation,
    'sigmoid': sigmoid_activation,
    'relu': relu_activation
}

results = {}
for name, func in activations.items():
    # Reset weights for each activation
    W = np.array([10, 0.2, -0.75])
    epochs, errors = perceptron_train(X, y, W, alpha, name)
    results[name] = (epochs, errors)

# Plot results
plt.figure(figsize=(10, 6))
for name, (epochs, errors) in results.items():
    plt.plot(range(1, epochs+1), errors, label=name)

plt.xlabel('Epochs')
plt.ylabel('Sum-Squared Error')
plt.title('Perceptron Learning with Different Activation Functions')
plt.legend()
plt.grid(True)
plt.show()

# Print convergence epochs
print("Epochs to converge:")
for name, (epochs, _) in results.items():
    print(f"{name}: {epochs} epochs")