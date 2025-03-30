import numpy as np

# a) Summation unit
def summation_unit(inputs, weights):
    """Calculates the weighted sum of inputs."""
    return np.dot(inputs, weights)

# b) Activation Unit
def step_activation(x):
    """Step activation function"""
    return 1 if x > 0 else 0

def bipolar_step_activation(x):
    """Bipolar step activation function"""
    return 1 if x > 0 else (-1 if x < 0 else 0)

def sigmoid_activation(x):
    """Sigmoid activation function"""
    return 1 / (1 + np.exp(-x))

def tanh_activation(x):
    """Hyperbolic tangent activation function"""
    return np.tanh(x)

def relu_activation(x):
    """ReLU activation function"""
    return max(0, x)

def leaky_relu_activation(x, alpha=0.01):
    """Leaky ReLU activation function"""
    return x if x > 0 else alpha * x

# c) Comparator unit for Error calculation
def calculate_error(target, predicted):
    """Calculates the error between target and predicted values."""
    return target - predicted

# Example usage
if __name__ == "__main__":
    # Test summation unit
    inputs = [1, 2, 3]
    weights = [0.2, 0.3, 0.4]
    print("Summation:", summation_unit(inputs, weights))
    
    # Test activation functions
    x = 0.5
    print("Step:", step_activation(x))
    print("Bipolar Step:", bipolar_step_activation(x))
    print("Sigmoid:", sigmoid_activation(x))
    print("TanH:", tanh_activation(x))
    print("ReLU:", relu_activation(x))
    print("Leaky ReLU:", leaky_relu_activation(x))
    
    # Test error calculation
    print("Error:", calculate_error(1, 0.8))