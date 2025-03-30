from sklearn.neural_network import MLPClassifier
import numpy as np

# AND Gate
X_and = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_and = np.array([0, 0, 0, 1])

# XOR Gate
X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_xor = np.array([0, 1, 1, 0])

# AND Gate classifier
mlp_and = MLPClassifier(hidden_layer_sizes=(2,), activation='logistic',solver='sgd', learning_rate_init=0.05, max_iter=10000,tol=0.002, random_state=1)
mlp_and.fit(X_and, y_and)

print("AND Gate Results:")
for i in range(X_and.shape[0]):
    print(f"Input: {X_and[i]}, Output: {mlp_and.predict([X_and[i]])[0]}, Expected: {y_and[i]}")

# XOR Gate classifier
mlp_xor = MLPClassifier(hidden_layer_sizes=(2,), activation='logistic',solver='sgd', learning_rate_init=0.05, max_iter=10000,tol=0.002, random_state=1)
mlp_xor.fit(X_xor, y_xor)

print("\nXOR Gate Results:")
for i in range(X_xor.shape[0]):
    print(f"Input: {X_xor[i]}, Output: {mlp_xor.predict([X_xor[i]])[0]}, Expected: {y_xor[i]}")