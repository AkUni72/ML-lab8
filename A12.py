import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('bloodtypes.csv')

# Data preprocessing
# Fill missing values with 0 (only for Egypt in this dataset)
data.fillna(0, inplace=True)

# Create a target variable - let's predict if a country is above median population
median_pop = data['Population'].median()
data['Above_Median_Pop'] = (data['Population'] > median_pop).astype(int)

# Prepare features (blood type percentages) and target
X = data.drop(['Country', 'Population', 'Above_Median_Pop'], axis=1)
y = data['Above_Median_Pop']

# Store country names for reference
countries = data['Country']

# Normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data (preserve indices)
X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X_scaled, y, range(len(X)), test_size=0.3, random_state=42)

# Create and train MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(50, 25), activation='relu',solver='adam', max_iter=1000, random_state=42)

mlp.fit(X_train, y_train)

# Predictions
y_pred = mlp.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Plot training loss
plt.plot(mlp.loss_curve_)
plt.title("MLP Training Loss Curve")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.show()

# Feature importance analysis
print("\nFirst layer weights shape:", mlp.coefs_[0].shape)
print("Output layer weights shape:", mlp.coefs_[1].shape)

# Visualize predictions with country names
results = pd.DataFrame({
    'Country': countries.iloc[idx_test].values,
    'Population': data['Population'].iloc[idx_test].values,
    'Actual': y_test.values,
    'Predicted': y_pred
})
print("\nSample Predictions:")
print(results.sample(10))