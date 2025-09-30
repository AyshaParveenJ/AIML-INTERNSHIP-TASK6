import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from matplotlib.colors import ListedColormap

# ==============================================================================
# 1. LOAD AND PREPARE DATASET (Normalization/Scaling)
# ==============================================================================

print("--- Step 1: Data Loading and Preparation ---")

# Load the built-in Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

# Split data into training and testing sets (70/30 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Normalize/Standardize Features (Crucial for KNN)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Dataset loaded. Training samples: {X_train_scaled.shape[0]}, Test samples: {X_test_scaled.shape[0]}")
print("Features standardized successfully.")

# ==============================================================================
# 2. EXPERIMENT WITH DIFFERENT VALUES OF K
# ==============================================================================

print("\n--- Step 2: Finding the Optimal K Value ---")

error_rates = []
k_range = range(1, 21) # Test K from 1 to 20

# Loop through K values and record the test error rate
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    y_pred_k = knn.predict(X_test_scaled)
    # Calculate error rate (1 - accuracy)
    error_rates.append(np.mean(y_pred_k != y_test))

# Find the K with the minimum error
optimal_k = k_range[np.argmin(error_rates)]
print(f"Optimal K found: {optimal_k}")

# Visualize K vs. Error Rate
plt.figure(figsize=(10, 6))
plt.plot(k_range, error_rates, color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=8)
plt.title('Error Rate vs. K Value')
plt.xlabel('K Value (Number of Neighbors)')
plt.ylabel('Error Rate')
plt.xticks(k_range)
plt.grid(True)
plt.show()

# ==============================================================================
# 3. EVALUATE FINAL MODEL (Accuracy, Confusion Matrix)
# ==============================================================================

print("\n--- Step 3: Model Evaluation ---")

# Train the final model using the optimal K
knn_final = KNeighborsClassifier(n_neighbors=optimal_k)
knn_final.fit(X_train_scaled, y_train)
y_final_pred = knn_final.predict(X_test_scaled)

# 4. Evaluate model using accuracy
accuracy = accuracy_score(y_test, y_final_pred)
print(f"Model Accuracy with Optimal K={optimal_k}: {accuracy:.4f}")

# Generate Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_final_pred)
print("\nConfusion Matrix:")
print(conf_matrix)

# Classification Report (Precision, Recall, F1-Score)
print("\nClassification Report:")
print(classification_report(y_test, y_final_pred, target_names=target_names))

# ==============================================================================
# 4. VISUALIZE DECISION BOUNDARIES (Using two features)
# ==============================================================================

print("\n--- Step 4: Visualizing Decision Boundaries ---")

# Use only the first two features (Sepal Length and Sepal Width) for 2D plotting
X_viz = X[:, :2]
y_viz = y
h = .02  # step size in the mesh

# Scale the 2-feature data
scaler_viz = StandardScaler()
X_viz_scaled = scaler_viz.fit_transform(X_viz)

# Train KNN on the 2-feature subset with optimal K
knn_viz = KNeighborsClassifier(n_neighbors=optimal_k)
knn_viz.fit(X_viz_scaled, y_viz)

# Create color maps
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

# Plot the decision boundary
x_min, x_max = X_viz_scaled[:, 0].min() - 1, X_viz_scaled[:, 0].max() + 1
y_min, y_max = X_viz_scaled[:, 1].min() - 1, X_viz_scaled[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# Predict class for each point in the mesh
Z = knn_viz.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(10, 7))
plt.pcolormesh(xx, yy, Z, cmap=cmap_light, shading='auto')

# Plot the data points
plt.scatter(X_viz_scaled[:, 0], X_viz_scaled[:, 1], c=y_viz, cmap=cmap_bold,
            edgecolor='k', s=20)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title(f"KNN Decision Boundaries (K={optimal_k})")
plt.xlabel(feature_names[0])
plt.ylabel(feature_names[1])
plt.show()