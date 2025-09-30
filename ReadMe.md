# K-Nearest Neighbors (KNN) Classification Summary

## Objective

The primary goal was to implement and evaluate the K-Nearest Neighbors (KNN) algorithm for classification. This involved mastering the full machine learning workflow: data preparation (including crucial feature normalization), hyperparameter tuning (finding the optimal K), and comprehensive model evaluation and visualization.

## Key Insights

**Normalization is Crucial**: Feature scaling (standardization) was essential. Since KNN is distance-based, normalization prevented features with larger values from unfairly dominating the calculation.

**Optimal K Selection**: The best value for K minimizes the test error, balancing the trade-off between high variance (low K, overfitting) and high bias (high K, oversmoothing).

**Non-Linear Boundaries**: KNN creates non-linear, piecewise decision boundaries, confirming its nature as a non-parametric and lazy learner that classifies based on local neighborhoods.