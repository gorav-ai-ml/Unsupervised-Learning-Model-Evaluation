
# Week 3: Unsupervised Learning & Model Evaluation

## 1. Objective
The objective of this project is to understand clustering algorithms and evaluate clustering performance using Python.

## 2. Dataset
Dataset: Mall_Customers.csv

Features used:
- Annual Income (k$)
- Spending Score (1-100)

The features were standardized using StandardScaler before clustering.

## 3. Algorithms Implemented
- K-Means Clustering
- Hierarchical Clustering using Ward linkage

## 4. Model Evaluation
The Silhouette Score was used to evaluate cluster separation.

K-Means Silhouette Score: [Enter your actual score]

Hierarchical Clustering Silhouette Score: [Enter your actual score]

## 5. Hyperparameter Tuning
K-Means was tested with cluster counts from 2 to 7. Silhouette Scores were compared to identify a suitable number of clusters.

Best cluster count: [Enter the result from your output]

## 6. Results
The clustering algorithms grouped customers based on annual income and spending score. Scatter plots and a dendrogram were generated to visualize the clusters.

The results can help identify customer segments for further analysis.

## 7. Conclusion
This project provided practical experience with unsupervised learning, feature scaling, clustering visualization, Silhouette Score evaluation, and hyperparameter tuning.

## 8. Files Generated
- unsupervised_learning.py
- Mall_Customers.csv
- clustering_results.csv
- kmeans_clusters.png
- hierarchical_dendrogram.png
