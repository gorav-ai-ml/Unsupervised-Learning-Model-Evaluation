# Unsupervised Learning & Model Evaluation

## Project Overview
This project demonstrates unsupervised machine learning techniques using Python. It focuses on customer segmentation using K-Means and Hierarchical Clustering.

## Objectives
- Understand K-Means clustering.
- Implement Hierarchical Clustering.
- Standardize numerical features.
- Evaluate clustering using the Silhouette Score.
- Compare different cluster counts through hyperparameter tuning.

## Dataset
The project uses `Mall_Customers.csv` with the following features:
- Annual Income (k$)
- Spending Score (1-100)

## Technologies Used
- Python
- Pandas
- Matplotlib
- Scikit-learn
- SciPy

## Project Files
- `unsupervised_learning.py` — clustering implementation and evaluation.
- `Mall_Customers.csv` — sample dataset.
- `Model_Evaluation_Report.md` — model evaluation report.
- `clustering_results.csv` — clustering results.
- `kmeans_clusters.png` — K-Means visualization.
- `hierarchical_dendrogram.png` — hierarchical clustering dendrogram.

## How to Run
Install the required libraries:

```bash
python -m pip install pandas matplotlib scikit-learn scipy
```

Run the program:

```bash
python unsupervised_learning.py
```

## Evaluation
The Silhouette Score is used to assess cluster separation. K-Means models with different cluster counts are compared to find a suitable configuration.

## Conclusion
This project provides practical experience in clustering, data preprocessing, visualization, and unsupervised model evaluation.
