
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

data = pd.read_csv("Mall_Customers.csv")

X = data[["Annual Income (k$)", "Spending Score (1-100)"]]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
data["KMeans_Cluster"] = kmeans.fit_predict(X_scaled)

linked = linkage(X_scaled, method="ward")
data["Hierarchical_Cluster"] = fcluster(
    linked, t=5, criterion="maxclust"
)

print(data.head(10))
print("\nK-Means Cluster Counts:")
print(data["KMeans_Cluster"].value_counts().sort_index())

print("\nHierarchical Cluster Counts:")
print(data["Hierarchical_Cluster"].value_counts().sort_index())

plt.figure(figsize=(8, 5))
plt.scatter(
    X["Annual Income (k$)"],
    X["Spending Score (1-100)"],
    c=data["KMeans_Cluster"],
    cmap="viridis"
)
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("K-Means Customer Clusters")
plt.savefig("kmeans_clusters.png")
plt.show()

plt.figure(figsize=(10, 5))
dendrogram(linked, truncate_mode="lastp", p=10)
plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Cluster Groups")
plt.ylabel("Distance")
plt.tight_layout()
plt.savefig("hierarchical_dendrogram.png")
plt.show()

data.to_csv("clustering_results.csv", index=False)
print("\nResults saved to clustering_results.csv")


from sklearn.metrics import silhouette_score


kmeans_score = silhouette_score(X_scaled, data["KMeans_Cluster"])
print("\nK-Means Silhouette Score:", round(kmeans_score, 3))

hierarchical_score = silhouette_score(
    X_scaled, data["Hierarchical_Cluster"]
)
print("Hierarchical Silhouette Score:", round(hierarchical_score, 3))



print("\nHyperparameter Tuning for K-Means")

for k in range(2, 8):
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    print(f"Clusters: {k}, Silhouette Score: {score:.3f}")


