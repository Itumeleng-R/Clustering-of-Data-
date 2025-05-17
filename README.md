# Clustering-of-Data-
"""
Author: [Itumeleng]   
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Step 1: Generate synthetic fitness tracker data
np.random.seed(42)  # For reproducibility

def generate_cluster(min_steps, max_steps, min_sleep, max_sleep, num_users):
    steps = np.random.randint(min_steps, max_steps, num_users)
    sleep = np.random.uniform(min_sleep, max_sleep, num_users)
    return np.column_stack((steps, sleep))

# Generate random user count for each cluster
num_users_1 = np.random.randint(100, 1000)
num_users_2 = np.random.randint(100, 1000)
num_users_3 = np.random.randint(100, 1000)

# Generate data for each activity level
cluster1 = generate_cluster(8000, 12000, 7, 9, num_users_1)  # Active Users
cluster2 = generate_cluster(5000, 8000, 6, 7.5, num_users_2)  # Moderately Active
cluster3 = generate_cluster(2000, 5000, 5, 6.5, num_users_3)  # Least Active

data = np.vstack((cluster1, cluster2, cluster3))

# Step 2: Visualize the raw data
plt.figure(figsize=(8, 6))
plt.scatter(data[:, 0], data[:, 1], alpha=0.5, color='gray')
plt.xlabel("Steps per Day")
plt.ylabel("Hours Slept per Night")
plt.title("Fitness Tracker Data Visualization")
plt.show()

# Step 3: Apply KMeans Clustering
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(data)
labels = kmeans.labels_

# Step 4: Visualize the clustered data
plt.figure(figsize=(8, 6))
colors = ['red', 'blue', 'green']
for i in range(3):
    plt.scatter(data[labels == i, 0], data[labels == i, 1], alpha=0.6, color=colors[i], label=f'Cluster {i+1}')
plt.xlabel("Steps per Day")
plt.ylabel("Hours Slept per Night")
plt.title("KMeans Clustering of Fitness Tracker Data")
plt.legend()
plt.show()

# Step 5: Cluster Analysis
for i in range(3):
    cluster_data = data[labels == i]
    avg_steps = np.mean(cluster_data[:, 0])
    avg_sleep = np.mean(cluster_data[:, 1])
    print(f"Cluster {i+1}: Avg Steps = {avg_steps:.2f}, Avg Sleep = {avg_sleep:.2f} hours")

# Insights and Observations
print("\n--- Insights & Observations ---")
print("1. The Active users (Cluster 1) have a higher step count and tend to sleep 7-9 hours per night.")
print("2. Moderately Active users (Cluster 2) have a medium step count and slightly lower sleep hours.")
print("3. Least Active users (Cluster 3) have the lowest step count and sleep fewer hours on average.")
print("4. The clustering confirms expected patterns in activity and sleep behavior.")
