# DSC SRM TECHNICAL TASK
# K-Means clustering to segment customers based on PURCHASING BEHAVIOUR
# Dataset: https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns


#Load the data
data = pd.read_csv(r'C:/Users/TIJI JOJI/Downloads/Mall_Customers.csv')
print(data.info())
print(data.head())


data.drop('CustomerID', axis=1)

# Encode Gender, male - 0, female - 1
data['Gender'] = data['Gender'].map({'Male': 0, 'Female': 1})

#Create features for Cluster
X = data[['Gender', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

#Apply Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#Apply Elbow method to find the optimal number of clusters (K)
inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init='auto')
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# Elbow curve
plt.figure(figsize=(7, 5))
plt.plot(range(1, 11), inertia, marker='o', linestyle='--')
plt.title('Elbow Curve')
plt.xlabel('K value')
plt.ylabel('Inertia')
plt.show()

#Apply K Means on the optimal number of clusters (K = 4)
optimal_k = 4
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42, n_init='auto')
y_kmeans = kmeans.fit_predict(X_scaled)

#Visualize the clusters
plt.figure(figsize=(10, 7))
sns.scatterplot(x=X_scaled[:, 2], y=X_scaled[:, 3], hue=y_kmeans, palette="viridis", s=100)
# We are plotting the Annual Income (x-axis) against the Spending Score (y-axis) for each customer.
plt.title('Customer Segments (K Means Cluster)')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.show()

#Add the clusters information for each row to original data
data['Clusters'] = y_kmeans
print(data)