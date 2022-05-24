#Importing the Dependencies
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt

#Data Collection & Analysis

# loading the data from csv file to a Pandas DataFrame
data= pd.read_csv('/content/Mall_Customers.csv')
# finding shape of the dataframe
data.shape

# first 5 rows in the dataframe
data.head()

# getting some informations about the dataset
data.info()

#Choosing the Annual Income Column & Spending Score column
x= data.iloc[:,[3,4]].values
x

#Choosing the number of clusters

#WCSS -> Within Clusters Sum of Squares

# finding wcss value for different number of clusters
wcss=[]
for i in range(1,11):
  kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
  kmeans.fit(x)

  wcss.append(kmeans.inertia_)

# plot an elbow graph
sns.set()
plt.plot(range(1,11), wcss)
plt.title('Elbow Label Graph')
plt.xlabel('Number of Clusters')
plt.ylabel('Within Clusters Sum Of Squares')
plt.show()

#Optimum Number of Clusters = 5

#Training the k-Means Clustering Model

kmeans = KMeans(n_clusters=5, init='k-means++', random_state=0)

# return a label for each data point based on their cluster
y=kmeans.fit_predict(x)
print(y)

# plotting all the clusters
plt.Figure(figsize=(8,8))
plt.scatter(x[y==0,0], x[y==0,1], s=50, c='green', label='Cluster 1')
plt.scatter(x[y==1,0], x[y==1,1], s=50, c='red', label='Cluster 2')
plt.scatter(x[y==2,0], x[y==2,1], s=50, c='blue', label='Cluster 3')
plt.scatter(x[y==3,0], x[y==3,1], s=50, c='orange', label='Cluster 4')
plt.scatter(x[y==4,0], x[y==4,1], s=50, c='yellow', label='Cluster 5')
plt.legend()
plt.title('Customer Groups')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.show()