# K-Means Clustering

[w3school](https://www.w3schools.com/python/python_ml_k-means.asp)

- Clustering is grouping similar data points together
- K means is unsupervised learning method for clustering data points
- Algorithm divides the data points into K amount of clusters

- A data point is randomly assigned to one of the K clusters
- Compute the centroid (Calc 2)
- Reassign the current data point to the closest centroid
- Continoually repeat this process of reassignment for each data point until they do not change

- Elbow method lets us graph the [inertia](https://www.codecademy.com/learn/machine-learning/modules/dspath-clustering/cheatsheet)

- Good model has low intertia and low number of k clusters

- Inertia measures how well a dataset was clustered by K-Means. It is calculated by measuring the distance between each data point and its centroid, squaring this distance, and summing these squares across one cluster.

- Once the graph shows the elbow, then that means that you should divide the amount of clusters by 2
