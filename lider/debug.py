from sklearn.cluster import KMeans
import numpy as np

# 假设 data 是一个二维数组，每行是一个数据点，labels 是一个一维数组，包含每个数据点的标签
data = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])
labels = np.array(['a', 'b', 'c', 'd', 'e', 'f'])

kmeans = KMeans(n_clusters=2, random_state=0).fit(data)

# 输出每个簇的标签
for i in range(kmeans.n_clusters):
    print(f"Cluster {i}:")
    for j in np.where(kmeans.labels_ == i)[0]:#所有属于编号为 i 的簇的数据点的索引
        print(labels[j])

print(kmeans.labels_)