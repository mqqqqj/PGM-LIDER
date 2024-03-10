import numpy as np  
import pdb 
from sklearnex import patch_sklearn
from sklearn.cluster import KMeans 
import struct
import gc
import os
import time
# os.environ["OPENBLAS_NUM_THREADS"] = "1"
# os.environ["NUM_THREADS"] = "1"
# os.environ["OMP_NUM_THREADS"] = "1"

prefix = '/home/mqj/data/sift'
patch_sklearn()
data = [] 
# 打开二进制文件  
with open(prefix+'/sift_base.fvecs', 'rb') as f:  
    # 初始化一个空数组来存储数据  
    for i in range(1000000):
        f.read(4)
        vector = np.fromfile(f, dtype=np.float32, count=128)
        data.append(vector)

data = np.array(data)
print("finish read")
# pdb.set_trace()
gc.collect()

n_clusters = 800
cluster_directory = os.path.dirname(f'{prefix}/{n_clusters}-kmeans/')
if not os.path.exists(cluster_directory):
    try:
        os.makedirs(cluster_directory)
        print("创建目录成功")
    except Exception as e:
        print(f"创建目录失败：{e}")
else:
    print("目录已存在")

start_time = time.time()  # 获取当前时间
kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(data) 
end_time = time.time()  # 获取当前时间
print(f"finish kmeans: {end_time - start_time} seconds")

# 将每个类的数据分别存储到二进制文件中，并记录每个类所包含的向量数和聚类中心
for i in range(n_clusters):
    # 获取第i类数据
    cluster_data = data[kmeans.labels_ == i]
    cluster_indices = np.where(kmeans.labels_ == i)[0]
    # 将数据存储到二进制文件中
    with open(f'{cluster_directory}/cluster{i}_data.bin', 'wb') as f:
        f.write(cluster_data.tobytes())
    #把label存储到二进制文件中
    with open(f'{cluster_directory}/cluster{i}_indices.bin', 'wb') as f:
        f.write(cluster_indices.tobytes())
    # 输出每个类所包含的向量数和聚类中心
    with open(f'{cluster_directory}/clustersInfo.txt', 'a') as f:
        f.write(f'Cluster {i}: {len(cluster_data)} vectors\n')
        f.write(f'Centroid: {kmeans.cluster_centers_[i]}\n')
        f.write('\n')
    with open(f'{cluster_directory}/clustersInfo.num.vec', 'ab') as f:
        f.write(struct.pack('i',len(cluster_data)))
        f.write(struct.pack('f'*128,*kmeans.cluster_centers_[i]))
    print("finish " + str(i) + " / " + str(n_clusters) + " clusters")