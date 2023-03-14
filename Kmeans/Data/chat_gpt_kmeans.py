import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Đọc dữ liệu
data = pd.read_excel('D:\WorkSpace\Learn python\Kmeans\Data\Lab\Thông-tin-cá-nhân-Hành-vi-khách-hàng.xlsx', 
                     usecols=['Năm sinh', 'Ngành nghề', 'Mức Lương', 'Tỉnh', 'Giới', 'Tình_trạng_hôn_nhân'])

# Xử lý missing data
data.fillna(data.mean(), inplace=True)

# Chuẩn hóa dữ liệu
numeric_features = ['Tuổi', 'Lương', 'Thời gian làm việc']
data_numeric = data[numeric_features]
scaler = preprocessing.StandardScaler()
data_scaled = scaler.fit_transform(data_numeric)


# Áp dụng K-means
kmeans = KMeans(n_clusters=5, random_state=0, n_init='auto')
kmeans.fit(data_scaled)

# Vẽ biểu đồ 3D
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data_scaled[:, 0], data_scaled[:, 2], data_scaled[:, 4], c=kmeans.labels_, s=50, alpha=0.5)
ax.set_xlabel('Năm sinh')
ax.set_ylabel('Mức Lương')
ax.set_zlabel('Giới')
plt.show()
