import pandas as pd

home_data = pd.read_csv('.\Data\housing.csv', usecols = ['longitude', 'latitude', 'median_house_value'])
#print(home_data)

import seaborn as sns
sns.scatterplot(data = home_data, x = 'longitude', y = 'latitude', hue = 'median_house_value')

#sns.show()
#Lưu ý rằng, để sử dụng hàm show() 
# trong seaborn, ta cần import module 
# matplotlib.pyplot như sau:

import matplotlib.pyplot as plt
#plt.show()

from sklearn.model_selection import train_test_split

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(home_data[['latitude', 'longitude']], home_data[['median_house_value']], test_size=0.33, random_state=0)

from sklearn import preprocessing

X_train_norm = preprocessing.normalize(X_train)
X_test_norm = preprocessing.normalize(X_test)

#print(X_train)
#print(X_train_norm)

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters = 3, random_state = 0, n_init='auto')
kmeans.fit(X_train_norm)

sns.scatterplot(data = X_train, x = 'longitude', y = 'latitude', hue = kmeans.labels_)
plt.show()

#sns.boxplot(x = kmeans.labels_, y = y_train['median_house_value'])
#plt.show()

from sklearn.metrics import silhouette_score

eval = silhouette_score(X_train_norm, kmeans.labels_, metric='euclidean')

print(eval)