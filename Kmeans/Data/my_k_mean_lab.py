import pandas as pd
from sklearn import preprocessing
home_data = pd.read_excel('D:\WorkSpace\Learn python\Kmeans\Data\data.xlsx', 
                          usecols = ['Gender', 'YoB', 'Region', 'Job', 'Wage', 'Mariage', 'ClickCount', 'EndStatus'])
#print(home_data)

home_data.fillna(home_data.mean(), inplace = True)

for col in home_data:
    print(col, home_data[col].dtype)

import seaborn as sns
#sns.scatterplot(data = home_data, x = 'Ngành nghề', y = 'Mức Lương', hue = 'Năm sinh')

import matplotlib.pyplot as plt
#plt.show()

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(home_data.drop('EndStatus', axis = 1), home_data['EndStatus'], test_size=0.33, random_state=0)


X_train_norm = preprocessing.normalize(X_train)
X_test_norm = preprocessing.normalize(X_test)

#print(X_train)
#print(X_train_norm)

from sklearn.cluster import KMeans

#kmeans = KMeans(n_clusters = 2, random_state = 0, n_init='auto')
#kmeans.fit(X_train_norm)

#sns.scatterplot(data = X_train,x = "Năm sinh", y = 'Mức Lương', hue = kmeans.labels_)
#plt.show()

import matplotlib.pyplot as plt

# đếm số lượng mẫu trong mỗi nhóm phân cụm
#cluster_counts = [sum(kmeans.labels_ == i) for i in range(5)]

#plt.pie(cluster_counts, labels=['Cluster 0', 'Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4'], autopct='%1.1f%%')
#plt.title('Cluster Distribution')
#plt.show()

"""
from sklearn.cluster import KMeans

sse = {}

for k in range (1, 11):
    kmeans = KMeans(n_clusters= k, random_state= 42)
    kmeans.fit(X_train_norm)
    sse[k] = kmeans.inertia_

plt.title('The Elbow Method')
plt.xlabel('k')
plt.ylabel('SSE')
sns.pointplot(x = list(sse.keys()), y = list(sse.values()))
plt.show()

"""

from sklearn.metrics import silhouette_score


K = range(2, 11)
fits = []
score = []


for k in K:
    # train the model for current value of k on training data
    model = KMeans(n_clusters = k, random_state = 0, n_init='auto').fit(X_train_norm)
    
    # append the model to fits
    fits.append(model)
    
    # Append the silhouette score to scores
    score.append(silhouette_score(X_train_norm, model.labels_, metric='euclidean'))

"""
sns.lineplot(x = K, y = score)
plt.title('The Elbow Method')
plt.xlabel('k')
plt.ylabel('score')
plt.show() 

print(X_train['Wage']) """

#'Gender', 'YoB', 'Region', 'Job', 'Wage', 'Mariage', 'ClickCount', 'EndStatus'
kmeans = KMeans(n_clusters = 5, random_state = 0, n_init='auto')
kmeans.fit(X_train_norm)
sns.boxplot(x = kmeans.labels_, y = X_train['ClickCount'])
plt.title('Characteristic of k clusters')
plt.xlabel('cluster')
plt.ylabel('ClickCount')
plt.show()


"""
X_train['Cluster'] = kmeans.labels_
#print(X_train)

home_data.fillna(home_data.mean(numeric_only=True), inplace=True)

X_train.groupby('Cluster').agg(
    {
        'Gender': 'mean',
        'YoB': 'mean',
        'Region': 'mean',
        'Job': 'mean',
        'Wage': 'mean',
        'Mariage': 'mean'
    }
).round(2)

#print(X_train) 
"""


