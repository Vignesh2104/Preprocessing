# Clustering algorithm - K-means

# Importing libraries 
import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats
import scipy.stats as ss
import math 
import matplotlib.pyplot as plt
from scipy.stats import norm
# K-means
from sklearn import preprocessing
from sklearn.cluster import KMeans
import matplotlib.patches as patches

# Importing data
data = pd.read_excel (r'C:\Users\ishaa\Google Drive\Knowledge Management\Data Science\Algorithms\Test Data\Clustering Models.xlsx',sheet_name = 'MR')
data.dtypes
data.columns

# Normalisation based on z-scores
## Create the Scaler object
scaler = preprocessing.StandardScaler()
## Fitting data on the scaler object
numerical = ['Revenue', 'Gross Margin %', 'Transactions']
names = data[numerical].columns
data1 = scaler.fit_transform(data[numerical])
data1 = pd.DataFrame(data1, columns=names)
## Merging data frames 
data2 = pd.merge(data,data1,left_index=True, right_index=True)
## Renaming the variables to the original names  
data2.rename(columns={"Revenue_y":"Revenue","Gross Margin %_y":"Gross Margin %","Transactions_y":"Transactions"},inplace=True)

# Variable selection
x = data2['Revenue']
y = data2['Gross Margin %']
### z = data2['Transactions'], If a third variable is to be added
### add as z in line 42
### Plot generation shows a memory error 

# Setting up the data frame 
data_km = np.column_stack((x,y))

# Identifying the no. of clusters
inertia = []
for n in range(1,10):
    algorithm = (KMeans(n_clusters = n ,init='k-means++', n_init = 10 ,max_iter=300, 
                        tol=0.0001,  random_state= 111  , algorithm='elkan') )
    algorithm.fit(data_km)
    inertia.append(algorithm.inertia_)

# Plotting inertia and the no. of clusters
plt.figure(1 , figsize = (15 ,6))
plt.plot(np.arange(1,10) , inertia , 'o')
plt.plot(np.arange(1,10) , inertia , '-' , alpha = 0.5)
plt.xlabel('Number of Clusters') , plt.ylabel('Inertia')
plt.show()

# Selecting the no. of clusters and running the algorithm
algorithm = KMeans(n_clusters=4).fit(data_km)

# Checking the cluster centers
centroids = algorithm.cluster_centers_

# Checking the algorithm information 
def km_info(algorithm):
    ## Getting the sum of squared distances of samples to their closest cluster center 
    inertia = algorithm.inertia_
    ## Checking the no.of iterations run
    itr = algorithm.n_iter_
    print('Inertia is',inertia,'\nNo.of Iterations Run is',itr)

## Function
km_info(algorithm)

# Adding the lable of cluster to the dataframe 
labels = algorithm.labels_
data2['Cluster'] = pd.DataFrame(labels)

# Plotting the results
## Plot-1
plt.scatter(x,y)
plt.scatter(centroids[:,0],centroids[:,1],s=1000,alpha=0.25)

## Plot-2
h = 0.02
x_min, x_max = data_km[:, 0].min() - 1, data_km[:, 0].max() + 1
y_min, y_max = data_km[:, 1].min() - 1, data_km[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = algorithm.predict(np.c_[xx.ravel(), yy.ravel()]) 

plt.figure(1 , figsize = (15 , 7) )
plt.clf()
Z = Z.reshape(xx.shape)
plt.imshow(Z , interpolation='nearest', 
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap = plt.cm.Pastel2, aspect = 'auto', origin='lower')

plt.scatter( x = 'Revenue' ,y = 'Gross Margin %' , data = data2 , c = labels , s = 200 )
plt.scatter(x = centroids[: , 0] , y =  centroids[: , 1] , s = 300 , c = 'red' , alpha = 0.5)
plt.ylabel('Revenue') , plt.xlabel('Gross Margin %')
plt.show()

# Generating the final data frame
data2.drop(['Revenue', 'Gross Margin %', 'Transactions'], axis=1,inplace = True)
data2.rename(columns={"Revenue_x":"Revenue","Gross Margin %_x":"Gross Margin %","Transactions_x":"Transactions"},inplace=True)
