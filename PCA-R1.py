# Principal Component Analysis (PCA)

# Importing libraries 
import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats
import scipy.stats as ss
import math 
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.patches as patches

# Importing data
data = pd.read_excel (r'C:\Users\ishaa\Google Drive\Knowledge Management\Statistics\PCA.xlsx',sheet_name = 'Sheet1')
data.dtypes
data.columns

# Problem statement 
## CLassification of vessel
## Using PCA to identify which features to use to classify vessels 

# Data transformation 
## Converting all nan to 0
data.fillna(0,inplace=True)

## Transpose the data set 
def transpose(data): 
    data1 = data.transpose()
    data1 = data1.rename(columns=data1.iloc[0]).drop(data1.index[0])
    data1.reset_index(inplace = True)
    data1 = data1.rename(columns = {'index':'Variables'})
    return data1
### Function
data1 = transpose(data)

## Separating features of the dataframe from the labels
X = data1.iloc[:,1:1000000].values
y = data1.iloc[:,0].values
data_col = data1['Variables']

# Standardisation
scaled_data = StandardScaler().fit_transform(X)

# PCA
## PCA 
pca = PCA()
pca.fit(scaled_data)
pca_data = pca.transform(scaled_data)
    
# Explained variance (based on eigen values of the features)
per_var = np.round(pca.explained_variance_ratio_*100,decimals=1)
labels = ['PC'+str(x) for x in range (1,len(per_var)+1)]

## Generating the PCA data frame with the vectors for each feature
pca_df = pd.DataFrame(pca_data,columns=labels)
pca_df = pd.merge(data_col,pca_df,left_index=True, right_index=True)
    
# Generating the variance data frame
per_var = np.round(pca.explained_variance_ratio_*100,decimals=1)
exp_var1 = pd.DataFrame(per_var)
labels_df = pd.DataFrame(labels)
exp_var = pd.merge(labels_df,exp_var1,left_index=True, right_index=True)
exp_var.rename(columns={"0_x":"Principal Component","0_y":"% of Explained Variance"},inplace=True)

# Scree plot
plt.style.use('ggplot')
x1 = exp_var['Principal Component']
y1 = exp_var['% of Explained Variance']
x_pos = [i for i, _ in enumerate(x1)]
plt.bar(x_pos, y1, color='blue')
plt.xlabel("Principal Component")
plt.ylabel("% of Explained Variance")
plt.title("Scree Plot")
plt.xticks(x_pos, x1)
plt.show()

# Loading scores for 2 PCs for all the variables
listA = data['Vessel Name']
loading_scores_pc1 = pd.Series(pca.components_[0],index=listA)
sorted_loading_scores_pc1 = loading_scores_pc1.abs().sort_values(ascending=False)
loading_scores_pc2 = pd.Series(pca.components_[1],index=listA)
sorted_loading_scores_pc2 = loading_scores_pc2.abs().sort_values(ascending=False)
loading = pd.DataFrame(sorted_loading_scores_pc1)
loading['PC2'] = pd.DataFrame(sorted_loading_scores_pc2)
loading.rename(columns={0:"PC1"},inplace=True)
loading.reset_index(inplace = True)

# PCA scatter plot
x=loading['PC1']
y=loading['PC2']
plt.scatter(x,y)
plt.title('PC Scatter Plot')
plt.xlabel('PC1 - {0}%'.format(per_var[0]))
plt.ylabel('PC2 - {0}%'.format(per_var[1]))
#--------------------------------------------------------------------------------------------#    
#--------------------------------------------------------------------------------------------#    
# Performing an additional k-mean clustering on the data 

# Variable selection
Xc = loading['PC1']
Yc = loading['PC2']

# Setting up the data frame 
data_km = np.column_stack((Xc,Yc))

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
algorithm = KMeans(n_clusters=3).fit(data_km)

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
loading['Cluster'] = pd.DataFrame(labels)

# Plotting the results
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

plt.scatter( x = 'PC1' ,y = 'PC2' , data = loading , c = labels , s = 200 )
plt.scatter(x = centroids[: , 0] , y =  centroids[: , 1] , s = 300 , c = 'red' , alpha = 0.5)
plt.ylabel('PC1') , plt.xlabel('PC2')
plt.show()

# Writing to excel 
writer = pd.ExcelWriter('PCA_Cluster.xlsx', engine='xlsxwriter')
loading.to_excel(writer,sheet_name= 'PCA_Cluster', index=False)
writer.save()
#--------------------------------------------------------------------------------------------#    
