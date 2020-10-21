# -*- coding: utf-8 -*-
"""
Created on Sat Jul 25 20:58:43 2020

@author: DELL
"""
#Import packages
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re

from sklearn.feature_selection import VarianceThreshold
import itertools

from sklearn.feature_selection import VarianceThreshold
    

df = pd.read_excel('C:/Users/DELL/Documents/Ultra Insights/OCEAN SPARKLE/sample.xlsx')

class Standards:
   def describe(self,df):
        desc = df.describe()  
        total = df._get_numeric_data().sum().reset_index().rename(columns={'index':'Variable',0:'Sum'})
        pivot_total = total.pivot_table(values='Sum', columns='Variable', aggfunc=np.sum)
        skew = df._get_numeric_data().skew().reset_index().rename(columns={'index':'Variable',0:'Skewness'})
        pivot_skew = skew.pivot_table(values='Skewness', columns='Variable', aggfunc=np.mean)
        kurt = df._get_numeric_data().kurt().reset_index().rename(columns={'index':'Variable',0:'Kurtosis'})
        pivot_kurt = kurt.pivot_table(values='Kurtosis', columns='Variable', aggfunc=np.mean)
        desc = pd.concat([desc,pivot_total,pivot_skew,pivot_kurt])
        return desc
    
   def categorize_by_quantiles(self,df, col):
        MT_Q1 = df[col].quantile(0.25).round(-1)
        MT_Q2 = df[col].quantile(0.50).round(-1)
        MT_Q3 = df[col].quantile(0.75).round(-1)    
        conditions = [(df[col] <=MT_Q1),
                      (df[col] >MT_Q1) & (df[col]<MT_Q3),
                      (df[col] >=MT_Q3)]
        categories = ['Low','Medium', 'High']
        df[str(col)+' Category'] = np.select(conditions, categories)
        df[str(col)+' Category'] = np.where(df[col].isnull(),'',df[str(col)+' Category'])
        return df[str(col)+' Category']
    
   def eighty_twenty_column(self,df, threshold = 80): 
        threshold = float(threshold)
        data_output =pd.DataFrame()
        cat_cols = list(df.loc[:,(df.dtypes == np.object)].columns)
        for i in range(len(cat_cols)): #giving those columns(length 12) in for loop
            L = pd.Series(df[cat_cols[i]]).value_counts().reset_index().rename(columns={0:"counts","index":"Level"})
            names = list(L.columns)
            Perc_Level_Count = round((L.iloc[:,1] * 100) /L.iloc[:,1].sum(),2)  
            Level_df         = pd.DataFrame({'Level' : L['Level'], 'Count': L.iloc[:,1], 'Perc_Count' : Perc_Level_Count})
            Level_df['Cum_Perc_Count'] = Level_df['Perc_Count'].cumsum()
            L_retained      = Level_df.query('Cum_Perc_Count <= @threshold')['Level']
            retained_levels = list(L_retained)        
            output_levels = pd.DataFrame({L.columns[1]:retained_levels})
            data_output = pd.concat([data_output,output_levels],axis=1)
        return data_output    
   
   def missing_values_table(self,df):
        mis_val = df.isnull().sum()  # Total missing values
        mis_val_percent = 100 * df.isnull().sum() / len(df) # Percentage of missing values
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1).rename(columns = {0 : 'Missing Values', 1 : '% of Total Values'})  #table with the results
        mis_val_table = mis_val_table[mis_val_table.iloc[:,1] != 0].sort_values('% of Total Values', ascending=False).round(1)  #Sort descending
        return mis_val_table
    
   missing = df.isnull().sum().sort_values(ascending=False)
   percentage = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)
   missing_data = pd.concat([missing, percentage], axis=1, keys=['Total Missing Values', '% Of Missing Data'])
   #f, ax = plt.subplots(figsize=(10, 6))
   #plt.xticks(rotation='90')
   #sns.barplot(x=missing_data.index, y=missing_data['% Of Missing Data'])
   #plt.xlabel('Features', fontsize=15)
   #plt.ylabel('% Of Missing Data', fontsize=15)
   #plt.title('% Of Missing Data By Features', fontsize=15)
   print(missing_data) 
   
   def percent_null(self,df, threshold, drop = True):
    missing_df = missing_values_table(df)
    missing_columns = list(missing_df[missing_df['% of Total Values'] > threshold].index)
    print('No. of columns with more than %d percent null values :'%threshold, len(missing_columns))
    print(missing_columns)
    if drop == True:
        df = df.drop(list(missing_columns), axis = 1)
    return df  
    
   def impute_null(self,df,col,impute): 
        if impute == 'mean':
            df[col] = df[col].fillna(df['column name'].mean())   
        elif impute == 'median':
            df[col] = df[col].fillna(df['column name'].median())
        elif impute == 'mode':
            df[col] = df[col].fillna(df['column name'].mode())  
        elif impute == 'drop':
            df[col] = df[col].dropna()
        return df[col]
   def outlier(self,data,col,iqrfactor,replace_by):
        Q1 = data[col].quantile(0.25)  
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        Min = Q1-(iqrfactor*IQR)
        Max = Q3+(iqrfactor*IQR)
        Mean = data[col][(data[col]>=Min) & (data[col]<=Max)].mean()
        Median = data[col][(data[col]>=Min)&(data[col]<=Max)].median()
        Mode = data[col][(data[col]>=Min)&(data[col]<=Max)].mode()[0]
        print('IQR is',IQR,'\nMin is',Min,'\nMax is',Max,'\nMean is',Mean,'\nMedian is',Median,'\nMode is',Mode)
        if replace_by == 'mean':        
            data[col]=np.where((data[col]<Min)|(data[col]>Max),Mean,data[col])
        elif replace_by == 'median':
            data[col]=np.where((data[col]<Min)|(data[col]>Max),Median,data[col])
        elif replace_by == 'blank':
            data[col]=np.where((data[col]<Min)|(data[col]>Max),np.nan,data[col])
        return data[col]
    
   def varianceSelection(self,data, THRESHOLD = .99):
        sel = VarianceThreshold(threshold=(THRESHOLD * (1 - THRESHOLD)))
        sel.fit_transform(data)
        return data[[c for (s, c) in zip(sel.get_support(), data.columns.values) if s]]

    
   def remove_feat_constants(self,data_frame):
        # Remove feature vectors containing one unique value,
        # because such features do not have predictive value.
        # Let's get the zero variance features by fitting VarianceThreshold
        # selector to the data, but let's not transform the data with
        # the selector because it will also transform our Pandas data frame into
        # NumPy array and we would like to keep the Pandas data frame. Therefore,
        # let's delete the zero variance features manually.
        n_features_originally = data_frame.shape[1]
        selector = VarianceThreshold()
        selector.fit(data_frame)
        # Get the indices of zero variance feats
        feat_ix_keep = selector.get_support(indices=True)
        orig_feat_ix = np.arange(data_frame.columns.size)
        feat_ix_delete = np.delete(orig_feat_ix, feat_ix_keep)
        print(feat_ix_delete)
        # Delete zero variance feats from the original pandas data frame
        data_temp = data_frame.columns[feat_ix_delete]
        data_frame = data_frame.drop(labels=data_frame.columns[feat_ix_delete],axis=1)
        
        # Print info
        n_features_deleted = feat_ix_delete.size
        print("  - Deleted %s / %s features (~= %.1f %%)" % (
            n_features_deleted, n_features_originally,
            100.0 * (np.float(n_features_deleted) / n_features_originally)))
        
        return (data_frame,data_temp)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
