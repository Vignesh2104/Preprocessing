#####################################  DATA OPERATIONS ###################################

'''
 1.  Importing packages
 2.  Data Cleaning - Unmerge cells in excel  
 3.  Basic functions and Date functions 
 4.  Null Handling
 5.  Grouping
 6.  Wriring to excel
 7.  Timing the code              
'''

#Import packages
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time

# =============================================================================
# Data Cleaning - Unmerge cells in excel
# =============================================================================
import re

def rolling_group(val):
    if pd.notnull(val): rolling_group.group +=1 #pd.notnull is signal to switch group
    return rolling_group.group
rolling_group.group = 0 #static variable

def joinFunc(g,column):
    col =g[column]
    joiner = "/" if column == "Action" else ","
    s = joiner.join([str(each) for each in col if pd.notnull(each)])
    s = re.sub("(?<=&)"+joiner," ",s) #joiner = " "
    s = re.sub("(?<=-)"+joiner,"",s) #joiner = ""
    s = re.sub(joiner*2,joiner,s)    #fixes double joiner condition
    return s

def opsclean(z):    
    #z.columns = z.iloc[0]
    #z = z[1:]
    groups = z.groupby(z['Vessel'].apply(rolling_group),as_index=False)
    groupFunct = lambda g: pd.Series([joinFunc(g,col) for col in g.columns],index=g.columns)
    z = groups.apply(groupFunct)
    return z


# =============================================================================
# Null Identification
# =============================================================================
#Option-1
#Null values with percentage of total values
def missing_values_table(df):
        mis_val = df.isnull().sum()  # Total missing values
        mis_val_percent = 100 * df.isnull().sum() / len(df) # Percentage of missing values
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1).rename(columns = {0 : 'Missing Values', 1 : '% of Total Values'})  #table with the results
        mis_val_table = mis_val_table[mis_val_table.iloc[:,1] != 0].sort_values('% of Total Values', ascending=False).round(1)  #Sort descending
        return mis_val_table


#Option-2
def missing_values_analysis(data):
    missing = data.isnull().sum().sort_values(ascending=False)
    percentage = (data.isnull().sum()/data.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([missing, percentage], axis=1, keys=['Total Missing Values', '% Of Missing Data'])
    f, ax = plt.subplots(figsize=(10, 6))
    plt.xticks(rotation='90')
    sns.barplot(x=missing_data.index, y=missing_data['% Of Missing Data'])
    plt.xlabel('Features', fontsize=15)
    plt.ylabel('% Of Missing Data', fontsize=15)
    plt.title('% Of Missing Data By Features', fontsize=15)
    return missing_data

# Get the columns with > 50% missing
def percent_null(df, threshold, drop = True):
    missing_df = missing_values_table(df)
    missing_columns = list(missing_df[missing_df['% of Total Values'] > threshold].index)
    print('No. of columns with more than %d percent null values :'%threshold, len(missing_columns))
    print(missing_columns)
    if drop == True:
        df = df.drop(list(missing_columns), axis = 1)
    return df        


#Null imputation
def impute_null(df,col,impute): 
    if impute == 'mean':
        df[col] = df[col].fillna(df['column name'].mean())   
    elif impute == 'median':
        df[col] = df[col].fillna(df['column name'].median())
    elif impute == 'mode':
        df[col] = df[col].fillna(df['column name'].mode())  
    elif impute == 'drop':
        df[col] = df[col].dropna()
    return df[col]


#----------------------------------------------------------------------------------------------------

"""To capture the end time in order to compute how long the entire code took to execute"""
start_time = time.time()
"""
Write the code to be timed here
"""
print("--- %s seconds ---" % (time.time() - start_time))
