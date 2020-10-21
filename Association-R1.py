# Association Analysis

# Importing Libraries 
import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats
import scipy.stats
import math 
import matplotlib.pyplot as plt
from scipy.stats import norm
import nltk
# Importing mlxtend for apriori implementation and extracting association rules
from mlxtend.frequent_patterns import apriori, association_rules

## Current example is for OSL. Modify the data set and feature selection accordingly 

# Importing data
data = pd.read_excel (r'C:\Users\ishaa\Google Drive\Knowledge Management\Statistics\M+R-Sample-June20 - ASS.xlsx',sheet_name = 'Sheet1 (3)')
data.dtypes

# Data transformation
## Feature selection. Ensure MECE 
data1 = data[['ID','Branch','Product','Customer','Revenue']]
#--------------------------------------------------------------------------------------------#    
## Data frame transformation for the algorithm
def trans(data1,ID_variable,quant_variable):
    data2=data1[[ID_variable,quant_variable]]
    data1=data1.drop(columns=[quant_variable])
    data1 = data1.set_index(ID_variable)
    data1 = data1.stack().reset_index(level=1, drop=True).reset_index()
    data3 =pd.DataFrame()
    data3 = data1.join(data2.set_index(ID_variable), on=ID_variable)
    data3 = data3.rename(columns={0: "Items"})
    return data3
# Function 
data3 = trans(data1,ID_variable='ID',quant_variable='Revenue')

## One hot encoding
data4 = data3.groupby(['ID', 'Items'])['Revenue'].sum().unstack().fillna(0)

## Converting all values >1 to 1 
def encode_units(x):
    if x <= 0:
        return 0
    if x >= 1:
        return 1
# Function
data_sets = data4.applymap(encode_units)
#--------------------------------------------------------------------------------------------#    

# Generating item sets
## Defining minimum support
min_support_count = 10
N = data4.shape[0]
support = min_support_count/N
print (support)
## Generating frequent itemsets based on minimum support
frequent_itemsets = apriori(data_sets, min_support=support, use_colnames=True, max_len=None, verbose=0, low_memory=False)

## Association mining rules
### Metric = lift or confidence
### Threshold - For lift = 1 - reads as >= . For confidence = 0.7  
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)

## Transfroming rules data frame by adding A & C lengths,lift & leverage categories & dropping conviction   
def rules_trans(rules):
    # Adding A & C lengths
    rules["antecedent_len"] = rules["antecedents"].apply(lambda x: len(x))
    rules["consequents_len"] = rules["consequents"].apply(lambda x: len(x))
    # Lift category 
    conditions = [
        (rules['lift'] == 1),
        (rules['lift'] > 1),
        (rules['lift'] < 1.6)]
    choices = ['A & C Are Independent', 'A & C Are Positively Related', 'A & C Are Negatively Related']
    rules['Lift Category'] = np.select(conditions, choices)
    # Leverage category
    conditions = [
        (rules['leverage'] == 0),
        (rules['leverage'] > 0),
        (rules['leverage'] < 0)]
    choices = ['A & C Are Independent', 'A & C Are Positively Related', 'A & C Are Negatively Related']
    rules['Leverage Category'] = np.select(conditions, choices)
    # Dropping conviction column 
    rules.drop(['conviction'], axis=1,inplace = True)
#Function
rules_trans(rules)

# Rearanging columns 
rules = rules[['antecedents', 'consequents','antecedent_len', 'consequents_len','antecedent support',
                   'consequent support', 'support', 'confidence', 'lift','Lift Category','leverage','Leverage Category']]
#--------------------------------------------------------------------------------------------#    
## Additional rules filer 
rules = rules[ (rules['antecedent_len'] >= 2) &
        (rules['confidence'] > 0.75) &
        (rules['lift'] > 1.2) ]
#--------------------------------------------------------------------------------------------#    
## Anlaysis
### Suport vs Confindence 
plt.scatter(rules['support'], rules['confidence'], alpha=0.5)
plt.xlabel('Support')
plt.ylabel('Confidence')
plt.title('Support vs Confidence')
plt.show()
### Support vs Lift
plt.scatter(rules['support'], rules['lift'], alpha=0.5)
plt.xlabel('Support')
plt.ylabel('Lift')
plt.title('Support vs Lift')
plt.show()
### Lift vs confidence
fit = np.polyfit(rules['lift'], rules['confidence'], 1)
fit_fn = np.poly1d(fit)
plt.plot(rules['lift'], rules['confidence'], 'yo', rules['lift'], 
fit_fn(rules['lift']))
#--------------------------------------------------------------------------------------------#    
## Writing to excel 
writer = pd.ExcelWriter('Association.xlsx', engine='xlsxwriter')
rules.to_excel(writer,sheet_name= 'Rules', index=False)
writer.save()
#--------------------------------------------------------------------------------------------#    
# Transforming the data frame for correlation analysis and Power BI
 
# Importing data
pbi_data = pd.read_excel (r'C:\Users\ishaa\Association.xlsx',sheet_name = 'Rules')

# Adding and transforming the itemset column
def item_trans(pbi_data):
    pbi_data['Itemset-1'] = pbi_data['antecedents'].replace(regex={"\\bfrozenset\\b":' '})
    pbi_data['Itemset-1'] = pbi_data['Itemset-1'].str.strip("()")
    pbi_data['Itemset-1'] = pbi_data['Itemset-1'].str.strip("{}")
    pbi_data['Itemset-1'] = pbi_data['Itemset-1'].str[3:]
    pbi_data['Itemset-2'] = pbi_data['consequents'].replace(regex={"\\bfrozenset\\b":' '})
    pbi_data['Itemset-2'] = pbi_data['Itemset-2'].str.strip("()")
    pbi_data['Itemset-2'] = pbi_data['Itemset-2'].str.strip("{}")
    pbi_data['Itemset-2'] = pbi_data['Itemset-2'].str[3:]
    pbi_data['Itemset'] = pbi_data['Itemset-1'] + ',' + pbi_data['Itemset-2']
    pbi_data.drop(['Itemset-1','Itemset-2'], axis=1,inplace = True)
    pbi_data = pbi_data[['Itemset','antecedents', 'consequents','antecedent_len', 'consequents_len','antecedent support',
                         'consequent support', 'support', 'confidence', 'lift','Lift Category','leverage','Leverage Category']]
    return pbi_data

# Function 
pbi_data = item_trans(pbi_data)
#--------------------------------------------------------------------------------------------#    
