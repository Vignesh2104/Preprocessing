#Descriptive & exploratory analytics functions
 
# Importing libraries 
import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats
import scipy.stats as ss
import math 
import matplotlib.pyplot as plt
from scipy.stats import norm
## For Pareto analysis 
from matplotlib.ticker import PercentFormatter
#From Stats Models, using ordinary least squares (ols) method
import statsmodels.api as sm
from statsmodels.formula.api import ols
# For ANOVOA Tukey's method 
from statsmodels.stats.multicomp import (pairwise_tukeyhsd,MultiComparison)
# For Chi-Square Test & Cramer's V
from scipy.stats import chi2_contingency
from scipy.stats import chi2
from sklearn import preprocessing
from collections import Counter

# Univariate data - Quantitative variables   

## Descriptive statistics 
def describe(df):
    desc = df.describe()  
    total = df._get_numeric_data().sum().reset_index().rename(columns={'index':'Variable',0:'Sum'})
    pivot_total = total.pivot_table(values='Sum', columns='Variable', aggfunc=np.sum)
    skew = df._get_numeric_data().skew().reset_index().rename(columns={'index':'Variable',0:'Skewness'})
    conditions = [(skew['Skewness'] >= -0.5) & (skew['Skewness'] <= 0.5),
              ((skew['Skewness'] >= -1) & (skew['Skewness'] < -0.5)) |  ((skew['Skewness'] > 0.5) & (skew['Skewness'] <= 1)),
              (skew['Skewness'] <-1) | (skew['Skewness'] > 1)]
    choices = ['Fairly symmetrical','Moderately skewed', 'Highly skewed']
    skew['Skewness Category'] = np.select(conditions,choices)
    kurt = df._get_numeric_data().kurt().reset_index().rename(columns={'index':'Variable',0:'Kurtosis'})
    conditions =  [kurt['Kurtosis'] == 3,kurt['Kurtosis'] > 3,kurt['Kurtosis'] < 3]
    choices = ['Mesokurtic-Similar to a normal dist', 'Leptokurtic-heavy tailed & has outliers','Platykurtic-light tailed & lacks outliers']
    kurt['Kurtosis Category'] = np.select(conditions,choices)
    kurt.drop(columns = 'Variable', inplace = True)
    x = pd.concat([skew,kurt], axis = 1).T
    x.columns = x.iloc[0]
    x = x.iloc[1:]
    desc = pd.concat([desc,pivot_total,x])
    return desc

## Variable distribution identification 
## If discrete, probability mass function (PMF) - use histograms 
## If continuous, probability density function (PDF) - use histograms and distribution plots  
#--------------------------------------------------------------------------------------------#
# Normal distribution
## Checking the distribution against the standard normal distribution ranges  
def normal_dist(data, variable):
    y = pd.DataFrame()
    y['SD From Mean'] = 1,2,3
    y['Range From'] = data[variable].mean() - (y['SD From Mean'] * data[variable].std())
    y['Range To'] = data[variable].mean() + (y['SD From Mean'] * data[variable].std())
    y['Count'] = [data[(data[variable] > y['Range From'][i]) & (data[variable] < y['Range To'][i])].count() for i in range(0,3)]
    y['Count'] = y['Count'].str[0]
    y['% of Data'] = np.round((y['Count'] / data[variable].count()),2)
    y['Required % of Data'] = 0.68, 0.95, 0.99
    return y

#--------------------------------------------------------------------------------------------#
# Univariate outlier analysis and treatment

## Replacing outliers by mean/median/mode and blanks as blanks
## IQR Factors = 1.00/1.15/1.50
def outlier(data,variable,iqrfactor,replace_by):
    SS = ((data[variable] - (data[variable].mean()))**2)
    MSE = (SS.sum())/len(data[variable])
    RMSE_Before = math.sqrt(MSE)
    Q1 = data[variable].quantile(0.25)  
    Q3 = data[variable].quantile(0.75)
    IQR = Q3 - Q1
    Min = Q1-(iqrfactor*IQR)
    Max = Q3+(iqrfactor*IQR)
    Mean = data[variable][(data[variable]>=Min) & (data[variable]<=Max)].mean()
    Median = data[variable][(data[variable]>=Min)&(data[variable]<=Max)].median()
    Mode = data[variable][(data[variable]>=Min)&(data[variable]<=Max)].mode()[0]
    print('IQR is',IQR,'\nMin is',Min,'\nMax is',Max,'\nMean is',Mean,'\nMedian is',Median,'\nMode is',Mode)
    if replace_by == 'mean':        
        data[variable]=np.where((data[variable]<Min)|(data[variable]>Max),Mean,data[variable])
    elif replace_by == 'median':
        data[variable]=np.where((data[variable]<Min)|(data[variable]>Max),Median,data[variable])
    elif replace_by == 'blank':
        data[variable]=np.where((data[variable]<Min)|(data[variable]>Max),np.nan,data[variable])
    SS = ((data[variable] - (data[variable].mean()))**2)
    MSE = (SS.sum())/len(data[variable])
    RMSE_After = math.sqrt(MSE)
    print('RMSE before is',RMSE_Before,'\nRMSE after is',RMSE_After)
    return data[variable]



## Root mean squared error (RMSE)
def RMSE(data,variable):
    SS = ((data[variable] - (data[variable].mean()))**2)
    MSE = (SS.sum())/len(data[variable])
    RMSE = math.sqrt(MSE)
    return RMSE

#--------------------------------------------------------------------------------------------#
# Categorization of quantitative data - only continuous variables 
## Categorization by quartiles - encoded (1 to 4 as numericals as 2>1 and so on)
def cat_quartile_encode(data,variable):
    Q1 = data[variable].quantile(0.25).round(-1)
    Q2 = data[variable].quantile(0.50).round(-1)
    Q3 = data[variable].quantile(0.75).round(-1)
    conditions = [(data[variable] <=Q1),(data[variable] >Q1) & (data[variable] <=Q2),
                  (data[variable] >Q2) & (data[variable] <=Q3),(data[variable] >Q3)]
    categories = [1,2,3,4]
    data[variable+' Category Encode'] = np.select(conditions, categories)
    data[variable+' Category Encode'] = np.where(data[variable].isnull(),'',data[variable+' Category Encode'])
    data[variable+' Category Encode'] = pd.to_numeric(data[variable+' Category Encode'],errors='coerce')
    
#--------------------------------------------------------------------------------------------# 
# S-curve
def scurve(data,sortby,variable):
### Sorting data  
    data.sort_values(by=sortby, ascending=False).head()
#Cumulative 
    data['Cumulative'+variable] = data[variable].cumsum()
    data.plot(sortby,'Cumulative'+variable,kind='line')

#--------------------------------------------------------------------------------------------#
# Cumulative functions
## Apply conditions where necessary
## Cumulative density function (CDF)
def CDF(data,variable):
    freq= data[variable].value_counts()
    freq= freq.reset_index()
    freq.rename(columns={"index":variable,variable:"Frequency"},inplace=True)
    freq.sort_values(by=variable, ascending=True, inplace=True)
    # Probability density function (PDF)
    freq['PDF'] = freq['Frequency'] / sum(freq['Frequency'])  
    # Cumulative density function (CDF)
    freq['CDF'] = freq['PDF'].cumsum()
    freq.plot(x = variable, y = ['CDF'], grid = True)
    return freq

#--------------------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------------------#
# Univariate data - Categorical variables 

##Frequency tabel for categorical data 

def freq(data,variable):
    freq= data[variable].value_counts()
    freq= freq.reset_index()
    freq.rename(columns={"index":variable,variable:"Frequency"},inplace=True)
    freq.sort_values(by=variable, ascending=True, inplace=True)
    # Probability density function (PDF)
    freq['PDF (%)'] = (freq['Frequency'] / sum(freq['Frequency']))*100
    return freq


## Based on the problem statement, group by 2 or 3 categorical variables for further analysis
#--------------------------------------------------------------------------------------------#    
# Bivariate data - Quantitative variables  

# Correlation between variables (Pearson's R)

## Linear correlations  
def correlation_1(data,variable):
    cr = data[variable].corr().unstack().reset_index()
    cr.columns = ['X Variable','Y Variable','Correlation Coefficient - R']
### Correlation coefficient - R category 
    conditions = [
        (cr['Correlation Coefficient - R'] >=  0.7),
        (cr['Correlation Coefficient - R'] <=  -0.7),
        (cr['Correlation Coefficient - R'] >  -0.7) & (cr['Correlation Coefficient - R'] < 0.7)]
    choices = ['Positive Linear Correlation', 'Negative Linear Correlation', 'No Linear Correlation']
    cr['Correlation Coefficient - R Category'] = np.select(conditions, choices)
### Dropping rows with same X & Y variable
    cr.drop(cr.loc[cr['X Variable']==cr['Y Variable']].index, inplace=True)
### Dropping rows with blank values
    cr.dropna(axis=0, subset=['Correlation Coefficient - R'], inplace=True)
    return cr



## Scatter plot
### Identify the type (liner/parabolic/circular/others), direction (positive/negative)
### and strength (density of points scattered along any line drawn).Lesser the density,higher the liner correlation 
def scatterplot(data,variable): 
    sns.set()
    sns.pairplot(data[variable], size = 5)
    plt.show();
    
#--------------------------------------------------------------------------------------------#
## Correlataion heat map 
def correlation_heatmap(data,variable):
    hm = data[variable].corr()
    ax = sns.heatmap(
        hm, 
        vmin=-1, vmax=1, center=0,
        cmap=sns.diverging_palette(20, 220, n=200),
        square=True
    )
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=45,
        horizontalalignment='right'
    );
#--------------------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------------------#
# Bivariate data - Categorical variables 

# Contingecy tables
# Contingency table across 2 variables 
def contin2(data,yvariable,xvariable):
    data_contin = pd.crosstab(data[yvariable],data[xvariable],margins = True)
    return data_contin 


# Contingency table % across 2 variables
## Normalize = If ‘all’, normalize over all values. If ‘index’, normalize over each row. If ‘columns’, normalize over each column
def contin2_per(data,yvariable,xvariable,normalize_type):
    data_contin_per = pd.crosstab(data[yvariable],data[xvariable],margins = True,normalize=normalize_type)
    return data_contin_per 


# Contingency table across 3 variables 
def contin3(data,yvariable,xvariable,zvariable):
    data_contin = pd.crosstab([data[yvariable], data[zvariable]],data[xvariable], margins = True)  
    return data_contin 


# Contingency table % across 3 variables
## Normalize = If ‘all’, normalize over all values. If ‘index’, normalize over each row. If ‘columns’, normalize over each column
def contin3_per(data,yvariable,xvariable,zvariable,normalize_type):
    data_contin_per = pd.crosstab([data[yvariable], data[zvariable]],data[xvariable], margins = True,normalize=normalize_type)  
    return data_contin_per 
#--------------------------------------------------------------------------------------------#    
# Chi-square test (at 95% confidence level)
def chi_square(data_cat,var1,var2):
  # Creating the contingency table  
  crosstab =np.array(pd.crosstab(data_cat[var1],data_cat[var2], rownames=None, colnames=None))
  # Calculating the chi-square value
  stat = chi2_contingency(crosstab)[0]
  print('Chi Square Value = ',stat)
  # Calculating the critical chi-square value
  ## Degrees of freedom
  k = (len(crosstab)-1)*(crosstab.shape[1]-1)
  ## p-value at 95% confidence level
  p = 0.05 
  ## Critical chi-square value
  critical_chi = chi2.isf(df=k, q=p)
  print('Critical Chi Square Value = ',critical_chi)
  # Can the null hypothesis be rejected?
  if stat > critical_chi:
     print("Reject null hypothesis - There is an unequal distribution among categories")
  elif stat <= critical_chi:
     print("Can not reject null hypothesis - There is an equal distribution among categories")
  # Cramers V
  ## Counting the total number of records 
  obs = np.sum(crosstab)
  ## Taking the minimum value between the no. of columns and rows
  mini = min(crosstab.shape)-1 
  # Calculating Cramers V 
  cramers_V_square = stat/(obs*mini)
  cramers_V = math.sqrt(cramers_V_square)
  print('Cramers V = ',cramers_V)
  
#--------------------------------------------------------------------------------------------#    
# Cramers V - for all the categorical variables 

## Data frame preparation - lable encoding
def label_encode(data_cat):
    label = preprocessing.LabelEncoder()
    data_encoded = pd.DataFrame() 
    for i in data_cat.columns :
      data_encoded[i]=label.fit_transform(data_cat[i])
    return data_encoded

## Building of the Cramer's V function
def cramers_V(var1,var2) :
  crosstab =np.array(pd.crosstab(var1,var2, rownames=None, colnames=None))
  stat = chi2_contingency(crosstab)[0] 
  obs = np.sum(crosstab)
  mini = min(crosstab.shape)-1
  cramers_V_square = stat/(obs*mini)
  cramers_V_value = math.sqrt(cramers_V_square)
  return (cramers_V_value)


## Generating the data frame-main
def cramers_df (cramers):
    cramers_df = cramers.unstack().reset_index()
    cramers_df.columns = ['X Variable','Y Variable','Cramers V']
    conditions = [
            (cramers_df['Cramers V'] ==  0),
            (cramers_df['Cramers V'] >  0) & (cramers_df['Cramers V'] <= 0.25),
            (cramers_df['Cramers V'] >  0.25) & (cramers_df['Cramers V'] <= 0.75),
            (cramers_df['Cramers V'] >  0.75) & (cramers_df['Cramers V'] < 1),
            (cramers_df['Cramers V'] ==  1)]
    choices = ['No Correlation', 'Weak Correlation', 'Moderate Correlation','Strong Correlation','Perfect Correlation']
    cramers_df['Cramers V Category'] = np.select(conditions, choices)
    ### Dropping rows with same X & Y variable
    cramers_df.drop(cramers_df.loc[cramers_df['X Variable']==cramers_df['Y Variable']].index, inplace=True)
    return cramers_df

#--------------------------------------------------------------------------------------------#    
# Theils U (Uncertainity Co-efficient)

## Defining the function for conditional entropy 
def conditional_entropy(x,y):
    # entropy of x given y
    y_counter = Counter(y)
    xy_counter = Counter(list(zip(x,y)))
    total_occurrences = sum(y_counter.values())
    entropy = 0
    for xy in xy_counter.keys():
        p_xy = xy_counter[xy] / total_occurrences
        p_y = y_counter[xy[1]] / total_occurrences
        entropy += p_xy * math.log2(p_y/p_xy)
    return entropy
## Calculating conditional entropy, full group entropy and theils u
def theil_u(x,y):
    s_xy = conditional_entropy(x,y)
    x_counter = Counter(x)
    total_occurrences = sum(x_counter.values())
    p_x = list(map(lambda n: n/total_occurrences, x_counter.values()))
    s_x = ss.entropy(p_x, base=2)
    if s_x == 0:
        return 1
    else:
        return (s_x - s_xy) / s_x

## Generating the data frame
def theilu_df1 (data_cat,xvariable):
    theilu = pd.DataFrame(index=[xvariable],columns=data_cat.columns)
    columns = data_cat.columns
    for j in range(0,len(columns)):
        u = theil_u(data_cat[xvariable].tolist(),data_cat[columns[j]].tolist())
        theilu.loc[:,columns[j]] = u
    theilu.fillna(value=np.nan,inplace=True)
    return theilu

## Generating the data frame-main
def theilu_df2 (theilu_df1):
    theilu_df2 = theilu_df1.unstack().reset_index()
    theilu_df2.columns = ['Y Variable','X Variable','Theils U']
    conditions = [
            (theilu_df2['Theils U'] ==  0),
            (theilu_df2['Theils U'] >  0) & (theilu_df2['Theils U'] <= 0.25),
            (theilu_df2['Theils U'] >  0.25) & (theilu_df2['Theils U'] <= 0.75),
            (theilu_df2['Theils U'] >  0.75) & (theilu_df2['Theils U']< 1),
            (theilu_df2['Theils U'] ==  1)]
    choices = ['No Correlation', 'Weak Correlation', 'Moderate Correlation','Strong Correlation','Perfect Correlation']
    theilu_df2['Theils U Category'] = np.select(conditions, choices)
    ### Dropping rows with same X & Y variable
    theilu_df2.drop(theilu_df2.loc[theilu_df2['X Variable']==theilu_df2['Y Variable']].index, inplace=True)
    theilu_df2=theilu_df2[['X Variable', 'Y Variable', 'Theils U', 'Theils U Category']]
    return theilu_df2


#--------------------------------------------------------------------------------------------#
#One Way ANOVA by Using Ordinary Least Squares (OLS) Method From Stats Models 

# Columns have to be in the format 'Variable~Group'
def anova(data, variable, group):
    combo = variable+'~'+group
    mod = ols(combo,data=data).fit()              
    anova_table = sm.stats.anova_lm(mod, typ=2)

#Comparing the F value with the critical F value 
##Transforming the ANOVA table 
    anova_table = anova_table.reset_index()
    anova_table.rename(columns={"index":"Description"},inplace=True)
##F Value
    Fvalue = anova_table.iloc[0]['F']
    print('F Value = ',Fvalue)
#Degrees of freedom
## No. of groups
    k = len(pd.unique(data[group]))
## Total no. of records
    n = len(data.values) 
## Df Between
    Df_Between = k-1
## Df Within
    Df_Within = n-k
#Critical F Value
    Confidence_Interval=0.05
    CriticalF = ss.f.ppf(q=1-Confidence_Interval, dfn=Df_Between, dfd=Df_Within)
    print('Critical F Value = ',CriticalF)

#Can the null hypothesis be rejected?
    if Fvalue > CriticalF:
        print("Reject null hypothesis - The means are not from a similar sample set")
    elif Fvalue <= CriticalF:
        print("Can not reject null hypothesis - The means are from a similar sample set")

#Pair-wise comparison using Tukey's method (if the null hypothesis is rejected)

##Set up the dataframe for comparison (creates a specialised object)
    MultiComp = MultiComparison(data[variable],data[group])

##Show all pair-wise comparisons:
    print(MultiComp.tukeyhsd().summary())

#--------------------------------------------------------------------------------------------#    
## Pareto 80-20 analysis 
### Preparing data frame and cumulative values  
def pareto_analysis(data, group_by, column):
    df = data.groupby(group_by)[column].sum().reset_index()
    df = df.sort_values(by=column,ascending=False)
    df["Cumulative%"] = df[column].cumsum()/df[column].sum()*100
    print(df)
### Generating plot 
    fig, ax = plt.subplots(figsize=(20,5))
    ax.bar(df[group_by], df[column], color="C0")
    ax2 = ax.twinx()
    ax2.plot(df[group_by], df["Cumulative%"], color="C1", marker="D", ms=7)
    ax2.yaxis.set_major_formatter(PercentFormatter())
    ax.tick_params(axis="y", colors="C0")
    ax2.tick_params(axis="y", colors="C1")
    ax2.set_ylim([0,110])
    plt.show()

#--------------------------------------------------------------------------------------------#
# Categorization of quantitative data
## Categorization by quartiles
def cat_quartile(data,variable):
    Q1 = data[variable].quantile(0.25).round(-1)
    Q2 = data[variable].quantile(0.50).round(-1)
    Q3 = data[variable].quantile(0.75).round(-1)
    conditions = [(data[variable] <=Q1),(data[variable] >Q1) & (data[variable] <=Q2),
                  (data[variable] >Q2) & (data[variable] <=Q3),(data[variable] >Q3)]
    categories = ['Quarter-1','Quarter-2','Quarter-3','Quarter-4']
    data[variable+' Category'] = np.select(conditions, categories)
    data[variable+' Category'] = np.where(data[variable].isnull(),'',data[variable+' Category'])

## Categorization by median (Q2)
def cat_quartiles(data,variable): 
    Q2 = data[variable].quantile(0.50).round(-1)
    conditions = [(data[variable] <=Q2),(data[variable] >Q2)]
    categories = ['Below Median','Above Median']
    data[variable+' Category'] = np.select(conditions, categories)
    data[variable+' Category'] = np.where(data[variable].isnull(),'',data[variable+' Category'])

## Categorization by defined limits
def cat_custom(data,variable):
    conditions = [
        (data[variable] >= 0.12) & (data[variable] < 10),
        (data[variable] >= 0.08) & (data[variable] < 0.12),
        (data[variable] >= 0) & (data[variable] < 0.08),
        (data[variable] >= -0.08) & (data[variable] < 0),
        (data[variable] >= -10) & (data[variable] < -0.08)]
    groups = ['A: >12%', 'B:8%-12%', 'C:0%-8%', 'D:-8%-0%', 'E:>-8%']
    data[variable+' Category'] = np.select(conditions, groups)
    data[variable+' Category'] = np.where(data[variable].isnull(),'',data[variable+' Category'])
#--------------------------------------------------------------------------------------------#
