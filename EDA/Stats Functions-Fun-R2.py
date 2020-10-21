#Descriptive & exploratory analytics functions

# Univariate data - Quantitative variables

## Descriptive statistics 
descriptive = describe(data)

## Variable distribution identification 
## If discrete, probability mass function (PMF) - use histograms 
## If continuous, probability density function (PDF) - use histograms and distribution plots  
#--------------------------------------------------------------------------------------------#

## Histograms 
numerical = ['A','B']
data[numerical].hist(bins=10, figsize=(30, 20), layout=(5, 6));

## Distributions
sns.distplot(data['Revenue'], kde=True, bins=10)

### Distribution - with normal plot 
sns.distplot(data['Revenue'], fit=norm);

## Variable distribution identification 
## If discrete, probability mass function (PMF) - use histograms 
## If continuous, probability density function (PDF) - use histograms and distribution plots  
#--------------------------------------------------------------------------------------------#

# Normal distribution
## Checking the distribution against the standard normal distribution ranges  
for i in ['Revenue','Cost']:
    locals()['ND '+i] = normal_dist(data, variable = i)
#--------------------------------------------------------------------------------------------#
# Univariate outlier analysis and treatment
## Replacing outliers by mean/median/mode and blanks as blanks
## IQR Factors = 1.00/1.15/1.50
data['Revenue'] = outlier(data,variable = 'Revenue',iqrfactor = 1.5, replace_by = 'mean')

# Multivariate outlier analysis and treatment
unique = data['Product'].unique()
list_of_values = unique.tolist()
for i in list_of_values:                             
    data[i] = outlier(data[data['Product'] == i],variable = 'Revenue',iqrfactor = 1.5, replace_by = 'mean')
    data[i].fillna(0, inplace = True)                     
data['Revenue-Product'] = data['OE']+ data['OI'] + data['CH']+ data['AE']+data['AI']
data.drop(columns = list_of_values, inplace = True)

#RMSE
RMSE(data,variable = 'Revenue')
#--------------------------------------------------------------------------------------------#

# Categorization of quantitative data - only continuous variables 
## Categorization by quartiles - encoded (1 to 4 as numericals as 2>1 and so on)
cat_quartile_encode(data,variable = 'Revenue')
#--------------------------------------------------------------------------------------------#

#S-Curve 
scurve(data,sortby = 'Job Date',variable = 'Revenue')

#--------------------------------------------------------------------------------------------#
# Cumulative functions
## Apply conditions where necessary
## Cumulative density function (CDF)
CDF_Revenue = CDF(data,variable = 'Revenue')
#--------------------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------------------#

# Univariate data - Categorical variables 

##Frequency tabel for categorical data 
data.nunique(axis = 0)
Freq_Product = freq(data,variable = 'Product')
## Based on the problem statement, group by 2 or 3 categorical variables for further analysis
#--------------------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------------------#

# Bivariate data - Quantitative variables  
# Correlation between variables (Pearson's R)
## Selecting the required variables 
quant1 = ['Revenue','Cost']  
quant2 = ['Revenue Category Encode', 'Cost Category Encode'] 
# Function
cr_sales = correlation(data,variable = quant1)
#--------------------------------------------------------------------------------------------#    

# Creating data sub-sets
## Stratify the data based on categorical variables for further exploration & analysis
## For one categorical variable and one sub-set 
data1 = data[data['Product'] == 'OE']
## For one categorical variable exluding one sub-set
data2 = data[data['Product'] != 'OE'] 
## For two categorical varibales and two subsets 
data3 = data[(data['Product'] == 'OE') & (data['Branch'] == 'M+R-CHE')]
#--------------------------------------------------------------------------------------------#

## Scatter plot
### Identify the type (liner/parabolic/circular/others), direction (positive/negative)
### and strength (density of points scattered along any line drawn).Lesser the density,higher the liner correlation 
scatterplot(data,variable = quant1)

## Correlataion heat map 
correlation_heatmap(data,variable = quant1)

#--------------------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------------------#

# Bivariate data - Categorical variables  

## Box plots (default IQR factor is 1.5)
### For single variables 
box_revenue = sns.boxplot(x=data['Revenue'], orient="h", palette="Set2")
numerical_box = ['Revenue','Cost']
box = sns.boxplot(data=data[numerical_box], orient="h", palette="Set2")

### Quantitative variable across 1 categorical variable
sns.boxplot(x='Product',y='Revenue',data=data)

## Violin plots
sns.violinplot(x='Product', y='Revenue', data=data)
### Violin plot with two variables
sns.violinplot(x='Product', y='Revenue', data=data, hue='Product Category')
#--------------------------------------------------------------------------------------------# 

# Contingecy tables
## Contingency table across 2 variables 
data_contin = contin2(data,yvariable='Product',xvariable='Branch')

## Contingency table % across 2 variables
### Normalize = If ‘all’, normalize over all values. If ‘index’, normalize over each row. If ‘columns’, normalize over each column
data_contin_per = contin2_per(data,yvariable='Product',xvariable='Branch',normalize_type='all')

## Contingency table across 3 variables 
data_contin = contin3(data,yvariable='Product',zvariable='Business Type',xvariable='Branch')   

## Contingency table % across 3 variables
### Normalize = If ‘all’, normalize over all values. If ‘index’, normalize over each row. If ‘columns’, normalize over each column
data_contin = contin3_per(data,yvariable='Product',zvariable='Business Type',xvariable='Branch',normalize_type='all')   
#---------------------------------------------------------------------------------------------#

# Chi-square test (at 95% confidence level)
data_cat = data[['Product','Product Category']]   
chi_square(data_cat,var1="Product",var2="Product Category")
#---------------------------------------------------------------------------------------------#

# Cramers V - for all the categorical variables 
## Data frame preparation - lable encoding
data_encoded = label_encode(data_cat)

## Building the matrix
rows= []
for var1 in data_encoded:
  col = []
  for var2 in data_encoded : 
    cramers =cramers_V(data_encoded[var1], data_encoded[var2])
    col.append(round(cramers,2))  
  rows.append(col)

## Generating the data frame   
cramers_results = np.array(rows)
cramers = pd.DataFrame(cramers_results, columns = data_encoded.columns, index =data_encoded.columns)

## Generating the data frame-main
cramers_df (cramers)

## Heat map
mask = np.zeros_like(cramers, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
with sns.axes_style("white"):
  ax = sns.heatmap(cramers, mask=mask,vmin=0., vmax=1, square=True)
plt.show()

#--------------------------------------------------------------------------------------------#    
# Theils U (Uncertainity Co-efficient)

## Defining the function and calculating conditional entropy full group entropy & theils u
### xvariable = If we know x, what is the the probability that we can accurately predict y
theilu_df1 = theilu_df1 (data_cat,xvariable = "Golf")

## Generating the analysis data frame
theilu_main = theilu_df2 (theilu_df1)

## Heat map
plt.figure(figsize=(20,1))
sns.heatmap(theilu_df1,annot=True,fmt='.2f')
plt.show()

#--------------------------------------------------------------------------------------------#

# One Way ANOVA by Using Ordinary Least Squares (OLS) Method From Stats Models 
anova(data, variable = 'Revenue', group = 'Product')
#--------------------------------------------------------------------------------------------#    

# Pareto 80-20 analysis 
pareto_analysis(data, group_by = 'Product', column = 'Revenue')

#--------------------------------------------------------------------------------------------#
# Categorization of quantitative data

## Categorization by quartiles
cat_quartile(data,variable = 'Revenue')

## Categorization by median (Q2)
cat_quartiles(data,variable = 'Cost')

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
### Function
cat_custom(data,variable = 'Cost')
#--------------------------------------------------------------------------------------------#
