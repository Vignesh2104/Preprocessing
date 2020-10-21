# =============================================================================
# Data Cleaning - Unmerge cells in excel
# =============================================================================
df = opsclean(df)


#===============================================================================
#Count and datatype information
df.info()           
#Drop unwanted columns
df.drop('Column Name', axis = 1)    
#Rename column names
df.rename(columns = {'':''})       
#Remove trailing spaces from column names 
df.column = df.columns.str.strip()
#Frequency table 
df.value_counts()
#Punctuation & Space removal
df['Categorical Column'] = df['Categorical Column'].str.replace("[^\w\s]",'').str.strip()

#Date Functions
df['Day'] = df['Date'].dt.day
df['Month'] = df['Date'].dt.month
df['Year'] = df['Date'].dt.year
df['Week'] = df['Date'].dt.week
df['Day of Week'] = df['Date'].dt.weekday_name

#Conversion of object dtype column(dd/mm/yy) to datetime column 
df['Date'] = pd.to_datetime(df['Date'],dayfirst = True)



# =============================================================================
# Null Identification
# =============================================================================
df.isnull().sum()   #Total null values in each column
#Option-1
#Null values with percentage of total values
missing_values_table(df) #returning mising values of each column

#Option-2
missing_values_analysis(df)

# Get the columns with > 50% missing
df = percent_null(df,threshold = 50, drop = False)

#Null imputation
df['Commodity'] = impute_null(df,col = 'Commodity', impute = 'drop')



#==============================================================================
df.head()           #Top 5 rows
df.tail()           #Bottom 5 rows
df.shape            #No. of rows and columns
df[df.duplicated()] #Check for duplicate rows

#Drop dulicate rows
df = df.drop_duplicates()  

#==========================================================================================
#Groupby columns 
grp = df.groupby('Column').agg({'Numeric column':'sum','Catergory column':'count'})

#Writing to excel
writer = pd.ExcelWriter('Filename.xlsx', engine='xlsxwriter')  #Filename
df.to_excel(writer,sheet_name= 'Sheet Name', index=False)      #Sheetname
writer.save() 


df.describe() - df1.describe()  #Difference of 2 descriptives
