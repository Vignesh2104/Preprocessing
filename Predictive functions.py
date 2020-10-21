#####################################  PREDICTIVE FUNCTIONS ###################################

#Data transformation -> 80/20 rule
"""
Description : Here we use Pareto Principle (80/20 Rule) which states
              80 percent of consequences come from 20 percent of the causes, or an unequal relationship between inputs and outputs.
              This principle serves as a general reminder that the relationship between inputs and outputs is not balanced.
Purpose : To reduce the number of levels in each categorical variables(levels >15).
          If our dataset is widely distributed across the categorical variables with higher number of levels , 
          the value in these variables cannot much contribute to the model,
          hence we apply 80/20 rule and reduce the number of levels in the categorical variables.
Steps involved : Check for the value counts of categorical variables mentioned in the file input,
                 Replace the value of lowest 20% values with the default value 999999 and create a new bucket of level
Assumption : Variable names in the file input will only be performed 80/20 rule.
"""

def eighty_twenty_column(df, threshold = 80): 
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
df1 = eighty_twenty_column(df, threshold = 80)




# =============================================================================
# To remove features with only 1% variance in data
# =============================================================================

from sklearn.feature_selection import VarianceThreshold
def varianceSelection(data, THRESHOLD = .99):
    sel = VarianceThreshold(threshold=(THRESHOLD * (1 - THRESHOLD)))
    sel.fit_transform(data)b 
    return data[[c for (s, c) in zip(sel.get_support(), data.columns.values) if s]]

data = varianceSelection(data,0.99)


"""Check for Zero variance 
Purpose : Zero variance means there is no deviation in the data, the data is constant
          Hence we drop the variables near zero variance , since it does not contribute to the model"""

from sklearn.feature_selection import VarianceThreshold
import itertools

def remove_feat_constants(data_frame):
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

print("Calculating the zero variance and dropping the variables from the dataframe .....")
data ,dropped_columns= remove_feat_constants(data) #calling the function of remove features of constant values

#####################################  DATA OPERATIONS ###################################