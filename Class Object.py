# -*- coding: utf-8 -*-
"""
Created on Sat Jul 25 21:04:10 2020

@author: DELL
"""

from classcode import Standards

df = pd.read_excel('C:/Users/DELL/Documents/Ultra Insights/OCEAN SPARKLE/sample.xlsx')
object1 = Standards()
object1.describe(df)
object1.categorize_by_quantiles(df,'hi')
object1.eighty_twenty_column(df, threshold = 80)
object1.missing_values_table(df)
#object1.percent_null(df,threshold = 50, drop = False)
object1.impute_null(df,col = 'hi', impute = 'drop')
object1.outlier(df,col = 'hi',iqrfactor = 1.5, replace_by = 'mean')
object1.varianceSelection(df,0.99)




