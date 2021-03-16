# -*- coding: utf-8 -*-
"""
Kathryn Ozgun
Created on Sun Feb 28 17:42:54 2021

"""


#!/usr/bin/env python
# coding: utf-8

# import statements

import pandas as pd
import matplotlib.pyplot as plt


# reads excel into the dataframe
movie_ratings = pd.read_csv('Top-1000-Movies-List-2.csv')
print( movie_ratings.columns)

# Get rid of all of the rows with all nans
# axis=0 means removing rows
temp1 = movie_ratings.dropna(axis = 0, how = 'all')

num_indvl_ratings = dict(1000-temp1.iloc[:,3:11].isna().sum())


ax = plt.subplot(111)
ax.bar(num_indvl_ratings.keys(), num_indvl_ratings.values(), width=0.2, align='center')
plt.title('Number of 1000 movies they have seen')


# lets look at the columns that we have and their types
temp1.dtypes
temp1.index
temp1.columns

# lets look at the first and last rows of the data
temp1.head(4)
temp1.tail(3)

# temp1.describe() get a quick summary of our data
# temp1.T will transpose the data
# temp1.sort_values(by="B") to sort by a specific column
# temp1.sort_index(axis=1, ascending=False) to sort by an index

# to get a single column  temp1["A"] or temp1.A
# to select a slice use temp1[0:3] to get cols 0-3 inclusive















