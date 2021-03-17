# -*- coding: utf-8 -*-
"""
Kathryn Ozgun
Created on Sun Feb 28 17:42:54 2021

"""
# spider / radar plots: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.copy.html

#!/usr/bin/env python
# coding: utf-8

# import statements

import pandas as pd
import matplotlib.pyplot as plt
import numpy
import re
from math import pi

# reads excel into the dataframe
movie_ratings = pd.read_csv('Top-1000-Movies-List-2.csv')
print( movie_ratings.columns)

# Get rid of all of the rows with all nans
# axis=0 means removing rows
core_ratings = movie_ratings.dropna(axis = 0, how = 'all')
core_ratings = core_ratings[:1001]


num_indvl_ratings = dict(1000-core_ratings.iloc[:,3:11].isna().sum())

# Show a total sum of the movies that people have seen
ax = plt.subplot(111)
ax.bar(num_indvl_ratings.keys(), num_indvl_ratings.values(), width=0.2, align='center')
plt.title('Number of 1000 movies they have seen')


# lets look at the columns that we have and their types
core_ratings.dtypes
core_ratings.index
core_ratings.columns

# lets look at the first and last rows of the data
core_ratings.head(4)
core_ratings.tail(3)

# core_ratings.describe() get a quick summary of our data
# core_ratings.T will transpose the data
# core_ratings.sort_values(by="B") to sort by a specific column
# core_ratings.sort_index(axis=1, ascending=False) to sort by an index

# to get a single column  core_ratings["A"] or core_ratings.A
# to select a slice use core_ratings[0:3] to get cols 0-3 inclusive

# Create a dataframe that contains the different rating types
genre_list = core_ratings["Genres"].str.split(', ', expand=True).stack().unique()

# Gather the genre data for each friend   
genre_data = []
genre_total = []
for friend_name in core_ratings.columns[3:11]:
    movies_watched = core_ratings[friend_name].str.match('X', flags=re.IGNORECASE)
    movies_watched = movies_watched.fillna(False)
    
    gr_list = [friend_name]
    for genre_type in genre_list:
        specific_genre = core_ratings["Genres"].str.match(genre_type)
        ismatch = numpy.logical_and(movies_watched, specific_genre)
        gr_list.append(sum(ismatch))
        if friend_name == core_ratings.columns[3]:
            genre_total.append(sum(specific_genre))
        
    genre_data.append(gr_list)

# Create a dataframe with our data
genre_df = pd.DataFrame(genre_data, columns=numpy.insert(genre_list, 0, 'Name'))

# Create a copy of the genre data without genres none of us watch
genre_df_subset = genre_df.copy('deep=True')
for genre_type in genre_df.columns[1:]:
    if(genre_df[genre_type].sum()==0):
        del genre_df_subset[genre_type]


N = genre_df_subset.shape[1]-1 # number of genres
angles = [n / float(N) * 2 * pi for n in range(N)]  # intialize angles
angles += angles[:1]
for fr_idx in range( genre_df_subset.shape[0]):
    
    values  = genre_df_subset.loc[fr_idx].drop('Name').values.flatten().tolist()
    values += values[:1] # repeat last value  
    
    ax = plt.subplot(111, polar=True)
     
    # Draw one axe per variable + add labels
    plt.xticks(angles[:-1], genre_df_subset.columns[1:], color='grey', size=8)
     
    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([10,20,30], ["10","20","30"], color="grey", size=7)
    plt.ylim(0,40)
     
    # Plot data
    ax.plot(angles, values, linewidth=1, linestyle='solid')
     
    # Fill area
    ax.fill(angles, values, 'b', alpha=0.1)
    
    # Show the graph
    plt.show()
    





