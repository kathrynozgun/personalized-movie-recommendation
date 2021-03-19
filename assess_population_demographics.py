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
movie_ratings = pd.read_csv('Top-1000-Movies-List-032021.csv')
print( movie_ratings.columns)

# Get rid of all of the rows with all nans
# axis=0 means removing rows
core_ratings = movie_ratings.dropna(axis = 0, how = 'all')
core_ratings = core_ratings[:1001]  # keep the first 1000 rows

# friend names to consider
friend_idx = range(3,11)

#%%  Movie Totals 

num_indvl_ratings = dict(1000-core_ratings.iloc[:,friend_idx].isna().sum())

# Show a total sum of the movies that people have seen
ax = plt.subplot(111)
ax.bar(num_indvl_ratings.keys(), num_indvl_ratings.values(), width=0.2, align='center')
plt.title('Number of 1000 movies they have seen', fontsize=  16)



#%% Movie Genres

# Create a dataframe that contains the different rating types
genre_list = core_ratings["Genres"].str.split(', ', expand=True).stack().unique()

# Gather the genre data for each friend   
genre_data = []
genre_total = []
for friend_name in core_ratings.columns[friend_idx]:
    movies_watched = core_ratings[friend_name].str.match('X', flags=re.IGNORECASE)
    movies_watched = movies_watched.fillna(False)
    
    gr_list = []
    for genre_type in genre_list:
        specific_genre = core_ratings["Genres"].str.contains(genre_type)
        ismatch = numpy.logical_and(movies_watched, specific_genre)
        gr_list.append(sum(ismatch))
        if friend_name == core_ratings.columns[3]:
            genre_total.append(sum(specific_genre))
        
    genre_data.append(gr_list)
genre_data.append(genre_total)

# Create a dataframe with our data
genre_df = pd.DataFrame(genre_data, columns=genre_list, index=numpy.insert(core_ratings.columns[friend_idx], len(friend_idx),'Total'))

genre_total_df = pd.DataFrame([genre_total], columns=genre_list)

# Create a copy of the genre data without genres none of us watch
genre_df_subset = genre_df.copy('deep=True')
genre_total_subset = genre_total_df.copy('deep=True')
for genre_type in genre_df.columns[1:]:
    if(genre_df[genre_type].sum()==0):
        del genre_df_subset[genre_type]
        del genre_total_subset[genre_type]
        
# Manually remove niche genres we don't really care about
del genre_df_subset['War'];  del genre_total_subset['War']
del genre_df_subset['Film-Noir'];  del genre_total_subset['Film-Noir']
del genre_df_subset['Western'];  del genre_total_subset['Western']
del genre_df_subset['Sport'];  del genre_total_subset['Sport']
del genre_df_subset['Musical'];  del genre_total_subset['Musical']
del genre_df_subset['Drama'];  del genre_total_subset['Drama']  #basically every movie is a drama

N = genre_df_subset.shape[1] # number of genres
angles = [n / float(N) * 2 * pi for n in range(N)]  # intialize angles
angles += angles[:1]

fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(12, 7), dpi=150)
fig.suptitle('Percent of Movies in Top 1000 Watched By Genre', fontsize = 16)
fig.tight_layout() 

for fr_idx in range( genre_df_subset.shape[0]-1):
    
    values = genre_df_subset.loc[genre_df_subset.index[fr_idx]].values.flatten().tolist()
    # lets get a percent of the total number 
   # values = [ values[idx] / genre_total_subset.loc[0].values.tolist()[idx] * 100 for idx in range(len(values)) ]
    values += values[:1] # repeat last value  
    
    ax = plt.subplot(2,4,fr_idx+1, polar=True)
     
    # Draw one axe per variable + add labels
    plt.xticks(angles[:-1], genre_df_subset.columns[:], color='grey', size=9)
     
    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([10,30,50], ["10","30","50"], color="grey", size=10);  plt.ylim(0,55)
    plt.title(genre_df_subset.index[fr_idx])
     
    # Plot data
    ax.plot(angles, values, linewidth=1, linestyle='solid')
    ax.fill(angles, values, 'b', alpha=0.1) # Fill area
    
    # Show the graph
plt.subplots_adjust( wspace=0.7, hspace=0.3)
plt.show()
    

#%% Actor/Actress Preferences

movie_details_df  = pd.read_pickle("./movie_details.pkl")
movie_details_df = movie_details_df.sort_values(by=['imdbID'])

core_ratings['ttcode'] = core_ratings['URL'].str.replace('https://www.imdb.com/title/','')
core_ratings['ttcode'] = core_ratings['ttcode'].str.replace('/','')
core_ratings = core_ratings.sort_values('ttcode')

movie_details_df = movie_details_df.set_index('imdbID')
core_ratings = core_ratings.set_index('ttcode')


actor_list = []
for actor in movie_details_df["Actors"].str.split(', ', expand=True).stack().unique():
        if re.search(' ', actor):
            actor_list.append(actor)


# Gather the actor data for each friend   
actor_data = []
actor_total = []
for friend_name in core_ratings.columns[friend_idx]:
    movies_watched = core_ratings[friend_name].str.match('X', flags=re.IGNORECASE)
    movies_watched = movies_watched.fillna(False)
    
    ac_list = []
    for actor_name in actor_list:
        specific_actor = movie_details_df["Actors"].str.contains(actor_name)
        specific_country = movie_details_df["Country"].str.contains('USA')
        ismatch = numpy.logical_and(movies_watched, specific_actor)
        ismatch = numpy.logical_and(ismatch, specific_country)
        ac_list.append(sum(ismatch))
        if friend_name == core_ratings.columns[3]:
            actor_total.append(sum(numpy.logical_and(specific_actor, specific_country)))
        
    actor_data.append(ac_list)
actor_data.append(actor_total)



# Create a dataframe with our data
actor_df = pd.DataFrame(actor_data, columns=actor_list, index=numpy.insert(core_ratings.columns[friend_idx], len(friend_idx),'Total'))

actor_df.head()
actors_sorted = actor_df.sort_values(by = 'Total', axis = 1, ascending = False) 
actors_sorted.head(10)

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(20, 6), dpi=150,)
actors_sorted_small = actors_sorted.iloc[:, 0:50]

plt.pcolor(actors_sorted_small)
plt.yticks(numpy.arange(0.5, len(actors_sorted_small.index), 1), actors_sorted_small.index)
plt.xticks(numpy.arange(0.5, len(actors_sorted_small.columns), 1), actors_sorted_small.columns, rotation=90)
plt.colorbar()
plt.xlabel('Actor')
plt.ylabel('Friend')
plt.title('Frequency of US Movies Starring Actor/Actresses', fontsize=16)
plt.show()


#%% Lets look at movie decades

decade_list = [] 
actor_data = []
actor_total = []
for friend_name in core_ratings.columns[friend_idx]:
    movies_watched = core_ratings[friend_name].str.match('X', flags=re.IGNORECASE)
    movies_watched = movies_watched.fillna(False)
    
    decade_list.append(core_ratings["Year"][movies_watched].tolist());
    
    
bins = numpy.linspace(1900,2015, 24)
str_ticks = [ str(round(bin)) for bin in bins if bin%20==0]
ticks = [ round(bin) for bin in bins if bin%20==0]

fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(16, 8), dpi=150)
fig.suptitle('Movie decade preference', fontsize = 16)
fig.tight_layout()
for fr_idx in range(len(core_ratings.columns[friend_idx])):
    ax = plt.subplot(2,4,fr_idx+1)
    plt.hist(decade_list[fr_idx], bins, color='b', alpha=0.5, density=True, stacked=True)
    plt.title(core_ratings.columns[friend_idx][fr_idx])
    plt.xticks( ticks, str_ticks, color="grey", size=10);  plt.xlim(1900, 2020)
    plt.ylim(0, 0.08)
 
plt.subplots_adjust( wspace=0.3, hspace=0.5)
plt.show()




