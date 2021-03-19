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


# reads friend's movie watch sheet into the dataframe
# friends were instructed to use a 'x' or 'X' to indicate they watched a movie
movie_ratings = pd.read_csv('Top-1000-Movies-List-032021.csv')
movie_ratings = movie_ratings.dropna(axis = 0, how = 'all')

# sort by the imbd code
movie_ratings['ttcode'] = movie_ratings['URL'].str.replace('https://www.imdb.com/title/','')
movie_ratings['ttcode'] = movie_ratings['ttcode'].str.replace('/','')
movie_ratings = movie_ratings.sort_values('ttcode')
movie_ratings = movie_ratings[:1000]  # keep the first 1000 rows


# load in movie details obtained from the omdb api
movie_details  = pd.read_pickle("./movie_details.pkl")
movie_details = movie_details.sort_values(by=['imdbID'])
movie_details = movie_details[:1000]  # keep the first 1000 rows

# since the movie title varies between dataframes 
# e.g. english/alt language, accents, etc, lets sort by the imdb ID
movie_details = movie_details.set_index('imdbID')
movie_ratings = movie_ratings.set_index('ttcode')


# friend names to consider
friend_idx = range(3,11)
print('Assessing Population Demographics for ', movie_ratings.columns[friend_idx].tolist(), '\n')
print('Movie Details: ', movie_details.columns.tolist(), '\n')


#%%  Movie Totals 

num_indvl_ratings = dict(1000-movie_ratings.iloc[:,friend_idx].isna().sum())

# Show a total sum of the movies that people have seen
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7, 5), dpi=300)
ax.bar(num_indvl_ratings.keys(), num_indvl_ratings.values(), edgecolor=[0.3, 0.3, 0.3], width=0.2, align='center')
plt.title('Number of Top 1000 Movies Watched', fontsize = 16)
plt.savefig('barplot_movies_watched_by_friends.png')
plt.show()

#%% Movie Data Topics

# Create an array that contains the different rating types
genre_list = movie_ratings["Genres"].str.split(', ', expand=True).stack().unique()

# Create an array of all actors
actor_list = []
for actor in movie_details["Actors"].str.split(', ', expand=True).stack().unique():
        if re.search(' ', actor): # Let's only look at actors with a first and last name to prevent errors
            actor_list.append(actor)
            

#%% Compile Movie Data

# Gather the data topics for each friend   
genre_data = []; genre_total = []
actor_data = []; actor_total = []
decade_list = [] 

for friend_name in movie_ratings.columns[friend_idx]:
    
    movies_watched = movie_ratings[friend_name].str.match('X', flags=re.IGNORECASE) 
    movies_watched = movies_watched.fillna(False)
 
    decade_list.append(movie_ratings["Year"][movies_watched].tolist());
    
    gr_list = []
    for genre_type in genre_list:
        specific_genre = movie_ratings["Genres"].str.contains(genre_type)
        ismatch = numpy.logical_and(movies_watched, specific_genre)
        gr_list.append(sum(ismatch))
        if friend_name == movie_ratings.columns[friend_idx[0]]:
            genre_total.append(sum(specific_genre))
    genre_data.append(gr_list)
    
    ac_list = []
    for actor_name in actor_list:
        specific_actor = movie_details["Actors"].str.contains(actor_name)
        # suppose we are interested in movies from the USA specifically
        specific_country = movie_details["Country"].str.contains('USA')
        ismatch = numpy.logical_and(movies_watched, specific_actor)
        ismatch = numpy.logical_and(ismatch, specific_country)
        ac_list.append(sum(ismatch))
        if friend_name == movie_ratings.columns[friend_idx[0]]:
            actor_total.append(sum(numpy.logical_and(specific_actor, specific_country)))        
    actor_data.append(ac_list)

genre_data.append(genre_total)
actor_data.append(actor_total)  


# Create a dataframe with our data
genre_df = pd.DataFrame(genre_data, columns=genre_list, index=numpy.insert(movie_ratings.columns[friend_idx], len(friend_idx),'Total'))
actor_df = pd.DataFrame(actor_data, columns=actor_list, index=numpy.insert(movie_ratings.columns[friend_idx], len(friend_idx),'Total'))


#%%% Genres

# Create a copy of the genre data to create a subset of genres that are important to us
genre_df_subset = genre_df.copy('deep=True')
for genre_type in genre_df.columns[1:]:
    if(genre_df[genre_type].sum()==0):
        del genre_df_subset[genre_type]
        
# Manually remove niche genres we don't really care about
del genre_df_subset['War'];  
del genre_df_subset['Film-Noir']; 
del genre_df_subset['Western'];  
del genre_df_subset['Sport'];  
del genre_df_subset['Musical']; 
del genre_df_subset['History']; 
del genre_df_subset['Biography']; 

# Nearly every movie is a drama, so this genre isn't that important to us
del genre_df_subset['Drama'];  


# Generate a radar plot
N_angles = genre_df_subset.shape[1] # number of genres
angles = [n / float(N_angles) * 2 * pi for n in range(N_angles)]  # intialize angles
angles += angles[:1]

fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(12, 7), dpi=300)
fig.suptitle('Number Movies in Top 1000 Watched By Genre', fontsize = 16)
fig.tight_layout() 

for fr_idx in range( genre_df_subset.shape[0]-1):
    
    values = genre_df_subset.loc[genre_df_subset.index[fr_idx]].values.flatten().tolist()
    # lets get a percent of the total number 
    #values = [ values[idx] / genre_df_subset.loc[genre_df_subset.index[-1]].values.tolist()[idx] * 100 for idx in range(len(values)) ]
    values += values[:1] # repeat last value  
    
    ax = plt.subplot(2,4,fr_idx+1, polar=True)
     
    # Draw one axe per variable + add labels
    plt.xticks(angles[:-1], genre_df_subset.columns[:], color=[0.3,0.3,0.3], size=9)
    ax.tick_params(axis='x', which='major', pad=12)

    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([20,40,60], ["20","40","60"], color=[0.3,0.3,0.3], size=10);  plt.ylim(0,65)
    plt.title(genre_df_subset.index[fr_idx])
     
    # Plot data
    ax.plot(angles, values, linewidth=1, linestyle='solid')
    ax.fill(angles, values, 'b', alpha=0.1) # Fill area
    
    # Show the graph
plt.subplots_adjust( wspace=0.7, hspace=0.3)
plt.savefig('popularity_by_genre.png')
plt.show()



#%% Actor/Actress Preferences

# Create a dataframe with our data
actor_df.head()
actors_sorted = actor_df.sort_values(by = 'Total', axis = 1, ascending = False) 
actors_sorted.head(10)

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(20, 6), dpi=300,)
# Lets look at the top 50 Actors and Actresses 
actors_sorted_small = actors_sorted.iloc[:, 0:50]

plt.pcolor(actors_sorted_small)
plt.yticks(numpy.arange(0.5, len(actors_sorted_small.index), 1), actors_sorted_small.index)
plt.xticks(numpy.arange(0.5, len(actors_sorted_small.columns), 1), actors_sorted_small.columns, rotation=90)
plt.colorbar()
plt.xlabel('Actor')
plt.ylabel('Friend')
plt.title('Frequency of US Movies Starring Top 50 Actor/Actresses', fontsize=16)
plt.savefig('popularity_by_actor.png')
plt.show()



#%% Lets look at the popularity of  movie decades between 1900 and 2015

bins = numpy.linspace(1900,2015, 24)
str_ticks = [ str(round(bin)) for bin in bins if bin%20==0]
ticks = [ round(bin) for bin in bins if bin%20==0]

fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(16, 8), dpi=300)
fig.suptitle('Movie decade preference', fontsize = 16)
fig.tight_layout()
for fr_idx in range(len(movie_ratings.columns[friend_idx])):
    ax = plt.subplot(2,4,fr_idx+1)
    plt.hist(decade_list[fr_idx], bins, edgecolor=[0.3, 0.3, 0.3], density=True, stacked=True)
    plt.title(movie_ratings.columns[friend_idx][fr_idx])
    plt.xticks( ticks, str_ticks, color="grey", size=10);  plt.xlim(1900, 2020)
    plt.ylim(0, 0.08)
 
plt.subplots_adjust( wspace=0.3, hspace=0.5)
plt.savefig('popularity_by_decade.png')
plt.show()




