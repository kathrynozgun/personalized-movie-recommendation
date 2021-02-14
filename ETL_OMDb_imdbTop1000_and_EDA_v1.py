#!/usr/bin/env python
# coding: utf-8



# import statements

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import sys
sys.path.insert(0,'C:\\Users\\kathr\\Desktop\\personalized-movie-recommendation\\test\\')
from get_api_key import get_API_key
from get_from_OMBd import get_from_OMBd
import json
import time



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


# Remove web address from the IMBD TTcode
temp2 = temp1.copy() 
temp2['ttcode'] = temp2['URL'].str.replace('https://www.imdb.com/title/','')
temp2['ttcode'] = temp2['ttcode'].str.replace('/','')

print(temp2.shape)

temp2.head()


# import .py files
tempkey = get_API_key()


# request OMDb database details based on IMDB code (ttcode), append json dict to list, create dataframe from list
movie_list = []

for iid in temp2['ttcode'][0:100]:
    
    abc = get_from_OMBd(tempkey,iid)
    
    # ignore Ratings key, value = another embedded dict
    abc['Ratings'] = 'embedded'
    
    movie_list.append(abc)
    
    time.sleep(0.3)
    
df = pd.DataFrame.from_dict(movie_list, orient='columns')

df.head()

# merge together the csv table with manual input from viewers
# AND the OMDb requested table with movie details

big_df = pd.merge(temp2,df,left_on='ttcode',right_on='imdbID')

print(big_df.shape)
big_df.head()









