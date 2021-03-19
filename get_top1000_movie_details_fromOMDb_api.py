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
import time


# reads excel into the dataframe
movie_ratings = pd.read_csv('Top-1000-Movies-List-032021.csv')
print( movie_ratings.columns)

top1000 = movie_ratings[:1001]  # keep the first 1000 rows


# Remove web address from the IMBD TTcode
top1000['ttcode'] = top1000['URL'].str.replace('https://www.imdb.com/title/','')
top1000['ttcode'] = top1000['ttcode'].str.replace('/','')

print(top1000.shape)
top1000.head(3)


# get our API key to access OMBD data
tempkey = get_API_key()


# request OMDb database details based on IMDB code (ttcode), append json dict to list, create dataframe from list
movie_list = []

for iid in top1000['ttcode'][0:1000]:
    
    movie_details = get_from_OMBd(tempkey,iid)
    
    # ignore Ratings key, value = another embedded dict
    movie_details['Ratings'] = 'embedded'
    
    movie_list.append(movie_details)
    
    time.sleep(0.3)
    
movie_details_df = pd.DataFrame.from_dict(movie_list, orient='columns')

movie_details_df.head()


movie_details_df.to_pickle("./movie_details.pkl")






