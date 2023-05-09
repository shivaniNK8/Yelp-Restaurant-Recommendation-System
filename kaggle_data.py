# The yelp data files were gigantic so we decided to subset the data for few states and users with more than 5 review counts
# Filtered the data for businesses which falls under category 'Restaurant'
# Below is the script that was run in Kaggle to create the required csv file

import pandas as pd
import numpy as np
import json

#Reading the json files
review_json_path = '/kaggle/input/yelp-dataset/yelp_academic_dataset_review.json'
business_json_path = '/kaggle/input/yelp-dataset/yelp_academic_dataset_business.json'
df_b = pd.read_json(business_json_path, lines=True)

user_json_path = '/kaggle/input/yelp-dataset/yelp_academic_dataset_user.json'
df_user = pd.read_json(user_json_path, lines=True)

# Filtering the data for four states
df_states=df_b[df_b['state'].isin(['IL','CA','NJ','AZ'])]

#Only keep the businesses that are still open in the dataset
# 1 = open, 0 = closed
df_b = df_b[df_b['is_open']==1]
drop_columns = ['hours','is_open']
df_b = df_b.drop(drop_columns, axis=1)

df_explode = df_states.assign(categories = df_states.categories
                         .str.split(', ')).explode('categories')

#Only keeping the business which falls under restaurant category
df_rest=df_explode[(df_explode['categories']=='Restaurants') ]

#Loading the review data file in chunks
size = 4000000
review = pd.read_json(review_json_path, lines=True,
                      dtype={'review_id':str,'user_id':str,
                             'business_id':str,'stars':int,
                             'date':str,'text':str,'useful':int,
                             'funny':int,'cool':int},
                      chunksize=size)

# There are multiple chunks to be read
chunk_list = []
for chunk_review in review:
    # Drop columns that aren't needed
    chunk_review = chunk_review.drop(['review_id','useful','funny','cool'], axis=1)
    # Renaming column name to avoid conflict with business overall star rating
    chunk_review = chunk_review.rename(columns={'stars': 'review_stars'})
    # Inner merge with edited business file so only reviews related to the business remain
    chunk_merged = pd.merge(df_rest, chunk_review, on='business_id', how='inner')
    # Show feedback on progress
    print(f"{chunk_merged.shape[0]} out of {size:,} related reviews")
    chunk_list.append(chunk_merged)
# After trimming down the review file, concatenate all relevant data back to one dataframe
df = pd.concat(chunk_list, ignore_index=True, join='outer', axis=0)

#writing the csv file for review dataset
csv_name = "yelp_reviews_rest.csv"
df.to_csv(csv_name, index=False)

#Working to subset the user data json file for users with mroe than 5 reviews
valid_user=df_user[df_user['review_count']>5]
valid_user_data=valid_user[['user_id', 'name', 'review_count', 'yelping_since', 'useful', 'funny', 'cool']]

#writing the csv file for user dataset
csv_name = "yelp_user_subset.csv"
valid_user_data.to_csv(csv_name, index=False)