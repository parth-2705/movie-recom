import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# pass in column names for each CSV as the column name is not given in the file and read them using pandas.
# You can check the column names from the readme file

#Reading users file:
u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
users = pd.read_csv('ml-100k/u.user', sep='|', names=u_cols,encoding='latin-1')

#Reading ratings file:
r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings = pd.read_csv('ml-100k/u.data', sep='\t', names=r_cols,encoding='latin-1')

#Reading items file:
i_cols = ['movie id', 'movie title' ,'release date','video release date', 'IMDb URL', 'unknown', 'Action', 'Adventure',
'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
items = pd.read_csv('ml-100k/u.item', sep='|', names=i_cols,
encoding='latin-1')


print(users.shape)
print(users.head())

print(ratings.shape)
print(ratings.head())

print(items.shape)
print(items.head())


r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings_train = pd.read_csv('ml-100k/ua.base', sep='\t', names=r_cols, encoding='latin-1')
ratings_test = pd.read_csv('ml-100k/ua.test', sep='\t', names=r_cols, encoding='latin-1')

print(ratings_train.shape, ratings_test.shape)

# Building collaborative filtering model from scratch

# We will recommend movies based on user-user similarity and item-item similarity. For that, first we need to calculate the number of unique users and movies.
n_users = ratings.user_id.unique().shape[0]
n_items = ratings.movie_id.unique().shape[0]


# Now, we will create a user-item matrix which can be used to calculate the similarity between users and items.
data_matrix = np.zeros((n_users, n_items))
for line in ratings.itertuples():
    data_matrix[line[1]-1, line[2]-1] = line[3]

# Now, we will calculate the similarity. We can use the pairwise_distance function from sklearn to calculate the cosine similarity.

from sklearn.metrics.pairwise import pairwise_distances 
user_similarity = pairwise_distances(data_matrix, metric='cosine')
item_similarity = pairwise_distances(data_matrix.T, metric='cosine')


# This gives us the item-item and user-user similarity in an array form.

# print(user_similarity)
# print(item_similarity)

# The next step is to make predictions based on these similarities. Let’s define a function to do just that.

def predict(ratings, similarity, type='user'):
    if type == 'user':
        mean_user_rating = ratings.mean(axis=1)
        #We use np.newaxis so that mean_user_rating has same format as ratings
        ratings_diff = (ratings - mean_user_rating[:, np.newaxis])
        pred = mean_user_rating[:, np.newaxis] + similarity.dot(ratings_diff) / np.array([np.abs(similarity).sum(axis=1)]).T
    elif type == 'item':
        pred = ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])
    return pred

# Finally, we will make predictions based on user similarity and item similarity.

user_prediction = predict(data_matrix, user_similarity, type='user')
item_prediction = predict(data_matrix, item_similarity, type='item')

print(user_prediction)


# Building a simple popularity and collaborative filtering model using Turicreate

import turicreate
train_data = turicreate.SFrame(ratings_train)
test_data = turicreate.SFrame(ratings_test)

# We have user behavior as well as attributes of the users and movies, so we can make content based as well as collaborative filtering algorithms. We will start with a simple popularity model and then build a collaborative filtering model.

# First we’ll build a model which will recommend movies based on the most popular choices, i.e., a model where all the users receive the same recommendation(s). We will use the turicreate recommender function popularity_recommender for this.

popularity_model = turicreate.popularity_recommender.create(train_data, user_id='user_id', item_id='movie_id', target='rating')

popularity_recomm = popularity_model.recommend(users=[1,2,3,4,5],k=5)
popularity_recomm.print_rows(num_rows=25)

# Note that the recommendations for all users are the same – 1536, 1201, 1189, 1122, 814. And they’re all in the same order! This confirms that all the recommended movies have an average rating of 5, i.e. all the users who watched the movie gave it a top rating. Thus our popularity system works as expected.

# After building a popularity model, we will now build a collaborative filtering model. Let’s train the item similarity model and make top 5 recommendations for the first 5 users.

#Training the model
item_sim_model = turicreate.item_similarity_recommender.create(train_data, user_id='user_id', item_id='movie_id', target='rating', similarity_type='cosine')

#Making recommendations
item_sim_recomm = item_sim_model.recommend(users=[1,2,3,4,5],k=8)
item_sim_recomm.print_rows(num_rows=40)

# Here we can see that the recommendations (movie_id) are different for each user. So personalization exists, i.e. for different users we have a different set of recommendations.

