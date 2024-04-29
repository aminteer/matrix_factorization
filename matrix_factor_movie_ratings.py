#!/usr/bin/env python

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
from sklearn.model_selection import train_test_split
from scipy.sparse import coo_matrix, csr_matrix
from scipy.spatial.distance import jaccard, cosine 
from sklearn.decomposition import NMF
from sklearn.impute import SimpleImputer
from pytest import approx
import unittest
from collections import namedtuple
from sklearn.metrics import pairwise_distances

class matrix_recommender_system():
    def __init__(self,data):
        self.data=data
        self.allusers = list(self.data.users['uID'])
        self.allmovies = list(self.data.movies['mID'])
        self.genres = list(self.data.movies.columns.drop(['mID', 'title', 'year']))
        self.mid2idx = dict(zip(self.data.movies.mID,list(range(len(self.data.movies)))))
        self.uid2idx = dict(zip(self.data.users.uID,list(range(len(self.data.users)))))
        self.Mr=self.rating_matrix()
        self.Mm=None 
        self.sim=np.zeros((len(self.allmovies),len(self.allmovies)))
    
    def setUp(self):

        # Creating Sample test data
        MV_users = pd.read_csv('data/users.csv')
        MV_movies = pd.read_csv('data/movies.csv')
        train = pd.read_csv('data/train.csv')
        test = pd.read_csv('data/test.csv')
        
        Data = namedtuple('Data', ['users','movies','train','test'])
        self.data = Data(MV_users, MV_movies, train, test)
        
        np.random.seed(42)
        self.sample_train = train[:30000]
        self.sample_test = test[:30000]


        self.sample_MV_users = MV_users[(MV_users.uID.isin(self.sample_train.uID)) | (MV_users.uID.isin(self.sample_test.uID))]
        self.sample_MV_movies = MV_movies[(MV_movies.mID.isin(self.sample_train.mID)) | (MV_movies.mID.isin(self.sample_test.mID))]


        self.sample_data = Data(self.sample_MV_users, self.sample_MV_movies, self.sample_train, self.sample_test)
        
    def rating_matrix(self):
        """
        Convert the rating matrix to numpy array of shape (#allusers,#allmovies)
        """
        ind_movie = [self.mid2idx[x] for x in self.data.train.mID] 
        ind_user = [self.uid2idx[x] for x in self.data.train.uID]
        rating_train = list(self.data.train.rating)
        
        return np.array(coo_matrix((rating_train, (ind_user, ind_movie)), shape=(len(self.allusers), len(self.allmovies))).toarray())
    
    def predict_from_sim(self,uid,mid):
        """
        Predict a user rating on a movie given userID and movieID
        """
        # Predict user rating as follows:
        # 1. Get entry of user id in rating matrix
        # 2. Get entry of movie id in sim matrix
        # 3. Employ 1 and 2 to predict user rating of the movie
        # your code here
        usr_idx = self.uid2idx[uid]
        user_ratings = self.Mr[usr_idx]
        movie_idx = self.mid2idx[mid]
        sim_movie = self.sim[movie_idx]
        #need to divide by count of valid ratings to minimize bias
        pred = np.dot(user_ratings, sim_movie) / np.dot(user_ratings != 0, sim_movie)
        return pred
    
    def predict(self):
        """
        Predict ratings in the test data. Returns predicted rating in a numpy array of size (# of rows in testdata,)
        """
        # your code here
        test_preds = []
        for i in range(len(self.data.test)):
            test_preds.append(self.predict_from_sim(self.data.test.uID[i], self.data.test.mID[i]))
        return np.array(test_preds)
    
    def rmse(self,yp):
        yp[np.isnan(yp)]=3 #In case there is nan values in prediction, it will impute to 3.
        yt=np.array(self.data.test.rating)
        return np.sqrt(((yt-yp)**2).mean())
    
    def calc_matrixfactor(self):    
        """
        Calculates item-item similarity for all pairs of items using matrix factorization with sklearn's NMF
        Returns a matrix of size (#all movies, #all movies)
        """
        # Return a sim matrix by calculating item-item similarity for all pairs of items using matrix factorization
        
        # get movie ratings array
        movie_ratings_matrix = self.Mr
        # replace Nan with zerios
        ratings_imputed = np.nan_to_num(movie_ratings_matrix)   
        
        # # Replace zeros with NaN for imputation
        # ratings = np.where(movie_ratings_matrix == 0, np.nan, movie_ratings_matrix)
        # # impute missing values with the mean
        # imputer = SimpleImputer(strategy='mean')
        # ratings_imputed = imputer.fit_transform(ratings)

        # Create an NMF model with 2 components and random initialization
        nmf_model = NMF(n_components=2, init='random', random_state=0)

        # Factorize the imputed matrix
        W = nmf_model.fit_transform(ratings_imputed)  # User features
        H = nmf_model.components_                     # Movie features

        # Predict ratings
        predicted_ratings = np.dot(W, H)
        
        self.sim = predicted_ratings

if __name__ == "__main__":
    # Creating Sample test data
    MV_users = pd.read_csv('data/users.csv')
    MV_movies = pd.read_csv('data/movies.csv')
    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')
    
    Data = namedtuple('Data', ['users','movies','train','test'])
    #data_list = Data(MV_users, MV_movies, train, test)
    
    np.random.seed(42)
    sample_train = train[:30000]
    sample_test = test[:30000]


    sample_MV_users = MV_users[(MV_users.uID.isin(sample_train.uID)) | (MV_users.uID.isin(sample_test.uID))]
    sample_MV_movies = MV_movies[(MV_movies.mID.isin(sample_train.mID)) | (MV_movies.mID.isin(sample_test.mID))]


    sample_data = Data(sample_MV_users, sample_MV_movies, sample_train, sample_test)
    
    nmf = matrix_recommender_system(sample_data)
    #nmf.setUp()
    nmf.calc_matrixfactor()
    pred = nmf.predict()
    print("RMSE:", nmf.rmse(pred))