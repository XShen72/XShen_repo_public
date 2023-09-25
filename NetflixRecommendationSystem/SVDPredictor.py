import pickle
from tqdm import tqdm
import numpy as np
import pandas as pd
import os
import pathlib
from surprise import Reader, Dataset
from surprise import SVD
from surprise import SVDpp
from surprise.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

class SVDPredictor:
    error_table = pd.DataFrame(columns = ["Model", "Train_RMSE", "Test_RMSE"])
    
    #Class takes in final.csv as a whole as a DataFrame
    def __init__(self, data, titles):
        self.movie = data
        self.titles = titles
        self._createAlgorithmFromData()
        
    def _createAlgorithmFromData(self):
        #check if algo and trainset/train_data files are already created
        file = pathlib.Path('svd.pickle')
        if not file.exists():
            self._reduceDataSize()
            self._splitMovie()
            self._createTrainSet()
        self._run_surprise()
        
    def recommendFor(self, customerID, count):
        preds = []
        ids = []
        for mov in self.movie.MovieID.unique().tolist():
            preds.append(self.predict(customerID, mov).est)
            ids.append(mov)
            
        movieAndRating = {}
        copyPreds = preds[:]
        for i in range(count):
            index = copyPreds.index(max(copyPreds))
            maxPred = max(copyPreds)
            mov = ids[index]
            title = movie_title.iloc[mov-1:mov]['title'][mov-1]
            movieAndRating[title] = maxPred
            copyPreds.pop(index)
        return movieAndRating
        
    def predict(self, userID, movieID):
        #use algo to predict rating. Return predicted rating
        return self.algo.predict(userID, movieID)
    
    def _splitMovie(self):
        self.movie = self.movie.iloc[:1500000]
        
    def _createTrainSet(self):
        reader = Reader(rating_scale=(1,5))
        movieInput = pd.DataFrame()
        movieInput['CustomerID'] = self.movie['CustomerID']
        movieInput['MovieID'] = self.movie['MovieID']
        movieInput['Rating'] = self.movie['Rating']

        self.train_data = Dataset.load_from_df(movieInput, reader)
        self.trainset = self.train_data.build_full_trainset()
        #write to a file
    
    def _reduceDataSize(self):
        self.movie['Date'] = self.movie['Date'].astype('category')
        self.movie['MovieID'] = self.movie['MovieID'].astype('int16')
        self.movie['CustomerID'] = self.movie['CustomerID'].astype('int32')
        self.movie['Rating'] = self.movie['Rating'].astype('int8')
    
    def _run_surprise(self):
        file = pathlib.Path('svd.pickle')
        if file.exists():
            with open('svd.pickle', 'rb') as f:
                self.algo = pickle.load(f)
        else:
            self.algo = SVD(n_factors = 5, biased=True, verbose=True)
            self.algo = self.algo.fit(self.trainset)
            with open('svd.pickle', 'wb') as f:
                pickle.dump(self.algo, f)
