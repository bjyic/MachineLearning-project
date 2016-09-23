import sklearn as skl
from sklearn import pipeline
from sklearn import ensemble
import pandas as pd
import dill

class ColumnTransformer(skl.base.BaseEstimator, skl.base.TransformerMixin):
    def __init__(self, columns=[]):
        self.columns = columns
    
    def fit(self, X, y=None):
        # fit the transformation
        return self
    
    def transform(self, X):
        return X[self.columns]# transformation


latlong = ColumnTransformer(['longitude','latitude'])

pipe = pipeline.Pipeline([
	('getcols',latlong),
	('forest',ensemble.RandomForestRegressor(max_depth=10))])

ratings = pd.read_json(open('yelp_train_academic_dataset_business.formatted.json'))

pipe.fit(ratings,ratings['stars'])
with open('ll_model.pkl','wb') as f:
	dill.dump(pipe,f)
