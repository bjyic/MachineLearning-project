import sklearn as skl
from sklearn import pipeline
from sklearn import ensemble
from sklearn import feature_extraction
from sklearn import linear_model
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

class ListDictifier(skl.base.BaseEstimator, skl.base.TransformerMixin):
    def __init__(self, columns=None):
        self.columns = columns
    
    def fit(self, X, y=None):
        # fit the transformation
        return self
    
    def transform(self, X):
    	if self.columns:
    		#only apply to selected columns
    		raise "Not Implemented"
    	else:
    		return [{el:1 for el in L} for L in X]

categories = ColumnTransformer('categories')
dictify = ListDictifier()
vectorize = feature_extraction.DictVectorizer()
LR = linear_model.LinearRegression()

pipe = pipeline.Pipeline([
	('getcols',categories),
	('dictify', dictify),
	('vectorize', vectorize),
	('LR', LR)])

ratings = pd.read_json(open('yelp_train_academic_dataset_business.formatted.json'))

pipe.fit(ratings,ratings['stars'])
with open('cat_model.pkl','wb') as f:
	dill.dump(pipe,f)