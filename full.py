import sklearn as skl
from sklearn import pipeline
from sklearn import ensemble
from sklearn import feature_extraction
from sklearn import linear_model
import pandas as pd
import dill
import numbers

class ETWrapper(skl.base.BaseEstimator, skl.base.TransformerMixin):
    def __init__(self, est):
        self.estimator = est
    
    def fit(self, X, y=None):
        # fit the transformation
        return self
    
    def transform(self, X):
        import pandas as pd
        return pd.DataFrame(self.estimator.predict(X)) # transformation


ll_model =  dill.load(open('ll_model.pkl'))
cat_model =  dill.load(open('cat_model.pkl'))
attr_model =  dill.load(open('attr_model.pkl'))

transformers = [
    ('ll', ETWrapper(ll_model)),
    ('cat', ETWrapper(cat_model)),
    ('attr', ETWrapper(attr_model))]

union = pipeline.FeatureUnion(transformers)
LR = linear_model.LinearRegression()

pipe = pipeline.Pipeline([
	('union', union),
	('LR', LR)])

ratings = pd.read_json(open('yelp_train_academic_dataset_business.formatted.json'))

pipe.fit(ratings,ratings['stars'])
with open('full_model.pkl','wb') as f:
	dill.dump(pipe,f)