from __future__ import absolute_import

import toolz

import pandas as pd
import dill
import numbers

import typecheck
import fellow
from .data import test_json

import os, sys
base_dir = os.path.dirname(__file__)
sys.path.insert(0,base_dir)

def pick(whitelist, dicts):
    return [toolz.keyfilter(lambda k: k in whitelist, d)
            for d in dicts]

def exclude(blacklist, dicts):
    return [toolz.keyfilter(lambda k: k not in blacklist, d)
            for d in dicts]


@fellow.batch(name="ml.city_model")
@typecheck.test_cases(record=pick({"city"}, test_json))
@typecheck.returns("number")
def city_model(record):
    city = record['city']
    ms = pd.Series.from_csv(base_dir + '/cities.csv')
    m = 3.6729137013
    if city in ms:
        return ms[city]
    else:
        return m

@fellow.batch(name="ml.lat_long_model")
@typecheck.test_cases(record=pick({"longitude", "latitude"}, test_json))
@typecheck.returns("number")
def lat_long_model(record):
    model = dill.load(open(base_dir + '/ll_model.pkl'))
    X = pd.DataFrame.from_records([record])
    return model.predict(X)[0]


@fellow.batch(name="ml.category_model")
@typecheck.test_cases(record=pick({"categories"}, test_json))
@typecheck.returns("number")
def category_model(record):
    model = dill.load(open(base_dir + '/cat_model.pkl'))
    X = pd.DataFrame.from_records([record])
    return model.predict(X)[0]


@fellow.batch(name="ml.attribute_knn_model")
@typecheck.test_cases(record=pick({"attributes"}, test_json))
@typecheck.returns("number")
def attribute_knn_model(record):
    model = dill.load(open(base_dir + '/attr_model.pkl'))
    X = pd.DataFrame.from_records([record])
    return model.predict(X)[0]


@fellow.batch(name="ml.full_model")
@typecheck.test_cases(record=exclude({"stars"}, test_json))
@typecheck.returns("number")
def full_model(record):
    model = dill.load(open(base_dir + '/full_model.pkl'))
    X = pd.DataFrame.from_records([record])
    return model.predict(X)[0]
