#!/usr/bin/env python
'''
This script is to train several undersampling algorithms
and save the trained models.
'''

from collections import Counter
from imblearn.under_sampling import CondensedNearestNeighbour
from imblearn.under_sampling import OneSidedSelection
from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import TomekLinks
from pickle import dump
from sklearn.base import TransformerMixin
from sklearn.preprocessing import OrdinalEncoder
import numpy as np
import os
import pandas as pd

# Impute missing values. Mean for numerical column and MPV for categorical column.
# A helper class to do exactly this.
class DataFrameImputer(TransformerMixin):
    def __init__(self):
        '''
        Impute missing values.
        Columns of dtype object are imputed with the most frequent value 
        in column.
        Columns of other types are imputed with mean of column.
        '''
    def fit(self, X, y=None):
        self.fill = pd.Series([X[c].value_counts().index[0]
            if X[c].dtype.name == 'category' else X[c].mean() if X[c].dtype == np.float64
            else int(X[c].median()) for c in X],
            index=X.columns)
        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)

class transaction_table:
    '''
    A class to handle the transaction tables.
    '''
    def __init__(self, data_fpn):
        '''
        Load a data file into the the class.
        '''
        # member variables
        self.cat_vars = ['ProductCD'] + [f'card{i+1}' for i in range(6)] +\
                        ['addr1', 'addr2'] + ['P_emaildomain', 'R_emaildomain'] +\
                        [f'M{i+1}' for i in range(9)]
        self.df = pd.read_csv(data_fpn)
        self.df_imputer = DataFrameImputer() # fit when preprocessing
        self.ord_enc = OrdinalEncoder() # fit when preprocessing

        # variable to store fitted undersampling transforms
        self.undersamples = dict()

        # variables to store undersampled data
        self.X_unders = dict()
        self.y_unders = dict()

        # initialize object
        self.preprocess_data()
    
    def preprocess_data(self):
        '''
        Preprocess data.
        1. Force data types.
        2. Impute missing values.
        3. Ordinally encode categorical variables.
        '''
        # force categorical data
        for col in self.cat_vars:
            self.df[col] = self.df[col].astype('category')
        # impute missing values
        self.df = self.df_imputer.fit_transform(self.df)
        # ordinally encode categorical variables
        self.df[self.cat_vars] = self.ord_enc.fit_transform(self.df[self.cat_vars])
    
    def add_undersampling_transform(self, name, transform):
        '''
        Given a (name, transform) pair, fit the undersampling transform
        with the identifier 'name'.
        '''
        self.undersamples[name] = transform
        self.X_unders[name], self.y_unders[name] = transform.fit_resample(
            self.df.drop('isFraud', axis=1), self.df.isFraud)
        print(f'Undersampler {name} training complete.')
        print(Counter(self.y_unders[name]))
    
    def save_trained_transforms(self):
        '''
        Save all trained transforms to file.
        To load a transform, use, for example,
            model = load(open('model.pkl', 'rb'))

        ref: https://machinelearningmastery.com/how-to-save-and-load-models-and-data-preparation-in-scikit-learn-for-later-use/
        '''
        # create the output folder
        out_dir = os.path.join('../trained_models', os.path.basename(__file__).rstrip('.py'))
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        
        # save preprocessing transforms
        dump(self.df_imputer, open(os.path.join(out_dir, 'df_imputer.pkl'), 'wb'))
        dump(self.ord_enc, open(os.path.join(out_dir, 'ord_enc.pkl'), 'wb'))

        # save undersampling transforms
        for key, transform in self.undersamples.items():
            dump(transform, open(os.path.join(out_dir, f'undersampler_{key}.pkl'), 'wb'))


if __name__ == '__main__':
    trans_data = transaction_table('../data/train_transaction.csv')
    print(trans_data.df.head(10))
    trans_data.add_undersampling_transform('random',
        RandomUnderSampler(sampling_strategy='majority', random_state=0))
    trans_data.add_undersampling_transform('CNN',
        CondensedNearestNeighbour(n_neighbors=1, random_state=0, n_jobs=-1))
    trans_data.add_undersampling_transform('Tomek',
        TomekLinks(n_jobs=-1))
    trans_data.add_undersampling_transform('OSS',
        OneSidedSelection(n_neighbors=1, n_seeds_S=200, random_state=0, n_jobs=-1))
    trans_data.save_trained_transforms()
    