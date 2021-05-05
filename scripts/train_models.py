#!/usr/bin/env python
'''
This script is to train models to predict fraud.
'''

from undersample import transaction_table

from imblearn.under_sampling import RandomUnderSampler
from pickle import dump, load
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import multiprocessing
import os

class reduced_transaction_table(transaction_table):
    def __init__(self, data_fpn):
        super().__init__(data_fpn)
    
    def select_features(self, name, ranking):
        '''
        Given a ranking array and the name of the undersampling method,
        do the feature selection.
        '''
        cols = self.X_unders[name].columns
        sel_cols = [cols[i] for i in range(len(cols)) if ranking[i] == 1 and cols[i] not in ['TransactionID', 'TransactionDT']]
        self.X_unders[name] = self.X_unders[name][sel_cols]


if __name__ == '__main__':
    # load data table
    red_data = reduced_transaction_table('../data/train_transaction.csv')

    und_samp_name = 'random'
    red_data.add_undersampling_transform(und_samp_name,
        RandomUnderSampler(sampling_strategy='majority', random_state=0))
    
    # load selected features
    rankings = load(open('../trained_models/select_features/rankings.pkl', 'rb'))
    # it is known that 80 features gives the best accuracy
    red_data.select_features(und_samp_name, rankings[80])
    
    # reference for fitting model and getting feature importance:
    # https://machinelearningmastery.com/calculate-feature-importance-with-python/

    # define dataset
    X, y = red_data.X_unders[und_samp_name], red_data.y_unders[und_samp_name]
    
    # split train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    # define the model
    print('Model initialization...')
    model = RandomForestClassifier(n_jobs=max(int(multiprocessing.cpu_count()*0.8), 1))
    # fit the model
    print('Start fitting model...')
    model.fit(X_train, y_train)
    # get importance
    importance = model.feature_importances_
    y_pred = model.predict(X_train)
    print('Train accuracy:', accuracy_score(y_pred, y_train))

    # create the output folder
    out_dir = os.path.join('../trained_models', os.path.basename(__file__).rstrip('.py'))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    # save model
    dump(model, open(os.path.join(out_dir, 'model_rfc.pkl'), 'wb'))

    # check performance
    y_test_pred = model.predict(X_test)
    print('Test accuracy:', accuracy_score(y_test_pred, y_test))

    # randomly sample 10000 records from the table and try again
    df_test2 = red_data.df.sample(10000, random_state=45)
    cols = [col for col in red_data.df.columns if col != 'isFraud']
    sel_cols = [cols[i] for i in range(len(cols)) if rankings[80][i] == 1 and cols[i] not in ['TransactionID', 'TransactionDT']]
    X_test2 = df_test2[sel_cols]
    y_test2 = df_test2.isFraud
    y_pred2 = model.predict(X_test2)
    print('Test accuracy 2:', accuracy_score(y_pred2, y_test2))