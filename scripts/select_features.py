#!/usr/bin/env python

from undersample import transaction_table

from imblearn.under_sampling import RandomUnderSampler
from pickle import dump
from sklearn.feature_selection import RFE
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

import argparse
import numpy as np
import os

if __name__ == '__main__':
    # command line argument
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--start_nfeatures', help='Log2 of the number of features to start',
                        type=int, default=2)
    args = parser.parse_args()

    trans_data = transaction_table('../data/train_transaction.csv')
    print(trans_data.df.head(10))

    trans_data.add_undersampling_transform('random',
        RandomUnderSampler(sampling_strategy='majority', random_state=0))
    
    # run through different number of features
    start_nf = args.start_nfeatures
    nfeature_trials = [int(i) for i in np.logspace(start_nf, 8, num=9-start_nf, base=2)] +\
                      [trans_data.X_unders['random'].shape[1]]
    acc = dict()
    rankings = dict()
    for n_features_to_select in nfeature_trials:
        # rename descriptors and target
        X = trans_data.X_unders['random']
        y = trans_data.y_unders['random']

        # construct a pipeline
        rfe = RFE(estimator=DecisionTreeClassifier(), n_features_to_select=n_features_to_select)
        model = DecisionTreeClassifier()
        pipeline = Pipeline(steps=[('s', rfe), ('m', model)])
        # evaluate model
        cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
        n_scores = cross_val_score(pipeline, X, y,
                                scoring='accuracy', cv=cv, n_jobs=20, error_score='raise')
        print('Accuracy with %i features selected: %.3f (%.3f)' % (n_features_to_select, np.mean(n_scores), np.std(n_scores)))
        acc[n_features_to_select] = np.mean(n_scores)

        # fit rfe to data
        # ref: https://stackoverflow.com/questions/51412684/attributeerror-rfecv-object-has-no-attribute-ranking
        rfe_data = pipeline.named_steps['s'].fit(X, y)

        rankings[n_features_to_select] = rfe_data.ranking_
    print(acc)

    # save all results
    # create the output folder
    out_dir = os.path.join('../trained_models', os.path.basename(__file__).rstrip('.py'))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    dump(acc, open(os.path.join(out_dir, 'acc.pkl'), 'wb'))
    dump(rankings, open(os.path.join(out_dir, 'rankings.pkl'), 'wb'))
