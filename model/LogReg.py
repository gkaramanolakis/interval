"""
Anonymized INTERVAL Code
"""

import argparse
import json
import logging
import os
import random
import joblib
import numpy as np
import multiprocessing as mp
from functools import partial
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import scipy

class LogRegTrainer:
    # Trainer Class
    # has to implement: __init__, train, evaluate, save, load
    def __init__(self, args, logger=None):
        self.args = args
        self.logger = logger
        self.clf = LogisticRegression(random_state=args.seed)
        self.gridsearch = True if not args.no_gridsearch else False

    def preprocess(self):
        pass

    def train(self, X_train, y_train, X_valid, y_valid, preprocessed_X_train=None, preprocessed_X_valid=None,
              sample_weights=None):
        if self.gridsearch:
            # find best parameters via grid-search
            best_params = self.tune_hyperparams(X_train, y_train, X_valid, y_valid,
                                                preprocessed_X_train=None, preprocessed_X_valid=None,
                                                sample_weights=sample_weights)
            self.clf = LogisticRegression(random_state=self.args.seed, **best_params)
            self.args.best_student_params = best_params

        self.clf.fit(X_train, y_train, sample_weight=sample_weights)
        return

    def train_pseudo(self, X_pseudo, y_pseudo, X_train, y_train, X_valid, y_valid,
              preprocessed_X_pseudo=None, preprocessed_X_train=None, preprocessed_X_valid=None,
                     weak_vs_clean_weight=1):
        if weak_vs_clean_weight == 1:
            sample_weights = None

        if y_train is not None and len(y_train) > 0:
            # cross-validation using the provided train and validation tests
            if len(y_pseudo) == 0:
                # in the case where no pseudo data are provided, this reduces to training on labeled-only data.
                X, y = X_train, y_train
            else:
                # cross-validation using the provided train and validation tests
                if scipy.sparse.issparse(X_train):
                    X = scipy.sparse.vstack((X_pseudo, X_train))
                else:
                    X = np.vstack((X_pseudo, X_train))
                y = np.array(list(y_pseudo) + list(y_train))
                if weak_vs_clean_weight != 1:
                    self.logger.info("CLEAN DATA WEIGHT = 1.0")
                    self.logger.info(f"WEAK DATA WEIGHT = {weak_vs_clean_weight}")
                    sample_weights = np.array([weak_vs_clean_weight] * len(y_pseudo) + [1.0] * len(y_train))
        else:
            X, y = X_pseudo, y_pseudo

        if self.gridsearch:
            # find best parameters via grid-search
            best_params = self.tune_hyperparams(X, y, X_valid, y_valid,
                                                preprocessed_X_train=None, preprocessed_X_valid=None,
                                                sample_weights=sample_weights)
            self.clf = LogisticRegression(random_state=self.args.seed, **best_params)
            self.args.best_student_params = best_params
        self.clf.fit(X, y, sample_weight=sample_weights)
        return

    def tune_hyperparams(self, X_train, y_train, X_valid, y_valid,
              preprocessed_X_train=None, preprocessed_X_valid=None, sample_weights=None):

        # Create custom cross-validation splits
        if y_valid is not None and len(y_valid) > 0:
            # cross-validation using the provided train and validation tests
            if scipy.sparse.issparse(X_train):
                X = scipy.sparse.vstack((X_train, X_valid))
            else:
                X = np.vstack((X_train, X_valid))
            y = np.array(list(y_train) + list(y_valid))
            if sample_weights is not None:
                # assign maximum weight to validation instances
                valid_weight = max(sample_weights)
                sample_weights = list(sample_weights) + [valid_weight] * len(y_valid)
                assert len(sample_weights) == len(y)
            all_inds = np.arange(len(y))
            train_inds = all_inds[:len(y_train)]
            dev_inds = all_inds[-len(y_valid):]
            cv_splits = [(train_inds, dev_inds)]
            refit = False

        else:
            # default cross-validation by sklearn
            X = X_train
            y = y_train
            cv_splits=None
            refit = True

        param_grid = [{
            'C': np.hstack((np.array([1]), np.logspace(-4, 4, 10))),
            'penalty': ['l2'], #  # penalty='elasticnet', 'l1'
        }]
        self.logger.info("tuning hyperparameters. param_grid:\n\t{}".format(param_grid))

        # Note: Cannot use "f1" as a scoring metric because it returns nan in multi-class classification
        gridsearch = GridSearchCV(self.clf, param_grid, scoring="f1_macro", cv=cv_splits, refit=refit)
        gridsearch.fit(X, y, sample_weight=sample_weights)
        best_params = gridsearch.best_params_
        best_score = gridsearch.best_score_
        self.logger.info("best params: {} (score={})".format(best_params, best_score))
        if np.isnan(best_score):
            raise(BaseException('GridSearch failed'))
        return best_params

    def predict(self, features):
        preds = self.clf.predict(features)  # , sample_weight=sample_weights)
        soft_proba = self.clf.predict_proba(features)
        res = {
            'preds': preds,
            'proba': soft_proba,
            'features': features
        }
        return res

    def save(self, savefolder):
        self.logger.info("saving student at {}".format(savefolder))
        joblib.dump(self.clf, os.path.join(savefolder, 'logreg.pkl'))

    def load(self, savefolder):
        self.logger.info("loading student from {}".format(savefolder))
        self.clf = joblib.load(os.path.join(savefolder, 'logreg.pkl'))
