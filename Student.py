"""
Anonymized INTERVAL Code
"""

import os
from model import LogRegTrainer, wrenchTrainer

preprocessed_dataset_list = ['trec', 'youtube', 'sms', 'census', 'mitr']
supported_trainers = {
    'logreg': LogRegTrainer,
    'wrench': wrenchTrainer, 
}


class TrivialCLF:
    """ A classifier that predicts a single class"""

    def __init__(self, class_ind=0):
        self.class_ind = class_ind

    def predict(self, x):
        return np.array([self.class_ind for xx in x])

class Student:
    def __init__(self, args, logger=None):
        self.args = args
        self.logger = logger
        self.name = args.student_name
        self.dh = args.dh # datahandler
        assert self.name in supported_trainers, "Student not supported: <{}>".format(self.name)
        self.trainer_class = supported_trainers[self.name]
        self.trainer = self.trainer_class(args=self.args, logger=self.logger)
        self.preprocess = self.trainer.preprocess

    def train(self, X_train, y_train, X_valid=None, y_valid=None, sample_weights=None):
        if len(set(y_train)) == 1 or len(set(y_train)) == 0:
            self.logger.info('ERROR: cannot train student because y_train is: {}'.format(set(y_train)))
            return None
        _ = self.trainer.train(X_train=X_train, preprocessed_X_train=None, y_train=y_train,
            X_valid=X_valid, preprocessed_X_valid=None, y_valid=y_valid, sample_weights=sample_weights)
        return 1

    def train_pseudo(self, X_pseudo, y_pseudo, X_train=None, y_train=None, X_valid=None, y_valid=None):
        if len(set(y_pseudo)) == 1 and y_train is None:
            self.logger.info('ERROR: CANNOT TRAIN THE STUDENT BECAUSE RULES PREDICT JUST A SINGLE CLASS: {}'.format(set(y_train)))
            return None
        elif len(set(y_pseudo)) == 0 and y_train is None:
            self.logger.info('ERROR: CANNOT TRAIN THE STUDENT BECAUSE ALL RULES ABSTAIN.')
            return None

        _ = self.trainer.train_pseudo(
            X_pseudo=X_pseudo, preprocessed_X_pseudo=None, y_pseudo=y_pseudo,
            X_train=X_train, preprocessed_X_train=None, y_train=y_train,
            X_valid=X_valid, preprocessed_X_valid=None, y_valid=y_valid,
            weak_vs_clean_weight = self.args.weak_vs_clean_weight,
        )
        return 1

    def predict(self, X, just_preds=False):
        res = self.trainer.predict(X)
        return res['preds'] if just_preds else res


