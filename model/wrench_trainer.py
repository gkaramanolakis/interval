import torch
import sys
import os
from os.path import expanduser
import numpy as np
home = expanduser("~")


wrench_folder = os.path.join(home, "wrench")
assert os.path.exists(wrench_folder), 'You need to download wrench first (see "ext" directory)'
sys.path.append(wrench_folder)
from wrench.endmodel import MLPModel 

class wrenchTrainer:
    # Trainer Class
    # has to implement: __init__, train, evaluate, save, load
    def __init__(self, args, logger=None):
        self.args = args
        self.logger = logger
        self.device = torch.device('cuda:0')
        n_steps = args.n_steps
        batch_size = 128  # 128
        test_batch_size = 1000
        self.patience = 20
        self.evaluation_step = 100
        self.target = 'acc'
        self.logger.info('initializing MLPModel')
        self.clf = MLPModel(n_steps=n_steps, batch_size=batch_size, test_batch_size=test_batch_size)
        self.train_pseudo = self.train

        # Create dummy data with WRENCH format that can be used during training/prediction
        self.dummy_data = self.args.dh.load_modified_wrench_data(method='train')

    def preprocess(self):
        pass

    def create_dummy_wrench_data(self, features, labels=None):
        # Load a dummy wrench dataset and use it for training the student
        num_examples = features.shape[0]
        data = self.dummy_data
        data.features = features
        data.labels = labels if labels is not None else [-1] * num_examples
        data.ids = list(np.arange(num_examples))
        data.examples = [""] * num_examples
        return data

    def train(self, X_train, y_train, X_valid, y_valid,
              preprocessed_X_train=None, preprocessed_X_valid=None, sample_weights=None):
        fetures, labels = X_train, y_train
        data = self.create_dummy_wrench_data(features, labels)
        history = self.clf.fit(dataset_train=data, y_train=labels, device=self.device, metric=self.target,
                               patience=self.patience, evaluation_step=self.evaluation_step)
        self.history = history

    def predict(self, features):
        data = self.create_dummy_wrench_data(features, labels=None)
        preds = self.clf.predict(data, device=self.device)
        soft_proba = self.clf.predict_proba(data, device=self.device)
        res = {
            'preds': preds,
            'proba': soft_proba,
            'features': features
        }
        return res

    def save(self, savefolder):
        self.logger.info("saving student at {}".format(savefolder))
        joblib.dump(self.clf, os.path.join(savefolder, 'wrench_model.pkl'))

    def load(self, savefolder):
        self.logger.info("loading student from {}".format(savefolder))
        self.clf = joblib.load(os.path.join(savefolder, 'wrench_model.pkl'))