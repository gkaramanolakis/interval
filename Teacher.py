"""
Anonymized INTERVAL Code
"""

import os
import numpy as np
import random
import pandas as pd
import joblib
from collections import defaultdict
from snorkel.utils import probs_to_preds
from sklearn.preprocessing import OneHotEncoder, LabelBinarizer
import scipy

class Teacher:
    """
    Teacher:
        (1) considers multiple weak sources (1) multiple weak (heuristic) rules, (2) Student
        (2) aggregates weak sources with an aggregation model to compute a single pseudo-label
    """

    def __init__(self, args, logger=None, name=None):
        self.name = args.teacher_name if name is None else name
        self.datapath = args.datapath
        self.args = args
        self.logger = logger
        self.seed = args.seed
        self.num_classes = args.dh.num_classes
        np.random.seed(self.seed)
        self.convert_abstain_to_random = args.convert_abstain_to_random
        self.label_model = self.get_label_model()
        self.abstain_ind = -1
        self.autorules_apply_fn = None
        self.rule_enc = None  # used for ASTRA
        self.student = None  # used for ASTRA

    def get_label_model(self):
        # print(os.path.abspath(wrench.__file__))
        if self.name == 'majority':
            from wrench.labelmodel import MajorityVoting
            return MajorityVoting()
        elif self.name == 'linear':
            from Student import Student
            return Student(self.args, self.logger)
        try:
            from wrench import labelmodel
            return getattr(labelmodel, self.name)()
        except:
            raise(BaseException("label model not available: {}".format(self.name)))

    def get_rule_encodings(self, rule_preds, student_proba=None):
        if rule_preds is None or rule_preds.shape[1] == 0:
            return None

        rule_preds = to_one_hot(rule_preds, self.rule_enc, sparse_output=False)
        if student_proba is not None:
            assert student_proba.shape[0] == rule_preds.shape[0]
            rule_preds = np.hstack((rule_preds, student_proba))
        return rule_preds

    def train(self, train_data=None, dev_data=None, u_data=None,
              train_student_proba=None, dev_student_proba=None, u_student_proba=None,
              rule_col='rule_preds', label_col='labels'):
        # this is used for 'linear' teacher
        """
        :param train_data: train data dict
        :param dev_data: development data dict (optional)
        :param u_data: unlabeled data dict (optional)
        :param train_student_proba: used for ASTRA
        :param dev_student_proba: used for ASTRA
        :param label_col: column of labels on the training set (in the dev set we always use 'labels')
        :return:
        """
        if self.name not in ['linear']:
            #_ = self.label_model.fit(u_data[rule_col], n_class=self.num_classes)
            #import pdb; pdb.set_trace()
            _ = self.apply_wrench(rule_preds=u_data[rule_col], fit=True, student_proba=u_student_proba)
            return 1
        assert train_data is not None, 'to train Teacher="linear" you need to provide training data'

        # Converts rule predictions into vectors and trains a linear classifier.
        self.rule_enc = LabelBinarizer(sparse_output=False)
        self.rule_enc.classes_ = [-1] + np.arange(self.num_classes).tolist()
        rule_enc_train = self.get_rule_encodings(train_data[rule_col], train_student_proba)
        rule_enc_dev = self.get_rule_encodings(dev_data[rule_col], dev_student_proba)
        if rule_enc_train is None:
            return 1
        self.label_model.train(X_train=rule_enc_train, y_train=train_data[label_col], X_valid=rule_enc_dev, y_valid=dev_data['labels'])
        return 1

    def apply(self, rule_preds, fit=False, student_proba=None):
        if self.name in 'linear':
            assert fit == False, 'to train linear Teacher you need to explicitly run teacher.train()'
            #if fit == True:
            #if rule_preds_train is None: rule_preds_train = rule_preds
            #self.train(train_data=train_data, dev_data=dev_data, student_proba)
            return self.apply_linear(rule_preds=rule_preds, student_proba=student_proba)
        else:
            return self.apply_wrench(rule_preds=rule_preds, fit=fit, student_proba=student_proba)

    def apply_linear(self, rule_preds, student_proba=None, abstain_on_uncovered=True):
        # assert that linear classifier is trained
        assert self.rule_enc is not None
        #if self.name == 'astra':
        #    student_proba = get_student_proba(data, self.student, student_proba)
        num_rules = rule_preds.shape[1]
        num_rules += 1 if student_proba is not None else 0
        self.logger.info(f'Applying Teacher with {num_rules} rules on {rule_preds.shape[0]} data')
        rule_enc = self.get_rule_encodings(rule_preds, student_proba)
        if rule_enc is None:
            self.logger.info("Teacher has access to no rules... returning trivial predictions")
            return self.get_trivial_res(N=rule_preds.shape[0])


        label_model_res = self.label_model.predict(rule_enc)
        soft_label = label_model_res['proba']
        hard_label = label_model_res['preds']
        if student_proba is None and abstain_on_uncovered == True:
            # predict "ABSTAIN" in examples that are not covered
            #hard_label = probs_to_preds(soft_label)
            abstain = self.check_abstain(rule_preds)
            hard_label[abstain] = self.abstain_ind
            soft_label[abstain] = np.ones((1, self.num_classes)) * (1.0 / self.num_classes)

        res = {"proba": soft_label, "preds": hard_label}
        return res

    def get_trivial_res(self, N):
        hard_labels = (np.ones((N, 1)) * self.abstain_ind).astype(int)
        soft_labels = np.ones((N, self.num_classes)) * (1.0 / self.num_classes)
        return {"proba": soft_labels, "preds": hard_labels}

    def apply_wrench(self, rule_preds, fit=False, student_proba=None):
        if student_proba is not None:
            # ASTRA: we use Student as an extra rule
            assert student_proba.shape[0] == rule_preds.shape[0], f'rule_preds={student_proba.shape} BUT student_proba={student_proba.shape}'
            student_preds = np.argmax(student_proba, axis=1)
            student_preds = np.expand_dims(student_preds, 1)
            rule_preds = np.hstack((rule_preds, student_preds))

        if rule_preds.shape[1] == 0:
            self.logger.info("Teacher has access to no rules... returning trivial predictions")
            return self.get_trivial_res(rule_preds.shape[0])

        if rule_preds.shape[1] < 3:
            # make sure there are at least 3 labeling functions to not raise any errors
            rule_preds = np.repeat(rule_preds, 3, axis=1)

        if fit:
            # self.label_model.fit(rule_preds, n_class=self.num_classes)
            try:
                self.label_model.fit(rule_preds, n_class=self.num_classes)
            except:
                # An error is raised in case the loss is NaN.
                self.logger.info("There was an error raised when training teacher.label_model")
                return None
        self.logger.info(f'Applying Teacher with {rule_preds.shape[1]} rules on {rule_preds.shape[0]} data')
        soft_label = self.label_model.predict_proba(rule_preds)
        hard_label = probs_to_preds(soft_label)
        abstain = self.check_abstain(rule_preds)
        hard_label[abstain] = self.abstain_ind
        res = {"proba": soft_label, "preds": hard_label}
        return res


    def remove_abstain(self, pseudodata, col):
        assert col in pseudodata, f"{col} not in pseudodata {pseudodata.keys()}"
        df = pd.DataFrame()
        df['idx'] = np.arange(pseudodata['features'].shape[0])
        df['labels'] = pseudodata[col]
        self.logger.info("\n{}".format(df['labels'].value_counts()))
        df = df[df['labels'] != self.abstain_ind]
        keep_ind = df['idx'].tolist()
        if len(keep_ind) == 0:
            self.logger.info("WARNING: all rules abstain...")
            return defaultdict(list)
        data = {x: np.array(y)[keep_ind] for x, y in pseudodata.items() if x != 'features'}
        if 'features' in pseudodata:
            data['features'] = pseudodata['features'][keep_ind]
        if 'texts' in data:
            data['texts'] = data['texts'].tolist()
        return data

    def check_abstain(self, rule_preds):
        # rule_preds: num_examples x num_rules
        # return a boolean array of size = num_examples
        # For each example, abstains = True iff all rules abstain (i.e., predict -1)
        abstains = rule_preds.max(axis=1) == self.abstain_ind
        return abstains

def to_one_hot(rules, enc, sparse_output=False):
    """
    Convert rule predictions to one-hot
    Note: abstain (-1) is interpreted as an extra class
    :param rules: an array of size num_examples x num_rules
    :return: an array of size num_examples x (num_rules * num_classes+1)
    """
    rule_enc = [enc.transform(rules[:, j]) for j in range(rules.shape[1])]
    if sparse_output:
        return scipy.sparse.hstack(rule_enc)
    else:
        return np.hstack(rule_enc)


def postprocess_teacher_data(features, labels):
    df = pd.DataFrame()
    df['idx'] = np.arange(features.shape[0])
    df['labels'] = labels
    print(df['labels'].value_counts())
    df = df[df['labels'] != -1]
    keep_ind = df['idx']

    keep_feat = features[keep_ind, :]
    keep_labels = labels[keep_ind]
    return keep_feat, keep_labels