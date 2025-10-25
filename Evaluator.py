"""
Anonymized INTERVAL Code
"""
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_recall_fscore_support
from collections import defaultdict
implemented_metrics = ['acc', 'prec', 'rec', 'f1', 'weighted_acc', 'weighted_f1']

import pandas as pd

class Evaluator:
    # A class that implements all evaluation metrics and prints relevant statistics
    def __init__(self, args, logger=None):
        args.ev = self
        self.args = args
        self.logger = logger
        self.metric = args.metric
        assert self.metric in implemented_metrics, "Evaluation metric not implemented: {}".format(self.metric)

    def evaluate_student(self, student, dev_data, test_data, feature_col='features'):
        if len(dev_data['texts']) > 0:
            valid_preds = student.predict(dev_data[feature_col], just_preds=True)
            valid_res = self.evaluate(gt=dev_data['labels'], pred=valid_preds, verbose=True)
        else:
            valid_res = {self.args.metric: -1}
        test_preds = student.predict(test_data[feature_col], just_preds=True)
        res = self.evaluate(gt=test_data['labels'], pred=test_preds, verbose=True)
        results = {"frac": -1, "teacher": -1, "student": res[self.args.metric], "teacher_res": {}, "student_res": res,
                   "valid_student": valid_res[self.args.metric], "valid_student_res": valid_res}
        return results

    def evaluate(self, gt, pred, verbose=True, beta=None):
        assert len(gt) == len(pred)
        beta = beta if beta is not None else self.args.fbeta
        orig_prec, orig_rec, orig_f, orig_sup = precision_recall_fscore_support(y_true=gt, y_pred=pred)
        test_df = pd.DataFrame()
        test_df['gt'] = gt
        test_df['pred'] = pred

        total = test_df.shape[0]
        ignore = test_df[test_df['pred'] == -1].shape[0]
        ignore_percent = ignore / float(total)
        coverage = 100 * (total - ignore) / float(total)

        if verbose:
            print("coverage: {}".format(coverage))
            print("ignored {}/{} samples ({:.1f}%)".format(ignore, total, 100 * ignore_percent))

        test_df = test_df[test_df['pred'] != -1]

        if verbose:
            print("GT:")
            print(test_df['gt'].value_counts())
            print("\nPRED:")
            print(test_df['pred'].value_counts())
        acc = 100 * accuracy_score(y_true=test_df['gt'], y_pred=test_df['pred'])
        f1 = 100 * f1_score(y_true=test_df['gt'], y_pred=test_df['pred'], average='macro')

        prec, rec, f, sup = precision_recall_fscore_support(y_true=test_df['gt'], y_pred=test_df['pred'],
                                                            average='macro')
        # clf_report = classification_report(y_true=test_df['gt'], y_pred=test_df['pred'])
        prec *= 100
        rec *= 100
        if verbose:
            print("\n\nAccuracy: {:.2f}%".format(acc))
            print("F1 score: {:.2f}%".format(f1))
            print("Precision: {:.2f}%".format(prec))
            print("Recall: {:.2f}%".format(rec))
            try:
                print(classification_report(y_true=test_df['gt'], y_pred=test_df['pred']))
            except:
                pass

        adjusted_acc = acc * (total - ignore) / total  # adjusting accuracy to include non-assignments
        adjusted_f1 = f1 * (total - ignore) / total  # adjusting f1 to include non-assignments
        adjusted_recall = rec * (total - ignore) / total  # adjusting recall to include non-assignments

        # compute Fb score with weights
        fb = 100 * compute_fb(precision=prec / 100, recall=adjusted_recall / 100, beta=beta)

        if verbose:
            print("\n\nAdjusted Accuracy: {:.2f}%".format(adjusted_acc))
            print("Adjusted F1 score: {:.2f}%".format(adjusted_f1))
        return {
            'coverage': coverage,
            'acc': adjusted_acc,
            'f1': adjusted_f1,
            'precision': prec,
            'recall': adjusted_recall,
            'orig_prec': orig_prec,
            'orig_rec': orig_rec,
            'orig_f': orig_f,
            'fb': fb,
        }

    def evaluate_old(self, preds, labels, proba=None, comment="", verbose=True):
        assert len(preds) == len(labels), "pred should have same length as true: pred={} gt={}".format(
            len(preds),
            len(labels)
        )

        preds = np.array(preds)
        labels = np.array(labels)

        total_num = len(preds)
        self.logger.info("Evaluating {} on {} examples".format(comment, total_num))

        # Ignore pred == -1 but also penalize by considering all of them as wrong predictions...
        ignore_ind = preds == -1
        keep_ind = preds != -1
        ignore_num = np.sum(ignore_ind)
        ignore_perc = ignore_num / float(total_num)
        if ignore_num > 0:
            self.logger.info("Ignoring {:.4f}% ({}/{}) predictions".format(100*ignore_perc, ignore_num, total_num))

        preds = preds[keep_ind]
        labels = labels[keep_ind]
        if proba is not None:
            proba = proba[keep_ind]
        if len(preds) == 0:
            self.logger.info("Passed empty {} list to Evaluator. Skipping evaluation".format(comment))
            return defaultdict(int)

        pred = list(preds)
        true = list(labels)
        acc = accuracy_score(y_true=true, y_pred=pred)
        f1 = f1_score(y_true=true, y_pred=pred, average='macro')
        prec, rec, fscore, support = precision_recall_fscore_support(y_true=true, y_pred=pred, average='macro')
        conf_mat = confusion_matrix(y_true=true, y_pred=pred)
        clf_report = classification_report(y_true=true, y_pred=pred)

        weighted_acc, weighted_f1 = compute_weighted_acc_f1(y_true=true, y_pred=pred)
        adjust_coef = (total_num - ignore_num) / float(total_num)

        res = {
            'acc': 100 * acc * adjust_coef,
            'weighted_acc': 100 * weighted_acc * adjust_coef,
            'prec': 100 * prec * adjust_coef,
            'rec': 100 * rec * adjust_coef,
            'f1': 100 * f1 * adjust_coef,
            'weighted_f1': 100 * weighted_f1 * adjust_coef,
            'ignored': ignore_num,
            'total': total_num
        }
        res["perf"] = res[self.metric]

        self.logger.info("{} performance: {:.2f}".format(comment, res["perf"]))
        if verbose:
            self.logger.info("{} confusion matrix:\n{}".format(comment, conf_mat))
            self.logger.info("{} report:\n{}".format(comment, clf_report))

        return res


def compute_weighted_acc_f1(y_true, y_pred):
    prec, rec, fscore, support = precision_recall_fscore_support(y_true=y_true, y_pred=y_pred)
    weighted_acc = np.sum(support * rec) / np.sum(support)
    weighted_f1 = np.mean(fscore)
    return weighted_acc, weighted_f1

def compute_fb(precision, recall, beta=1.0):
    # Compute F-beta score, i.e., weighted harmonic mean of precision and recall
    # b=0 uses only the precision, b=1 leads to F1 score, b=infinite uses only the recall
    if np.isposinf(beta):
        return recall
    beta2 = beta ** 2
    denom = beta2 * precision + recall
    if denom == 0:
        denom = 1
    fb_score = (1 + beta**2) * precision * recall / denom
    return fb_score


