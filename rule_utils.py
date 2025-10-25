"""
Anonymized INTERVAL Code
# Utils for using the imodels library (https://github.com/csinva/imodels) as a component of our framework
"""

import os
import glob
import pandas as pd
import joblib
import re
import numpy as np
from collections import defaultdict

# utility functions for rule-based algorithms
def get_printable_version(rule, pred=-1, op='AND'):
    # for feat, comp, val in rule:
    rulestr = ' {} '.format(op).join(
        ['{} {} {}'.format(feat, comp, val) for feat, comp, val in rule]) + ' ==> {}'.format(pred)
    return rulestr


def get_rule_description_ruleset(rule, vocab):
    original_desc = str(rule)
    featurelist = original_desc.split('feature_')
    featurelist = [x.replace('and', '').strip() for x in featurelist if x != '']

    features = [vocab[int(x.split(' ')[0])] for x in featurelist]
    comparisons = [x.split(' ')[1].strip() for x in featurelist]
    values = [x.split(' ')[2].strip() for x in featurelist]
    pred = score2class(rule.args[0])
    rule = list(zip(features, comparisons, values))
    rulestr = get_printable_version(rule, pred=pred)
    return rulestr


def get_rule_description_rulelist(rule, vocab):
    def check_rule_flip(rule):
        if not 'flip' in rule:
            return ''
        if rule['flip']:
            return '~'
        else:
            return ''

    prefix = check_rule_flip(rule)

    original_desc = rule['col']
    featurelist = original_desc.split('feat ')
    featurelist = [x.replace('and', '').strip() for x in featurelist if x != '']

    features = [prefix + vocab[int(x.split(' ')[0])] for x in featurelist]
    comparisons = ['>=' for x in featurelist]
    values = [rule['cutoff'] for x in featurelist]
    pred = score2class_list(rule['val_right'])
    # values = [x.split(' ')[2].strip() for x in featurelist]
    rule = list(zip(features, comparisons, values))
    rulestr = get_printable_version(rule, pred=pred)
    return rulestr


def get_rule_descriptions_ruleset(rule_clf, vocab):
    selected_rules = rule_clf.rules_without_feature_names_
    rule_descs = [get_rule_description_ruleset(x, vocab) for x in selected_rules]
    return rule_descs


def get_rule_descriptions_rulelist(rule_clf, vocab):
    selected_rules = rule_clf.rules_
    # skip 'else' statement
    selected_rules = selected_rules[:-1]
    rule_descs = [get_rule_description_rulelist(x, vocab) for x in selected_rules]
    return rule_descs


def score2class(score):
    if score > 0:
        return 1
    elif score < 0:
        return 0
    return


def score2class_list(score):
    if score >= 0.5:
        return 1
    else:
        return 0
    return


def apply_rules(X, rule_clf, rule_type):
    if 'corels' in rule_type:
        return apply_corels_rule_list(X, rule_clf)
    if 'ruleset' in rule_type:
        return apply_rule_set(X, rule_clf)
    elif 'rulelist' in rule_type:
        return apply_rule_list(X, rule_clf)
    else:
        raise (BaseExeption('cannot provide predictions for: {}'.format(rule_type)))
    return


def get_rule_descriptions(rule_clf, rule_type, vocab):
    if 'corels' in rule_type:
        return -1
    if 'ruleset' in rule_type:
        return get_rule_descriptions_ruleset(rule_clf, vocab)
    elif 'rulelist' in rule_type:
        return get_rule_descriptions_rulelist(rule_clf, vocab)
    else:
        raise (BaseExeption('cannot provide predictions for: {}'.format(rule_type)))
    return


def apply_corels(X, rule_clf):
    rule_clf.rl_.rules
    return


def apply_rule_set(X, rule_clf):
    selected_rules = rule_clf.rules_without_feature_names_
    df = pd.DataFrame(X, columns=rule_clf.feature_placeholders)  # alternatively: columns=vocab
    rule_preds = np.ones((X.shape[0], len(selected_rules))) * -1
    for jj, r in enumerate(selected_rules):
        features_r_uses = list(map(lambda x: x[0], r.agg_dict.keys()))
        rule_preds[df[features_r_uses].query(str(r)).index.values, jj] = score2class(r.args[0])
    rule_preds = rule_preds.astype(int)
    return rule_preds


def apply_rule_list(X, rule_clf):
    X = np.array(X)
    n = X.shape[0]
    rules = rule_clf.rules_
    rules = rules[:-1]

    num_rules = len(rules)
    probs = np.zeros((n, num_rules))
    rule_preds = np.ones((n, num_rules)) * -1
    for i in range(n):
        x = X[i]
        for j, rule in enumerate(rules):
            if j == len(rules) - 1:
                rule_preds[i, j] = score2class_list(rule['val'])
            elif x[rule['index_col']] >= rule['cutoff']:
                rule_preds[i, j] = score2class_list(rule['val_right'])
    rule_preds = rule_preds.astype(int)
    return rule_preds