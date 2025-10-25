"""
Anonymized INTERVAL Code
"""

import warnings
warnings.filterwarnings('ignore')

import os
import glob
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import re
import numpy as np
from collections import defaultdict
import argparse
import json
import logging
import os
import glob
from os.path import expanduser
from pathlib import Path
import numpy as np
import random
import shutil
import torch
import joblib
from DataHandler import MyDataHandler, filter_inds
from Student import Student
from Teacher import Teacher, postprocess_teacher_data
from Evaluator import Evaluator
from datetime import datetime
from copy import deepcopy
from utils import to_one_hot, evaluate_ran, analyze_rule_attention_scores, evaluate, evaluate_test
from utils import save_and_report_results, summarize_results, backup_src

from InstanceRanker import InstanceRanker
from RuleSuggester import RuleSuggester
from Oracle import Oracle
home = expanduser("~")
import sys
import torch


def interval(args, logger):
    labeled_samples_per_category = None if 'full' in args.type else args.labeled_samples_per_category
    rule_choice_strategy = None if args.rule_choice_strategy == 'full' else args.rule_choice_strategy

    dh = MyDataHandler(args, logger)
    ev = Evaluator(args, logger)
    student = Student(args, logger)
    teacher = Teacher(args, logger)
    ir = InstanceRanker(args, logger, active_method='hierarchical_subsampling_entropy')
    rulesuggester = RuleSuggester(args, logger)
    oracle = Oracle(args, logger)

    train_data, u_data, dev_data, test_data = dh.load_all_data(labeled_samples_per_category=labeled_samples_per_category, rule_choice_strategy=rule_choice_strategy)
    oracle.init(train_data, u_data)

    logger.info("Applying RuleSuggester to generate rules...")
    u_candidate_rule_preds = rulesuggester.generate_rules(u_data=u_data, labeled_data=train_data)
    if u_candidate_rule_preds is None:
        return {"frac": 1.0, "teacher": 0.0, "student": 0.0, "teacher_res": defaultdict(int),
                "student_res": defaultdict(int)}
    logger.info(f"RuleSuggester generated {u_candidate_rule_preds.shape[1]} accurate rules.")

    # Train Student on (few) labeled data
    logger.info("Training student on few labeled data...")
    student_res = student.train(X_train=train_data['features'], y_train=train_data['labels'], X_valid=dev_data['features'], y_valid=dev_data['labels'])
    if student_res is None:
        return {"frac": 1.0, "teacher": 0.0, "student": 0.0, "teacher_res": defaultdict(int),
                "student_res": defaultdict(int)}

    results = ev.evaluate_student(student, dev_data, test_data)
    logger.info("[iter 0]. Student={:.2f}".format(results["student"]))


    for data in train_data, dev_data, u_data:
        # initialize "autorule_preds" with empty rules
        data['autorule_preds'] = (np.ones((len(data['texts']), 0)) * teacher.abstain_ind).astype(int)

    active_results = [results]
    rule_query_history = [-1]
    for i in range(1, args.num_iter + 1):
        if u_data['features'].shape[0] < args.annotations_per_it:
            logger.info("No more unlabeled data remained. Stopping Active Learning...")
            break

        logger.info(f"[iter {i}] applying student on unlabeled/train/dev data")
        for data in train_data, dev_data, u_data:
            _student_res = student.predict(data['features'])
            data['student_proba'] = _student_res['proba']
            data['student_preds'] = _student_res['preds']

        logger.info(f"[iter {i}] training teacher on unlabeled/train/dev data")
        _teacher_res = teacher.train(train_data=train_data,
                                     train_student_proba=train_data['student_proba'] if 'astra' in args.type else None,
                                     dev_data=dev_data,
                                     dev_student_proba=dev_data['student_proba'] if 'astra' in args.type else None,
                                     u_data=u_data,
                                     u_student_proba=u_data['student_proba'] if 'astra' in args.type else None,
                                     rule_col='autorule_preds')
        if _teacher_res is None:
            return {"frac": 1.0, "teacher": 0.0, "student": 0.0, "teacher_res": defaultdict(int),
                    "student_res": defaultdict(int)}

        logger.info(f"[iter {i}] applying teacher on all data")
        for data in train_data, dev_data, u_data:
            _teacher_res = teacher.apply(rule_preds=data['autorule_preds'],
                                         student_proba=data['student_proba'] if 'astra' in args.type else None)
            data['teacher_preds'] = _teacher_res['preds']
            data['teacher_proba'] = _teacher_res['proba']

        logger.info(f"[iter {i}] applying rule suggester on all data")
        for data in train_data, dev_data, u_data:
            candidate_preds = rulesuggester.suggest_rules(data, return_rule_info=False)
            num_candidate_rules = candidate_preds.shape[1]
            logger.info(f"\t\t---> {num_candidate_rules} candidate accurate rules")
            data['candidate_preds'] = candidate_preds


        logger.info(f"[iter {i}] choosing top instances")
        sampled_inds = ir.active_select(u_proba=u_data['student_proba'], train_proba=train_data['student_proba'],
                                        u_data=u_data, train_data=train_data)

        logger.info(f"[iter {i}] applying rule suggester on unlabeled data")
        u_candidate_preds, rule_info = rulesuggester.suggest_rules(u_data, return_rule_info=True)
        # joblib.dump(rule_info, f"{args.dataset}_rule_info_interval.pkl")
        oracle_candidate_preds = rulesuggester.suggest_rules(oracle.data)
        num_candidate_rules = u_candidate_preds.shape[1]
        logger.info(f"\t\t----> {num_candidate_rules} candidate accurate rules")

        # suggest K candidate rules that are active for those instances
        # sort rules by precision
        B = len(sampled_inds)
        B_j = args.rule_queries_per_instance

        num_rule_queries = 0
        num_accepted_rule_queries = 0
        already_queried_inds = set()   # storing a set of inds that have already been inspected in the current iteration
        all_oracle_feedbacks = {}
        query_history = []
        for s_ind_i, s_ind in enumerate(sampled_inds):
            text_i = u_data['texts'][s_ind]
            label_i = u_data['orig_labels'][s_ind]
            query_history.append(('instance', label_i))
            selected_rule_preds = u_candidate_preds[[s_ind]]
            has_same_label = selected_rule_preds[0] == label_i
            selected_rule_inds = np.arange(len(has_same_label))[has_same_label].tolist()
            selected_rule_inds = [i for i in selected_rule_inds if i not in already_queried_inds]

            if len(selected_rule_inds) == 0:
                logger.info(f"[iter {i}] chose instance {s_ind_i}/{B}:\n\tTEXT = {text_i}\n\tLABEL = {label_i}")
                logger.info("No candidate rules apply to this instance. Skipping rule suggestion...")
                continue

            def get_list(criterion):
                # return lists and negate them for sorting purposes.
                # note: assumes access to rule_info and selected_rule_inds
                return np.array([x[criterion] for x in rule_info])[selected_rule_inds]

            list_to_rank = list(zip(-get_list('precision'), -get_list('coverage_labeled'), -get_list('coverage_unlabeled')))
            list_to_rank = np.array(list_to_rank, dtype=[('prec', '<i4'), ('l_cov', '<i4'), ('u_cov', '<i4')])
            sorted_inds = np.argsort(list_to_rank, order=('prec', 'l_cov', 'u_cov'))
            sorted_correct_rule_inds = np.array(selected_rule_inds)[sorted_inds]

            top_correct_rule_inds = sorted_correct_rule_inds[:B_j]
            logger.info(f"[iter {i}] chose instance {s_ind_i}/{B}:\n\tTEXT = {text_i}\n\tLABEL = {label_i}\n\tRules:\tCORRECT={len(sorted_correct_rule_inds)}\tKEPT={len(top_correct_rule_inds)} (budget={B_j})")

            #### get feedback on candidate rules by the Oracle
            exp_dict = {1: "*** ACCEPT ***", 0: ""}
            oracle_feedback = {}
            for ind in top_correct_rule_inds:
                info = rule_info[ind]
                oracle_input = oracle_candidate_preds[:, ind]
                oracle_ans, oracle_exp = oracle.answer(oracle_input)
                oracle_feedback[ind] = {'decision': oracle_ans, 'explanation': oracle_exp}
                logger.info("rule {} (prec={:.2f}, l_cov={}, u_cov={}): {}\t==> {} ({})".format(ind, info['precision'], info['coverage_labeled'], info['coverage_unlabeled'], info['description'], exp_dict[ oracle_ans], oracle_exp))
                query_history.append(('rule', oracle_ans, info['description'], info['precision']))
            num_oracle_accepted = sum([x['decision'] for x in oracle_feedback.values()])

            logger.info(f'\n\n\t\tOracle accepted {num_oracle_accepted}/{len(oracle_feedback)} rules\n\n\n')
            num_rule_queries += len(top_correct_rule_inds)
            num_accepted_rule_queries += num_oracle_accepted
            already_queried_inds.update(set(top_correct_rule_inds))
            all_oracle_feedbacks.update(oracle_feedback)

        ### Update step
        # update training data
        logger.info(f"[iter {i}] Increasing labeled set by {len(sampled_inds)} examples")
        train_data, u_data = dh.augment_train_data(train_data, u_data, sampled_inds)

        # Update RuleSuggester: update rule accuracies
        rulesuggester.update(labeled_data=train_data, oracle_feedback=all_oracle_feedbacks)   


        # add new rules in Teacher. Here Teacher2 = RuleSuggester.
        for data in train_data, dev_data, u_data:
            data['autorule_preds'] = rulesuggester.apply_accepted_rules(data)
            logger.info("accepted autorule predictions: {}".format(data['autorule_preds'].shape))

        # Re-train Teacher
        _teacher_res = teacher.train(train_data=train_data,
                                     train_student_proba=train_data['student_proba'] if 'astra' in args.type else None,
                                     dev_data=dev_data,
                                     dev_student_proba=dev_data['student_proba'] if 'astra' in args.type else None,
                                     u_data=u_data,
                                     u_student_proba=u_data['student_proba'] if 'astra' in args.type else None,
                                     rule_col='autorule_preds')
        if _teacher_res is None:
            return {"frac": 1.0, "teacher": 0.0, "student": 0.0, "teacher_res": defaultdict(int),
                    "student_res": defaultdict(int)}
        teacher_res = teacher.apply(rule_preds=u_data['autorule_preds'],
                                    student_proba=u_data['student_proba'] if 'astra' in args.type else None)

        # Re-train Student
        teacher_col = 'teacher_preds'
        pseudodata = deepcopy(u_data)
        assert len(pseudodata['texts']) == teacher_res['preds'].shape[0]
        pseudodata.update({f'teacher_{x}': y for x, y in teacher_res.items()})
        pseudodata = teacher.remove_abstain(pseudodata, col=teacher_col)
        if args.balance:
            pseudodata = dh.balance(pseudodata, balance_col=teacher_col)

        logger.info('training student classifier')
        student_res = student.train_pseudo(X_pseudo=pseudodata['features'], y_pseudo=pseudodata[teacher_col],
                                           X_train=train_data['features'], y_train=train_data['labels'],
                                           X_valid=dev_data['features'], y_valid=dev_data['labels'])
        if student_res is None:
            return {"frac": 1.0, "teacher": 0.0, "student": 0.0, "teacher_res": defaultdict(int),
                    "student_res": defaultdict(int)}

        # Test student
        results = ev.evaluate_student(student, dev_data, test_data)
        logger.info("[iter {}]. Student={:.2f}".format(i, results["student"]))

        results['num_oracle_accepted'] = num_accepted_rule_queries
        results['num_oracle_queries'] = num_rule_queries  # len()  # num_rule_queries
        if num_rule_queries != len(all_oracle_feedbacks):
            logger.info(f"ORACLE WARNING: Oracle feedback = {len(all_oracle_feedbacks)} but num_rule_queries={num_rule_queries}")


        active_results.append(results)
        rule_query_history.append({"num_rule_queries": num_rule_queries,
                                   "num_accepted_rule_queries": num_accepted_rule_queries,
                                   "history": query_history,
                                   })

    # Return the iteration that has best performance on the validation set
    best_valid_ind = np.argmax([res["valid_student"] for res in active_results])
    logger.info(f"Best iteration according to dev set: {best_valid_ind}")
    res = active_results[best_valid_ind]
    res["best_iteration"] = best_valid_ind
    res["active_results"] = active_results
    res["rule_query_history"] = rule_query_history
    return res