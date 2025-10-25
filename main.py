"""
Anonymized INTERVAL Code
"""

import warnings
warnings.filterwarnings('ignore')

import os
import glob
import pandas as pd
import joblib
import numpy as np
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
from datetime import datetime
home = expanduser("~")

from Logger import get_logger, close
from interval import interval


def main():
    parser = argparse.ArgumentParser()

    # Data processing arguments
    parser.add_argument("--datapath", help="Path to base dataset folder", type=str, default='../data')
    parser.add_argument("--dataset", help="Dataset name", type=str, default='youtube')
    parser.add_argument("--labeled_samples_per_category", nargs="?", type=int, default=20, help="number of labeled samples to consider per category (in training and validation set)")
    parser.add_argument("--logdir", help="Experiment log directory", type=str, default='./')
    parser.add_argument("--dont_trim_validation_set", action="store_true", help="Keep the original validation set without reducing its size")
    parser.add_argument("--rule_choice_strategy", help="Strategy for choosing a subset of rules", type=str, default='full')

    # Main Arguments
    parser.add_argument("--type", help="Dataset name", type=str, default='supervised') # Type of experiment: rulefrac vs. labelfrac vs. supervised vs. low_supervised vs. selftrain vs. weaksup
    parser.add_argument("--rule_eval_set_perc", help="Downsample labeled dataset used to evaluate rule metrics for ranking", type=float, default=1.0)
    parser.add_argument("--supervised", action="store_true", help="Run supervised experiment")

    # Teacher arguments
    parser.add_argument("--teacher_name", help="Teacher short name", type=str, default='astra')
    parser.add_argument("--convert_abstain_to_random", action="store_true", help="In Teacher, if rules abstain on dev/test then flip a coin")
    parser.add_argument("--weighted_mv", action='store_true', help='aggregate rule predictions via *weighted* majority voting')

    # Student arguments
    parser.add_argument("--encoder", help="Encoder type: bert, bow, tfidf", type=str, default='default')  # default / logreg / bert / bow
    parser.add_argument("--rule_metric", help="Metric for sorting rules", type=str, default='random')
    parser.add_argument("--student_rule_metric", help="Metric for sorting rules using Student's labels on unlabeled data", type=str, default='random') # if random then continue
    parser.add_argument("--num_runs", help="Number of runs for the same experiment (in case of random rule selection)", type=int, default=10)
    parser.add_argument("--balance", action="store_true", help="Balance labels before training the student")
    parser.add_argument("--no_gridsearch", action="store_true", help="Deactivate hyper-parameter tuning in Student via grid-search ")
    parser.add_argument("--weak_vs_clean_weight", help="Relative weight of weak vs. clean labeled data (0: fine-tuning using just clean data, 1: equal weight)", type=float, default=1.0)

    # Rule Extractor arguments
    parser.add_argument("--lf_generator", help="Type of LF generator: original, accurate, exhaustive", type=str, default='original')

    # Evaluator arguments
    parser.add_argument("--metric", help="Evaluation metric", type=str, default='f1')
    parser.add_argument("--fbeta", default=1.0, type=float, help="beta parameter for computing F-beta score (weighted harmonic mean of precision and recall)")
    parser.add_argument("--min_propensity", default=0.01, type=float, help="minimum propensity for automatically generated rules")

    parser.add_argument("--student_name", help="Student short name", type=str, default='logreg')
    parser.add_argument("--n_steps", nargs="?", type=int, default=1000, help="number of steps for training student")


    # Extra Arguments
    parser.add_argument("--experiment_folder", help="Dataset name", type=str, default='../experiments/')
    parser.add_argument("--num_iter", help="Number of self/co-training iterations", type=int, default=25)
    parser.add_argument("--num_supervised_trials", nargs="?", type=int, default=5,
                        help="number of different trials to start self-training with")
    parser.add_argument('-ws', '--weak_sources', help="List of weak sources name for Teacher", nargs='+')
    parser.add_argument("--downsample", help="Downsample labeled train & dev datasets randomly stratisfied by label",
                        type=float, default=1.0)
    parser.add_argument("--oversample", help="Oversample labeled train datasets", type=int, default=1)
    parser.add_argument("--tokenizer_method", help="Tokenizer method (for LogReg student)", type=str, default='clean')
    parser.add_argument("--num_epochs", default=70, type=int, help="Total number of training epochs for student.")
    parser.add_argument("--num_unsup_epochs", default=25, type=int, help="Total number of training epochs for training student on unlabeled data")
    parser.add_argument("--debug", action="store_true", help="Activate debug mode")
    parser.add_argument("--soft_labels", action="store_true", help="Use soft labels for training Student")
    parser.add_argument("--loss_weights", action="store_true", help="Use instance weights in loss function according to Teacher's confidence")
    parser.add_argument("--hard_student_rule", action="store_true",
                        help="When using Student as a rule in Teacher, use hard (instead of soft) student labels")
    parser.add_argument("--train_batch_size", help="Train batch size", type=int, default=16)
    parser.add_argument("--eval_batch_size", help="Dev batch size", type=int, default=128)
    parser.add_argument("--unsup_batch_size", help="Unsupervised batch size", type=int, default=128)
    parser.add_argument("--max_size",
                        help="Max size of unlabeled data for training the student if balance_maxsize==True", type=int,
                        default=1000)
    parser.add_argument("--max_seq_length", help="Maximum sequence length (student)", type=int, default=64)
    parser.add_argument("--max_rule_seq_length",
                        help="Maximum sequence length of rule predictions (i.e., max # rules that can cover a single instance)",
                        type=int, default=10)
    parser.add_argument("--no_cuda", action="store_true", help="Do not use CUDA")
    parser.add_argument("--lower_case", action="store_true", help="Use uncased model")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite dataset if exists")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay for student")
    parser.add_argument("--learning_rate", default=1e-3, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--finetuning_rate", default=1e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--logging_steps", type=int, default=500, help="Log every X updates steps.")
    parser.add_argument("--fp16", action='store_true', help='whehter use fp16 or not')
    parser.add_argument("--sample_size", nargs="?", type=int, default=16384, help="number of unlabeled samples for evaluating uncetainty on in each self-training iteration")
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    # More arguments
    parser.add_argument("--use_clean_labeled", action="store_true", help="Use few clean labeled data for training student")

    # arguments for active learning component (InstanceRanker.py)
    parser.add_argument("--reverse", action="store_true", help="Choose opposite data points")
    parser.add_argument("--indicator", required=False, default=None, type=str, help="Experiment indicator")
    parser.add_argument("--annotations_per_it", type=int, default=10, help="Number of annotations per Active Learning iteration")
    parser.add_argument("--max_unlabeled", default=10000, type=float, help="maximum number of unlabeled data to consider (for CAL method)")
    parser.add_argument("--num_nei", default=10, type=float, help="number of kNN to find (for CAL method)")
    parser.add_argument("--ce", default=False, type=bool, help="if True choose cross entropy for scoring (for CAL method), otherwise use KL div")
    parser.add_argument("--operator", default="mean", type=str, help="operator to combine scores of neighbours (for CAL method)")
    # parser.add_argument("--conf_mask", default=False, type=bool, help="if True mask neighbours with confidence score")

    # Arguments for Rule Suggester
    parser.add_argument("--min_rule_coverage", default=10, type=int, help="Minimum coverage that a rule must have on the unlabeled set to be considered")
    parser.add_argument("--min_rule_precision", default=0.75, type=float, help="Minimum precision that a rule must have on the labeled set to be considered")
    parser.add_argument("--rule_family", default="ngram", type=str, help="Family of automatically extracted rules (default = ngram)")  # ngram, dnf
    parser.add_argument("--topk_prompt", default=10, type=int, help="Topk tokens (ranked by RoBERTa) to keep for prompt-based features")

    # Arguments for Oracle
    parser.add_argument("--oracle_precision_threshold", default=0.75, type=float, help="Minimum Oracle precision that a rule must have for the Oracle to accept it")
    parser.add_argument("--rule_queries_per_instance", type=int, default=1, help="Number of rule queries to submit to Oracle per instance")

    args = parser.parse_args()
    np.random.seed(args.seed)
    random.seed(args.seed)


    # Start Experiment
    now = datetime.now()
    date_time = now.strftime("%Y_%m_%d-%H_%M")

    args.experiment_folder = os.path.join(args.experiment_folder, args.dataset)
    args.logdir = os.path.join(args.experiment_folder, args.logdir)
    experiment_dir = str(Path(args.logdir).parent.absolute())

    if args.supervised:
        args.type = "supervised"

    os.makedirs(args.logdir, exist_ok=True)
    logger = get_logger(logfile=os.path.join(args.logdir, 'log.log'))
    backup_src(args)

    # Setup CUDA, GPU & distributed training
    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.train_batch_size = args.train_batch_size * max(1, args.n_gpu)
    args.eval_batch_size = args.train_batch_size * max(1, args.n_gpu)


    logger.info("\n\n\t\t *** NEW EXPERIMENT ***\nargs={}".format(args))
    random_seeds = random.sample(range(1000), args.num_runs) if args.num_runs > 1 else [args.seed]
    args.seed_list = random_seeds
    logger.info("random seeds = {}".format(random_seeds))
    joblib.dump(vars(args), os.path.join(args.logdir, 'args.pkl'))

    all_res = []
    args.best_student_params = {}

    try:
        for run in range(args.num_runs):
            args.seed = random_seeds[run]
            logger.info(f"\n\n\t\t *** run = {run+1} / {args.num_runs} (seed={args.seed}) ***")
            res = interval(args, logger)
            all_res.append(res)
    except Exception as e:
        logger.exception(e)

    report_experiments(args, logger, all_res)
    joblib.dump(all_res, os.path.join(args.logdir, 'rule_res.pkl'))
    joblib.dump(args.best_student_params, os.path.join(args.logdir, 'best_student_params.pkl'))
    logger.info("results stored at {}".format(args.logdir))
    close(logger)


if __name__ == "__main__":
    main()

