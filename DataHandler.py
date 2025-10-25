"""
Anonymized INTERVAL Code
"""

import os
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from copy import deepcopy
from datasets import list_datasets
import joblib
import scipy

huggingface_datasets_list = list_datasets()
preprocessed_dataset_list = ['trec', 'youtube', 'sms']


from os.path import expanduser
home = expanduser("~")
import sys
wrench_folder = os.path.join(home, "wrench")
assert os.path.exists(wrench_folder), 'You need to download wrench first (see "ext" directory)'
sys.path.append(wrench_folder)
from wrench.dataset import load_dataset as load_wrench_dataset
from rule_utils import apply_rules, get_rule_descriptions


class MyDataHandler:
    def __init__(self, args, logger):
        args.dh = self
        self.args = args
        self.logger = logger
        self.dataset = args.dataset
        self.seed = args.seed
        self.encoder_name = args.encoder
        self.lf_generator = args.lf_generator
        self.num_classes = None
        self.sample_validation_set = True if not args.dont_trim_validation_set else False

        astra_datasets = ['TREC', 'YOUTUBE', 'SMS']
        wrench_datasets = ['youtube', 'sms', 'imdb', 'yelp', 'agnews']
        if self.dataset in astra_datasets:
            self.load_data = self.load_data_astra
        elif self.dataset in wrench_datasets:
            self.init_wrench()
            self.load_data = self.load_data_wrench
        else:
            raise(BaseException('dataset not supported by datahandler: {}'.format(self.dataset)))

    def load_all_data(self, labeled_samples_per_category=None, rule_choice_strategy=None):
        self.logger.info('sampling rules')
        train_data = self.load_data(method='train')
        if rule_choice_strategy is not None and rule_choice_strategy != 'full':
            keep_rule_inds = self.choose_rules(method='train', rule_choice_strategy=rule_choice_strategy)
        self.logger.info('unlabeled/train/dev/test set split')
        train_data, u_data = self.split_labeled_unlabeled(train_data, labeled_samples_per_category=labeled_samples_per_category, random_state=self.seed)
        train_data['orig_labels'] = train_data['labels']
        u_data['orig_labels'] = u_data['labels']
        u_data['labels'] = [None] * len(list(u_data['labels']))

        dev_data = self.load_data(method='dev')
        if self.sample_validation_set:
            # Sample labeled data from validation set
            # if we consider interactive methods, then we consider a larger validation set (since we have more resources)
            valid_labeled_samples_per_category = 10 * labeled_samples_per_category if 'active' in self.args.type or 'iws' in self.args.type else labeled_samples_per_category
            dev_data, _ = self.split_labeled_unlabeled(dev_data, labeled_samples_per_category=valid_labeled_samples_per_category, random_state=self.seed)
        test_data = self.load_data(method='test')

        if rule_choice_strategy is not None and rule_choice_strategy != 'full':
            for data in [train_data, u_data, dev_data, test_data]:
                data['rule_preds'] = data['rule_preds'][:, keep_rule_inds]

        self.logger.info(f"Data statistics\n\ttrain={len(train_data['texts'])} \trules={train_data['rule_preds'].shape}\
        \n\tunlabeled={len(u_data['texts'])} \trules={u_data['rule_preds'].shape}\
        \n\tdev={len(dev_data['texts'])} \trules={dev_data['rule_preds'].shape}\
        \n\ttest={len(test_data['texts'])} \trules={test_data['rule_preds'].shape}")
        return train_data, u_data, dev_data, test_data

    def choose_rules(self, method='train', rule_choice_strategy="best_f"):
        try:
            ranking, metric = tuple(rule_choice_strategy.split("_"))
            topk = 1
        except:
            ranking, metric, topk = tuple(rule_choice_strategy.split("_"))
            topk = int(topk)

        metric = f"orig_{metric}" if metric in ['f', 'prec'] else metric
        # metric = "orig_f"
        # ranking = "best"
        self.logger.info(f"Choosing a subset of {topk} rules ({rule_choice_strategy}): ranking={ranking}, metric={metric}")

        from Evaluator import Evaluator
        ev = Evaluator(self.args, self.logger)
        data = self.load_data(method=method)
        all_rules = data['rule_preds'].shape[1]
        summary, lf_summary = self.get_wrench_summary(method="train")
        lf_summary = lf_summary.to_dict(orient='records')
        all_results = []
        for i in range(all_rules):
            res = ev.evaluate(gt=data['labels'], pred=data['rule_preds'][:, [i]], verbose=False)
            all_results.append({
                "rule": i,
                "results": res,
                "lf_summary": lf_summary[i],
            })

        polarities = [res["lf_summary"]["Polarity"] for res in all_results]
        self.logger.info(f"\t\tSorting rules by {metric}")
        kept_rule_inds = []
        for class_ind in range(summary['n_class']):
            if "orig" in metric:
                kept_res = [res["results"][metric][class_ind + 1] for res in all_results]
            else:
                kept_res = [res["results"][metric] for res in all_results]

            sorted_ind = np.argsort(kept_res)[::-1]
            sorted_ind = [i for i in sorted_ind if class_ind in polarities[i]]
            sorted_ind = [i for i in sorted_ind if kept_res[i] >= 0.001]

            if ranking == "best":
                selected_ind = sorted_ind[0:topk]
            elif ranking == "worst":
                selected_ind = sorted_ind[-topk:]
            elif ranking == "median":
                N = len(sorted_ind) // 2
                selected_ind = sorted_ind[max(N - topk // 2, 0) : N + int(np.ceil(topk / 2))]
            self.logger.info("\n" + "\n".join(["(class {}) rule {}: {:.4f}".format(polarities[i], i, kept_res[i]) for i in selected_ind]))
            kept_rule_inds.extend(selected_ind)

        self.logger.info("selected rule inds: {}".format(kept_rule_inds))
        for rule_ind in kept_rule_inds:
            self.logger.info("\n\nrule {}:\n\t{},\n\t{}".format(rule_ind, all_results[rule_ind]["results"], lf_summary[rule_ind]))
        return kept_rule_inds

    def augment_train_data(self, train_data, u_data, inds):
        """
        Move the examples from u_data (indexed by inds) to train_data
        :param train_data: train data
        :param u_data: unlabeled data
        :param inds: indices of unlabeled data to move
        :return:
        """
        inds_set = set(inds)
        not_selected_inds = [i for i in range(len(u_data['labels'])) if i not in inds_set]
        selected_data = filter_inds(u_data, inds)

        # update the true labels of the selected labeled data
        selected_data['labels'] = selected_data['orig_labels']
        new_u_data = filter_inds(u_data, not_selected_inds)
        new_train_data = self.concatenate_data(train_data, selected_data)
        return new_train_data, new_u_data

    def sample(self, data, samples):
        N = data['features'].shape[0]

        if samples > N:
            return data

        df = pd.DataFrame()
        all_indices = np.arange(N)
        df['idx'] = all_indices
        df = df.sample(n=samples, random_state=self.seed)
        keep_inds = df['idx'].tolist()
        keep_data = filter_inds(data, keep_inds)
        return keep_data

    def split_labeled_unlabeled(self, data, frac=1.0, samples=None, labeled_samples_per_category=None, random_state=None):
        """
            Splits training data into training and unlabeled data to simulate low-resource setting
        :param data: training data dict
        :param frac:
        :param samples:
        :param labeled_samples_per_category:
        :param random_state:
        :return: train_data, unlabeled_data
        """
        features, labels = data['features'], data['labels']
        df = pd.DataFrame()
        all_indices = np.arange(features.shape[0])
        df['idx'] = all_indices
        df['labels'] = labels

        if frac < 1:
            # sample a fraction of labeled data
            train_df = df.sample(frac=frac, random_state=random_state)
            self.logger.info("Sampled data:\n{}".format(train_df['labels'].value_counts()))
        elif samples is not None:
            # sample a number of labeled data
            train_df = df.sample(n=samples, random_state=random_state)
            self.logger.info("Sampled data:\n{}".format(train_df['labels'].value_counts()))
        elif labeled_samples_per_category is not None:
            # sample a number of labeled data per category
            min_class_support = df['labels'].value_counts().min()
            if min_class_support < labeled_samples_per_category:
                self.logger.info(f"min_class_support = {min_class_support} < {labeled_samples_per_category}. reducing labeled_samples_per_category")
                labeled_samples_per_category = min_class_support
            train_df = df.groupby('labels', group_keys=False).apply(
                lambda x: x.sample(labeled_samples_per_category, random_state=random_state))
            self.logger.info("Sampled data:\n{}".format(train_df['labels'].value_counts()))
        else:
            # keeping all labeled data
            train_df = df

        train_inds = train_df['idx'].tolist()
        train_inds_set = set(train_inds)
        unlabeled_inds = [x for x in all_indices if x not in train_inds_set]

        train_data = filter_inds(data, train_inds)
        unlabeled_data = filter_inds(data, unlabeled_inds)
        return train_data, unlabeled_data

    def init_wrench(self):
        self.logger.info('loading wrench data')
        dataset_home = os.path.join(wrench_folder, 'datasets')
        dataset = self.dataset
        self.wrench_dataset_home = os.path.join(dataset_home, dataset)
        extract_fn = self.encoder_name
        model_name = 'bert-base-cased' if extract_fn == 'bert' else extract_fn
        cache_name = "wrench_{}".format(extract_fn) if extract_fn != 'bert' else extract_fn
        train_data, valid_data, test_data = load_wrench_dataset(dataset_home, dataset, extract_feature=True,
                                                         extract_fn=extract_fn,
                                                         cache_name=cache_name, model_name=model_name)
        self.num_classes = len(set(train_data.labels))

        self.wrench_data = {
            'train': train_data,
            'dev': valid_data,
            'test': test_data,
        }
        self.modified_wrench_data = {
            'train': deepcopy(train_data),
            'dev': deepcopy(valid_data),
            'test': deepcopy(test_data),
        }

        if self.args.lf_generator == 'pool':
            self.rule_clf_types = ['rulefit_ruleset', 'boostedrules_ruleset', 'skope_ruleset', 'oneR_rulelist', 'greedy_rulelist']  # , 'corels_rulelist'] #, 'slim_linear']
            self.log_extracted_rule_pool()
        return


    def log_extracted_rule_pool(self):
        all_rule_descs = []
        all_rule_types = []

        # load human-created LFs
        method = 'test'
        data = self.wrench_data[method]
        rule_preds = np.array(data.weak_labels)
        rule_descs = ['human{}'.format(i) for i in range(rule_preds.shape[1])]
        all_rule_descs.extend(rule_descs)
        all_rule_types.extend(['human'] * len(rule_descs))


        # exhaustive rules (WRENCH)
        method = 'test'
        if self.args.min_propensity == 0.01:
            fpath = os.path.join(self.wrench_dataset_home, 'generated_LFs_exhaustive_{}.pkl'.format(method))
        else:
            fpath = os.path.join(self.wrench_dataset_home, 'generated_LFs_exhaustive_minsupport{}_{}.pkl'.format(self.args.min_propensity, method))
        weak_labels = joblib.load(fpath)
        rule_preds = np.array(weak_labels)
        rule_descs = ['exhaustive{}'.format(i) for i in range(rule_preds.shape[1])]
        all_rule_descs.extend(rule_descs)
        all_rule_types.extend(['exhaustive'] * len(rule_descs))

        # Rules extracted from rule based classifiers
        rulefolder = os.path.join(home, 'iws/ext/rule_based_classifiers')
        clf_fname = os.path.join(rulefolder, '{}_rule_based_classifiers.pkl'.format(self.dataset))
        vectorizer_fname = os.path.join(rulefolder, '{}_vectorizer.pkl'.format(self.dataset))
        vectorizer_rules = joblib.load(vectorizer_fname)

        vocab = vectorizer_rules.get_feature_names()
        rule_clfs = joblib.load(clf_fname)

        for ruletype in self.rule_clf_types:
            # load LFs from rule-based classifiers that were already trained.
            clf = rule_clfs[ruletype]
            rule_descs = get_rule_descriptions(rule_clf=clf, rule_type=ruletype, vocab=vocab)
            all_rule_descs.extend(rule_descs)
            all_rule_types.extend([ruletype] * len(rule_descs))

        # save rule info
        df = pd.DataFrame()
        df['desc'] = all_rule_descs
        df['type'] = all_rule_types
        self.logger.info("{} rules:\n{}".format(df.shape[0], df['type'].value_counts()))
        joblib.dump(df, os.path.join(self.args.logdir, 'rules_df.pkl'))

    def load_modified_wrench_data(self, method):
        # This is a help function used by the Teacher because many label models require this WRENCH data structure
        if method == 'unlabeled':
            method = 'train'
        return self.modified_wrench_data[method]

    def get_wrench_summary(self, method):
        if method == 'unlabeled':
            method = 'train'
        data = self.wrench_data[method]
        summary = data.summary()
        lf_summary = data.lf_summary()
        if self.args.type == 'rulestats':
            self.logger.info("{} summary: \n{}\n{}".format(method, summary, lf_summary))
        elif self.args.type == 'datastats':
            self.logger.info("{} summary: \n{}".format(method, summary))
        return summary, lf_summary

    def load_data_wrench(self, method):
        if method == 'unlabeled':
            method = 'train'
        data = self.wrench_data[method]

        if self.args.type == 'rulestats':
            summary = data.summary()
            lf_summary = data.lf_summary()
            self.logger.info("{} summary: \n{}\n{}".format(method, summary, lf_summary))
        elif self.args.type == 'datastats':
            summary = data.summary()
            self.logger.info("{} summary: \n{}".format(method, summary))

        self.logger.info('loading human rules')
        if self.lf_generator == 'original':
            # load human-created LFs
            rule_preds = np.array(data.weak_labels)
        elif self.lf_generator == 'exhaustive':
            # load exhaustively generated LFs
            if self.args.min_propensity == 0.01:
                fpath = os.path.join(self.wrench_dataset_home, 'generated_LFs_exhaustive_{}.pkl'.format(method))
            else:
                fpath = os.path.join(self.wrench_dataset_home, 'generated_LFs_exhaustive_minsupport{}_{}.pkl'.format(self.args.min_propensity, method))
            weak_labels = joblib.load(fpath)
            rule_preds = np.array(weak_labels)
        elif self.lf_generator == 'accurate':
            # load top 100 accurate generated LFs
            if self.args.min_propensity == 0.01:
                fpath = os.path.join(self.wrench_dataset_home, 'generated_LFs_accurate_{}.pkl'.format(method))
            else:
                fpath = os.path.join(self.wrench_dataset_home, 'generated_LFs_accurate_minsupport{}_{}.pkl'.format(self.args.min_propensity, method))
            weak_labels = joblib.load(fpath)
            rule_preds = np.array(weak_labels)
        elif 'rulebased' in self.lf_generator:

            ruletype = self.lf_generator.replace('rulebased_', '')
            # load LFs from rule-based classifiers that were already trained.
            rulefolder = os.path.join(home, 'iws/ext/rule_based_classifiers')
            clf_fname = os.path.join(rulefolder, '{}_rule_based_classifiers.pkl'.format(self.dataset))
            vectorizer_fname = os.path.join(rulefolder, '{}_vectorizer.pkl'.format(self.dataset))
            vectorizer_rules = joblib.load(vectorizer_fname)
            clfs = joblib.load(clf_fname)

            vocab = vectorizer_rules.get_feature_names()

            texts = [ex['text'] for ex in data.examples]
            feature_vectors = vectorizer_rules.transform(texts).todense()

            clf = clfs[ruletype]
            rule_descs = get_rule_descriptions(rule_clf=clf, rule_type=ruletype, vocab=vocab)

            #print(ruletype)
            #print('\n'.join(rule_descs))
            rule_preds = apply_rules(X=feature_vectors, rule_clf=clf, rule_type=ruletype)
            logger.info('loaded {} rules extracted from {}'.format(rule_preds.shape[1], ruletype))
        elif self.lf_generator == 'pool':
            all_rule_preds = []

            # load human-created LFs
            rule_preds = np.array(data.weak_labels)
            all_rule_preds.append(rule_preds)

            # exhaustive rules (WRENCH)
            if self.args.min_propensity == 0.01:
                fpath = os.path.join(self.wrench_dataset_home, 'generated_LFs_exhaustive_{}.pkl'.format(method))
            else:
                fpath = os.path.join(self.wrench_dataset_home, 'generated_LFs_exhaustive_minsupport{}_{}.pkl'.format(self.args.min_propensity,method))
            weak_labels = joblib.load(fpath)
            all_rule_preds.append(np.array(weak_labels))

            # rules extracted from rule-based models
            rulefolder = os.path.join(home, 'iws/ext/rule_based_classifiers')
            texts = [ex['text'] for ex in data.examples]
            clf_fname = os.path.join(rulefolder, '{}_rule_based_classifiers.pkl'.format(self.dataset))
            vectorizer_fname = os.path.join(rulefolder, '{}_vectorizer.pkl'.format(self.dataset))
            vectorizer_rules = joblib.load(vectorizer_fname)
            vocab = vectorizer_rules.get_feature_names()
            feature_vectors = vectorizer_rules.transform(texts).todense()
            clfs = joblib.load(clf_fname)


            for ruletype in self.rule_clf_types:
                # load LFs from rule-based classifiers that were already trained.
                clf = clfs[ruletype]
                # rule_descs = get_rule_descriptions(rule_clf=clf, rule_type=ruletype, vocab=vocab)
                rule_preds1 = apply_rules(X=feature_vectors, rule_clf=clf, rule_type=ruletype)
                all_rule_preds.append(rule_preds1)

            # concatenate all rule predictions
            rule_preds = np.hstack(all_rule_preds)
        else:
            raise (BaseException('rule type {} does not exist'.format(lf_generator)))

        texts = [ex['text'] for ex in data.examples]
        features = data.features
        labels = data.labels
        res = {'texts': texts,
               'features': features,
               'labels': labels,
               'rule_preds': rule_preds,
               'ids': range(len(texts)),
               }

        if self.args.rule_family == "dnf":
            self.logger.info('loading high-level features')
            prompt_features = self.load_prompt_features(method=method, topk=self.args.topk_prompt)
            spacy_features = self.load_spacy_features(method=method)
            promptfeats = [" ".join(f) for f in prompt_features]
            spacyfeats = [" ".join(f) for f in spacy_features]
            assert len(texts) == len(prompt_features) and len(texts) == len(spacy_features)
            res['prompt_features'] = promptfeats
            res['spacy_features'] = spacyfeats
        return res

    def load_prompt_features(self, method, topk=100):
        promptfolder = os.path.join(self.args.datapath, 'prompt_based_features')
        prompt_res_cache_dir = os.path.join(promptfolder, f'{self.dataset}_{method}_cached_top{topk}.pkl')

        if os.path.exists(prompt_res_cache_dir):
            self.logger.info("loading cached prompt features from {}".format(prompt_res_cache_dir))
            prompt_features = joblib.load(prompt_res_cache_dir)
            return prompt_features.tolist()


        prompt_res = joblib.load(os.path.join(promptfolder, f'{self.dataset}_{method}.pkl'))
        prompt_features = []
        for template_name, res in prompt_res.items():
            tokens, proba, template_str = res['top_tokens'], res['proba'], res['template_str']
            tokens = [token_list[:topk] for token_list in tokens]
            # Sometimes the model predicts nonzero proba for < topk tokens so we need to make sure all examples have topk tokens
            tokens = [token_list if len(token_list) == topk else token_list + ['NONE'] * (topk - len(token_list)) for token_list in tokens]
            # check that I have topk tokens
            #for token_list in tokens:
            #    if len(token_list) != topk:
            #        print("error for template={}".format(template_name))

            tokens = [[f"PROMPT_{template_name.upper()}_{t}" for t in token_list] for token_list in tokens]
            prompt_features.append(tokens)

        #xxx = [set([len(x) for x in prompt_features[iii]]) for iii in range(len(prompt_features))]
        #print(xxx)

        prompt_features = np.hstack(prompt_features)
        self.logger.info("saving prompt features at {}".format(prompt_res_cache_dir))
        joblib.dump(prompt_features, prompt_res_cache_dir)
        return prompt_features.tolist()

    def load_spacy_features(self, method):
        spacyfolder = os.path.join(self.args.datapath, 'spacy_features')
        spacy_ner_features = joblib.load(os.path.join(spacyfolder, f'{self.dataset}_{method}.pkl'))
        spacy_features = [[f"SPACY_ENT_{t}" for t in ner_list] for ner_list in spacy_ner_features]
        return spacy_features

    def load_data_astra(self, method):
        basefolder = '/home/code/ASTRA/data'
        features = joblib.load(os.path.join(basefolder, "{}/seed{}/{}_x.pkl".format(self.dataset, self.seed, method)))
        labels = joblib.load(os.path.join(basefolder, "{}/seed{}/{}_labels.pkl".format(self.dataset, self.seed, method)))
        labels = labels.flatten()
        rule_preds = joblib.load(os.path.join(basefolder, "{}/seed{}/{}_rule_preds.pkl".format(self.dataset, self.seed, method)))
        abstain = rule_preds.max()
        rule_preds[rule_preds == abstain] = -1
        return features, labels, rule_preds

    def balance(self, data, balance_col='label', min_samples_per_class=20):
        # res: a dictionary returned by Teacher or Student.
        assert balance_col in data, f'res dict must does not contain {balance_col}. It has: {data.keys()}'
        df = pd.DataFrame()
        all_indices = np.arange(data[balance_col].shape[0])
        df['idx'] = all_indices
        df['labels'] = data[balance_col]
        self.logger.info("Data BEFORE balancing\n{}".format(df['labels'].value_counts()))
        min_class_support = df['labels'].value_counts().min()
        if min_class_support < min_samples_per_class:
            # there is at least one class that has less than min_samples_per_class samples
            # we will sample with replacement
            balanced_df = df.groupby('labels', group_keys=False).apply(lambda x: x.sample(min_samples_per_class, random_state=self.seed, replace=True))
        else:
            balanced_df = df.groupby('labels', group_keys=False).apply(lambda x: x.sample(min_class_support, random_state=self.seed))
        self.logger.info("Data AFTER balancing\n{}".format(balanced_df['labels'].value_counts()))

        keep_inds = balanced_df['idx'].tolist()
        balanced_data = filter_inds(data, keep_inds)
        return balanced_data

    def concatenate_data(self, data1, data2):
        """
        Filtering specific indices from a dataset
        :param data: dataset in dictionary format
        :param keep_inds: indices to keep
        :return: new dataset
        """
        data1 = deepcopy(data1)
        assert data1.keys() == data2.keys(), f'cannot concatenate different datasets: {data1.keys()}, {data1.keys()}'
        for k in data2.keys():
            try:
                data1[k].extend(data2[k])
            except:
                if scipy.sparse.issparse(data1[k]):
                    data1[k] = scipy.sparse.vstack((data1[k], data2[k]))
                elif len(data1[k].shape) == 1:
                    data1[k] = np.hstack((data1[k], data2[k]))
                else:
                    data1[k] = np.vstack((data1[k], data2[k]))

        if 'texts' in data1[k]:
            # make sure we convert texts back to a list of strings
            data1['texts'] = list(data1['texts'])
        return data1

    def sort_data(self, data, col=None):
        assert col in data, 'column={} not available in data'
        assert len(set(data[col])) == data['features'].shape[0], 'found duplicate indices...'
        sorted_ind = np.argsort(data['ids'])
        sorted_data = filter_inds(data, sorted_ind.tolist())
        return sorted_data

def filter_inds(data, keep_inds):
    """
    Filtering specific indices from a dataset
    :param data: dataset in dictionary format
    :param keep_inds: indices to keep
    :return: new dataset
    """
    new_data = {}
    for x, y in data.items():
        try:
            new_data[x] = y[keep_inds]
        except:
            try:
                new_data[x] = np.array(y)[keep_inds]
            except:
                new_data[x] = y.tocsr()[keep_inds]

    if 'texts' in new_data:
        # make sure we convert texts back to a list of strings
        new_data['texts'] = new_data['texts'].tolist()
    return new_data


