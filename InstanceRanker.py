"""
Anonymized INTERVAL Code
Parts of the code where borrowed from https://github.com/mourga/contrastive-active-learning
"""

import collections
from tqdm import tqdm
import numpy as np
import scipy
import torch
import torch.nn.functional as F


# functions that require many arguments
from InstanceRanker_utils import select_alps, badge, bertKM, BB_acquisition

# functions that require logits
from InstanceRanker_utils import least_confidence, max_entropy, bald, mean_std

# functions that require probability distributions (note: not implemented)
from InstanceRanker_utils import margin_of_confidence, ratio_of_confidence, _proba_uncertainty, _proba_neg_margin, _proba_entropy
from HierarchicalSampling import HierarchicalSampling as HS
from HierarchicalSampling import Dataset as HDataset
from HierarchicalSampling import UncertaintySampling

def proba_to_logits(proba):
    try:
        logits = np.log(proba / (1 - proba))
    except:
        self.logger.info("ERROR: could not convert probabilities into logits")
        logits = proba
    return logits

class InstanceRanker:
    def __init__(self, args, logger=None, active_method=None):
        self.args = args
        self.seed = args.seed
        self.logger = logger
        self.num_classes = args.dh.num_classes
        assert 'active_' in args.type or active_method is not None, 'Instance ranker needs --type active_*'
        self.active_method = args.type.split('active_')[-1] if active_method is None else active_method
        self.logger.info(f"Setting up InstanceRanker={self.active_method}")
        self.qs = None


    def active_select(self, train_data, u_data, u_proba, train_proba):
        if 'new' in self.active_method:
            return self.active_select_(u_proba)
        elif 'hierarchical' in self.active_method or 'albl' in self.active_method or 'eer' in self.active_method:
            return self.hierarchical_sampling(train_data, u_data,
                                              u_proba, train_proba)
        elif self.active_method.split('_')[-1] == 'cal':
            return self.contrastive_acquisition(train_data, u_data, u_proba, train_proba)
        else:
            return self.calculate_uncertainty(u_proba)


    def prepare_data(self, train_data, u_data, u_proba, train_proba):
        # Sort examples based on their ids to ensure that they have always the same order across iterations
        args = self.args
        train_data['proba'] = train_proba
        u_data['proba'] = u_proba
        data = args.dh.concatenate_data(train_data, u_data)
        data = args.dh.sort_data(data, col='ids')
        return data

    def get_X_y(self, data):
        if scipy.sparse.issparse(data['features']):
            X = list(data['features'].toarray())
        else:
            X = list(data['features'])
        y = list(data['labels'])
        return X, y

    def init_hierarchical(self, data):
        args = self.args
        classes = np.arange(self.num_classes).tolist()
        method = self.active_method.split('hierarchical_')[-1].split('_')[0]
        assert method in ['random', 'subsampling', 'nosubsampling'], f'unknown hierarchical sampling method: {method}'

        X, y = self.get_X_y(data)
        self.ds = HDataset(X, y)

        active_selecting = True if method not in ['random'] else False
        subsample_qs = None if method not in ['subsampling'] else 'uncertainty'
        self.logger.info(f"active: {active_selecting}")
        self.logger.info(f"subsample: {subsample_qs}")
        if subsample_qs is not None:
            subsample_arg = self.active_method.split('_')[-1]
            assert subsample_arg in ['lc', 'sm', 'entropy'], f'invalid subsample_arg: {subsample_arg}'
            subsample_qs = UncertaintySampling(self.ds, method=subsample_arg)
            self.logger.info(f"subsample function : uncertainty-{subsample_arg}")

        self.qs = HS(self.ds, classes, active_selecting=active_selecting, subsample_qs=subsample_qs,
                     random_state=self.seed, logger=self.logger)
        return

    def hierarchical_sampling(self, train_data, u_data, u_proba, train_proba):
        # source: https://github.com/ntucllab/libact/blob/2432b512bca3bdf97df6d2d7c40c5ffba5b46094/libact/query_strategies/multiclass/hierarchical_sampling.py
        """
        Supported hierarchical methods
         -- active_hierarchical_random: random passive subsampling
         -- active_hierarchical_nosubsampling: random active subsampling
         -- active_hierarchical_subsampling: use entropy-based uncertainty to subsample

         Supported active learning by learning method
         -- albl
        :return:
        """
        args = self.args
        sample_size = args.annotations_per_it

        data = self.prepare_data(train_data, u_data, u_proba, train_proba)

        if self.qs is None:
            if 'hierarchical' in self.active_method:
                self.init_hierarchical(data)
            else:
                raise(BaseException('not implemented'))

        global_sampled_ind = []
        for _ in range(sample_size):
            ask_inds = self.qs.make_query(proba=data['proba'], num_samples=1)
            assert len(ask_inds) == 1, 'algorithm returned more than 1 sample'
            ask_ind = ask_inds[0]
            gt_label = data['orig_labels'][ask_ind]
            #self.logger.info(f"UPDATE {ask_ind}: {gt_label}")
            self.ds.update(ask_ind, gt_label)
            global_sampled_ind.append(ask_ind)


        assert all([x >= 0 for x in global_sampled_ind]), "Indexing ERROR: there exist negative indices"

        # re-construct the right indices for unlabeled data
        sampled_ids = data['ids'][global_sampled_ind]
        assert len(set(train_data['ids']) & set(sampled_ids)) == 0, "Algorithm ERROR: sampled data from the already labeled data"
        u_data_ids = u_data['ids'].tolist()
        unlabeled_ind = [u_data_ids.index(x) for x in sampled_ids.tolist()]
        sampled_ind = unlabeled_ind
        if len(sampled_ind) < sample_size:
            self.logger.info("WARNING: returned fewer sample data")
        return sampled_ind

    def active_select_(self, u_proba):
        args = self.args
        method = self.active_method
        if 'entropy' in method:
            ret = _proba_entropy(u_proba)
        elif 'margin' in method:
            ret = _proba_neg_margin(u_proba)
        elif 'uncertainty' in method:
            ret = _proba_uncertainty(u_proba)
        else: NotImplementedError
        scores = ret
        sampled_ind = get_sampled_ind(scores, args.annotations_per_it, args.reverse)
        return sampled_ind

    def contrastive_acquisition(self, train_data, u_data, u_proba, train_proba):
        args = self.args
        u_proba = torch.from_numpy(u_proba)
        train_proba = torch.from_numpy(train_proba)
        u_logits = proba_to_logits(u_proba)
        train_logits = proba_to_logits(train_proba)

        ret = contrastive_acquisition(
            args=args,
            annotations_per_iteration=args.annotations_per_it,
            logits_dpool=u_logits,
            logits_train=train_logits,
            dtrain_reprs=train_data['features'],
            dpool_reprs=u_data['features'],
            dtrain_labels=train_data['labels'])
        sampled_ind = ret
        return sampled_ind

    def calculate_uncertainty(self, proba):
        args = self.args
        proba = torch.from_numpy(proba)
        try:
            logits = np.log(proba / (1 - proba))
        except:
            self.logger.info("ERROR: could not convert probabilities into logits")
            logits = proba
        ret = calculate_uncertainty(
            args=args,
            logits=logits,
            prob_dist=proba,
            method=self.active_method,
            annotations_per_it=args.annotations_per_it,
            device=args.device,
            iteration=None,
            task=args.dataset,
            representations=None,
            candidate_inds=None, #X_train_remaining_inds,
            labeled_inds=None, # X_train_current_inds,
            discarded_inds=None, # X_discarded_inds,
            original_inds=None, #X_train_original_inds,
            model=None, # train_results['model'], # (a transformers model instance)
            X_original=None,
            y_original=None
        )
        sampled_ind = ret
        return sampled_ind


def get_sampled_ind(uncertainty_scores, annotations_per_it, reverse):
    sampled_ind = np.argpartition(uncertainty_scores, -annotations_per_it)[-annotations_per_it:]
    if reverse:
            sampled_ind = np.argpartition(uncertainty_scores, annotations_per_it)[:annotations_per_it]
    return sampled_ind

def multi_argmax(values: np.ndarray, n_instances: int = 1) -> np.ndarray:
    """
    Selects the indices of the n_instances highest values.
    This functions does the same job as get_sampled_ind() but returns a different order of the results
    Args:
        values: Contains the values to be selected from.
        n_instances: Specifies how many indices to return.
    Returns:
        The indices of the n_instances largest values.
    """
    assert n_instances <= values.shape[0], 'n_instances must be less or equal than the size of utility'

    max_idx = np.argpartition(-values, n_instances-1, axis=0)[:n_instances]
    return max_idx


def calculate_uncertainty(args, method, logits, prob_dist, annotations_per_it, device,
                          iteration, task=None,
                          oversampling=False,
                          representations=None,
                          candidate_inds=None,
                          labeled_inds=None,
                          discarded_inds=None,
                          original_inds=None,
                          model=None,
                          X_original=None, y_original=None):
    """
    Selects and performs uncertainty-based acquisition.
    :param method: uncertainty-based acquisition function. options:
        - 'least_conf' for least confidence
        - 'margin_conf' for margin of confidence
        - 'ratio_conf' for ratio of confidence
    :param prob_dist: output probability distribution
    :param logits: output logits
    :param annotations_per_it: number of samples (to be sampled)
    :param D_lab: [(X_labeled, y_labeled)] labeled data
    :param D_unlab: [(X_unlabeled, y_unlabeled)] unlabeled data
    :return:
    """
    #

    init_unlabeled_data = logits.shape[0]

    if method not in ['random', 'alps', 'badge', 'FTbertKM']:
        if type(logits) is list and logits != []:
            assert init_unlabeled_data == logits[0].size(0), "logits {}, inital unlabaled data {}".format(logits[0].size(0), init_unlabeled_data)
        elif type(logits) != []:
            assert init_unlabeled_data == len(logits)

    if method == 'random':
        sampled_ind = np.random.choice(init_unlabeled_data, annotations_per_it, replace=False)
        return sampled_ind
    elif method == 'least_conf':
        uncertainty_scores = least_confidence(logits)
    elif method == 'ratio_conf':
        uncertainty_scores = ratio_of_confidence(prob_dist)
    elif method == 'margin_conf':
        uncertainty_scores = margin_of_confidence(prob_dist)
    elif method == 'entropy':
        uncertainty_scores = max_entropy(logits)
    elif method == 'std':
        uncertainty_scores = mean_std(logits)
    elif method == 'bald':
        uncertainty_scores = bald(logits)
    elif method == 'batch_bald':
        uncertainty_scores, sampled_ind = BB_acquisition(logits.unsqueeze(1), device, annotations_per_it)
        assert len(set(sampled_ind)) == len(sampled_ind), "unique {}, total {}".format(len(set(sampled_ind)), len(sampled_ind))
        assert len(sampled_ind) == annotations_per_it, "sampled ind {}, acquisition size {}".format(len(sampled_ind), annotations_per_it)
    else:
        raise ValueError('Acquisition function {} not implemented yet check again!'.format(method))
    sampled_ind = get_sampled_ind(uncertainty_scores, annotations_per_it, args.reverse)
    return sampled_ind

from sklearn.neighbors import KNeighborsClassifier
def contrastive_acquisition(args, annotations_per_iteration, logits_dpool, logits_train,
                            dtrain_reprs=None, dpool_reprs=None,  dtrain_labels=None):
    """

    :param args: arguments (such as flags, device, etc)
    :param annotations_per_iteration: acquisition size
    :param X_original: list of all data
    :param y_original: list of all labels
    :param labeled_inds: indices of current labeled/training examples
    :param candidate_inds: indices of current unlabeled examples (pool)
    :param discarded_inds: indices of examples that should not be considered for acquisition/annotation
    :param original_inds: indices of all data (this is a list of indices of the X_original list)
    :param tokenizer: tokenizer
    :param train_results: dictionary with results from training/validation phase (for logits) of training set
    :param results_dpool: dictionary with results from training/validation phase (for logits) of unlabeled set (pool)
    :param logits_dpool: logits for all examples in the pool
    :param bert_representations: representations of pretrained bert (ablation)
    :param train_dataset: the training set in the tensor format
    :param model: the fine-tuned model of the iteration
    :param tfidf_dtrain_reprs: tf-idf representations of training set (ablation)
    :param tfidf_dpool_reprs: tf-idf representations of unlabeled set (ablation)
    :return:
    """
    """
    CAL (Contrastive Active Learning)
    Acquire data by choosing those with the largest KL divergence in the predictions between a candidate dpool input
     and its nearest neighbours in the training set.
     Our proposed approach includes:
     args.cls = True
     args.operator = "mean"
     the rest are False. We use them (True) in some experiments for ablation/analysis
     args.mean_emb = False
     args.mean_out = False
     args.bert_score = False 
     args.tfidf = False 
     args.reverse = False
     args.knn_lab = False
     args.ce = False
    :return:
    """

    #####################################################
    # Contrastive Active Learning (CAL)
    #####################################################
    neigh = KNeighborsClassifier(n_neighbors=args.num_nei)
    neigh.fit(X=dtrain_reprs, y=dtrain_labels)
    criterion = torch.nn.KLDivLoss(reduction='none') if not args.ce else torch.nn.CrossEntropyLoss()

    kl_scores = []
    #num_adv = 0
    distances = []
    for unlab_i, candidate in enumerate(tqdm(zip(dpool_reprs, logits_dpool), desc="Finding neighbours for every unlabeled data point")):
        # find indices of closesest "neighbours" in train set
        #unlab_representation, unlab_logit = candidate

        if scipy.sparse.issparse(candidate[0]):
            distances_, neighbours = neigh.kneighbors(X=candidate[0], return_distance=True)
        else:
            distances_, neighbours = neigh.kneighbors(X=[candidate[0]], return_distance=True)
        distances.append(distances_[0])

        # calculate score
        neigh_prob = F.softmax(logits_train[neighbours], dim=-1)

        """
        if args.ce:
            kl = np.array([criterion(unlab_logit.view(-1, args.num_classes), label.view(-1)) for label in
                           labeled_neighbours_labels])
        else:
        """
        uda_softmax_temp = 1
        candidate_log_prob = F.log_softmax(candidate[1] / uda_softmax_temp, dim=-1)
        kl = np.array([torch.sum(criterion(candidate_log_prob, n), dim=-1).numpy() for n in neigh_prob])

        # confidence masking
        #if args.conf_mask:
        #    conf_mask = torch.max(neigh_prob, dim=-1)[0] > args.conf_thresh
        #    conf_mask = conf_mask.type(torch.float32)
        #    kl = kl * conf_mask.numpy()
        if args.operator == "mean":
            kl_scores.append(kl.mean())
        elif args.operator == "max":
            kl_scores.append(kl.max())
        elif args.operator == "median":
            kl_scores.append(np.median(kl))

    #distances = np.array([np.array(xi) for xi in distances])

    #logger.info('Total Different predictions for similar inputs: {}'.format(num_adv))

    if args.reverse:  # if True select opposite (ablation)
        selected_inds = np.argpartition(kl_scores, annotations_per_iteration)[:annotations_per_iteration]
    else:
        selected_inds = np.argpartition(kl_scores, -annotations_per_iteration)[-annotations_per_iteration:]
    #############################################################################################################################


    # map from dpool inds to original inds
    sampled_ind = list(np.arange(dpool_reprs.shape[0])[selected_inds])
    return sampled_ind