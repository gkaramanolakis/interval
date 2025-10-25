"""
Anonymized INTERVAL Code
"""

from copy import deepcopy
from Evaluator import Evaluator

class Oracle:
    """
    Oracle: simulates human response
    """

    def __init__(self, args, logger=None, name=None):
        self.args = args
        self.logger = logger
        self.seed = args.seed
        self.num_classes = args.dh.num_classes
        self.data = None
        self.evaluator = Evaluator(args, logger)
        #self.precision_threshold = 0.75
        self.precision_threshold = args.oracle_precision_threshold

    def init(self, train_data, u_data):
        oracle_data = deepcopy(u_data)
        oracle_data['labels'] = oracle_data['orig_labels']
        oracle_data = self.args.dh.concatenate_data(oracle_data, train_data)
        self.data = oracle_data

    def answer(self, preds):
        """
        A binary accept/reject answer depending on certain criteria.
        :return:
        """
        active = (preds != -1)

        if sum(active) == 0:
            self.logger.info("WARNING: asked Oracle to accept a rule with coverage=0. Rejecting it...")
            return 0, "coverage=0"

        precision = sum(preds[active] == self.data['labels'][active]) / sum(active)
        if precision >= self.precision_threshold:
            ret, exp = 1, "prec = {:.2f}>={:.2f}".format(precision, self.precision_threshold)
        else:
            ret, exp = 0, "prec = {:.2f}<{:.2f}".format(precision, self.precision_threshold)
        return ret, exp


