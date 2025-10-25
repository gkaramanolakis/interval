"""
Anonymized INTERVAL Code
"""
import numpy as np
from collections import Counter

def calc_acc(y):
    return np.sum(y) / len(y)

class RuleSuggester:
    """
    RuleSuggester: suggests candidate rules to a human
        * Step 1: generate rules with minimum coverage
        * Step 2: filter rules based on criteria
        * Step 3: up
    """

    def __init__(self, args, logger=None):
        self.args = args
        self.logger = logger
        self.seed = args.seed
        self.num_classes = args.dh.num_classes
        self.rule_family = args.rule_family
        self.abstain_ind = -1
        self.uncovered_precision = -1  # precision of a rule that does not cover any examples in the labeled dataset
        self.min_support = args.min_rule_coverage  # minimum rule coverage (absolute number)
        self.min_precision = args.min_rule_precision # 50 %
        self.rule_generator = self.init_rule_generator()

        self.all_rules = None
        self.suggested_rules = None
        self.accepted_rules = []

    def init_rule_generator(self):
        if self.rule_family == 'ngram':
            self.rule_generator = MyNGramLFGenerator(logger=self.logger, num_classes=self.num_classes,
                                                     vectorizer=None, ngram_range=(1, 3), min_acc_gain=None,
                                                     min_support=self.min_support, random_state=self.seed)
        elif self.rule_family == 'dnf':
            self.rule_generator = MyDNFGenerator(logger=self.logger, num_classes=self.num_classes,
                                                     vectorizer=None, ngram_range=(1, 3), min_acc_gain=None,
                                                     min_support=self.min_support, random_state=self.seed)
        else:
            raise(NotImplementedError(f'rule family is not supported: {self.rule_family}'))
        return self.rule_generator

    def apply(self, lf_list, data, return_rule_info=False):
        if len(lf_list) == 0:
            self.logger.info('No rules... RuleSuggester returning empty predictions')
            rule_preds = np.ones((len(data['texts']), 0))   # better than returning (N, 0) size array
        else:
            rule_preds = self.rule_generator.apply(lf_list, data=data)

        if not return_rule_info:
            return rule_preds
        rule_info = self.get_rule_info(lf_list)
        return rule_preds, rule_info

    def generate_rules(self, u_data, labeled_data):
        # Step 1: generate high-coverage rules
        self.all_rules = self.rule_generator.generate_rules(data=u_data)
        if self.all_rules is None or len(self.all_rules) == 0:
            self.logger.info("ERROR: RuleSuggester generated 0 rules... You need to adapt min_rule_coverage and min_rule_precision arguments")
            return None

        for r in self.all_rules:
            r.metadata['oracle_feedback'] = None

        # logger.info("In total, there are {} rules: {}".format(rulegen.n_feature, rulegen.idx_to_ngram))

        # Step2: update rule accuracies and filter high-accuracy rules using labeled data
        self.update(labeled_data=labeled_data)
        return self.apply(self.suggested_rules, u_data)

    def apply_accepted_rules(self, data, return_rule_info=False):
        return self.apply(self.accepted_rules, data, return_rule_info)

    def suggest_rules(self, data, return_rule_info=False):
        return self.apply(self.suggested_rules, data, return_rule_info)

    def update(self, labeled_data=None, oracle_feedback=None):
        # store Oracle feedback
        if oracle_feedback is not None:
            for rule_ind, rule_feedback in oracle_feedback.items():
                self.suggested_rules[rule_ind].metadata['oracle_feedback'] = rule_feedback
                if rule_feedback['decision'] == 1:
                    self.accepted_rules.append(self.suggested_rules[rule_ind])

        candidate_lfs = self.all_rules
        if labeled_data is not None:
            self.update_rule_info(candidate_lfs, labeled_data)
            self.update_rule_info(self.accepted_rules, labeled_data)

        self.print_rule_stats(lf_list=candidate_lfs, acc_thres=self.min_precision)
        candidate_lfs = [lf for lf in candidate_lfs if lf.acc >= self.min_precision]
        self.suggested_rules = [lf for lf in candidate_lfs if lf.metadata['oracle_feedback'] is None]

    def __get_rule_preds__(self, lf_list, data):
        rule_preds = self.rule_generator.apply(lf_list, data=data)
        assert len(lf_list) == rule_preds.shape[1]
        return rule_preds


    def update_rule_info(self, lf_list, labeled_data):
        if len(lf_list) == 0:
            return
        self.logger.info('updating rule accuracies')
        labels = labeled_data['labels']

        assert -1 not in set(labels), 'you need to provide labeled data to estimate rule accuracies'

        # Update rule accuracy estimates based on the given labeled data
        rule_preds = self.__get_rule_preds__(lf_list, data=labeled_data)

        for rule_ind in range(rule_preds.shape[1]):
            covered_idx = rule_preds[:, rule_ind] != self.abstain_ind
            coverage_labeled = np.sum(covered_idx)
            y = np.array(rule_preds[:, rule_ind] == labels, dtype=np.int)
            exist_acc = calc_acc(y[covered_idx])

            exist_acc = exist_acc if not np.isnan(exist_acc) else self.uncovered_precision  # 1.0
            lf_list[rule_ind].acc = exist_acc
            lf_list[rule_ind].coverage_labeled = coverage_labeled
        #self.print_rule_stats(lf_list=lf_list, acc_thres=self.min_precision)
        return

    def get_rule_info(self, lf_list):
        rule_info = []
        for lf in lf_list:
            info = {
                'description': lf.description,
                'label': lf.label,
                'precision': lf.acc,
                'coverage_unlabeled': lf.coverage_unlabeled,
                'coverage_labeled': lf.coverage_labeled,
            }
            rule_info.append(info)
        return rule_info

    def print_rule_stats(self, lf_list, acc_thres=0.75):
        rule_accs = [rule.acc for rule in lf_list]
        acc_histogram, axis = np.histogram(rule_accs, bins=10, range=(0, 1))

        self.logger.info("\t\t **** accepted rules ****")
        if len(self.accepted_rules) == 0:
            self.logger.info('no accepted rules')
        for rule_ind in range(len(self.accepted_rules)):
            lf = self.accepted_rules[rule_ind]
            self.logger.info("rule {} (prec={:.2f}, l_cov={}, u_cov={}): {}".format(lf.id, lf.acc, lf.coverage_labeled, lf.coverage_unlabeled, lf.description))

        self.logger.info("\t\t **** candidate rules ****")
        num_nan_acc = len([x for x in rule_accs if x == -1])
        self.logger.info("nan:\t{}".format(num_nan_acc))

        for i, h in enumerate(acc_histogram):
            self.logger.info("[{:.1f}-{:.1f}]: {}".format(axis[i], axis[i + 1], h))

        self.logger.info("accurate rules per class:")
        rule_labels = [rule.label for rule in lf_list if rule.acc >= acc_thres]
        c = Counter(rule_labels)
        for label in range(self.num_classes):
            self.logger.info("class {}: {} rules".format(label, c[label]))