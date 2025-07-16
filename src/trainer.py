import sys

sys.path.append("..")
import time
import copy
from numba import jit
import heapq
import numpy as np
from data_utils import ColdStartDataBuilder
from eval_utils import ranking_evaluation, mdg_i, get_gini_div


@jit(nopython=True)
def find_k_largest(K, candidates):
    n_candidates = []
    for iid, score in enumerate(candidates[:K]):
        n_candidates.append((score, iid))
    heapq.heapify(n_candidates)
    for iid, score in enumerate(candidates[K:]):
        if score > n_candidates[0][0]:
            heapq.heapreplace(n_candidates, (score, iid + K))
    n_candidates.sort(key=lambda d: d[0], reverse=True)
    ids = [item[1] for item in n_candidates]
    k_largest_scores = [item[0] for item in n_candidates]
    return ids, k_largest_scores


class BaseColdStartTrainer(object):
    def __init__(
        self,
        args,
        training_set,
        warm_valid_set,
        cold_valid_set,
        overall_valid_set,
        warm_test_set,
        cold_test_set,
        overall_test_set,
        user_num,
        item_num,
        warm_user_idx,
        warm_item_idx,
        cold_user_idx,
        cold_item_idx,
        device,
        user_content=None,
        item_content=None,
        **kwargs,
    ):
        super(BaseColdStartTrainer, self).__init__()
        self.args = args
        self.data = ColdStartDataBuilder(
            training_set,
            warm_valid_set,
            cold_valid_set,
            overall_valid_set,
            warm_test_set,
            cold_test_set,
            overall_test_set,
            user_num,
            item_num,
            warm_user_idx,
            warm_item_idx,
            cold_user_idx,
            cold_item_idx,
            user_content,
            item_content,
        )
        self.bestPerformance = []
        top = args.topN.split(",")
        self.topN = [int(num) for num in top]
        self.max_N = max(self.topN)
        self.model_name = args.model
        self.dataset_name = args.dataset
        self.emb_size = args.emb_size
        self.maxEpoch = args.epochs
        self.batch_size = args.bs
        self.lr = args.lr
        self.reg = args.reg
        self.device = device
        self.result = []

    def print_basic_info(self):
        print("*" * 80)
        print("Model: ", self.model_name)
        print("Dataset: ", self.dataset_name)
        print("Embedding Dimension:", self.emb_size)
        print("Maximum Epoch:", self.maxEpoch)
        print("Learning Rate:", self.lr)
        print("Batch Size:", self.batch_size)
        print("*" * 80)

    def timer(self, start=True):
        if start:
            self.train_start_time = time.time()
        else:
            self.train_end_time = time.time()

    def train(self):
        pass

    def predict(self, u):
        pass

    def save(self):
        pass

    def valid(self, valid_type="all", final=False):
        if valid_type == "warm":
            valid_set = self.data.warm_valid_set
        elif valid_type == "cold":
            valid_set = self.data.cold_valid_set
        elif valid_type == "all":
            valid_set = self.data.overall_valid_set
        else:
            raise ValueError("Invalid valid type!")

        rec_list = {}
        user_count = len(valid_set)
        for i, user in enumerate(valid_set):
            candidates = self.predict(user)
            if final:
                candidates = copy.deepcopy(candidates)
            rated_list, li = self.data.user_rated(user)
            if len(rated_list) != 0:
                candidates[self.data.get_item_id_list(rated_list)] = -10e8
            if valid_type == "warm" and self.args.cold_object == "item":
                candidates[self.data.mapped_cold_item_idx] = -10e8
            if valid_type == "cold" and self.args.cold_object == "item":
                candidates[self.data.mapped_warm_item_idx] = -10e8

            ids, scores = find_k_largest(self.max_N, candidates)
            item_names = [self.data.id2item[iid] for iid in ids]
            rec_list[user] = list(zip(item_names, scores))
        return rec_list

    def test(self, test_type="warm"):
        if test_type == "warm":
            test_set = self.data.warm_test_set
        elif test_type == "cold":
            test_set = self.data.cold_test_set
        elif test_type == "all":
            test_set = self.data.overall_test_set
        else:
            raise ValueError("Invalid test type!")

        rec_list = {}
        user_count = len(test_set)
        for i, user in enumerate(test_set):
            candidates = copy.deepcopy(self.predict(user))
            rated_list, li = self.data.user_rated(user)
            if len(rated_list) != 0:
                candidates[self.data.get_item_id_list(rated_list)] = -10e8
            if test_type == "warm" and self.args.cold_object == "item":
                candidates[self.data.mapped_cold_item_idx] = -10e8
            if test_type == "cold" and self.args.cold_object == "item":
                candidates[self.data.mapped_warm_item_idx] = -10e8
            ids, scores = find_k_largest(self.max_N, candidates)
            item_names = [self.data.id2item[iid] for iid in ids]
            rec_list[user] = list(zip(item_names, scores))
        return rec_list

    def fast_evaluation_quiet(self, epoch, valid_type="all"):
        if valid_type == "warm":
            valid_set = self.data.warm_valid_set
        elif valid_type == "cold":
            valid_set = self.data.cold_valid_set
        elif valid_type == "all":
            valid_set = self.data.overall_valid_set
        else:
            raise ValueError("Invalid evaluation type!")
        rec_list = self.valid(valid_type)
        measure, _ = ranking_evaluation(valid_set, rec_list, [self.max_N])
        if len(self.bestPerformance) > 0:
            count = 0
            performance = {}
            for m in measure[1:]:
                k, v = m.strip().split(":")
                performance[k] = float(v)
            for k in self.bestPerformance[1]:
                if self.bestPerformance[1][k] > performance[k]:
                    count += 1
                else:
                    count -= 1
            if count < 0:
                self.bestPerformance[1] = performance
                self.bestPerformance[0] = epoch + 1
                self.save()
        else:
            self.bestPerformance.append(epoch + 1)
            performance = {}
            for m in measure[1:]:
                k, v = m.strip().split(":")
                performance[k] = float(v)
            self.bestPerformance.append(performance)
            self.save()
        measure = [m.strip() for m in measure[1:]]
        return measure

    def full_evaluation(self, rec_list, test_type="cold"):
        test_set = self.data.cold_test_set

        self.result, test_performance = ranking_evaluation(
            test_set, rec_list, self.topN
        )
        model_mdg = mdg_i(test_set, rec_list)
        mdg_sorted = sorted(list(model_mdg.values()))

        mdg_min = np.mean(mdg_sorted[: int(len(model_mdg) * 0.8)])
        mdg_max = np.mean(mdg_sorted[int(len(model_mdg) * 0.95) :])
        mdg_all = np.mean(mdg_sorted)
        gini_div = get_gini_div(rec_list, holdout_items=self.data.cold_test_set_item)

        self.result += "MDG-Min80:" + str(round(mdg_min, 5)) + "\n"
        self.result += "MDG-Max5:" + str(round(mdg_max, 5)) + "\n"
        self.result += "MDG-All:" + str(round(mdg_all, 5)) + "\n"
        self.result += "Gini-Div:" + str(round(gini_div, 5)) + "\n"

        print("*" * 80)
        print(
            f"[{test_type} setting] The result of %s:\n%s"
            % (self.model_name, "".join(self.result))
        )

    def eval_test(self):
        cold_rec_list = self.test(test_type="cold")
        self.full_evaluation(cold_rec_list, test_type="cold")
