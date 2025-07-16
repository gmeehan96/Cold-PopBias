import math
import numpy as np
from collections import defaultdict
import pandas as pd


class Metric(object):
    def __init__(self):
        pass

    @staticmethod
    def hits(origin, res):
        hit_count = {}
        for user in origin:
            items = list(origin[user].keys())
            predicted = [item[0] for item in res[user]]
            hit_count[user] = len(set(items).intersection(set(predicted)))
        return hit_count

    @staticmethod
    def hit_ratio(origin, hits):
        """
        Note: This type of hit ratio calculates the fraction:
         (# retrieved interactions in the test set / #all the interactions in the test set)
        """
        total_num = 0
        for user in origin:
            items = list(origin[user].keys())
            total_num += len(items)
        hit_num = 0
        for user in hits:
            hit_num += hits[user]
        return round(hit_num / total_num, 5)

    @staticmethod
    def precision(hits, N):
        prec = sum([hits[user] for user in hits])
        return round(prec / (len(hits) * N), 5)

    @staticmethod
    def recall(hits, origin):
        recall_list = [hits[user] / len(origin[user]) for user in hits]
        recall = round(sum(recall_list) / len(recall_list), 5)
        return recall

    @staticmethod
    def F1(prec, recall):
        if (prec + recall) != 0:
            return round(2 * prec * recall / (prec + recall), 5)
        else:
            return 0

    @staticmethod
    def MAE(res):
        error = 0
        count = 0
        for entry in res:
            error += abs(entry[2] - entry[3])
            count += 1
        if count == 0:
            return error
        return round(error / count, 5)

    @staticmethod
    def RMSE(res):
        error = 0
        count = 0
        for entry in res:
            error += (entry[2] - entry[3]) ** 2
            count += 1
        if count == 0:
            return error
        return round(math.sqrt(error / count), 5)

    @staticmethod
    def NDCG(origin, res, N):
        sum_NDCG = 0
        for user in res:
            DCG = 0
            IDCG = 0
            # 1 = related, 0 = unrelated
            for n, item in enumerate(res[user]):
                if item[0] in origin[user]:
                    DCG += 1.0 / math.log(n + 2, 2)
            for n, item in enumerate(list(origin[user].keys())[:N]):
                IDCG += 1.0 / math.log(n + 2, 2)
            sum_NDCG += DCG / IDCG
        return round(sum_NDCG / len(res), 5)

    @staticmethod
    def MAP(origin, res, N):
        sum_prec = 0
        for user in res:
            hits = 0
            precision = 0
            for n, item in enumerate(res[user]):
                if item[0] in origin[user]:
                    hits += 1
                    precision += hits / (n + 1.0)
            sum_prec += precision / min(len(origin[user]), N)
        return round(sum_prec / len(res), 5)


def ranking_evaluation(origin, res, N):
    measure = []
    performance = [[] for _ in N]
    for i, n in enumerate(N):
        predicted = {}
        for user in res:
            predicted[user] = res[user][:n]
        indicators = []
        if len(origin) != len(predicted):
            print(
                f"ground-truth set size: {len(origin)}, predicted set size: {len(predicted)}"
            )
            print("The Lengths of ground-truth set and predicted set do not match!")
            exit(-1)
        hits = Metric.hits(origin, predicted)

        recall = Metric.recall(hits, origin)
        indicators.append("Recall:" + str(recall) + "\n")

        NDCG = Metric.NDCG(origin, predicted, n)
        indicators.append("NDCG:" + str(NDCG) + "\n")

        measure.append("Top " + str(n) + "\n")
        measure += indicators

        performance[i].append(recall)
        performance[i].append(NDCG)
    return measure, performance


def gini_coefficient(x):
    """Compute Gini coefficient of array of values"""
    diffsum = 0
    for i, xi in enumerate(x[:-1], 1):
        diffsum += np.sum(np.abs(xi - x[i:]))
    return diffsum / (len(x) ** 2 * np.mean(x))


def mdg_i(origin, res):
    mdgs = defaultdict(float)
    item_pops = defaultdict(float)
    for user, pred in res.items():
        for item in origin[user]:
            item_pops[item] += 1
        for n, item in enumerate(pred):
            if item[0] in origin[user]:
                mdgs[item[0]] += 1 / math.log(n + 2, 2)
    return {i: mdgs[i] / pop for i, pop in item_pops.items()}


def get_gini_div(model_recs, holdout_items):
    model_rec_items = [i[0] for v in model_recs.values() for i in v]
    pred_counts = pd.Series(model_rec_items).value_counts()
    pred_counts_rel = np.array(
        [pred_counts.loc[i] if i in pred_counts.index else 0 for i in holdout_items]
    )

    pred_ratios = pred_counts_rel / pred_counts_rel.sum()
    return 1 - gini_coefficient(pred_ratios)
