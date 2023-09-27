from math import log


def calculate(all_exps, outputs, topk=10):
    idcg, dcg, ndcg, pre, rec = 0.0, 0.0, 0.0, 0.0, 0.0
    for exps, rank_list in zip(all_exps, outputs):
        hits = 0.0
        dcg = 0.0
        idcg = 0.0
        for idx, exp in enumerate(rank_list):
            if exp in exps:
                hits += 1
                dcg += 1 / (log(idx + 2) / log(2))
        for i in range(min(len(exps), len(rank_list))):
            idcg += 1 / (log(i + 2) / log(2))
        ndcg += dcg / idcg
        pre += hits / topk
        rec += hits / len(exps)
    return ndcg, pre, rec, len(outputs)
