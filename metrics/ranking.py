'''
ZFJ 2022
Code for Maximal Marginal Ranking
'''
import numpy as np
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity


def reduce_batch(scores_dict, batch_size, limit):
    scores_reduced_by_max = dict()
    scores_reduced_by_max_sorted = dict()
    for s_type in scores_dict.keys():
        scores = scores_dict[s_type]
        divided_scores = [scores[i:i+batch_size] for i in range(0, len(scores), batch_size)]
        scores_reduced_by_max[s_type] = [max(s[:limit]) for s in divided_scores]
        scores_reduced_by_max_sorted[s_type] = [0 for _ in divided_scores]
    return scores_reduced_by_max, scores_reduced_by_max_sorted


def reduce_batch_with_rank(scores_dict, predictions, batch_size, limit):
    divided_predictions = [predictions[i:i+batch_size] for i in range(0, len(predictions), batch_size)]
    _, sorted_batch_indexs = ranked_predictions(divided_predictions, limit)
    scores_reduced_by_max = dict()
    scores_reduced_by_max_sorted = dict()
    for s_type in scores_dict.keys():
        scores = scores_dict[s_type]
        divided_scores = [scores[i:i+batch_size] for i in range(0, len(scores), batch_size)]
        scores_reduced_by_max[s_type] = [max(s[:limit]) for s in divided_scores]
        sorted_batch_scores = []
        for batch_idx, inner_idx in enumerate(sorted_batch_indexs):
            sorted_batch_scores.append([divided_scores[batch_idx][i] for i in inner_idx])
        scores_reduced_by_max_sorted[s_type] = [max(s[:limit]) for s in sorted_batch_scores]
    return scores_reduced_by_max, scores_reduced_by_max_sorted


def build_bi_gram_vector(predictions):
    bi_gram_predictions = [] 
    bi_grams = []
    for predict in predictions:
        bi_gram_result = list(zip(*[predict[i:] for i in range(2)]))
        bi_grams += bi_gram_result
        bi_gram_predictions.append(bi_gram_result)
    bi_gram_count = Counter(bi_grams) 
    bi_gram_ids = {} 
    for i, bi_gram_key in enumerate(bi_gram_count.keys()):
        bi_gram_ids[bi_gram_key] = i
    prediction_vectors = []
    for p in bi_gram_predictions:
        initial = np.zeros(len(bi_gram_count.keys()))
        for b in p:
            initial[bi_gram_ids[b]] = bi_gram_count[b]
        prediction_vectors.append(initial)
    return np.array(prediction_vectors)


def ranked_predictions(batched_predictions, limit):
    ranked_batch_predictions = []
    ranked_batch_indexs = []
    for i in range(len(batched_predictions)):
        p_batch = batched_predictions[i]
        ranked_predictions, ranked_indexs = bi_gram_cosine_mmr(p_batch, limit)
        ranked_batch_predictions.append(ranked_predictions)
        ranked_batch_indexs.append(ranked_indexs)
    return ranked_batch_predictions, ranked_batch_indexs



def scoring_prediction_bi_gram(predictions):
    with_weight = []
    bi_gram_predictions = []
    bi_grams = []
    for predict in predictions:
        bi_gram_result = list(zip(*[predict[i:] for i in range(2)]))
        bi_grams += bi_gram_result
        bi_gram_predictions.append(bi_gram_result)
    bi_gram_count = Counter(bi_grams)
    for i in range(len(predictions)):
        weight = sum([bi_gram_count[bg] for bg in bi_gram_predictions[i]])
        with_weight.append((predictions[i], i, weight))
    return with_weight


def mmr_inner(all_preds, sampled_preds, sampled_indexs, sampled_cos):
    cos_scores_with_last = cosine_similarity([sampled_preds[-1]], all_preds)  # shape (1, n_sample)
    sampled_cos[sampled_indexs[-1]] = cos_scores_with_last[0]

    cos_scores_all = []
    for idx in range(len(all_preds)):
        candidate_score = 0
        for j in sampled_indexs:
            candidate_score += sampled_cos[j][idx]
        cos_scores_all.append(candidate_score)

    least_similar_idx = 0
    least_score = 1
    for i, score in enumerate(cos_scores_all):
        if i in sampled_indexs:
            continue
        if score < least_score:
            least_score = score
            least_similar_idx = i

    sampled_preds.append(all_preds[least_similar_idx])
    sampled_indexs.append(least_similar_idx)
    return sampled_preds, sampled_indexs, sampled_cos


def get_max_bgf(preds):
    # preds nparray (n_sample, n_bi_grams)
    bgf_values = np.array(list(map(lambda x: sum(x), preds)))
    max_index = np.argmax(bgf_values)
    sampled_preds = [preds[max_index]]
    return [max_index], sampled_preds


def bi_gram_cosine_mmr(predictions, limit):
    pred_vectors = build_bi_gram_vector(predictions) 
    sampled_indexs, sampled_preds = get_max_bgf(pred_vectors)  
    sampled_cos = dict()
    while len(sampled_preds) < limit:
        sampled_preds, sampled_indexs, sampled_cos = mmr_inner(pred_vectors, sampled_preds, sampled_indexs, sampled_cos)

    return sampled_preds, sampled_indexs
