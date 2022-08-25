"""
ZFJ 2022
used for evaluation during training
"""
from rouge_score import rouge_scorer, scoring

def compute_rouge(predictions, references, rouge_types=None, use_agregator=True, use_stemmer=False):
    if rouge_types is None:
        rouge_types = ["rouge1", "rouge2", "rougeL", "rougeLsum"]

    scorer = rouge_scorer.RougeScorer(rouge_types=rouge_types, use_stemmer=use_stemmer)
    if use_agregator:
        aggregator = scoring.BootstrapAggregator()
    else:
        scores = []

    for ref, pred in zip(references, predictions):
        score = scorer.score(ref, pred)
        if use_agregator:
            aggregator.add_scores(score)
        else:
            scores.append(score)

    if use_agregator:
        result = aggregator.aggregate()
    else:
        result = {}
        for key in scores[0]:
            result[key] = list(score[key] for score in scores)

    return result
