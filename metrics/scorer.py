'''
ZFJ 2022
Score with BLEU, Rouge
Scored with lower case, tokenize in our own way
'''
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge
from joblib import Parallel, delayed, cpu_count
import json
import numpy as np
import pandas as pd
import os

from ranking import reduce_batch_with_rank, reduce_batch


CHOOSEN_LANGUAGES = set([
    '<javascript>', '<python>', '<java>', '<c#>', '<php>', '<c>', '<go>', '<ruby>'
])


special_tokens_id = list(range(33, 48))
special_tokens_id += list(range(58, 65))
special_tokens_id += list(range(91, 97))
special_tokens_id += list(range(123, 127))
special_tokens = [chr(i) for i in special_tokens_id]


def convention_tokenize(text):
    for st in special_tokens:
        text = f' {st} '.join(text.split(st)).strip()
    tokens = text.split()
    if not tokens:
        tokens = ['nothingishere']
    return tokens


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def dump_dataframe(lines, path):
    df = pd.DataFrame(lines)
    df.to_csv(path, mode='a', index=False, header=not os.path.exists(path))


def load_label_file(label_path):
    label_lines = []
    lang_lines = []
    with open(label_path, 'r', encoding='utf-8') as f:
        for line in f:
            line_js = json.loads(line.strip().lower())
            label_lines.append(convention_tokenize(line_js['docstring_tokens']))
            lang_lines.append(line_js['language'])
            if LINES_FOR_TEST and len(lang_lines) == LINES_FOR_TEST:
                break
    return label_lines, lang_lines


def load_pred_file(pred_path, return_nums):
    # read txt files
    pred_lines = []
    with open(pred_path, 'r', encoding='utf-8') as f:
        for line in f:
            pred_lines.append(convention_tokenize(line.lower()))
            if LINES_FOR_TEST and len(pred_lines) == LINES_FOR_TEST*return_nums:
                break
    return pred_lines


def load_data(pred_path, label_path, return_nums):
    labels, langs = load_label_file(label_path)
    preds = load_pred_file(pred_path, return_nums)
    labels_extended = [label for label in labels for _ in range(return_nums)]
    langs_extended = [lang for lang in langs for _ in range(return_nums)]
    assert len(labels_extended) == len(preds)
    return preds, langs_extended, labels_extended


def nltk_bleu(hypotheses, references):
    '''return float'''
    scores = []
    smoothing = SmoothingFunction().method2
    for i, hyp in enumerate(hypotheses):
        ref = [references[i]]
        score = sentence_bleu(ref, hyp, weights=(0.25, 0.25, 0.25, 0.25),
                              smoothing_function=smoothing)
        scores.append(score)
    return scores


def py_rouge(hypotheses, references, metric):
    '''
    metrics: 'rouge-1'/'rouge-2'/'rouge-l'
    '''
    rouge = Rouge(metrics=[metric])
    scores = []
    for i, hyp in enumerate(hypotheses):
        score = rouge.get_scores([' '.join(hyp)], [' '.join(references[i])])[0]
        scores.append(score[metric]['f'])
    return scores


def parallel_score(hypotheses, references, scorer, scorer_args=dict(), bulk_size=1000):
    n_jobs = cpu_count() - 2
    with Parallel(n_jobs=n_jobs) as pool:
        chunk_results = pool(
            delayed(scorer)(hypotheses[i:i+bulk_size], references[i:i+bulk_size], **scorer_args) for i in range(0, len(hypotheses), bulk_size)
        )
    results = [i for chunk in chunk_results for i in chunk]
    return results


def _statistics_per_lang_type(config, scores_dict, predictions):
    lines = []
    scores_reduced_by_max, scores_reduced_by_max_sorted = reduce_batch_with_rank(scores_dict, predictions, config['num_return_sequences'], config['limit'])
    # scores_reduced_by_max, scores_reduced_by_max_sorted = reduce_batch(scores_dict, config['num_return_sequences'], config['limit'])
    for s_type in scores_dict.keys():
        line = config.copy()
        line['score_type'] = s_type
        line['without_sort'] = sum(scores_reduced_by_max[s_type])/len(scores_reduced_by_max[s_type])
        line['with_sort'] = sum(scores_reduced_by_max_sorted[s_type])/len(scores_reduced_by_max_sorted[s_type])
        lines.append(line)
    return lines


def statistics(config, hypotheses, scores, langs):
    lines = []
    predictions_array = np.array(hypotheses, dtype=object)
    for lang_tag in CHOOSEN_LANGUAGES:
        lang_indexes = [i for i, x in enumerate(langs) if x == lang_tag]
        if len(lang_indexes) == 0:
            continue
        lang_socres_dict = dict()
        for s_type in scores.keys():
            scores_array = np.array(scores[s_type])
            lang_scores = scores_array[lang_indexes].tolist()
            lang_socres_dict[s_type] = lang_scores

        lang_predictions = predictions_array[lang_indexes].tolist()
        lang_config = config.copy()
        lang_config['language'] = lang_tag[1:-1]
        lines += _statistics_per_lang_type(lang_config, lang_socres_dict, lang_predictions)
    return lines


def cal_per_config(config):
    label_path = f'dataset/new_json/{config["tag"]}.8_langs.test.json'
    pred_path = f'./out/{config["model"]}_{config["tag"]}/{config["file_name"]}'
    print(f'start calculation: {pred_path}')
    hypotheses, langs, references = load_data(pred_path, label_path, config["num_return_sequences"])
    bleu = parallel_score(hypotheses, references, nltk_bleu)
    rouge1s = parallel_score(hypotheses, references, py_rouge, scorer_args={'metric': 'rouge-1'})
    rouge2s = parallel_score(hypotheses, references, py_rouge, scorer_args={'metric': 'rouge-2'})
    rougeLs = parallel_score(hypotheses, references, py_rouge, scorer_args={'metric': 'rouge-l'})
    scores = {
        'bleu': bleu,
        'rouge1': rouge1s,
        'rouge2': rouge2s,
        'rougeL': rougeLs
    }
    return statistics(config, hypotheses, scores, langs)


def load_configs(model, tag, config_files):
    raw_configs = []
    for file in config_files:
        raw_configs += load_json(file)
    configs = []
    for config in raw_configs:
        config['model'] = model
        config['tag'] = tag
        configs.append(config)
    return configs

if __name__ == '__main__':
    LINES_FOR_TEST = None
    configs = [{
        'num_return_sequences': 100,
        'model': 'bart',
        'tag': 'no_tags',
        'file_name': 'pred_100_None_0.9.txt',
        'limit': 2,
        'note': 'with_rank'
    }]
    for config in configs:
        dump_dataframe(cal_per_config(config), 'bart_rmns_0dot9.csv')
