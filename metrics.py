import collections
import math
import torch
from typing import List

def ngrams_iterator(token_list, ngrams):
    '''
    返回一个迭代器，该迭代器生成给定的token和它们的ngrams。
    Args:
        token_list: 一个包含tokens的列表。
        ngrams: ngrams的数量。
    '''
    def _get_ngrams(n):
        return zip(*[token_list[i:] for i in range(n)])
    for x in token_list:
        yield x
    for n in range(2, ngrams + 1):
        for x in _get_ngrams(n):
            yield " ".join(x)

def _compute_ngram_counter(tokens, max_n):
    '''
    计算给定的tokens的ngrams的计数器。
    Args:
        tokens: 一个可迭代对象，包含tokens。
        max_n: 最大的ngram的数量。
    Returns:
        一个collections.Counter，包含ngrams的计数器。
    '''
    ngrams_counter = collections.Counter(tuple(x.split(" ")) for x in ngrams_iterator(tokens, max_n))
    return ngrams_counter

def bleu_score(candidate_corpus: List[List[str]], references_corpus: List[List[str]], max_n=4, weights=[0.25] * 4):
    '''
    计算候选翻译语料库和参考翻译语料库之间的BLEU分数。基于https://www.aclweb.org/anthology/P02-1040.pdf
    Args:
        candidate_corpus: 候选翻译的可迭代对象。
        references_corpus: 参考翻译的可迭代对象。
        max_n: 最大的ngram的数量。
        weights: 权重的列表，用于计算BLEU分数。
    Returns:
        BLEU分数。
    '''
    clipped_counts = torch.zeros(max_n)
    total_counts = torch.zeros(max_n)
    weights = torch.tensor(weights)
    candidate_len = 0.0
    refs_len = 0.0
    for (candidate, refs) in zip(candidate_corpus, references_corpus):
        current_candidate_len = len(candidate)
        candidate_len += current_candidate_len
        refs_len_list = [float(len(ref)) for ref in refs]
        refs_len += min(refs_len_list, key=lambda x: abs(current_candidate_len - x))
        reference_counters = _compute_ngram_counter(refs[0], max_n)
        for ref in refs[1:]:
            reference_counters = reference_counters | _compute_ngram_counter(ref, max_n)
        candidate_counter = _compute_ngram_counter(candidate, max_n)
        clipped_counter = candidate_counter & reference_counters
        for ngram, count in clipped_counter.items():
            clipped_counts[len(ngram) - 1] += count
        for i in range(max_n):
            total_counts[i] += max(current_candidate_len - i, 0)
    if min(clipped_counts) == 0:
        return 0.0
    else:
        pn = clipped_counts / total_counts
        log_pn = weights * torch.log(pn)
        score = torch.exp(sum(log_pn))
        bp = math.exp(min(1 - refs_len / candidate_len, 0))
        return bp * score.item()