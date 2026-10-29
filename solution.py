#!/usr/bin/env python3
import argparse
import csv
import math
import os
import re
from collections import Counter
from itertools import product
from time import perf_counter


def tokenize(text):
    return re.findall(r'\w+', text.lower())


def read_queries(path):
    queries = {}
    with open(path, encoding='utf-8') as f:
        for line in f:
            query_id, text = line.rstrip('\n').split('\t', 1)
            queries[int(query_id)] = list(dict.fromkeys(tokenize(text)))
    return queries


def read_qrels(path):
    qrels = {}
    with open(path, encoding='utf-8') as f:
        for line in f:
            query_id, _, doc_id, label = line.split()
            qrels.setdefault(int(query_id), {})[doc_id] = int(label)
    return qrels


def read_docs(path, doc_ids, terms):
    docs = {}
    df = Counter()
    total_title = total_body = n = 0
    with open(path, encoding='utf-8') as f:
        for line in f:
            fields = line.rstrip('\n').split('\t', 3)
            fields += [''] * (4 - len(fields))
            doc_id, _, title, body = fields[:4]
            title_words = tokenize(title)
            body_words = tokenize(body)
            title_counts = Counter(w for w in title_words if w in terms)
            body_counts = Counter(w for w in body_words if w in terms)
            df.update(title_counts.keys() | body_counts.keys())
            total_title += len(title_words)
            total_body += len(body_words)
            n += 1
            if doc_id in doc_ids:
                docs[doc_id] = (
                    len(title_words), len(body_words), title_counts, body_counts
                )
    avg_title = max(total_title / n, 1) if n else 1
    avg_body = max(total_body / n, 1) if n else 1
    idf = {w: math.log(1 + (n - df[w] + 0.5) / (df[w] + 0.5)) for w in terms}
    return docs, idf, avg_title, avg_body


def bm25f(words, doc, idf, avg_title, avg_body, k1, b, title_weight):
    title_len, body_len, title_counts, body_counts = doc
    title_norm = 1 - b + b * title_len / avg_title
    body_norm = 1 - b + b * body_len / avg_body
    score = 0.0
    for word in words:
        tf = title_weight * title_counts.get(word, 0) / title_norm
        tf += body_counts.get(word, 0) / body_norm
        if tf:
            score += idf[word] * (k1 + 1) * tf / (k1 + tf)
    return score


def ndcg(queries, qrels, docs, idf, avg_title, avg_body, params):
    k1, b, title_weight = params
    total = 0.0
    discounts = [1 / math.log2(i + 2) for i in range(10)]
    empty_doc = (0, 0, {}, {})
    for query_id, labels in qrels.items():
        if query_id not in queries:
            continue
        ideal = sorted(labels.values(), reverse=True)[:10]
        best = sum((2 ** rel - 1) * d for rel, d in zip(ideal, discounts))
        if not best:
            continue
        ranked = sorted(
            ((bm25f(queries[query_id], docs.get(doc_id, empty_doc), idf,
                    avg_title, avg_body, k1, b, title_weight), rel)
             for doc_id, rel in labels.items()),
            key=lambda item: item[0], reverse=True,
        )[:10]
        dcg = sum((2 ** rel - 1) * d for (_, rel), d in zip(ranked, discounts))
        total += dcg / best
    return total / len(qrels) if qrels else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--submission_file', required=True)
    parser.add_argument('data_dir')
    args = parser.parse_args()
    start = perf_counter()

    train_queries = read_queries(os.path.join(args.data_dir, 'vkmarco-doctrain-queries.tsv'))
    test_queries = read_queries(os.path.join(args.data_dir, 'vkmarco-doceval-queries.tsv'))
    train_qrels = read_qrels(os.path.join(args.data_dir, 'vkmarco-doctrain-qrels.tsv'))
    with open(os.path.join(args.data_dir, 'objects.csv'), encoding='utf-8') as f:
        objects = [(r['ObjectId'], int(r['QueryId']), r['DocumentId']) for r in csv.DictReader(f)]

    terms = {word for words in list(train_queries.values()) + list(test_queries.values()) for word in words}
    doc_ids = {doc_id for labels in train_qrels.values() for doc_id in labels}
    doc_ids.update(doc_id for _, _, doc_id in objects)
    docs, idf, avg_title, avg_body = read_docs(
        os.path.join(args.data_dir, 'vkmarco-docs.tsv'), doc_ids, terms
    )

    best_params = None
    best_score = -1
    for params in product((0.8, 1.2, 1.8), (0.25, 0.55, 0.85), (1.5, 3.0, 6.0, 10.0)):
        score = ndcg(train_queries, train_qrels, docs, idf, avg_title, avg_body, params)
        if score > best_score:
            best_score = score
            best_params = params

    print(f'best params: {best_params}, train nDCG@10: {best_score:.4f}')
    empty_doc = (0, 0, {}, {})
    with open(args.submission_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['ObjectId', 'Score'])
        for object_id, query_id, doc_id in objects:
            score = bm25f(test_queries[query_id], docs.get(doc_id, empty_doc), idf,
                          avg_title, avg_body, *best_params)
            writer.writerow([object_id, score])
    print(f'finished, elapsed = {perf_counter() - start:.3f}')


if __name__ == '__main__':
    main()
