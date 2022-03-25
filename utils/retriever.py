from typing import List
from typing import Tuple
from transformers import AutoTokenizer, AutoModel, AutoConfig
import numpy as np
import torch
from torch.nn.functional import normalize
import pickle

tokenizer_labse = AutoTokenizer.from_pretrained("setu4993/LaBSE")
model_labse = AutoModel.from_pretrained("setu4993/LaBSE")

with open('IDFs.pkl', 'rb') as f:
    words2IDF = pickle.load(f)
with open('title_to_embeddings.pkl') as f:
    title_to_embeddings = pickle.load(f)
N_DOC = len(title_to_embeddings.key())


def get_embeddings(sentece):
    """
    Return embeddings based on encoder model

    :param sentence: input sentence(s)
    :type sentence: str or list of strs
    :return: embeddings
    """
    tokenized = tokenizer_labse(sentece,
                                return_tensors="pt",
                                padding=True)
    with torch.no_grad():
        embeddings = model_labse(**tokenized)
    
    return np.squeeze(np.array(embeddings.pooler_output))


def calc_idf_score(sentence) -> float:
    """
    Calculate the mean idf score for given sentence.

    :param sentence: input sentence
    :type sentence: str
    :return: mean idf score of sentence token
    """
    tokenzied_sentence = [s.lower() for s in tokenizer_labse.tokenize(sentence)]
    score = 0
    for token in tokenzied_sentence:
        if token in words2IDF:
            score += words2IDF[token]
        else:
            score += np.log(N_DOC)
    return score / len(tokenzied_sentence)


def predict_labelwise_doc_at_history_ordered(queries, title_embeddings, k=1) -> Tuple[List[float], List[str]]:
    """
    Predict which document is matched to the given query.

    :param queries: input queries in time reversed order (latest first)
    :type queries: str (or list[str])
    :param title_embeddings: list of title embeddings
    :type title_embeddings: list[str]
    :param k: number of returning docs
    :type k: int 
    :return: return the document names and accuracies
    """
    similarities = np.array(list(map(lambda x: 0.0, title_embeddings)))
    coef_sum = 0
    for i, query in enumerate(queries):
        query_embd = get_embeddings(query)
        query_sim = list(map(lambda x: np.dot(x, query_embd) /
                            (np.linalg.norm(query_embd) * np.linalg.norm(x)),
                            title_embeddings))
        query_sim = np.array(query_sim)

        coef = 2**(-i) * calc_idf_score(query)
        coef_sum += coef
        similarities += coef * query_sim

    similarities = similarities / coef_sum
    best_k_idx = similarities.argsort()[::-1][:k]
    accuracy = similarities[best_k_idx]
    return (accuracy, best_k_idx)


def get_documents(docs, queries, k=10) -> List[str]:
    "returns list of related document IDs"
    titles = list(set(docs).intersection(title_to_embeddings.keys()))
    title_embeddings = [title_to_embeddings[title] for title in titles]
    acc, best_k_idx = predict_labelwise_doc_at_history_ordered(queries, title_embeddings, k)
    return [titles[i] for i in best_k_idx]
