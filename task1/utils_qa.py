# coding=utf-8
# Copyright 2020 The HuggingFace Team All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Post-processing utilities for question answering.
"""
import collections
import json
import logging
import os
from typing import Optional, OrderedDict, Tuple

import numpy as np
from tqdm.auto import tqdm
import sys

from transformers import AutoTokenizer, AutoModel, AutoConfig, AutoModelWithLMHead, AutoTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import torch
from torch.nn.functional import normalize
import pickle
import json
from tqdm import tqdm

answer_max_length = 30
tokenizer_t5 = AutoTokenizer.from_pretrained("mrm8488/t5-base-finetuned-summarize-news")
model_t5 = AutoModelWithLMHead.from_pretrained("mrm8488/t5-base-finetuned-summarize-news")

tokenizer_labse = AutoTokenizer.from_pretrained("setu4993/LaBSE")
model_labse = AutoModel.from_pretrained("setu4993/LaBSE")

logger = logging.getLogger(__name__)

if any(['unseen' in item for item in sys.argv]):
    DOC_FILEPATH = "/home/alavian/Documents/stvt/CAiRE/dataset/multidialdoc/dialdoc2022_sharedtask/MDD-UNSEEN/multidoc2dial_doc_cdccovid.json"
else:
    DOC_FILEPATH = "/home/alavian/Documents/stvt/CAiRE/dataset/multidialdoc/multidoc2dial/multidoc2dial_doc.json"

if '--generative' in sys.argv:
    GENERATIVE = True
else:
    GENERATIVE = False

tfidfVectorizer = None
tfidf_wm = None

def tfIDF_fitting(path2doc) -> None:
    global tfidfVectorizer, tfidf_wm

    with open(path2doc, 'r') as f:
        multidoc2dial_doc = json.load(f)
    
    doc_texts_train = []
    for domain in multidoc2dial_doc['doc_data']:
        for title in multidoc2dial_doc['doc_data'][domain]:
            doc_texts_train.append(multidoc2dial_doc['doc_data'][domain]\
                                          [title]['doc_text'].strip())
    
    tfidfVectorizer = TfidfVectorizer(strip_accents=None,
                                    analyzer='char',
                                    ngram_range=(2, 8),
                                    norm='l2',
                                    use_idf=True,
                                    smooth_idf=True)
    tfidf_wm = tfidfVectorizer.fit_transform(doc_texts_train)


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


def get_best_answer_for_question(answers, question, beta=1) -> str:
    """
    answers: List
    question: Str

    Returns answer: Str
    """
    question_embd = get_embeddings(question)
    answers_embds = list(map(get_embeddings, answers))
    answer_sim = list(map(lambda x: np.dot(x, question_embd) /
                            (np.linalg.norm(question_embd) * np.linalg.norm(x)),
                            answers_embds))
    question_trasform = np.squeeze(np.asarray(tfidf_wm @ tfidfVectorizer.transform([question]).todense().T))
    tfidf_sim = list(map(lambda x: np.dot(x, question_trasform) /
                            (np.linalg.norm(question_trasform) * np.linalg.norm(x)),
                            np.squeeze(np.asarray(tfidf_wm @ tfidfVectorizer.transform(answers).todense().T)).T))
    sim = np.array(answer_sim) + beta * np.array(tfidf_sim)
    return answers[np.argmax(sim)]


def summarize(predictions, k=3):
    text = ".".join(predictions[:k])
    input_ids = tokenizer_t5.encode(text, return_tensors="pt", add_special_tokens=True)
    generated_ids = model_t5.generate(input_ids=input_ids, num_beams=2, max_length=answer_max_length,  repetition_penalty=2.5, length_penalty=1.0, early_stopping=True)
    preds = [tokenizer_t5.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
    return preds[0]


def final_postprocess_qa_predictions(
    example_id_to_index, 
    examples, 
    all_predictions,
    initial_predictions=None
):
    """
    remove duplicate answers for questions by getting attention between questions and the answers

    :example_id_to_index: Dict{id: examples_index}
    :examples: Dataset(id, context, question, domain, title)
    :all_predictions: Dict{id: prediction_text}

    Returns a dict of final predictions Dict{id: prediction_text}
    """
    
    if initial_predictions:
        id_best_probs = OrderedDict()
        output = OrderedDict()
        for id, index in tqdm(example_id_to_index.items()):
            new_id = "{}_{}".format(* id.split('_')[0:2])
            initial_prediction = initial_predictions[index]
            if new_id not in output:
                output[new_id] = all_predictions[id]
                id_best_probs[new_id] = initial_prediction['probability']
            else:
                if id_best_probs[new_id] > initial_prediction['probability']:
                    continue
                else:
                    output[new_id] = all_predictions[id]
                    id_best_probs[new_id] = initial_prediction['probability']
    else:
        predictions = collections.OrderedDict()
        # { # id : {"question": "", "predictions": [""]} }
        for id, index in tqdm(example_id_to_index.items()):
            new_id = "{}_{}".format(* id.split('_')[0:2])
            if new_id not in predictions:
                predictions[new_id] = {
                    "question": examples[index]["only-question"],
                    "predictions": [all_predictions[id]],
                }
            else:
                predictions[new_id]["predictions"].append(all_predictions[id])

        output = collections.OrderedDict()

        #   Fitting TF-IDF
        global DOC_FILEPATH
        tfIDF_fitting(DOC_FILEPATH)
        
        for id in tqdm(predictions, desc="getting best answer for each question"):
            if GENERATIVE:
                output[id] = summarize(predictions[id]["predictions"])
            else:
                output[id] = get_best_answer_for_question(predictions[id]["predictions"], predictions[id]["question"])
            
    return output


def postprocess_qa_predictions(
    examples,
    features,
    predictions: Tuple[np.ndarray, np.ndarray],
    version_2_with_negative: bool = False,
    n_best_size: int = 20,
    max_answer_length: int = 30,
    null_score_diff_threshold: float = 0.0,
    output_dir: Optional[str] = None,
    prefix: Optional[str] = None,
    is_world_process_zero: bool = True,
):
    """
    Post-processes the predictions of a question-answering model to convert them to answers that are substrings of the
    original contexts. This is the base postprocessing functions for models that only return start and end logits.

    Args:
        examples: The non-preprocessed dataset (see the main script for more information).
        features: The processed dataset (see the main script for more information).
        predictions (:obj:`Tuple[np.ndarray, np.ndarray]`):
            The predictions of the model: two arrays containing the start logits and the end logits respectively. Its
            first dimension must match the number of elements of :obj:`features`.
        version_2_with_negative (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not the underlying dataset contains examples with no answers.
        n_best_size (:obj:`int`, `optional`, defaults to 20):
            The total number of n-best predictions to generate when looking for an answer.
        max_answer_length (:obj:`int`, `optional`, defaults to 30):
            The maximum length of an answer that can be generated. This is needed because the start and end predictions
            are not conditioned on one another.
        null_score_diff_threshold (:obj:`float`, `optional`, defaults to 0):
            The threshold used to select the null answer: if the best answer has a score that is less than the score of
            the null answer minus this threshold, the null answer is selected for this example (note that the score of
            the null answer for an example giving several features is the minimum of the scores for the null answer on
            each feature: all features must be aligned on the fact they `want` to predict a null answer).

            Only useful when :obj:`version_2_with_negative` is :obj:`True`.
        output_dir (:obj:`str`, `optional`):
            If provided, the dictionaries of predictions, n_best predictions (with their scores and logits) and, if
            :obj:`version_2_with_negative=True`, the dictionary of the scores differences between best and null
            answers, are saved in `output_dir`.
        prefix (:obj:`str`, `optional`):
            If provided, the dictionaries mentioned above are saved with `prefix` added to their names.
        is_world_process_zero (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether this process is the main process or not (used to determine if logging/saves should be done).
    """
    assert len(predictions) == 2, "`predictions` should be a tuple with two elements (start_logits, end_logits)."
    all_start_logits, all_end_logits = predictions

    assert len(predictions[0]) == len(features), f"Got {len(predictions[0])} predictions and {len(features)} features."

    # Build a map example to its corresponding features.
    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(features):
        features_per_example[example_id_to_index[feature["example_id"]]].append(i)

    # The dictionaries we have to fill.
    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    if version_2_with_negative:
        scores_diff_json = collections.OrderedDict()
    all_positions = collections.OrderedDict()

    # Logging.
    logger.setLevel(logging.INFO if is_world_process_zero else logging.WARN)
    logger.info(f"Post-processing {len(examples)} example predictions split into {len(features)} features.")

    # Let's loop over all the examples!
    for example_index, example in enumerate(tqdm(examples)):
        # Those are the indices of the features associated to the current example.
        feature_indices = features_per_example[example_index]

        min_null_prediction = None
        prelim_predictions = []

        # Looping through all the features associated to the current example.
        for feature_index in feature_indices:
            # We grab the predictions of the model for this feature.
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]
            # This is what will allow us to map some the positions in our logits to span of texts in the original
            # context.
            offset_mapping = features[feature_index]["offset_mapping"]
            # Optional `token_is_max_context`, if provided we will remove answers that do not have the maximum context
            # available in the current feature.
            token_is_max_context = features[feature_index].get("token_is_max_context", None)

            # Update minimum null prediction.
            feature_null_score = start_logits[0] + end_logits[0]
            if min_null_prediction is None or min_null_prediction["score"] > feature_null_score:
                min_null_prediction = {
                    "offsets": (0, 0),
                    "score": feature_null_score,
                    "start_logit": start_logits[0],
                    "end_logit": end_logits[0],
                }

            # Go through all possibilities for the `n_best_size` greater start and end logits.
            start_indexes = np.argsort(start_logits)[-1 : -n_best_size - 1 : -1].tolist()
            end_indexes = np.argsort(end_logits)[-1 : -n_best_size - 1 : -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Don't consider out-of-scope answers, either because the indices are out of bounds or correspond
                    # to part of the input_ids that are not in the context.
                    if (
                        start_index >= len(offset_mapping)
                        or end_index >= len(offset_mapping)
                        or offset_mapping[start_index] is None
                        or offset_mapping[start_index] == []
                        or offset_mapping[end_index] is None 
                        or offset_mapping[end_index] == []
                    ):
                        continue
                    # Don't consider answers with a length that is either < 0 or > max_answer_length.
                    if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                        continue
                    # Don't consider answer that don't have the maximum context available (if such information is
                    # provided).
                    if token_is_max_context is not None and not token_is_max_context.get(str(start_index), False):
                        continue
                    prelim_predictions.append(
                        {
                            "offsets": (offset_mapping[start_index][0], offset_mapping[end_index][1]),
                            "score": start_logits[start_index] + end_logits[end_index],
                            "start_logit": start_logits[start_index],
                            "end_logit": end_logits[end_index],
                        }
                    )

        if version_2_with_negative:
            # Add the minimum null prediction
            prelim_predictions.append(min_null_prediction)
            null_score = min_null_prediction["score"]

        # Only keep the best `n_best_size` predictions.
        predictions = sorted(prelim_predictions, key=lambda x: x["score"], reverse=True)[:n_best_size]

        # Add back the minimum null prediction if it was removed because of its low score.
        if version_2_with_negative and not any(p["offsets"] == (0, 0) for p in predictions):
            predictions.append(min_null_prediction)

        # Use the offsets to gather the answer text in the original context.
        context = example["context"]
        for pred in predictions:
            offsets = pred["offsets"]
            pred["text"] = context[offsets[0] : offsets[1]]

        # In the very rare edge case we have not a single non-null prediction, we create a fake prediction to avoid
        # failure.
        if len(predictions) == 0 or (len(predictions) == 1 and predictions[0]["text"] == ""):
            predictions.insert(0, {"text": "", "start_logit": 0.0, "end_logit": 0.0, "score": 0.0, "offsets": (0,0)})

        # Compute the softmax of all scores (we do it with numpy to stay independent from torch/tf in this file, using
        # the LogSumExp trick).
        scores = np.array([pred.pop("score") for pred in predictions])
        exp_scores = np.exp(scores - np.max(scores))
        probs = exp_scores / exp_scores.sum()

        # Include the probabilities in our predictions.
        for prob, pred in zip(probs, predictions):
            pred["probability"] = prob

        # Pick the best prediction. If the null answer is not possible, this is easy.
        if not version_2_with_negative:
            all_predictions[example["id"]] = predictions[0]["text"]
            all_positions[example["id"]] = tuple(predictions[0]["offsets"])
        else:
            # Otherwise we first need to find the best non-empty prediction.
            i = 0
            all_empty = False
            while predictions[i]["text"] == "" or predictions[i]["text"] == "empty":
                i += 1
                if i == len(predictions):
                    all_empty = True
                    break
            
            if not all_empty:
                best_non_null_pred = predictions[i] 

                # Then we compare to the null prediction using the threshold.
                score_diff = null_score - best_non_null_pred["start_logit"] - best_non_null_pred["end_logit"]
                scores_diff_json[example["id"]] = float(score_diff)  # To be JSON-serializable.
                if score_diff > null_score_diff_threshold:
                    all_predictions[example["id"]] = ""
                    all_positions[example["id"]] = (0,0)
                else:
                    all_predictions[example["id"]] = best_non_null_pred["text"]
                    all_positions[example["id"]] = tuple(best_non_null_pred["offsets"])
            else:
                all_predictions[example["id"]] = ""
                all_positions[example["id"]] = (0,0)

        # Make `predictions` JSON-serializable by casting np.float back to float.
        all_nbest_json[example["id"]] = [
            {k: (float(v) if isinstance(v, (np.float16, np.float32, np.float64)) else v) for k, v in pred.items()}
            for pred in predictions
        ]

    if len(next(iter(example_id_to_index)).split('_')) == 3:
        all_predictions = final_postprocess_qa_predictions(example_id_to_index, examples, all_predictions, initial_predictions=predictions)

    # If we have an output_dir, let's save all those dicts.
    if output_dir is not None:
        assert os.path.isdir(output_dir), f"{output_dir} is not a directory."

        prediction_file = os.path.join(
            output_dir, "predictions.json" if prefix is None else f"predictions_{prefix}".json
        )
        nbest_file = os.path.join(
            output_dir, "nbest_predictions.json" if prefix is None else f"nbest_predictions_{prefix}".json
        )
        if version_2_with_negative:
            null_odds_file = os.path.join(
                output_dir, "null_odds.json" if prefix is None else f"null_odds_{prefix}".json
            )
        position_file = os.path.join(
            output_dir, "positions.json" if prefix is None else f"positions_{prefix}".json
        )

        logger.info(f"Saving predictions to {prediction_file}.")
        with open(prediction_file, "w") as writer:
            writer.write(json.dumps(all_predictions, indent=4) + "\n")
        logger.info(f"Saving nbest_preds to {nbest_file}.")
        with open(nbest_file, "w") as writer:
            writer.write(json.dumps(all_nbest_json, indent=4) + "\n")
        if version_2_with_negative:
            logger.info(f"Saving null_odds to {null_odds_file}.")
            with open(null_odds_file, "w") as writer:
                writer.write(json.dumps(scores_diff_json, indent=4) + "\n")

        logger.info(f"Saving positions to {position_file}.")
        with open(position_file, "w") as writer:
            writer.write(json.dumps(all_positions, indent=4) + "\n")

    return all_predictions


def postprocess_qa_predictions_with_beam_search(
    examples,
    features,
    predictions: Tuple[np.ndarray, np.ndarray],
    version_2_with_negative: bool = False,
    n_best_size: int = 20,
    max_answer_length: int = 30,
    start_n_top: int = 5,
    end_n_top: int = 5,
    output_dir: Optional[str] = None,
    prefix: Optional[str] = None,
    is_world_process_zero: bool = True,
):
    """
    Post-processes the predictions of a question-answering model with beam search to convert them to answers that are substrings of the
    original contexts. This is the postprocessing functions for models that return start and end logits, indices, as well as
    cls token predictions.

    Args:
        examples: The non-preprocessed dataset (see the main script for more information).
        features: The processed dataset (see the main script for more information).
        predictions (:obj:`Tuple[np.ndarray, np.ndarray]`):
            The predictions of the model: two arrays containing the start logits and the end logits respectively. Its
            first dimension must match the number of elements of :obj:`features`.
        version_2_with_negative (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not the underlying dataset contains examples with no answers.
        n_best_size (:obj:`int`, `optional`, defaults to 20):
            The total number of n-best predictions to generate when looking for an answer.
        max_answer_length (:obj:`int`, `optional`, defaults to 30):
            The maximum length of an answer that can be generated. This is needed because the start and end predictions
            are not conditioned on one another.
        start_n_top (:obj:`int`, `optional`, defaults to 5):
            The number of top start logits too keep when searching for the :obj:`n_best_size` predictions.
        end_n_top (:obj:`int`, `optional`, defaults to 5):
            The number of top end logits too keep when searching for the :obj:`n_best_size` predictions.
        output_dir (:obj:`str`, `optional`):
            If provided, the dictionaries of predictions, n_best predictions (with their scores and logits) and, if
            :obj:`version_2_with_negative=True`, the dictionary of the scores differences between best and null
            answers, are saved in `output_dir`.
        prefix (:obj:`str`, `optional`):
            If provided, the dictionaries mentioned above are saved with `prefix` added to their names.
        is_world_process_zero (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether this process is the main process or not (used to determine if logging/saves should be done).
    """
    assert len(predictions) == 5, "`predictions` should be a tuple with five elements."
    start_top_log_probs, start_top_index, end_top_log_probs, end_top_index, cls_logits = predictions

    assert len(predictions[0]) == len(
        features
    ), f"Got {len(predictions[0])} predicitions and {len(features)} features."

    # Build a map example to its corresponding features.
    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(features):
        features_per_example[example_id_to_index[feature["example_id"]]].append(i)

    # The dictionaries we have to fill.
    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    scores_diff_json = collections.OrderedDict() if version_2_with_negative else None

    # Logging.
    logger.setLevel(logging.INFO if is_world_process_zero else logging.WARN)
    logger.info(f"Post-processing {len(examples)} example predictions split into {len(features)} features.")

    # Let's loop over all the examples!
    for example_index, example in enumerate(tqdm(examples)):
        # Those are the indices of the features associated to the current example.
        feature_indices = features_per_example[example_index]

        min_null_score = None
        prelim_predictions = []

        # Looping through all the features associated to the current example.
        for feature_index in feature_indices:
            # We grab the predictions of the model for this feature.
            start_log_prob = start_top_log_probs[feature_index]
            start_indexes = start_top_index[feature_index]
            end_log_prob = end_top_log_probs[feature_index]
            end_indexes = end_top_index[feature_index]
            feature_null_score = cls_logits[feature_index]
            # This is what will allow us to map some the positions in our logits to span of texts in the original
            # context.
            offset_mapping = features[feature_index]["offset_mapping"]
            # Optional `token_is_max_context`, if provided we will remove answers that do not have the maximum context
            # available in the current feature.
            token_is_max_context = features[feature_index].get("token_is_max_context", None)

            # Update minimum null prediction
            if min_null_score is None or feature_null_score < min_null_score:
                min_null_score = feature_null_score

            # Go through all possibilities for the `n_start_top`/`n_end_top` greater start and end logits.
            for i in range(start_n_top):
                for j in range(end_n_top):
                    start_index = start_indexes[i]
                    j_index = i * end_n_top + j
                    end_index = end_indexes[j_index]
                    # Don't consider out-of-scope answers (last part of the test should be unnecessary because of the
                    # p_mask but let's not take any risk)
                    if (
                        start_index >= len(offset_mapping)
                        or end_index >= len(offset_mapping)
                        or offset_mapping[start_index] is None
                        or offset_mapping[end_index] is None
                    ):
                        continue
                    # Don't consider answers with a length negative or > max_answer_length.
                    if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                        continue
                    # Don't consider answer that don't have the maximum context available (if such information is
                    # provided).
                    if token_is_max_context is not None and not token_is_max_context.get(str(start_index), False):
                        continue
                    prelim_predictions.append(
                        {
                            "offsets": (offset_mapping[start_index][0], offset_mapping[end_index][1]),
                            "score": start_log_prob[i] + end_log_prob[j_index],
                            "start_log_prob": start_log_prob[i],
                            "end_log_prob": end_log_prob[j_index],
                        }
                    )

        # Only keep the best `n_best_size` predictions.
        predictions = sorted(prelim_predictions, key=lambda x: x["score"], reverse=True)[:n_best_size]

        # Use the offsets to gather the answer text in the original context.
        context = example["context"]
        for pred in predictions:
            offsets = pred.pop("offsets")
            pred["text"] = context[offsets[0] : offsets[1]]

        # In the very rare edge case we have not a single non-null prediction, we create a fake prediction to avoid
        # failure.
        if len(predictions) == 0:
            predictions.insert(0, {"text": "", "start_logit": -1e-6, "end_logit": -1e-6, "score": -2e-6, "offsets":(0,0)})

        # Compute the softmax of all scores (we do it with numpy to stay independent from torch/tf in this file, using
        # the LogSumExp trick).
        scores = np.array([pred.pop("score") for pred in predictions])
        exp_scores = np.exp(scores - np.max(scores))
        probs = exp_scores / exp_scores.sum()

        # Include the probabilities in our predictions.
        for prob, pred in zip(probs, predictions):
            pred["probability"] = prob

        # Pick the best prediction and set the probability for the null answer.
        all_predictions[example["id"]] = predictions[0]["text"]
        if version_2_with_negative:
            scores_diff_json[example["id"]] = float(min_null_score)

        # Make `predictions` JSON-serializable by casting np.float back to float.
        all_nbest_json[example["id"]] = [
            {k: (float(v) if isinstance(v, (np.float16, np.float32, np.float64)) else v) for k, v in pred.items()}
            for pred in predictions
        ]

    # If we have an output_dir, let's save all those dicts.
    if output_dir is not None:
        assert os.path.isdir(output_dir), f"{output_dir} is not a directory."

        prediction_file = os.path.join(
            output_dir, "predictions.json" if prefix is None else f"predictions_{prefix}".json
        )
        nbest_file = os.path.join(
            output_dir, "nbest_predictions.json" if prefix is None else f"nbest_predictions_{prefix}".json
        )
        if version_2_with_negative:
            null_odds_file = os.path.join(
                output_dir, "null_odds.json" if prefix is None else f"null_odds_{prefix}".json
            )

        print(f"Saving predictions to {prediction_file}.")
        with open(prediction_file, "w") as writer:
            writer.write(json.dumps(all_predictions, indent=4) + "\n")
        print(f"Saving nbest_preds to {nbest_file}.")
        with open(nbest_file, "w") as writer:
            writer.write(json.dumps(all_nbest_json, indent=4) + "\n")
        if version_2_with_negative:
            print(f"Saving null_odds to {null_odds_file}.")
            with open(null_odds_file, "w") as writer:
                writer.write(json.dumps(scores_diff_json, indent=4) + "\n")

    return all_predictions, scores_diff_json
