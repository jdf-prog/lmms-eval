import datetime
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Union

import cv2
import numpy as np
import yaml
from loguru import logger as eval_logger

from lmms_eval.tasks._task_utils.file_utils import generate_submission_file

hf_home = os.getenv("HF_HOME", "./~/.cache/huggingface")
# hf_home="/share/junjie/shuyan/lmms-eval/~/.cache/huggingface"
base_cache_dir = os.path.expanduser(hf_home)

with open(Path(__file__).parent / "lvbench.yaml", "r") as f:
    raw_data_test = f.readlines()
    safe_data_test = []
    for i, line in enumerate(raw_data_test):
        # remove function definition since yaml load cannot handle it
        if "!function" not in line:
            safe_data_test.append(line)
cache_name_test = yaml.safe_load("".join(safe_data_test))["dataset_kwargs"]["cache_dir"]
cache_dir_test = os.path.join(base_cache_dir, cache_name_test)

def lvbench_doc_to_visual_test(doc):
    video_path = doc["key"] + ".mp4"
    video_path = os.path.join(cache_dir_test, video_path)
    if os.path.exists(video_path):
        video_path = video_path
    else:
        sys.exit(f"video path:{video_path} does not exist, please check")
    return [video_path]


def lvbench_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    option_prompt="Please select the best answer from the options above and directly provide the letter representing your choice without giving any explanation."
    question = doc["question"] + "\n" + option_prompt
    return question


def extract_characters_regex(answer):
    answer = answer.strip()

    answer = answer.split(")")[0]
    answer = answer.strip()

    if "(" in answer:
        try:
            answer = answer.split("(")[1]
            answer = answer.strip()
        except:
            pass

    answer = answer.split(" ")[0]

    if len(answer)>0:
        return answer[0]
    return answer


def lvbench_process_results(doc, results):
    """
    Args:
        doc: a instance of the eval dataset
        results: [pred]
    Returns:
        a dictionary with key: metric name (in this case videomme score), value: metric value
    """
    pred = results[0]

    pred_ans = extract_characters_regex(pred)

    task_type = doc["question_type"]
    data_dict = {"question_id": doc["uid"], "task_type": task_type, "pred_answer": pred_ans, "answer": doc["answer"]}

    return {f"lvbench_percetion_score": data_dict}


def lvbench_aggregate_results_test(results):
    """
    Args:
        results: a list of values returned by process_results
    Returns:
        A score
    """
    
    
    total_qa_num = 0
    right_num = 0
    category_right = defaultdict(int)
    category_total = defaultdict(int)
    category_acc = defaultdict(int)
    
    for item in results:
        uid = str(item["question_id"])
        model_answer = item["pred_answer"]
        for category in item["task_type"]:
            category_total[category] += 1
            if model_answer == item["answer"]:
                category_right[category] += 1
        if model_answer == item["answer"]:
            right_num += 1
        total_qa_num += 1

    for key in category_total:
        category_acc[key] = category_right[key] / category_total[key]
        eval_logger.info(f"Evaluation on Task Categories: {key}: {category_acc[key]:.1f}%")

    acc = float(right_num) / total_qa_num
    category_acc.update({"acc": acc})
    eval_logger.info(f"Evaluation on Task Categories: Overall Accuracy: {acc:.1f}%")
    
    return category_acc
