import random 
import logging
import pandas as pd
import os
import numpy as np
import re
import pickle
import json
from collections import OrderedDict, defaultdict
from torch.utils.data import Dataset


def decision(probability):
    return random.random() < probability

def load_sessions(data_dir):
    
    with open(os.path.join(data_dir, "data_desc.json"), "r") as fr:
        data_desc = json.load(fr)
    
    with open(os.path.join(data_dir, "session_train.pkl"), "rb") as fr:
        session_train = pickle.load(fr) 
    with open(os.path.join(data_dir, "session_test.pkl"), "rb") as fr:
        session_test = pickle.load(fr)
    
    train_labels = [
        v["label"] if not isinstance(v["label"], list) else int(sum(v["label"]) > 0)

        for _, v in session_train.items()
    ]

    test_labels = [
        v["label"] if not isinstance(v["label"], list) else int(sum(v["label"]) > 0)
        for _, v in session_test.items()
    ]

    num_train = len(session_train)
    ratio_train = sum(train_labels) / num_train
    num_test = len(session_test)
    ratio_test = sum(test_labels) / num_test

    logging.info(f"Train sessions: {num_train}, ratio of anomalies: {ratio_train:.2f}")
    logging.info(f"Test sessions: {num_test}, ratio of anomalies: {ratio_test:.2f}")

    return session_train, session_test 

class log_dataset(Dataset):
    def __init__(self, session_dict, feature_type="semantics"):
        flatten_data_list = []

        for session_idx, data_dict in enumerate(session_dict.values()):
            features = data_dict["features"][feature_type]
            window_labels = data_dict["window_labels"]
            window_anomalies = data_dict["window_anomalies"]

            for window_idx in range(len(window_labels)):
                sample = {
                    "session_idx": session_idx,
                    "features": features[window_idx],
                    "window_labels": window_labels[window_idx],
                    "window_anomalies": window_anomalies[window_idx],
                }
                flatten_data_list.append(sample)
        self.flatten_data_list = flatten_data_list
    
    def __len__(self):
        return len(self.flatten_data_list)

    def __getitem__(self, index):
        return self.flatten_data_list[index]
    