import torch
from torch import nn

from base_model import ForcastBaseModel

class Transformer(ForcastBaseModel):

    def __init__(
        self,
        meta_data,
        embedding_dim,
        nhead,
        hidden_size=100,
        model_save_path="./transformer_models",
        feature_type="sequentials",
        label_type="next_log", # might want to change this (anomaly detection)
        eval_type="session",
        topk=5,
        use_tfidf=False,
        freeze=False,
        gpu=-1,
        **kwargs
    ):
        super().__init__(
            meta_data=meta_data,
            model_save_path=model_save_path,
            feature_type=feature_type,
            label_type=label_type,
            eval_type=eval_type,
            topk=topk,
            use_tfidf=use_tfidf,
            embedding_dim=embedding_dim,
            freeze=freeze,
            gpu=gpu,

        )
        