import os
import time
import logging
import numpy as np
import torch
import pandas as pd
from torch import nn
from collections import defaultdict
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from utils import set_device, tensor2flatten_arr

class Embedder(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        pretrain_matrix=None,
        freeze=False,
        use_tfidf=False
    ):
        super(Embedder, self).__init__()
        self.use_tfidf = use_tfidf
        if pretrain_matrix is not None:
            self.embedding_layer = nn.Embedding.from_pretrained(
                pretrain_matrix, padding_idx = 1, freeze=freeze
            )
        else:
            self.embedding_layer = nn.Embedding(
                vocab_size, embedding_dim, padding_idx=1
            )
    
    def forward(self, x):
        if self.use_tfidf:
            return torch.matmul(x, self.embedding_layer.weight.double())
        else:
            return self.embedding_layer(x.long())
        
        


class ForcastBaseModel(nn.Module):
    def __init__(
        self,
        meta_data,
        model_save_path,
        feature_type,
        label_type,
        eval_type,
        topk,
        use_tfidf,
        embedding_dim,
        freeze=False,
        gpu=-1,
        anomaly_ratio=None,
        patience=3,
        **kwargs
    ):
        super(ForcastBaseModel, self).__init__()
        self.device = set_device(gpu)
        self.topk = topk
        self.feature_type = feature_type
        self.label_type = label_type
        self.eval_type = eval_type
        self.anomaly_ratio = anomaly_ratio
        self.patience = patience
        self.time_tracker = {}

        os.makedirs(model_save_path, exist_ok=True)
        self.model_save_file = os.path.join(model_save_path, "model.pt")
        if feature_type in ["sequentials", "semantics"]:
            self.embedder = Embedder(
                meta_data["vocab_size"],
                embedding_dim=embedding_dim,
                pretrain_matrix=meta_data.get("pretrain_matrix", None),
                freeze=freeze,
                use_tfidf=use_tfidf
            )
        else:
            logging.info("Feature type not supported for embedding.")

    def evaluate(self, test_loader, dtype="test"):
        logging.info(f"Evaluating {dtype} set...")

        if self.label_type == "next_log":
            return self.__evaluate_next_log(test_loader, dtype=dtype)
        elif self.label_type == "anomaly":
            return self.__evaluate_anomaly(test_loader, dtype=dtype)
        elif self.label_type == "None":
            return self.__evaluate_recst(test_loader, dtype=dtype)
    
    def __evaluate_recst(self, test_loader, dtype="test"):
        self.eval()
        with torch.no_grad():
            y_pred = []
            store_dict = defaultdict(list)
            infer_start = time.time()
            for batch_input in test_loader:
                return_dict = self.forward(self.__input2device(batch_input))
                y_pred = return_dict["y_pred"]

                store_dict["session_idx"].extend(
                    tensor2flatten_arr(batch_input["session_idx"])
                )

                store_dict["window_anomalies".extend(
                    tensor2flatten_arr(batch_input["window_anomalies"])
                )]

                store_dict["window_preds"].extend(tensor2flatten_arr(y_pred))
            
            infer_end = time.time()
            logging.info(f"Inference time: {infer_end - infer_start:.2f} seconds")

            self.time_tracker["test"] = infer_end - infer_start

            store_df = pd.DataFrame(store_dict)

            use_cols = ["session_idx", "window_anomalies", "window_preds"]

            session_df = (
                store_df[use_cols].groupby("session_idx", as_index=False).max()
            )

            assert (
                self.anomaly_ratio is not None
            ), "Anomaly ratio must be set for evaluation."

            thre = np.percentile(
                session_df["window_preds"].values(), 100-self.anomaly_ratio * 100
            )

            pred = (session_df["windows_preds"] > thre).astype(int)

            y = (session_df["window_anomalies"] > 0).astype(int)

            eval_results = {
                "f1": f1_score(y, pred),
                "rc": recall_score(y, pred),
                "pc": precision_score(y, pred),
                "acc": accuracy_score(y, pred),
            }

            logging.info({k: f"{v:.4f}" for k, v in eval_results.items()})
            return eval_results
    def __evaluate_anomaly(self, test_loader, dtype="test"):
        self.eval()

        with torch.no_grad():
            y_pred = []
            store_dict = defaultdict(list)
            infer_start = time.time()

            for batch_input in test_loader:

                return_dict = self.forward(self.__input2device(batch_input))

                y_prob, y_pred = return_dict["y_pred"].max(dim=1)

                store_dict["session_idx"].extend(
                    tensor2flatten_arr(batch_input["session_idx"])
                )
                store_dict["window_anomalies"].extend(
                    tensor2flatten_arr(batch_input["window_anomalies"])
                )
                store_dict["window_preds"].extend(tensor2flatten_arr(y_pred))

                infer_end = time.time()
                logging.info(f"Inference time: {infer_end - infer_start:.2f} seconds")

                self.time_tracker["test"] = infer_end - infer_start

                store_df = pd.DataFrame(store_dict)

                use_cols = ["session_idx", "window_anomalies", "window_preds"]
                session_df = (
                    store_df[use_cols].groupby("session_idx", as_index=False).sum()
                )

                pred = (session_df["window_preds"] > 0).astype(int)
                y = (session_df["window_anomalies"] > 0).astype(int)

                eval_results = {
                    "f1": f1_score(y, pred),
                    "rc": recall_score(y, pred),
                    "pc": precision_score(y, pred),
                    "acc": accuracy_score(y, pred),
                }

                logging.info({k: f"{v:.4f}" for k, v in eval_results.items()})
                return eval_results
    
    # look into this a bit more
    def __evaluate_next_log(self, test_loader, dtype="test"):
        model = self.eval()

        with torch.no_grad():
            y_pred = []
            store_dict = defaultdict(list)
            infer_start = time.time()

            for batch_input in test_loader:
                return_dict = model.forward(self.__input2device(batch_input))
                y_pred = return_dict["y_pred"]

                y_prob_topk, y_pred_topk = torch.topk(y_pred, self.topk,)

                store_dict["session_idx"].extend(
                    tensor2flatten_arr(batch_input["session_idx"])
                )   
                store_dict["window_anomalies"].extend(
                    tensor2flatten_arr(batch_input["window_anomalies"])
                )

                # ground truth next tokens
                store_dict["window_labels"].extend(
                    tensor2flatten_arr(batch_input["window_labels"])
                )

                store_dict["x"].extend(batch_input["features"].data.cpu().numpy())
                store_dict["y_pred_topk"].extend(y_pred_topk.data.cpu().numpy())
                store_dict["y_prob_topk"].extend(y_prob_topk.data.cpu().numpy())

            infer_end = time.time()
            logging.info(f"Inference time: {infer_end - infer_start:.2f} seconds")
            self.time_tracker["test"] = infer_end - infer_start

            store_df = pd.DataFrame(store_dict)
            best_result = None
            best_f1 = -float("inf")
            count_start = time.time()

            topkdf = pd.DataFrame(store_df["y_pred_topk"].tolist())
            logging.info("Calculating acc sum")

            hit_df = pd.DataFrame()

            for col in sorted(topkdf.columns):
                topk = col + 1
                hit = (topkdf[col] == store_df["window_labels"]).astype(int)
                hit_df[topk] = hit

                if col == 0:
                    acc_sum = 2 ** topk * hit
                else:
                    acc_sum += 2 ** topk * hit
            acc_sum[acc_sum == 0] = 2 ** (1+(len(topkdf.columns)))
            hit_df["acc_num"] = acc_sum

            for col in sorted(topkdf.columns):
                topk = col+1
                check_num = 2 ** topk 

                store_df[f"window_pred_anomaly_{topk}"] = (
                    ~(hit_df["acc_num"] <= check_num)
                ).astype(int)   
            
            logging.info("Finish generating store_df")

            if self.eval_type == "session":
                use_cols = ["session_idx", "window_anomalies"] + [
                    f"window_pred_anomaly_{topk}" for topk in range(1, self.topk + 1)
                ]
                session_df = (
                    store_df[use_cols].groupby("session_idx", as_index=False).sum()
                )
            else:
                session_df = store_df
            
            for topk in range(1, self.topk+1):
                pred = (session_df[f"window_pred_anomaly_{topk}"] > 0).astype(int)
                y = (session_df["window_anomalies"] > 0).astype(int)
                window_topk_acc = 1 - store_df["window_anomalies"].sum() / len(store_df)

                eval_results = {
                    "f1": f1_score(y, pred),
                    "rc": recall_score(y, pred),
                    "pc": precision_score(y, pred),
                    f"top{topk}-acc": window_topk_acc
                }

                logging.info({k: f"{v:.4f}" for k, v in eval_results.items()})

                if eval_results["f1"] > best_f1:
                    best_result = eval_results
                    best_f1 = eval_results["f1"]
            
            count_end = time.time()
            logging.info(f"Count time: {count_end - count_start:.2f} seconds")
            return best_result
    
    def __input2device(self, batch_input):
        return {k: v.to(self.device) for k, v in batch_input.items()}

    def save_model(self):
        logging.info(f"Saving model to {self.model_save_file}")

        try:
            torch.save(
                self.state_dict(),
                self.model_save_file,
                _use_new_zipfile_serialization=False
            )
        except:
            torch.save(self.state_dict(), self.model_save_file)
    
    def load_model(self, model_save_file=""):
        logging.info(f"Loading model from {self.model_save_file}")
        self.load_state_dict(torch.load(model_save_file, map_location=self.device))
    

    def fit(self, train_loader, test_loader=None, epoches=10, learning_rate=1.0e-3):
        self.to(self.device)
        logging.info(f"Training model on {self.device}...")

        best_f1 = -float("inf")
        best_results = None
        worst_count = 0

        for epoch in range(1, epoches+1):
            epoch_time_start = time.time()

            model = self.train()
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

            batch_cnt = 0
            epoch_loss = 0

            for batch_input in train_loader:
                loss = model.forward(self.__input2device(batch_input))["loss"]

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                epoch_loss == loss.item()
                batch_cnt += 1

            epoch_loss /= batch_cnt
            epoch_time_elapsed = time.time() - epoch_time_start
            logging.info(f"Epoch {epoch} loss: {epoch_loss:.4f}, time: {epoch_time_elapsed:.2f} seconds")

            self.time_tracker["train"] = epoch_time_elapsed
            if test_loader is not None and epoch % 1 == 0:
                eval_results = self.evaluate(test_loader)

                if eval_results["f1"] > best_f1:
                    best_f1 = eval_results["f1"]
                    best_results = eval_results
                    best_results["converge"]= int(epoch)
                    self.save_model()
                    worst_count = 0
                else:
                    worst_count += 1

                if worst_count >= self.patience:
                    logging.info(f"Early stopping at epoch {epoch}.")
                    break    
        self.load_model(self.model_save_file)
        return best_results
