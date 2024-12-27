from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
from data_loader_extract import RSTDataset
import logging
import torch

# setlog
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("training.log")],
)


@dataclass
class DataPaths:
    rst_path: str
    nli_data_path: str
    pre_emb_path: str
    hyp_emb_path: str
    lexical_path: str
    pair_graph: str


class Config:
    def __init__(self, **kwargs):
        self.seed = kwargs.get("seed", 42)
        self.stage = kwargs.get("stage", "classification")
        self.mode = kwargs.get("mode", "train")
        # basic path
        self.base_dir = kwargs.get("base_dir", "/mnt/nlp/yuanmengying")
        self.batch_file_size = kwargs.get("batch_file_size", 1)

        # training set
        self.epochs = kwargs.get("epochs", 7)
        self.batch_size = kwargs.get("batch_size", 10)
        self.save_dir = kwargs.get("save_dir", "checkpoints")
        self.save_interval = kwargs.get("save_interval", 5)
        self.log_interval = kwargs.get("log_interval", 100)
        self.use_tensorboard = kwargs.get("use_tensorboard", True)
        self.tensorboard_dir = kwargs.get("tensorboard_dir", "runs")
        self.eval_interval = kwargs.get("eval_interval", 1)

        # optimizer setting
        self.warmup_ratio = kwargs.get("warmup_ratio", 0.1)
        self.lr = kwargs.get("lr", 0.001)
        self.total_steps = kwargs.get("total_steps", 1000)
        self.optimizer_type = kwargs.get("optimizer_type", "adamw")
        self.scheduler_type = kwargs.get("scheduler_type", "linear_warmup")
        self.device = kwargs.get(
            "device", torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        # model setting
        self.model_config = kwargs.get(
            "model_config",
            {
                "in_dim": 1024,
                "hidden_dim": 1024,
                "n_classes": 3,
                "rel_names": [
                    "Temporal",
                    "TextualOrganization",
                    "Joint",
                    "Topic-Comment",
                    "Comparison",
                    "Condition",
                    "Contrast",
                    "Evaluation",
                    "Topic-Change",
                    "Summary",
                    "Manner-Means",
                    "Attribution",
                    "Cause",
                    "Background",
                    "Enablement",
                    "Explanation",
                    "Same-Unit",
                    "Elaboration",
                    "span",
                    "lexical",
                ],
            },
        )

        # init path
        self._init_data_paths()

    def _init_data_paths(self):
        """init all data path"""
        self.paths = {
            "train": DataPaths(
                rst_path=(
                    r"/mnt/nlp/yuanmengying/ymy/data/new_2cd_nli/train/train1/new_rst_result.jsonl"
                ),
                nli_data_path=(
                    r"/mnt/nlp/yuanmengying/ymy/data/nli_type_data/v2/train_re_hyp.json"
                ),
                pre_emb_path=(
                    "/mnt/nlp/yuanmengying/ymy/data/new_2cd_nli/train/pre/node_embeddings.npz"
                ),
                hyp_emb_path=(
                    "/mnt/nlp/yuanmengying/ymy/data/new_2cd_nli/train/hyp/hyp_node_embeddings.npz"
                ),
                lexical_path="/mnt/nlp/yuanmengying/ymy/data/new_2cd_nli/graph_infos/train/lexical_matrixes.npz",
                pair_graph=r"/mnt/nlp/yuanmengying/ymy/data/new_2cd_nli/graph_pairs/train",
            ),
            "dev": DataPaths(
                rst_path=(
                    r"/mnt/nlp/yuanmengying/ymy/data/new_2cd_nli/dev/dev1/new_rst_result.jsonl"
                ),
                nli_data_path=(
                    r"/mnt/nlp/yuanmengying/ymy/data/nli_type_data/v2/dev_re_hyp.json"
                ),
                pre_emb_path=(
                    "/mnt/nlp/yuanmengying/ymy/data/new_2cd_nli/dev/pre/node_embeddings.npz"
                ),
                hyp_emb_path=(
                    "/mnt/nlp/yuanmengying/ymy/data/new_2cd_nli/dev/hyp/hyp_node_embeddings.npz"
                ),
                lexical_path="/mnt/nlp/yuanmengying/ymy/data/new_2cd_nli/graph_infos/dev/lexical_matrixes.npz",
                pair_graph=r"/mnt/nlp/yuanmengying/ymy/data/new_2cd_nli/graph_pairs/dev",
            ),
            "test": DataPaths(
                rst_path=(
                    r"/mnt/nlp/yuanmengying/ymy/data/new_2cd_nli/test/test1/new_rst_result.jsonl"
                ),
                nli_data_path=(
                    r"/mnt/nlp/yuanmengying/ymy/data/nli_type_data/v2/test_re_hyp.json"
                ),
                pre_emb_path=(
                    "/mnt/nlp/yuanmengying/ymy/data/new_2cd_nli/test/pre/node_embeddings.npz"
                ),
                hyp_emb_path=(
                    "/mnt/nlp/yuanmengying/ymy/data/new_2cd_nli/test/hyp/hyp_node_embeddings.npz"
                ),
                lexical_path="/mnt/nlp/yuanmengying/ymy/data/new_2cd_nli/graph_infos/test/lexical_matrixes.npz",
                pair_graph=r"/mnt/nlp/yuanmengying/ymy/data/new_2cd_nli/graph_pairs/test",
            ),
        }

    @classmethod
    def from_dict(cls, config_dict: Dict):
        """set instance from dict"""
        return cls(**config_dict)

    def to_dict(self):
        d = self.__dict__.copy()
        if "device" in d:
            d["device"] = str(d["device"])
        return d

    def get(self, key, default=None):
        return getattr(self, key, default)


def data_model_loader(device):
    # build config
    config = Config(
        save_dir="checkpoints/experiment1",
        tensorboard_dir="runs/experiment1",
        optimizer_type="adamw",
        scheduler_type="linear_warmup",
        device=device,
    )

    # create dataset
    logging.info("Processing train data")
    train_dataset = RSTDataset(
        config.paths["train"].rst_path,
        config.paths["train"].nli_data_path,
        config.paths["train"].pre_emb_path,
        config.paths["train"].lexical_path,
        config.paths["train"].hyp_emb_path,
        config.batch_file_size,
        save_dir=config.paths["train"].pair_graph,
    )

    logging.info("Processing dev data")
    dev_dataset = RSTDataset(
        config.paths["dev"].rst_path,
        config.paths["dev"].nli_data_path,
        config.paths["dev"].pre_emb_path,
        config.paths["dev"].lexical_path,
        config.paths["dev"].hyp_emb_path,
        config.batch_file_size,
        save_dir=config.paths["dev"].pair_graph,
    )

    logging.info("Processing test data")
    test_dataset = RSTDataset(
        config.paths["test"].rst_path,
        config.paths["test"].nli_data_path,
        config.paths["test"].pre_emb_path,
        config.paths["test"].lexical_path,
        config.paths["test"].hyp_emb_path,
        config.batch_file_size,
        save_dir=config.paths["test"].pair_graph,
    )
    config.total_steps = (
        222 // config.batch_size * config.epochs
    )  # data_loader'length * epochs
    config.warmup_steps = int(config.total_steps * config.warmup_ratio)
    config.steps_per_epoch = len(train_dataset) / 3 // config.batch_size
    # init model
    return config, train_dataset, dev_dataset, test_dataset


if __name__ == "__main__":
    config, train_dataset, dev_dataset, test_dataset = data_model_loader()
