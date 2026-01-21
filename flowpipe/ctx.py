import sys
import os
from abc import ABCMeta

'''
本地测试 为了再次能够运行ctxpipe 需要添加 正确的env环境
'''
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import env
import pandas as pd
import torch
from transformers import (
    AutoModel,
    AutoTokenizer,
    TapasForQuestionAnswering,
    TapasModel,
    TapasTokenizer,
)


class TableEmbedder(metaclass=ABCMeta):
    MAX_N_COLUMNS = 32
    MAX_N_ROWS = 20

    def embed(self, x: pd.DataFrame) -> torch.Tensor:
        raise NotImplementedError


class TextEmbedder(metaclass=ABCMeta):
    def embed(self, x: str) -> torch.Tensor:
        raise NotImplementedError


class GTEEmbedder(TextEmbedder):
    # CTX_MODEL_PATH = "./embed/gte-large"  # 位置更改
    # CTX_MODEL_PATH = "./embed/gte-large-en-v1.5" #  单独运行ctxpipe --位置更改
    CTX_MODEL_PATH = "/data1/data_1/anoy/anoy_pipe/embed/gte-large-en-v1-5"  # ← 用绝对路径

    def __init__(self) -> None:
        # 原文
        # self._ctx_tokenizer = AutoTokenizer.from_pretrained(self.CTX_MODEL_PATH)
        # self._ctx_model = AutoModel.from_pretrained(self.CTX_MODEL_PATH).to(env.DEVICE)

        # 1.5 版本 版本号
        self._ctx_tokenizer = AutoTokenizer.from_pretrained(self.CTX_MODEL_PATH, use_safetensors=True,
                                                            trust_remote_code=True)
        self._ctx_model = AutoModel.from_pretrained(self.CTX_MODEL_PATH, use_safetensors=True,
                                                    trust_remote_code=True).to(env.DEVICE)

    def embed(self, x: str) -> torch.Tensor:
        ctx_dict = self._ctx_tokenizer(
            [x],
            max_length=512,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(env.DEVICE)
        output = self._ctx_model(**ctx_dict)
        attn_mask: torch.Tensor = ctx_dict["attention_mask"]  # type:ignore
        embeddings = (
            self._average_pool(output.last_hidden_state, attn_mask).detach().cpu()
        )
        embeddings = embeddings.squeeze(dim=0)

        # logger.debug(f"embedding shape: {embeddings.shape}") # 1024

        return embeddings

    def embed1(self, x: str) -> torch.Tensor:
        ctx_dict = self._ctx_tokenizer(
            [x],
            max_length=512,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(env.DEVICE)
        output = self._ctx_model(**ctx_dict)
        attn_mask: torch.Tensor = ctx_dict["attention_mask"]  # type:ignore
        embeddings = (
            self._average_pool(output.last_hidden_state, attn_mask).detach().cpu()
        )
        embeddings = embeddings.squeeze(dim=0)

        # logger.debug(f"embedding shape: {embeddings.shape}") # 1024

        return embeddings

    def _average_pool(
            self, last_hidden_states: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        last_hidden = last_hidden_states.masked_fill(
            ~attention_mask[..., None].bool(), 0.0
        )
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


class TapasEmbedder(TableEmbedder):
    CTX_MODEL_PATH = "google/tapas-base"
    QA_MODEL_PATH = "google/tapas-base-finetuned-wtq"

    def __init__(self) -> None:
        self._ctx_tokenizer = TapasTokenizer.from_pretrained(self.CTX_MODEL_PATH)
        self._ctx_model = AutoModel.from_pretrained(self.CTX_MODEL_PATH).to(env.DEVICE)

    def embed(self, x: pd.DataFrame) -> torch.Tensor:
        queries = [
            "Summarize the table",
        ]

        x = x.fillna("NULL").astype(str)
        x.columns = x.columns.map(str)
        x = x.reset_index(drop=True)
        print(x)

        with torch.no_grad():
            ctx_dict = self._ctx_tokenizer(
                table=x, queries=queries, padding="max_length", return_tensors="pt"
            ).to(env.DEVICE)

            output = self._ctx_model(**ctx_dict)
            embeddings = (
                self._average_pool(
                    output.last_hidden_state.clone(), ctx_dict["attention_mask"]
                )
                .detach()
                .cpu()
            )
            embeddings = embeddings.squeeze(dim=0)

            del ctx_dict
            del output

            torch.cuda.empty_cache()

            # logger.debug(f"embedding shape: {embeddings.shape}") # 768

        return embeddings

    def _average_pool(
            self, last_hidden_states: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        last_hidden = last_hidden_states.masked_fill(
            ~attention_mask[..., None].bool(), 0.0
        )
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


embedder = GTEEmbedder()
# embedder = TapasEmbedder()


if __name__ == "__main__":
    data = {
        "Actors": ["Brad Pitt", "Leonardo Di Caprio", "George Clooney"],
        "Age": ["56", "45", "59"],
        "Number of movies": ["87", "53", "69"],
    }
    data = "hello world,wo shi zhong guoren "
    # table = pd.DataFrame.from_dict(data)
    print(env.DEVICE)
    e = GTEEmbedder()
    o = e.embed(data)
    o1 = e.embed("woshi1122")
    print(o == o1)
    print(o1)
    print(o.shape, o)
