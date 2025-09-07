import torch
from torch.utils.data import Dataset
import pandas as pd
from transformers import AutoTokenizer


class AihubTranslationDataset(Dataset):
    """
    AI Hub 번역 데이터셋 CSV 파일을 읽어서 (원문, 번역문)을 (src_text, tgt_text)로 가공.
    - 파일 경로를 받아 내부에서 pd.read_csv() 수행
    - 전처리 함수와 토크나이저를 적용하여 토큰화된(정수로 인코딩된) 시퀀스를 반환
    """

    def __init__(
        self,
        csv_path,
        preprocess_fn=None,  # callable(text, lang) -> str
        max_len=60,
        add_special_tokens=True,
    ):

        # CSV 읽기
        self.df = pd.read_csv(csv_path).reset_index(drop=True)
        self.preprocess_fn = preprocess_fn
        self.max_len = max_len
        self.add_special_tokens = add_special_tokens

        # 텍스트 전처리
        self.src_texts = []
        self.tgt_texts = []
        for i in range(len(self.df)):
            src = str(self.df.loc[i, "원문"])
            tgt = str(self.df.loc[i, "번역문"])
            if self.preprocess_fn is not None:
                src = self.preprocess_fn(src, lang="ko")
                tgt = self.preprocess_fn(tgt, lang="en")
            self.src_texts.append(src)
            self.tgt_texts.append(tgt)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        src_text = self.src_texts[idx]
        tgt_text = self.tgt_texts[idx]

        src_tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")
        tgt_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

        src_enc = src_tokenizer(
            src_text,
            truncation=True,
            max_length=self.max_len,
            add_special_tokens=self.add_special_tokens,
        )
        tgt_enc = tgt_tokenizer(
            tgt_text,
            truncation=True,
            max_length=self.max_len,
            add_special_tokens=self.add_special_tokens,
        )

        return {
            "src_ids": torch.tensor(src_enc["input_ids"], dtype=torch.long),
            "tgt_ids": torch.tensor(tgt_enc["input_ids"], dtype=torch.long),
        }
