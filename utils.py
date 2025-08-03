import re
import numpy as np
import torch
import torch.nn as nn


def preprocess_text(text, lang="ko"):
    text = re.sub(r"\(.*?\)", "", text)  # 괄호 안 텍스트 삭제
    text = re.sub(r'[·“”"-]', " ", text)  # 문장부호 (,-·“) 공백으로 대체
    text = re.sub(r"[,]", "", text)  # 쉼표 삭제

    if lang == "ko":
        text = re.sub(r"\$", " 달러 ", text)
        text = re.sub(r"(\d+)\%", r"\1 퍼센트 ", text)

    elif lang == "en":
        text = re.sub(r"\$", " dollars ", text)
        text = re.sub(r"(\d+)\%", r"\1 percent ", text)

    text = re.sub(r"\s+", " ", text).strip()  # 연속된 공백을 하나로

    return text


def get_causality_mask(input_len):
    q_grid, k_grid = torch.meshgrid(torch.arange(input_len), torch.arange(input_len))
    causality_mask = torch.where(
        q_grid < k_grid, float("-inf"), torch.zeros_like(q_grid, dtype=torch.float32)
    )
    return causality_mask


def get_padding_mask(input, pad_token_id=0):
    # (batch_size, 1, 1, input_len)
    mask = (input == pad_token_id).unsqueeze(1).unsqueeze(2)
    padding_mask = mask.float() * float("-inf")
    return padding_mask