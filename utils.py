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
