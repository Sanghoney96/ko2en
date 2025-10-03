import pandas as pd
import nltk.translate.bleu_score as bleu
from utils import preprocess_text
from transformers import AutoTokenizer
import torch


def translate(src_sentence, max_len, model, device):
    """
    원문 -> 전처리 -> 토큰화 -> 모델 추론 -> 디코딩
    """
    src_text = preprocess_text(src_sentence, lang="ko")

    src_tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")
    tgt_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    src_ids = src_tokenizer(
        src_text, truncation=True, max_length=max_len, add_special_tokens=True
    )
    src_ids = torch.tensor(src_ids["input_ids"], dtype=torch.long)

    model.eval()

    return


def get_bleu_score(predictions, targets):
    """
    원문을 번역한 문장(predictions)과 번역문(targets)의 pandas dataframe을 입력하여 전체 test set에 대한 bleu score 계산
    """
    s = pd.concat([predictions, targets], axis=1).dropna()

    predictions = s.iloc[:, 0].astype(str).tolist()
    targets = s.iloc[:, 1].astype(str).tolist()

    predictions = [prediction.strip().split() for prediction in predictions]
    targets = [[target.strip().split()] for target in targets]

    bleu_scores = bleu.corpus_bleu(targets, predictions)

    return bleu_scores


def beam_search():
    return
