import pandas as pd
import nltk.translate.bleu_score as bleu


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
