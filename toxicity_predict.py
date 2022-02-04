import time
from typing import Union, Dict, Optional, List
import json

import googleapiclient.errors
import pandas as pd
from detoxify import Detoxify
from googleapiclient import discovery
from tqdm import tqdm

from perspective_api_key import API_KEY


def toxic_bert_predict(texts: List[str], model: Optional[str]=None) -> Dict[str, List[float]]:
    if model is None:
        results = {}
        for m in ["unbiased", "original", "multilingual"]:
            ind_result = Detoxify(m).predict(texts)
            results.update({f"{model}_{k}": v for k,v in ind_result.items()})
    else:
        assert model in ["original", "unbiased", "multilingual"]
        results = Detoxify(model).predict(texts)
        results = {f"{model}_{k}":v for k,v in results.items()}
    return results


def perspective_predict(texts: List[str]) -> Dict[str, List[float]]:
    client = discovery.build(
        "commentanalyzer",
        "v1alpha1",
        developerKey=API_KEY,
        discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
        static_discovery=False,
    )
    output = {'TOXICITY': [], "SEVERE_TOXICITY": [], "IDENTITY_ATTACK": [], "INSULT": [],
                                    "PROFANITY": [], "THREAT": []}
    for text in tqdm(texts):
        analyze_request = {
            'comment': {'text': text},
            'requestedAttributes': {'TOXICITY': {}, "SEVERE_TOXICITY": {}, "IDENTITY_ATTACK": {}, "INSULT": {},
                                    "PROFANITY": {}, "THREAT": {}}
        }
        try:
            response = client.comments().analyze(body=analyze_request).execute()
        except googleapiclient.errors.HttpError as e:
            try:
                analyze_request["requestedAttributes"] = {"TOXICITY": {}}
                response = client.comments().analyze(body=analyze_request).execute()
            except googleapiclient.errors.HttpError as e:
                response = {"attributeScores": {'TOXICITY': {}, "SEVERE_TOXICITY": {}, "IDENTITY_ATTACK": {}, "INSULT": {},
                                                "PROFANITY": {}, "THREAT": {}}}
                print(f"Request HTTP Error: {e}")
        for attr, attr_score in response["attributeScores"].items():
            output[attr].append(attr_score.get("summaryScore", {}).get("value", -1.0))
        time.sleep(0.5)
    return output


def predict_instagram(content: str, model: str):
    if content == "caption":
        df = pd.read_csv("data/caption_4emotions_bertweetV2.csv")
        text = list(df["Contents"].fillna(""))
    else:
        df = pd.read_csv("data/cleaned_22676_4emotions_bertweetV2.csv")
        text = list(df["comment"].fillna(""))

    if model == "perspective":
        pred = perspective_predict(text)
    else:
        pred = toxic_bert_predict(text, model)

    pred_df = pd.DataFrame(pred)
    df = df.append(pred_df)
    return df


if __name__ == "__main__":
    # out = toxic_bert_predict(["Chinese people eat crazy things like chicken feet", "I hate chinese people"])
    # df = predict_instagram("caption", "original")
    df = predict_instagram("caption", "perspective")
    pass