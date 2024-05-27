# -*- coding: utf-8 -*-
import os

import datasets
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from ..core.path import dirparent


CB_DIR = os.path.join(dirparent(os.path.realpath(__file__), 3), "data", "cb")
NORM_DIR = os.path.join(dirparent(os.path.realpath(__file__), 3), "data", "norm")
# create a map to convert the norm names with their adher/viol to indices starting from 1 with
NORM_MAP = {
    "APOLOGY_ADHER": 1,
    "APOLOGY_VIOL": 2,
    "CRITICISM_ADHER": 3,
    "CRITICISM_VIOL": 4,
    "GREETING_ADHER": 5,
    "GREETING_VIOL": 6,
    "REQUEST_ADHER": 7,
    "REQUEST_VIOL": 8,
    "PERSUASION_ADHER": 9,
    "PERSUASION_VIOL": 10,
    "THANKING_ADHER": 11,
    "THANKING_VIOL": 12,
    "LEAVING_ADHER": 13,
    "LEAVING_VIOL": 14,
    "ADMIRATION_ADHER": 15,
    "ADMIRATION_VIOL": 16,
    "FINALIZE_DEAL_ADHER": 17,
    "FINALIZE_DEAL_VIOL": 18,
    "REFUSE_REQUEST_ADHER": 19,
    "REFUSE_REQUEST_VIOL": 20,
}



def load(class_type: int, mode: str) -> datasets.Dataset:
    # class types: 1: each norm, 2: each norm adh/vio, 3: adh/vio of ANY norm
    assert class_type in (1, 2, 3)
    assert mode in ("train", "test")
    if mode == "train":
        filename = "train_wh_utt.jsonl"
    elif mode == "test":
        filename = "test_wh_utt.jsonl"
    df = pd.read_json(os.path.join(NORM_DIR, filename), lines=True)

    # {"file_id": "O01008ZE1", "segment_id": "O01008ZE1_0001", "segment_overlap_pct": 0.1449333333333319, "start": 2396.353, "end": 2398.774, "text": "И весь современный мир — это сейчас микс.", "speaker": "SPEAKER_01", "transcript_overlap_pct": 0.8979760429575127, "NORM": {"APOLOGY_ADHERENCES": 0, "APOLOGY_VIOLATIONS": 0, "CRITICISM_ADHERENCES": 0, "CRITICISM_VIOLATIONS": 0, "GREETING_ADHERENCES": 0, "GREETING_VIOLATIONS": 0, "REQUEST_ADHERENCES": 0, "REQUEST_VIOLATIONS": 0, "PERSUASION_ADHERENCES": 0, "PERSUASION_VIOLATIONS": 0, "THANKING_ADHERENCES": 0, "THANKING_VIOLATIONS": 0, "LEAVING_ADHERENCES": 0, "LEAVING_VIOLATIONS": 0, "ADMIRATION_ADHERENCES": 0, "ADMIRATION_VIOLATIONS": 0, "FINALIZE_DEAL_ADHERENCES": 0, "FINALIZE_DEAL_VIOLATIONS": 0, "REFUSE_REQUEST_ADHERENCES": 0, "REFUSE_REQUEST_VIOLATIONS": 0}, "EMOTION": {"ANGER": 0, "ANTICIPATION": 0, "DISGUST": 0, "FEAR": 0, "JOY": 0, "SADNESS": 0, "SURPRISE": 0, "TRUST": 0, "MULTI_SPEAKER": "False"}, "VA": {"VALENCE_AVG": 543.0, "AROUSAL_AVG": 578.3333333333334, "VALENCE_BINNED_AVG": 3.0, "AROUSAL_BINNED_AVG": 3.333333333333333, "VALENCE_AVG_Z": 0.0240707300729762, "AROUSAL_AVG_Z": 0.1260399460227707}, "catalog_id": "LDC2024E15", "duration": 2.4209999999998217, "url": "https://vk.com/video-76218259_456239289"}
    # drop everything except the text and the norms
    df = df[["text", "NORM"]]

    if class_type == 1:
        # combine the adherences and violations for each norm in a single column per norm as a binary 
        # value: 0 for no adherence or violation, 1 for adherence or violation. 
        for norm in ["APOLOGY", "CRITICISM", "GREETING", "REQUEST", "PERSUASION", "THANKING", "LEAVING", "ADMIRATION", "FINALIZE_DEAL", "REFUSE_REQUEST"]:
            df['NORM'] = 1 if df['NORM'][f"{norm}_ADHERENCES"] + df['NORM'][f"{norm}_VIOLATIONS"] > 0 else 0
            df = df.drop(columns=[f"{norm}.{norm}_ADHERENCES", f"{norm}.{norm}_VIOLATIONS"])
    elif class_type == 2:
        # keep the adherences and violations for each norm as separate columns but change the value to 1 if
        # there is an adherence or violation, 0 otherwise
        for norm in ["APOLOGY", "CRITICISM", "GREETING", "REQUEST", "PERSUASION", "THANKING", "LEAVING", "ADMIRATION", "FINALIZE_DEAL", "REFUSE_REQUEST"]:
            df['NORM'][f"{norm}_ADHERENCES"] = 1 if df['NORM'][f"{norm}_ADHERENCES"] > 0 else 0
            df['NORM'][f"{norm}_VIOLATIONS"] = 1 if df['NORM'][f"{norm}_VIOLATIONS"] > 0 else 0
            # convert the multiple adherences and violations columns into a single NORM column with the norm name_ADHER or _VIOL if value is 1 in the respective column
            df['NORM'] = df['NORM'].apply(lambda x: f"{norm}_ADHER" if x[f"{norm}_ADHERENCES"] == 1 else f"{norm}_VIOL" if x[f"{norm}_VIOLATIONS"] == 1 else 0)
            # use NORM_MAP to convert the norm names with their adher/vio to indices starting from 1
            df['NORM'] = df['NORM'].map(NORM_MAP)
            df = df.drop(columns=[f"{norm}.{norm}_ADHERENCES", f"{norm}.{norm}_VIOLATIONS"])
    elif class_type == 3:
        # combine everything into two columns: ADHERENCES and VIOLATIONS, where the value is 1 if there is an
        # adherence or violation, 0 otherwise
        df["ADHERENCES"] = 1 if any(df[f"{norm}.{norm}_ADHERENCES"] > 0 for norm in ["APOLOGY", "CRITICISM", "GREETING", "REQUEST", "PERSUASION", "THANKING", "LEAVING", "ADMIRATION", "FINALIZE_DEAL", "REFUSE_REQUEST"]) else 0
        df["VIOLATIONS"] = 1 if any(df[f"{norm}.{norm}_VIOLATIONS"] > 0 for norm in ["APOLOGY", "CRITICISM", "GREETING", "REQUEST", "PERSUASION", "THANKING", "LEAVING", "ADMIRATION", "FINALIZE_DEAL", "REFUSE_REQUEST"]) else 0

        # drop the individual norm columns
        for norm in ["APOLOGY", "CRITICISM", "GREETING", "REQUEST", "PERSUASION", "THANKING", "LEAVING", "ADMIRATION", "FINALIZE_DEAL", "REFUSE_REQUEST"]:
            df = df.drop(columns=[f"{norm}.{norm}_ADHERENCES", f"{norm}.{norm}_VIOLATIONS"])
    # flatten the NORM column and then return the dataset
    df = df.assign(**df["NORM"].apply(pd.Series))
    df = df.drop(columns=["NORM"])

    return datasets.Dataset.from_pandas(df, preserve_index=False)

def load_data(class_type: int,seed: int = 42) -> datasets.DatasetDict:
    assert class_type in (1, 2, 3)
    norm_train = load(class_type, "train")
    norm_test = load(class_type, "test")
    return datasets.DatasetDict({
        "train": norm_train,
        "test": norm_test,
    })

def load_opensmile() -> pd.DataFrame:
    df = pd.read_csv(os.path.join(CB_DIR, "opensmile.csv"))
    df = df.assign(
        audio_file=df.file.str.replace("_features", "").str.replace(".csv", ".wav")
    )
    fcols = [c for c in df.columns if c.isnumeric()]
    features = df[fcols].to_numpy()
    return pd.DataFrame({
        "opensmile_features": list(features),
        "audio_file": df.audio_file
    })
