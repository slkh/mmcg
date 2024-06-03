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

NORMS = ["APOLOGY", "CRITICISM", "GREETING", "REQUEST", "PERSUASION", "THANKING", "LEAVING", "ADMIRATION", "FINALIZE_DEAL", "REFUSE_REQUEST"]
NORM_MAP = {
    "APOLOGY_ADHERENCES": 1,
    "APOLOGY_VIOLATIONS": 2,
    "CRITICISM_ADHERENCES": 3,
    "CRITICISM_VIOLATIONS": 4,
    "GREETING_ADHERENCES": 5,
    "GREETING_VIOLATIONS": 6,
    "REQUEST_ADHERENCES": 7,
    "REQUEST_VIOLATIONS": 8,
    "PERSUASION_ADHERENCES": 9,
    "PERSUASION_VIOLATIONS": 10,
    "THANKING_ADHERENCES": 11,
    "THANKING_VIOLATIONS": 12,
    "LEAVING_ADHERENCES": 13,
    "LEAVING_VIOLATIONS": 14,
    "ADMIRATION_ADHERENCES": 15,
    "ADMIRATION_VIOLATIONS": 16,
    "FINALIZE_DEAL_ADHERENCES": 17,
    "FINALIZE_DEAL_VIOLATIONS": 18,
    "REFUSE_REQUEST_ADHERENCES": 19,
    "REFUSE_REQUEST_VIOLATIONS": 20,
}


def group_lines(df: pd.DataFrame):
    # group lines by segment_id where the merged text is the concatenation of the text column.
    # the NORM is the minimum of the NORM column
    return df.groupby("segment_id").agg(
        text=("text", "\n".join),
        NORM=("NORM", "min"),
    ).reset_index()


def multiply_minority_classes(df: pd.DataFrame) -> pd.DataFrame:
    # Get the class with the fewest samples
    minority_class = df["NORM"].value_counts().idxmin()
    # Get the number of samples in the minority class
    minority_class_size = df["NORM"].value_counts().min()
    # Get the number of samples in the majority class
    majority_class_size = df["NORM"].value_counts().max()
    # Get the number of times the minority class needs to be multiplied to match the majority class
    multiplier = majority_class_size // minority_class_size
    # Get the remainder of the division
    remainder = majority_class_size % minority_class_size
    # Get the minority class samples
    minority_class_samples = df[df["NORM"] == minority_class]
    # Multiply the minority class samples
    df_multiplied = pd.concat([df, pd.concat([minority_class_samples] * multiplier)])
    # Add the remainder of the division
    df_multiplied = pd.concat([df_multiplied, minority_class_samples.sample(remainder)])
    # Reset the index to maintain a clean index
    df_multiplied.reset_index(drop=True, inplace=True)
    return df_multiplied

def load(num_labels: int, mode: str) -> datasets.Dataset:
    # class types: 1: each norm, 2: each norm adh/vio, 3: adh/vio of ANY norm
    assert num_labels in (2, 3, 20)
    assert mode in ("train", "test")
    if mode == "train":
        filename = "train_wh_utt.jsonl"
    elif mode == "test":
        filename = "test_wh_utt.jsonl"
    df = pd.read_json(os.path.join(NORM_DIR, filename), lines=True)

    # {"file_id": "O01008ZE1", "segment_id": "O01008ZE1_0001", "segment_overlap_pct": 0.1449333333333319, "start": 2396.353, "end": 2398.774, "text": "И весь современный мир — это сейчас микс.", "speaker": "SPEAKER_01", "transcript_overlap_pct": 0.8979760429575127, "NORM": {"APOLOGY_ADHERENCES": 0, "APOLOGY_VIOLATIONS": 0, "CRITICISM_ADHERENCES": 0, "CRITICISM_VIOLATIONS": 0, "GREETING_ADHERENCES": 0, "GREETING_VIOLATIONS": 0, "REQUEST_ADHERENCES": 0, "REQUEST_VIOLATIONS": 0, "PERSUASION_ADHERENCES": 0, "PERSUASION_VIOLATIONS": 0, "THANKING_ADHERENCES": 0, "THANKING_VIOLATIONS": 0, "LEAVING_ADHERENCES": 0, "LEAVING_VIOLATIONS": 0, "ADMIRATION_ADHERENCES": 0, "ADMIRATION_VIOLATIONS": 0, "FINALIZE_DEAL_ADHERENCES": 0, "FINALIZE_DEAL_VIOLATIONS": 0, "REFUSE_REQUEST_ADHERENCES": 0, "REFUSE_REQUEST_VIOLATIONS": 0}, "EMOTION": {"ANGER": 0, "ANTICIPATION": 0, "DISGUST": 0, "FEAR": 0, "JOY": 0, "SADNESS": 0, "SURPRISE": 0, "TRUST": 0, "MULTI_SPEAKER": "False"}, "VA": {"VALENCE_AVG": 543.0, "AROUSAL_AVG": 578.3333333333334, "VALENCE_BINNED_AVG": 3.0, "AROUSAL_BINNED_AVG": 3.333333333333333, "VALENCE_AVG_Z": 0.0240707300729762, "AROUSAL_AVG_Z": 0.1260399460227707}, "catalog_id": "LDC2024E15", "duration": 2.4209999999998217, "url": "https://vk.com/video-76218259_456239289"}
    # drop everything except the text and the norms
    df = df[["text", "NORM", "segment_id"]]
    df['text'] = df['text'].astype(str)
    # flatten the NORM column
    norm_df = pd.json_normalize(df['NORM'])
    df = pd.concat([df, norm_df], axis=1)
    df = df.drop(columns=["NORM"])
    if num_labels == 2:
        # value: 0 for no adherence or violation, 1 for adherence or violation. 
        df['NORM'] = df.apply(lambda row: 1 if any(row[f"{norm}_ADHERENCES"] > 0 or row[f"{norm}_VIOLATIONS"] > 0 for norm in NORMS) else 0, axis=1)
        # make NORM an integer column
        df['NORM'] = df['NORM'].astype(int)
        # drop the individual norm columns
        for norm in NORMS:
            df = df.drop(columns=[f"{norm}_ADHERENCES", f"{norm}_VIOLATIONS"])
        
        df = group_lines(df)
        df = df.drop(columns=["segment_id"])
        # if mode == "train":
        #     df = multiply_minority_classes(df)
    elif num_labels == 3:
#        df['NORM'] = 0
        # if there is an adherence, set to 1, if there is a violation, set to 2, else 0
        df['NORM'] = df.apply(lambda row: 1 if any(row[f"{norm}_ADHERENCES"] > 0 for norm in NORMS) else 2 if any(row[f"{norm}_VIOLATIONS"] > 0 for norm in NORMS) else 0, axis=1)

        for norm in NORMS:
            df = df.drop(columns=[f"{norm}_ADHERENCES", f"{norm}_VIOLATIONS"])
        df['NORM'] = df['NORM'].astype(int)
        df = group_lines(df)
        df = df.drop(columns=["segment_id"])

    elif num_labels == 20:
        # for each norm, whether there is an adherence or violation, set the NORM column to the corresponding index from the NORM_MAP
        df['NORM'] = df.apply(
        lambda row: next(
            (NORM_MAP[f"{norm}_ADHERENCES"] for norm in NORMS if row[f"{norm}_ADHERENCES"] > 0),
                next(
                (NORM_MAP[f"{norm}_VIOLATIONS"] for norm in NORMS if row[f"{norm}_VIOLATIONS"] > 0),
                0
        )
    ), 
    axis=1
)

        # drop the individual norm columns
        for norm in ["APOLOGY", "CRITICISM", "GREETING", "REQUEST", "PERSUASION", "THANKING", "LEAVING", "ADMIRATION", "FINALIZE_DEAL", "REFUSE_REQUEST"]:
            df = df.drop(columns=[f"{norm}_ADHERENCES", f"{norm}_VIOLATIONS"])
        df['NORM'] = df['NORM'].astype(int)
        df = group_lines(df)
        df = df.drop(columns=["segment_id"])

    return datasets.Dataset.from_pandas(df, preserve_index=False)

def load_data(num_labels: int,seed: int = 42) -> datasets.DatasetDict:
    assert num_labels in (2, 3, 20)
    norm_train = load(num_labels, "train")
    norm_test = load(num_labels, "test")
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
