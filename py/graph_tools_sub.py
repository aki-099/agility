import pandas as pd
import numpy as np
import scipy
import matplotlib.pyplot as plt
import os
import japanize_matplotlib

japanize_matplotlib.japanize()
import warnings

warnings.filterwarnings("ignore")


def calculate_body_part_center(df, target):
    for axis in ["x", "y", "z"]:
        if target != "MOUTH":
            df[f"CENTER_{target}_{axis}"] = df[
                [f"LEFT_{target}_{axis}", f"RIGHT_{target}_{axis}"]
            ].mean(axis="columns")
        else:
            df[f"CENTER_{target}_{axis}"] = df[
                [f"{target}_LEFT_{axis}", f"{target}_RIGHT_{axis}"]
            ].mean(axis="columns")
    return df


# 部位の中心を求める
def initialize_center_values(df):
    targets = ["MOUTH", "SHOULDER", "HIP", "ELBOW", "WRIST"]
    for target in targets:
        df = calculate_body_part_center(df, target)
    return df


# 部位のベクトルを取得する
def get_body_vectors(df, frame):
    target_dict = {}

    LEFT_EYE_x = df[["LEFT_EYE_INNER_x", "LEFT_EYE_OUTER_x"]].loc[frame]
    LEFT_EYE_y = df[["LEFT_EYE_INNER_y", "LEFT_EYE_OUTER_y"]].loc[frame]
    LEFT_EYE_z = df[["LEFT_EYE_INNER_z", "LEFT_EYE_OUTER_z"]].loc[frame]
    target_dict["LEFT_EYE_x"] = LEFT_EYE_x
    target_dict["LEFT_EYE_y"] = LEFT_EYE_y
    target_dict["LEFT_EYE_z"] = LEFT_EYE_z

    RIGHT_EYE_x = df[["RIGHT_EYE_INNER_x", "RIGHT_EYE_OUTER_x"]].loc[frame]
    RIGHT_EYE_y = df[["RIGHT_EYE_INNER_y", "RIGHT_EYE_OUTER_y"]].loc[frame]
    RIGHT_EYE_z = df[["RIGHT_EYE_INNER_z", "RIGHT_EYE_OUTER_z"]].loc[frame]
    target_dict["RIGHT_EYE_x"] = RIGHT_EYE_x
    target_dict["RIGHT_EYE_y"] = RIGHT_EYE_y
    target_dict["RIGHT_EYE_z"] = RIGHT_EYE_z

    MOUTH_x = df[["MOUTH_LEFT_x", "MOUTH_RIGHT_x"]].loc[frame]
    MOUTH_y = df[["MOUTH_LEFT_y", "MOUTH_RIGHT_y"]].loc[frame]
    MOUTH_z = df[["MOUTH_LEFT_z", "MOUTH_RIGHT_z"]].loc[frame]
    target_dict["MOUTH_x"] = MOUTH_x
    target_dict["MOUTH_y"] = MOUTH_y
    target_dict["MOUTH_z"] = MOUTH_z

    SHOULDER_x = df[["LEFT_SHOULDER_x", "RIGHT_SHOULDER_x"]].loc[frame]
    SHOULDER_y = df[["LEFT_SHOULDER_y", "RIGHT_SHOULDER_y"]].loc[frame]
    SHOULDER_z = df[["LEFT_SHOULDER_z", "RIGHT_SHOULDER_z"]].loc[frame]
    target_dict["SHOULDER_x"] = SHOULDER_x
    target_dict["SHOULDER_y"] = SHOULDER_y
    target_dict["SHOULDER_z"] = SHOULDER_z

    LEFT_UPPER_ARM_x = df[["LEFT_SHOULDER_x", "LEFT_ELBOW_x"]].loc[frame]
    LEFT_UPPER_ARM_y = df[["LEFT_SHOULDER_y", "LEFT_ELBOW_y"]].loc[frame]
    LEFT_UPPER_ARM_z = df[["LEFT_SHOULDER_z", "LEFT_ELBOW_z"]].loc[frame]
    target_dict["LEFT_UPPER_ARM_x"] = LEFT_UPPER_ARM_x
    target_dict["LEFT_UPPER_ARM_y"] = LEFT_UPPER_ARM_y
    target_dict["LEFT_UPPER_ARM_z"] = LEFT_UPPER_ARM_z

    RIGHT_UPPER_ARM_x = df[["RIGHT_SHOULDER_x", "RIGHT_ELBOW_x"]].loc[frame]
    RIGHT_UPPER_ARM_y = df[["RIGHT_SHOULDER_y", "RIGHT_ELBOW_y"]].loc[frame]
    RIGHT_UPPER_ARM_z = df[["RIGHT_SHOULDER_z", "RIGHT_ELBOW_z"]].loc[frame]
    target_dict["RIGHT_UPPER_ARM_x"] = RIGHT_UPPER_ARM_x
    target_dict["RIGHT_UPPER_ARM_y"] = RIGHT_UPPER_ARM_y
    target_dict["RIGHT_UPPER_ARM_z"] = RIGHT_UPPER_ARM_z

    LEFT_FOREARM_x = df[["LEFT_ELBOW_x", "LEFT_WRIST_x"]].loc[frame]
    LEFT_FOREARM_y = df[["LEFT_ELBOW_y", "LEFT_WRIST_y"]].loc[frame]
    LEFT_FOREARM_z = df[["LEFT_ELBOW_z", "LEFT_WRIST_z"]].loc[frame]
    target_dict["LEFT_FOREARM_x"] = LEFT_FOREARM_x
    target_dict["LEFT_FOREARM_y"] = LEFT_FOREARM_y
    target_dict["LEFT_FOREARM_z"] = LEFT_FOREARM_z

    RIGHT_FOREARM_x = df[["RIGHT_ELBOW_x", "RIGHT_WRIST_x"]].loc[frame]
    RIGHT_FOREARM_y = df[["RIGHT_ELBOW_y", "RIGHT_WRIST_y"]].loc[frame]
    RIGHT_FOREARM_z = df[["RIGHT_ELBOW_z", "RIGHT_WRIST_z"]].loc[frame]
    target_dict["RIGHT_FOREARM_x"] = RIGHT_FOREARM_x
    target_dict["RIGHT_FOREARM_y"] = RIGHT_FOREARM_y
    target_dict["RIGHT_FOREARM_z"] = RIGHT_FOREARM_z

    HIP_x = df[["LEFT_HIP_x", "RIGHT_HIP_x"]].loc[frame]
    HIP_y = df[["LEFT_HIP_y", "RIGHT_HIP_y"]].loc[frame]
    HIP_z = df[["LEFT_HIP_z", "RIGHT_HIP_z"]].loc[frame]
    target_dict["HIP_x"] = HIP_x
    target_dict["HIP_y"] = HIP_y
    target_dict["HIP_z"] = HIP_z

    LEFT_BODY_x = df[["LEFT_SHOULDER_x", "LEFT_HIP_x"]].loc[frame]
    LEFT_BODY_y = df[["LEFT_SHOULDER_y", "LEFT_HIP_y"]].loc[frame]
    LEFT_BODY_z = df[["LEFT_SHOULDER_z", "LEFT_HIP_z"]].loc[frame]
    target_dict["LEFT_BODY_x"] = LEFT_BODY_x
    target_dict["LEFT_BODY_y"] = LEFT_BODY_y
    target_dict["LEFT_BODY_z"] = LEFT_BODY_z

    RIGHT_BODY_x = df[["RIGHT_SHOULDER_x", "RIGHT_HIP_x"]].loc[frame]
    RIGHT_BODY_y = df[["RIGHT_SHOULDER_y", "RIGHT_HIP_y"]].loc[frame]
    RIGHT_BODY_z = df[["RIGHT_SHOULDER_z", "RIGHT_HIP_z"]].loc[frame]
    target_dict["RIGHT_BODY_x"] = RIGHT_BODY_x
    target_dict["RIGHT_BODY_y"] = RIGHT_BODY_y
    target_dict["RIGHT_BODY_z"] = RIGHT_BODY_z

    LEFT_THIGH_x = df[["LEFT_HIP_x", "LEFT_KNEE_x"]].loc[frame]
    LEFT_THIGH_y = df[["LEFT_HIP_y", "LEFT_KNEE_y"]].loc[frame]
    LEFT_THIGH_z = df[["LEFT_HIP_z", "LEFT_KNEE_z"]].loc[frame]
    target_dict["LEFT_THIGH_x"] = LEFT_THIGH_x
    target_dict["LEFT_THIGH_y"] = LEFT_THIGH_y
    target_dict["LEFT_THIGH_z"] = LEFT_THIGH_z

    RIGHT_THIGH_x = df[["RIGHT_HIP_x", "RIGHT_KNEE_x"]].loc[frame]
    RIGHT_THIGH_y = df[["RIGHT_HIP_y", "RIGHT_KNEE_y"]].loc[frame]
    RIGHT_THIGH_z = df[["RIGHT_HIP_z", "RIGHT_KNEE_z"]].loc[frame]
    target_dict["RIGHT_THIGH_x"] = RIGHT_THIGH_x
    target_dict["RIGHT_THIGH_y"] = RIGHT_THIGH_y
    target_dict["RIGHT_THIGH_z"] = RIGHT_THIGH_z

    LEFT_LOWER_LEG_x = df[["LEFT_KNEE_x", "LEFT_ANKLE_x"]].loc[frame]
    LEFT_LOWER_LEG_y = df[["LEFT_KNEE_y", "LEFT_ANKLE_y"]].loc[frame]
    LEFT_LOWER_LEG_z = df[["LEFT_KNEE_z", "LEFT_ANKLE_z"]].loc[frame]
    target_dict["LEFT_LOWER_LEG_x"] = LEFT_LOWER_LEG_x
    target_dict["LEFT_LOWER_LEG_y"] = LEFT_LOWER_LEG_y
    target_dict["LEFT_LOWER_LEG_z"] = LEFT_LOWER_LEG_z

    RIGHT_LOWER_LEG_x = df[["RIGHT_KNEE_x", "RIGHT_ANKLE_x"]].loc[frame]
    RIGHT_LOWER_LEG_y = df[["RIGHT_KNEE_y", "RIGHT_ANKLE_y"]].loc[frame]
    RIGHT_LOWER_LEG_z = df[["RIGHT_KNEE_z", "RIGHT_ANKLE_z"]].loc[frame]
    target_dict["RIGHT_LOWER_LEG_x"] = RIGHT_LOWER_LEG_x
    target_dict["RIGHT_LOWER_LEG_y"] = RIGHT_LOWER_LEG_y
    target_dict["RIGHT_LOWER_LEG_z"] = RIGHT_LOWER_LEG_z

    LEFT_INSTEP_x = df[["LEFT_ANKLE_x", "LEFT_FOOT_INDEX_x"]].loc[frame]
    LEFT_INSTEP_y = df[["LEFT_ANKLE_y", "LEFT_FOOT_INDEX_y"]].loc[frame]
    LEFT_INSTEP_z = df[["LEFT_ANKLE_z", "LEFT_FOOT_INDEX_z"]].loc[frame]
    target_dict["LEFT_INSTEP_x"] = LEFT_INSTEP_x
    target_dict["LEFT_INSTEP_y"] = LEFT_INSTEP_y
    target_dict["LEFT_INSTEP_z"] = LEFT_INSTEP_z

    RIGHT_INSTEP_x = df[["RIGHT_ANKLE_x", "RIGHT_FOOT_INDEX_x"]].loc[frame]
    RIGHT_INSTEP_y = df[["RIGHT_ANKLE_y", "RIGHT_FOOT_INDEX_y"]].loc[frame]
    RIGHT_INSTEP_z = df[["RIGHT_ANKLE_z", "RIGHT_FOOT_INDEX_z"]].loc[frame]
    target_dict["RIGHT_INSTEP_x"] = RIGHT_INSTEP_x
    target_dict["RIGHT_INSTEP_y"] = RIGHT_INSTEP_y
    target_dict["RIGHT_INSTEP_z"] = RIGHT_INSTEP_z

    LEFT_ACHILLES_x = df[["LEFT_ANKLE_x", "LEFT_HEEL_x"]].loc[frame]
    LEFT_ACHILLES_y = df[["LEFT_ANKLE_y", "LEFT_HEEL_y"]].loc[frame]
    LEFT_ACHILLES_z = df[["LEFT_ANKLE_z", "LEFT_HEEL_z"]].loc[frame]
    target_dict["LEFT_ACHILLES_x"] = LEFT_ACHILLES_x
    target_dict["LEFT_ACHILLES_y"] = LEFT_ACHILLES_y
    target_dict["LEFT_ACHILLES_z"] = LEFT_ACHILLES_z

    RIGHT_ACHILLES_x = df[["RIGHT_ANKLE_x", "RIGHT_HEEL_x"]].loc[frame]
    RIGHT_ACHILLES_y = df[["RIGHT_ANKLE_y", "RIGHT_HEEL_y"]].loc[frame]
    RIGHT_ACHILLES_z = df[["RIGHT_ANKLE_z", "RIGHT_HEEL_z"]].loc[frame]
    target_dict["RIGHT_ACHILLES_x"] = RIGHT_ACHILLES_x
    target_dict["RIGHT_ACHILLES_y"] = RIGHT_ACHILLES_y
    target_dict["RIGHT_ACHILLES_z"] = RIGHT_ACHILLES_z

    LEFT_FOOT_x = df[["LEFT_FOOT_INDEX_x", "LEFT_HEEL_x"]].loc[frame]
    LEFT_FOOT_y = df[["LEFT_FOOT_INDEX_y", "LEFT_HEEL_y"]].loc[frame]
    LEFT_FOOT_z = df[["LEFT_FOOT_INDEX_z", "LEFT_HEEL_z"]].loc[frame]
    target_dict["LEFT_FOOT_x"] = LEFT_FOOT_x
    target_dict["LEFT_FOOT_y"] = LEFT_FOOT_y
    target_dict["LEFT_FOOT_z"] = LEFT_FOOT_z

    RIGHT_FOOT_x = df[["RIGHT_FOOT_INDEX_x", "RIGHT_HEEL_x"]].loc[frame]
    RIGHT_FOOT_y = df[["RIGHT_FOOT_INDEX_y", "RIGHT_HEEL_y"]].loc[frame]
    RIGHT_FOOT_z = df[["RIGHT_FOOT_INDEX_z", "RIGHT_HEEL_z"]].loc[frame]
    target_dict["RIGHT_FOOT_x"] = RIGHT_FOOT_x
    target_dict["RIGHT_FOOT_y"] = RIGHT_FOOT_y
    target_dict["RIGHT_FOOT_z"] = RIGHT_FOOT_z

    CENTER_CORE_x = df[["CENTER_HIP_x", "CENTER_SHOULDER_x"]].loc[frame]
    CENTER_CORE_y = df[["CENTER_HIP_y", "CENTER_SHOULDER_y"]].loc[frame]
    CENTER_CORE_z = df[["CENTER_HIP_z", "CENTER_SHOULDER_z"]].loc[frame]

    target_dict["CENTER_CORE_x"] = CENTER_CORE_x
    target_dict["CENTER_CORE_y"] = CENTER_CORE_y
    target_dict["CENTER_CORE_z"] = CENTER_CORE_z

    return target_dict


def calculate_angle(df: pd.DataFrame, vnames: list, mode: str):
    if len(vnames) <= 1:
        raise ValueError("vnames should be 2 or 3 or 4")
    if len(vnames) == 2:
        vector_a = df[[f"{vnames[0]}_{axis}" for axis in mode]].values
        vector_b = df[[f"{vnames[1]}_{axis}" for axis in mode]].values
    if len(vnames) == 3:
        vector_a = df[[f"{vnames[0]}_{axis}" for axis in mode]].values
        vector_b = df[[f"{vnames[1]}_{axis}" for axis in mode]].values
        vector_c = df[[f"{vnames[2]}_{axis}" for axis in mode]].values
    if len(vnames) == 4:
        vector_a = df[[f"{vnames[0]}_{axis}" for axis in mode]].values
        vector_b = df[[f"{vnames[1]}_{axis}" for axis in mode]].values
        vector_c = df[[f"{vnames[2]}_{axis}" for axis in mode]].values
        vector_d = df[[f"{vnames[3]}_{axis}" for axis in mode]].values
    if len(vnames) >= 5:
        raise ValueError("vnames should be 2 or 3 or 4")

    if len(vnames) == 2:
        list1 = vector_a
        list2 = vector_b
    elif len(vnames) == 3:
        list1 = vector_a - vector_b
        list2 = vector_c - vector_b
    elif len(vnames) == 4:
        list1 = vector_a - vector_b
        list2 = vector_c - vector_d
    else:
        raise ValueError("vnames should be 2 or 3 or 4")

    degree_list = []
    for vec1, vec2 in zip(list1, list2):
        degree_list.append(calculate_angle_between_vectors(vec1, vec2))
    return degree_list


def calculate_angle_between_vectors(vec_a, vec_b):
    dot_product = np.dot(vec_a, vec_b)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)

    if norm_a == 0 or norm_b == 0:
        return np.NaN
    else:
        cos_theta = dot_product / (norm_a * norm_b)
        theta = np.arccos(cos_theta)
        return np.degrees(theta)


# 2D座標データから母指球の位置を計算
def cal_ball_foot(df):
    # 2D座標データから母指球の位置を計算
    for xyz in ["x", "y", "z"]:
        for lr in ["LEFT", "RIGHT"]:
            df[f"{lr}_BALL_FOOT_{xyz}"] = (
                df[f"{lr}_FOOT_INDEX_{xyz}"] * 3 + df[f"{lr}_HEEL_{xyz}"]
            ) / 4
    return df


# 角度とラベル名の定義
def define_angle() -> list:
    vectors = [
        [
            ["CENTER_SHOULDER_x", "CENTER_SHOULDER_y", "CENTER_SHOULDER_z"],
            ["CENTER_HIP_x", "CENTER_HIP_y", "CENTER_HIP_z"],
            ["LEFT_BALL_FOOT_x", "LEFT_BALL_FOOT_y", "CENTER_HIP_z"],
        ],
        [
            ["LEFT_HIP_x", "LEFT_HIP_y", "LEFT_HIP_z"],
            ["LEFT_KNEE_x", "LEFT_KNEE_y", "LEFT_KNEE_z"],
            ["LEFT_ANKLE_x", "LEFT_ANKLE_y", "LEFT_ANKLE_z"],
        ],
        [
            ["LEFT_KNEE_x", "LEFT_KNEE_y", "LEFT_KNEE_z"],
            ["LEFT_ANKLE_x", "LEFT_ANKLE_y", "LEFT_ANKLE_z"],
            ["LEFT_FOOT_INDEX_x", "LEFT_FOOT_INDEX_y", "LEFT_FOOT_INDEX_z"],
        ],
    ]

    labels = ["body", "knee", "ankle"]

    return vectors, labels


# 3D座標データから角度を計算する
def process(df: pd.DataFrame) -> pd.DataFrame:
    vectors, labels = define_angle()
    for i in range(len(labels)):
        df = cal_angle(df, vectors[i], labels[i])
    return df


# 2つのベクトルから角度を計算する
def cal_angle(df: pd.DataFrame, l: list, label: str) -> list:
    l0 = l[0]
    l1 = l[1]
    l2 = l[2]
    a = df[l0].values
    b = df[l1].values
    c = df[l2].values

    ab = a - b
    cb = c - b

    angle = []
    for a, b in zip(ab, cb):
        dot_product = np.dot(a, b)

        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        cos_theta = dot_product / (norm_a * norm_b)
        theta = np.arccos(cos_theta)
        angle_degrees = np.degrees(theta)

        angle.append(angle_degrees)
    df[label] = angle
    return df
