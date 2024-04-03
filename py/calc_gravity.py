import pandas as pd
import numpy as np


#  四角形の重心を求める関数
def calculate_gravity(xa, ya, za, xb, yb, zb, xc, yc, zc, xd, yd, zd):
    """
    四角形の重心を求める関数
    xa, ya, za : 頂点Aの座標
    xb, yb, zb : 頂点Bの座標
    xc, yc, zc : 頂点Cの座標
    xd, yd, zd : 頂点Dの座標
    G_x, G_y, G_z : 四角形の重心の座標
    """
    G_abc_x = (xa + xb + xc) / 3
    G_abc_y = (ya + yb + yc) / 3
    G_abc_z = (za + zb + zc) / 3

    G_adc_x = (xa + xd + xc) / 3
    G_adc_y = (ya + yd + yc) / 3
    G_adc_z = (za + zd + zc) / 3

    ab_x = xb - xa
    ab_y = yb - ya
    ab_z = zb - za
    ac_x = xc - xa
    ac_y = yc - ya
    ac_z = zc - za

    ad_x = xd - xa
    ad_y = yd - ya
    ad_z = zd - za

    # ベクトルABとベクトルACの外積を計算
    cross_product_x_abc = ab_y * ac_z - ab_z * ac_y
    cross_product_y_abc = ab_z * ac_x - ab_x * ac_z
    cross_product_z_abc = ab_x * ac_y - ab_y * ac_x

    cross_product_x_adc = ad_y * ac_z - ad_z * ac_y
    cross_product_y_adc = ad_z * ac_x - ad_x * ac_z
    cross_product_z_adc = ad_x * ac_y - ad_y * ac_x

    # 外積の大きさを求めて面積を計算
    S_abc = 0.5 * np.sqrt(
        cross_product_x_abc**2
        + cross_product_y_abc**2
        + cross_product_z_abc**2
    )
    S_adc = 0.5 * np.sqrt(
        cross_product_x_adc**2
        + cross_product_y_adc**2
        + cross_product_z_adc**2
    )

    ratio = S_adc / S_abc

    G_x = G_abc_x + (G_adc_x - G_abc_x) * ratio
    G_y = G_abc_y + (G_adc_y - G_abc_y) * ratio
    G_z = G_abc_z + (G_adc_z - G_abc_z) * ratio

    return G_x, G_y, G_z


# 各部位における重心を求める関数
def cal_center(df: pd.DataFrame) -> pd.DataFrame:
    """
    BlazePoseの出力から,各部位の重心を求める関数
    df: DataFrame
    """
    # 頭の重心
    df["center_head_x"] = df["NOSE_x"]
    df["center_head_y"] = df["NOSE_y"]
    df["center_head_z"] = df["NOSE_z"]

    # 左上腕の重心
    df["upper_arm_left_x"] = (df["LEFT_SHOULDER_x"] + df["LEFT_ELBOW_x"]) / 2
    df["upper_arm_left_y"] = (df["LEFT_SHOULDER_y"] + df["LEFT_ELBOW_y"]) / 2
    df["upper_arm_left_z"] = (df["LEFT_SHOULDER_z"] + df["LEFT_ELBOW_z"]) / 2

    # 右上腕の重心
    df["upper_arm_right_x"] = (
        df["RIGHT_SHOULDER_x"] + df["RIGHT_ELBOW_x"]
    ) / 2
    df["upper_arm_right_y"] = (
        df["RIGHT_SHOULDER_y"] + df["RIGHT_ELBOW_y"]
    ) / 2
    df["upper_arm_right_z"] = (
        df["RIGHT_SHOULDER_z"] + df["RIGHT_ELBOW_z"]
    ) / 2

    # 左下腕の重心
    df["bottom_arm_left_x"] = (df["LEFT_ELBOW_x"] + df["LEFT_WRIST_x"]) / 2
    df["bottom_arm_left_y"] = (df["LEFT_ELBOW_y"] + df["LEFT_WRIST_y"]) / 2
    df["bottom_arm_left_z"] = (df["LEFT_ELBOW_z"] + df["LEFT_WRIST_z"]) / 2

    # 右下腕の重心
    df["bottom_arm_right_x"] = (df["RIGHT_ELBOW_x"] + df["RIGHT_WRIST_x"]) / 2
    df["bottom_arm_right_y"] = (df["RIGHT_ELBOW_y"] + df["RIGHT_WRIST_y"]) / 2
    df["bottom_arm_right_z"] = (df["RIGHT_ELBOW_z"] + df["RIGHT_WRIST_z"]) / 2

    # 左上腿の重心
    df["upper_leg_left_x"] = (df["LEFT_HIP_x"] + df["LEFT_KNEE_x"]) / 2
    df["upper_leg_left_y"] = (df["LEFT_HIP_y"] + df["LEFT_KNEE_y"]) / 2
    df["upper_leg_left_z"] = (df["LEFT_HIP_z"] + df["LEFT_KNEE_z"]) / 2

    # 右上腿の重心
    df["upper_leg_right_x"] = (df["RIGHT_HIP_x"] + df["RIGHT_KNEE_x"]) / 2
    df["upper_leg_right_y"] = (df["RIGHT_HIP_y"] + df["RIGHT_KNEE_y"]) / 2
    df["upper_leg_right_z"] = (df["RIGHT_HIP_z"] + df["RIGHT_KNEE_z"]) / 2

    # 左下腿の重心
    df["bottom_leg_left_x"] = (df["LEFT_KNEE_x"] + df["LEFT_ANKLE_x"]) / 2
    df["bottom_leg_left_y"] = (df["LEFT_KNEE_y"] + df["LEFT_ANKLE_y"]) / 2
    df["bottom_leg_left_z"] = (df["LEFT_KNEE_z"] + df["LEFT_ANKLE_z"]) / 2

    # 右下腿の重心
    df["bottom_leg_right_x"] = (df["RIGHT_KNEE_x"] + df["RIGHT_ANKLE_x"]) / 2
    df["bottom_leg_right_y"] = (df["RIGHT_KNEE_y"] + df["RIGHT_ANKLE_y"]) / 2
    df["bottom_leg_right_z"] = (df["RIGHT_KNEE_z"] + df["RIGHT_ANKLE_z"]) / 2

    # 左足の重心
    df["center_foot_left_x"] = (
        df["LEFT_ANKLE_x"] + df["LEFT_FOOT_INDEX_x"] + df["LEFT_HEEL_x"]
    ) / 3
    df["center_foot_left_y"] = (
        df["LEFT_ANKLE_y"] + df["LEFT_FOOT_INDEX_y"] + df["LEFT_HEEL_y"]
    ) / 3
    df["center_foot_left_z"] = (
        df["LEFT_ANKLE_z"] + df["LEFT_FOOT_INDEX_z"] + df["LEFT_HEEL_z"]
    ) / 3

    # 右足の重心
    df["center_foot_right_x"] = (
        df["RIGHT_ANKLE_x"] + df["RIGHT_FOOT_INDEX_x"] + df["RIGHT_HEEL_x"]
    ) / 3
    df["center_foot_right_y"] = (
        df["RIGHT_ANKLE_y"] + df["RIGHT_FOOT_INDEX_y"] + df["RIGHT_HEEL_y"]
    ) / 3
    df["center_foot_right_z"] = (
        df["RIGHT_ANKLE_z"] + df["RIGHT_FOOT_INDEX_z"] + df["RIGHT_HEEL_z"]
    ) / 3

    # 右手の重心
    xa = df["RIGHT_WRIST_x"]
    ya = df["RIGHT_WRIST_y"]
    za = df["RIGHT_WRIST_z"]
    xb = df["RIGHT_THUMB_x"]
    yb = df["RIGHT_THUMB_y"]
    zb = df["RIGHT_THUMB_z"]
    xc = df["RIGHT_INDEX_x"]
    yc = df["RIGHT_INDEX_y"]
    zc = df["RIGHT_INDEX_z"]
    xd = df["RIGHT_PINKY_x"]
    yd = df["RIGHT_PINKY_y"]
    zd = df["RIGHT_PINKY_z"]
    G_x, G_y, G_z = calculate_gravity(
        xa, ya, za, xb, yb, zb, xc, yc, zc, xd, yd, zd
    )
    df["center_hand_right_x"] = G_x
    df["center_hand_right_y"] = G_y
    df["center_hand_right_z"] = G_z

    # 左手の重心
    xa = df["LEFT_WRIST_x"]
    ya = df["LEFT_WRIST_y"]
    za = df["LEFT_WRIST_z"]
    xb = df["LEFT_THUMB_x"]
    yb = df["LEFT_THUMB_y"]
    zb = df["LEFT_THUMB_z"]
    xc = df["LEFT_INDEX_x"]
    yc = df["LEFT_INDEX_y"]
    zc = df["LEFT_INDEX_z"]
    xd = df["LEFT_PINKY_x"]
    yd = df["LEFT_PINKY_y"]
    zd = df["LEFT_PINKY_z"]
    G_x, G_y, G_z = calculate_gravity(
        xa, ya, za, xb, yb, zb, xc, yc, zc, xd, yd, zd
    )
    df["center_hand_left_x"] = G_x
    df["center_hand_left_y"] = G_y
    df["center_hand_left_z"] = G_z

    # 胴体の重心
    xa = df["LEFT_SHOULDER_x"]
    ya = df["LEFT_SHOULDER_y"]
    za = df["LEFT_SHOULDER_z"]
    xb = df["RIGHT_SHOULDER_x"]
    yb = df["RIGHT_SHOULDER_y"]
    zb = df["RIGHT_SHOULDER_z"]
    xc = df["RIGHT_HIP_x"]
    yc = df["RIGHT_HIP_y"]
    zc = df["RIGHT_HIP_z"]
    xd = df["LEFT_HIP_x"]
    yd = df["LEFT_HIP_y"]
    zd = df["LEFT_HIP_z"]
    G_x, G_y, G_z = calculate_gravity(
        xa, ya, za, xb, yb, zb, xc, yc, zc, xd, yd, zd
    )
    df["center_body_x"] = G_x
    df["center_body_y"] = G_y
    df["center_body_z"] = G_z

    return df


# 各部位における重心を座標を用いて全身の重心を求める関数
def cal_center_of_gravity(df: pd.DataFrame, m: float) -> pd.DataFrame:
    """
    体の重心を求める関数
    df: DataFrame
    m: 体重
    """
    # 各部位の質量比
    HEAD = 6.9
    UPPER_ARM = 2.7
    BOTTOM_ARM = 1.6
    UPPER_LEG = 11.0
    BOTTOM_LEG = 5.1
    FOOT = 1.1
    HAND = 0.6
    BODY = 48.9

    G_x = (
        m
        * (
            HEAD * df["center_head_x"]
            + HAND * (df["center_hand_right_x"] + df["center_hand_left_x"])
            + BODY * df["center_body_x"]
            + FOOT * (df["center_foot_left_x"] + df["center_foot_right_x"])
            + UPPER_ARM * (df["upper_arm_left_x"] + df["upper_arm_right_x"])
            + BOTTOM_ARM * (df["bottom_arm_left_x"] + df["bottom_arm_right_x"])
            + UPPER_LEG * (df["upper_leg_left_x"] + df["upper_leg_right_x"])
            + BOTTOM_LEG * (df["bottom_leg_left_x"] + df["bottom_leg_right_x"])
        )
        / 100
    ) / m
    G_y = (
        m
        * (
            HEAD * df["center_head_y"]
            + HAND * (df["center_hand_right_y"] + df["center_hand_left_y"])
            + BODY * df["center_body_y"]
            + FOOT * (df["center_foot_left_y"] + df["center_foot_right_y"])
            + UPPER_ARM * (df["upper_arm_left_y"] + df["upper_arm_right_y"])
            + BOTTOM_ARM * (df["bottom_arm_left_y"] + df["bottom_arm_right_y"])
            + UPPER_LEG * (df["upper_leg_left_y"] + df["upper_leg_right_y"])
            + BOTTOM_LEG * (df["bottom_leg_left_y"] + df["bottom_leg_right_y"])
        )
        / 100
    ) / m
    G_z = (
        m
        * (
            HEAD * df["center_head_z"]
            + HAND * (df["center_hand_right_z"] + df["center_hand_left_z"])
            + BODY * df["center_body_z"]
            + FOOT * (df["center_foot_left_z"] + df["center_foot_right_z"])
            + UPPER_ARM * (df["upper_arm_left_z"] + df["upper_arm_right_z"])
            + BOTTOM_ARM * (df["bottom_arm_left_z"] + df["bottom_arm_right_z"])
            + UPPER_LEG * (df["upper_leg_left_z"] + df["upper_leg_right_z"])
            + BOTTOM_LEG * (df["bottom_leg_left_z"] + df["bottom_leg_right_z"])
        )
        / 100
    ) / m

    df["center_of_gravity_x"] = G_x
    df["center_of_gravity_y"] = G_y
    df["center_of_gravity_z"] = G_z

    return df
