import pandas as pd
from graph_tools_sub import (
    cal_ball_foot,
    initialize_center_values,
    process,
)


# グラフの描画を行うメイン関数
def graph_main(df: pd.DataFrame) -> pd.DataFrame:
    """
    input_path: データが格納されているパス(例: "examples")
    output_path: グラフを保存するパス(例: "examples/output_result")

    sprint関数を実行することで、グラフを一括で描画する
    df : 角度のグラフの作成
    df1 : 体の旋回角度のグラフの作成
    frames : 接地フレーム
    max_power : 最大加重時のフレーム
    """

    # データの前処理(graph_tools_sub.pyにて定義)
    df = cal_ball_foot(df)
    df = initialize_center_values(df)
    df = process(df)

    return df
