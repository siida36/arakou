"""Pythonによる重回帰分析のサンプルです。

次の流れで処理を実行します。

1. データ読み込み
2. 重回帰分析
3. 結果の表示
"""
from pathlib import Path
from typing import Dict

from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd


def read_data_dummy() -> Dict[str, np.array]:
    """デバッグ用のダミーデータです"""
    X = np.array([[1, 2, 3], [4, 5, 6]])
    y = np.array([0, 1])
    return {
        "X": normalize(X),
        "y": normalize(y)
    }


def read_data_from_csv(filepath: Path) -> Dict[str, np.array]:
    """ファイルからデータを読み込みます"""
    df = pd.read_csv(filepath)
    y = df["label"].to_numpy()
    del df["label"]
    X = df.to_numpy()

    return {
        "X": normalize(X),
        "y": normalize(y)
    }


def normalize(matrix: np.array) -> np.array:
    """標準化処理を行います"""
    mean = matrix.mean(axis=0)
    diff = (matrix - mean) ** 2
    std = np.sqrt(diff.mean())
    res = (matrix - mean) / std
    return res


def train(input_data) -> LinearRegression:
    """重回帰分析を行います"""
    model = LinearRegression()
    model.fit(input_data["X"], input_data["y"])
    return model


def print_params(model: LinearRegression):
    """重回帰分析の結果を表示します"""
    print(f"係数: {model.coef_}")
    print(f"切片: {model.intercept_}")


def main():
    """すべての処理を実行します"""
    input_data = read_data_from_csv("input/input_std.csv")
    model = train(input_data)
    print_params(model)


if __name__ == "__main__":
    main()
