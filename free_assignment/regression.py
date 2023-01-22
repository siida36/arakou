from typing import Dict

from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd


def read_data_dummy() -> Dict[str, np.array]:
    X = np.array([[1, 2, 3], [4, 5, 6]])
    y = np.array([0, 1])
    return {
        "X": normalize(X),
        "y": normalize(y)
    }


def read_data_from_csv() -> Dict[str, np.array]:
    df = pd.read_csv("input/input_std.csv")
    y = df["label"].to_numpy()
    del df["label"]
    X = df.to_numpy()
    return {
        "X": normalize(X),
        "y": normalize(y)
    }


def normalize(matrix: np.array) -> np.array:
    mean = matrix.mean(axis=0)
    diff = (matrix - mean) ** 2
    std = np.sqrt(diff.mean())
    res = (matrix - mean) / std
    return res


def train(input_data) -> LinearRegression:
    model = LinearRegression()
    model.fit(input_data["X"], input_data["y"])
    return model


def print_params(model: LinearRegression):
    print(f"coef: {model.coef_}")
    print(f"bias: {model.intercept_}")


def main():
    input_data = read_data_from_csv()
    model = train(input_data)
    print_params(model)


if __name__ == "__main__":
    main()
