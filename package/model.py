from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import os


class ExoplanetClassier:
    def __init__(
        self,
        data_path: str = "data",
        train_data_name: str = "train_data.parquet",
        test_data_name: str = "test_data.parquet",
    ):
        self.train_data_parquet_path = os.path.join(data_path, train_data_name)
        self.test_data_parquet_path = os.path.join(data_path, test_data_name)
        self.scaler: MinMaxScaler = None
        self.model: DecisionTreeClassifier = None

        self.df_train: pd.DataFrame = None
        self.df_test: pd.DataFrame = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

    def scale(self):
        self.scaler = MinMaxScaler()
        self.scaler = self.scaler.fit(self.df_train.iloc[:, 1:])

    def set_data(self):
        print("set_data")
        self.df_train = pd.read_parquet(self.train_data_parquet_path)
        self.df_test = pd.read_parquet(self.test_data_parquet_path)

        self.X_train = self.df_train.iloc[:, 1:]
        self.y_train = self.df_train["LABEL"]

        self.X_test = self.df_test.iloc[:, 1:]
        self.y_test = self.df_test["LABEL"]
        print("done!")


if __name__ == "__main__":
    classifier = ExoplanetClassier()
    classifier.set_data()