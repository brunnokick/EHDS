from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import joblib
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
        self.scaler: Scaler = None
        self.model: DecisionTreeClassifier = None

        self.df_train: pd.DataFrame = None
        self.df_test: pd.DataFrame = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

    def run_full(self):
        print("Run full")
        self.scaler = Scaler()
        self.set_data()
        feature_columns = self.X_train.columns

        self.scaler.fit(self.X_train)
        self.X_train = pd.DataFrame(
            columns=feature_columns, data=self.scaler.tranform(self.X_train)
        )
        self.X_test = pd.DataFrame(
            columns=feature_columns, data=self.scaler.tranform(self.X_test)
        )

        self.set_model()
        self.fit()
        self.predict()
        self.print_metrics()

    def set_data(self):
        print("set_data")
        self.df_train = pd.read_parquet(self.train_data_parquet_path)
        self.df_test = pd.read_parquet(self.test_data_parquet_path)

        self.X_train = self.df_train.iloc[:, 1:]
        self.y_train = self.df_train["LABEL"]

        self.X_test = self.df_test.iloc[:, 1:]
        self.y_test = self.df_test["LABEL"]
        print("done!")

    def set_model(self):
        print("set model")
        self.model = DecisionTreeClassifier()
        self.dump()

    def fit(self, dump: bool = True):
        print("fit")
        self.model.fit(self.X_train, self.y_train)

    def predict(self, X_test: pd.DataFrame = None, y_test: pd.DataFrame = None):
        print("predict")
        if X_test is None:
            X_test = self.X_test
        else:
            self.X_test = X_test

        if y_test is not None:
            self.y_test = y_test

        self.predicted = self.model.predict(X_test)
        return self.predicted

    def print_metrics(self):
        print("print_metrics")
        accuracy = round(accuracy_score(self.y_test, self.predicted), 4)
        print(accuracy)

    def dump(self, destination_folder: str = "model"):
        joblib.dump(self.model, os.path.join(destination_folder, "classfier.joblib"))


class Scaler:
    def __init__(self):
        self.scaler = MinMaxScaler()

    def fit(self, df: pd.DataFrame):
        self.scaler.fit(df)
        self.dump()

    def tranform(self, df: pd.DataFrame) -> np.ndarray:
        return self.scaler.transform(df)

    def dump(self, destination_folder: str = "model"):
        joblib.dump(self.scaler, os.path.join(destination_folder, "scaler.joblib"))


if __name__ == "__main__":
    classifier = ExoplanetClassier()
    classifier.run_full()