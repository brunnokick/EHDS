from zipfile import ZipFile
import pandas as pd
import numpy as np
import os


class PreProcessing:
    def __init__(self, env: str = "local"):
        self.env = env

    def run(
        self,
        train_data_zip: str = "exoTrain.csv.zip",
        test_data_zip: str = "exoTest.csv.zip",
        output_folder: str = "data",
    ):
        self.unzip(train_data_zip, test_data_zip, output_folder)

        # TRAIN DATA
        train_data_csv = os.path.join(output_folder, "exoTrain.csv")
        df_train = pd.read_csv(train_data_csv)
        df_train["LABEL"] = np.where(df_train["LABEL"] == 2, 1, 0)
        df_train.to_parquet(str(os.path.join(output_folder, "train_data.parquet")))

        # TEST DATA
        train_data_csv = os.path.join(output_folder, "exoTest.csv")
        df_test = pd.read_csv(train_data_csv)
        df_test["LABEL"] = np.where(df_test["LABEL"] == 2, 1, 0)
        df_test.to_parquet(str(os.path.join(output_folder, "test_data.parquet")))

    def unzip(self, train_data_path: str, test_data_path: str, output_folder: str):
        print("Unzipping..")
        with ZipFile(train_data_path, "r") as zipObj:
            print(f"train_data_path: {train_data_path}")
            unzip_path = os.path.join(output_folder)
            zipObj.extractall(unzip_path)
            print(f"train_data is unzipped in {unzip_path}")

        with ZipFile(test_data_path, "r") as zipObj:
            print(f"train_data_path: {test_data_path}")
            unzip_path = os.path.join(output_folder)
            zipObj.extractall(unzip_path)
            print(f"train_data is unzipped in {unzip_path}")


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(
    #     description="Configuracoes para o processamento"
    # )

    # parser.add_argument("enviroment", default="local", help="local or gcp")

    # args = parser.parse_args()

    pre_process = PreProcessing()
    pre_process.run()
