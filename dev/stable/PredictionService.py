import datetime
import os
import random

import joblib
import numpy as np
import tensorflow as tf
from influxdb_client import InfluxDBClient
from influxdb_client.client.write_api import SYNCHRONOUS
from tensorflow import keras

from Dataset import Dataset
from FeatureEngineeringService import FeatureEngineeringService
from FeatureEngineeringStrategy import *
from LinearComponentModel import LinearComponentModel
from NonLinearComponentModel import NonLinearComponentModel

HOME = os.environ['LIMA_HOME']
# set random seed
seed_value = 42
os.environ['PYTHONHASHSEED'] = str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)


class PredictionService:
    def build_model(self):
        fe_service = FeatureEngineeringService(
            ModelBuildingFeatureEngineeringStrategy())
        feature, label = fe_service.get_feature()
        dataset = Dataset(X=feature, y=label, idx=fe_service.idx)

        lin_model = LinearComponentModel(dataset)
        lin_model.set_model("LR")
        lin_model.train()
        pred_linear_y = lin_model.predict(dataset.feature)

        non_linear_y = dataset.label - pred_linear_y.reshape(
            dataset.label.shape)
        nonlinear_data = Dataset(X=feature,
                                 y=non_linear_y,
                                 idx=fe_service.idx,
                                 scaling=True)

        nonlin_model = NonLinearComponentModel(nonlinear_data)
        nonlin_model.set_model("BiGRU")
        nonlin_model.train()

        model_path = f"{HOME}/dev/stable/models"
        nonlin_model.model.save(f"{model_path}/nonlinear.model")
        joblib.dump(lin_model.model, f"{model_path}/lin_model.joblib")
        joblib.dump(nonlin_model.dataset.feature_scaler,
                    f"{model_path}/feature_scaler-10.joblib")
        joblib.dump(nonlin_model.dataset.label_scaler,
                    f"{model_path}/label_scaler-3.joblib")
        joblib.dump(fe_service.feature_selector,
                    f"{model_path}/feature_selector.joblib")
        joblib.dump(fe_service.strategy.news_feature_helper.lda_model,
                    f"{model_path}/lda_model.joblib")
        joblib.dump(fe_service.strategy.news_feature_helper.emb_scaler,
                    f"{model_path}/emb_scaler.joblib")

    def predict(self):
        model_path = f"{HOME}/dev/stable/models"
        trained_nonlin_model = keras.models.load_model(
            f"{model_path}/nonlinear.model")
        trained_lin_model = joblib.load(f"{model_path}/lin_model.joblib")
        trained_feature_scaler = joblib.load(
            f"{model_path}/feature_scaler-10.joblib")
        trained_label_scaler = joblib.load(
            f"{model_path}/label_scaler-3.joblib")
        trained_feature_selector = joblib.load(
            f"{model_path}/feature_selector.joblib")
        trained_lda_model = joblib.load(f"{model_path}/lda_model.joblib")
        trained_emb_scaler = joblib.load(f"{model_path}/emb_scaler.joblib")
        fe_service = FeatureEngineeringService(
            ForecastFeatureEngineeringStrategy())
        fe_service.strategy.news_feature_helper.set_emb_scaler(
            trained_emb_scaler)
        fe_service.strategy.news_feature_helper.set_lda_model(
            trained_lda_model)
        fe_service.strategy.set_feature_selector(trained_feature_selector)

        feature, label = fe_service.get_feature()

        dataset = Dataset(X=feature,
                          y=None,
                          idx=fe_service.idx,
                          scaling=True,
                          feature_scaler=trained_feature_scaler,
                          label_scaler=trained_label_scaler)
        pred_linear_y = trained_lin_model.predict(dataset.feature)
        pred_nonlinear_y = trained_nonlin_model.predict(
            dataset.train_X)  # normalized
        inverted_pred_nonlinear_y = dataset.label_scaler.inverse_transform(
            pred_nonlinear_y)
        pred_combined_nonlinear_y = pd.DataFrame(
            inverted_pred_nonlinear_y).apply(lambda x: x.sum(),
                                             axis=1).to_numpy().reshape(-1, 1)
        pred_final = pred_combined_nonlinear_y.ravel() + pred_linear_y.ravel()
        df_results = pd.DataFrame(pred_final,
                                  columns=["CLC1_forecast"],
                                  index=fe_service.idx)
        # original idx is (t)th day, result should be (t+1)th day
        df_results.index=df_results.index.map(lambda x: x+datetime.timedelta(days=1))
        return df_results

    def publish_db(self, df_results, window_size=20, mode="forecast"):
        def shift_weekend(x):
            if x.weekday() in [5, 6]:
                x = x + datetime.timedelta(days=7 - x.weekday())
            return x

        if len(df_results) > window_size:
            df_results = df_results.iloc[-window_size:]
        df_results.index = df_results.index.map(
            lambda x: datetime.datetime.combine(x, datetime.time(22, 0)))
        df_results.index = df_results.index.map(shift_weekend)
        df_results.columns = ["CLC1"]
        df_results["h"] = 1
        df_results["type"] = mode

        client = InfluxDBClient.from_config_file(
            f"{HOME}/dev/DB/influxdb_config.ini")
        write_api = client.write_api(write_options=SYNCHRONOUS)
        write_api.write(bucket="dummy",
                        record=df_results,
                        data_frame_measurement_name='WTI',
                        data_frame_tag_columns=["h", "type"])
        write_api.close()
        client.close()


def main():
    prediction_service = PredictionService()
    prediction_service.build_model()


if __name__ == "__main__":
    main()
