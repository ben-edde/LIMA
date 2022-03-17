class PredictionService:

    def build_model(self):
        fe_service=FeatureEngineeringService(mode="build")
        fe_service.feature_extraction(mode="build")
        fe_service.feature_selection(mode="build")
        dataset=Dataset(X=fe_service.X,y=fe_service.y)

        lin_model=LinearComponentModel(dataset)
        lin_model.set_model("LR")
        lin_model.train()
        pred_linear_y=lin_model.predict(data.feature)

        non_linear_y=dataset.label-pred_linear_y.reshape(y.shape)
        nonlinear_data=Dataset(X=fe_service.X,y=non_linear_y,scaling=True)

        nonlin_model=NonLinearComponentModel(nonlinear_data)
        nonlin_model.set_model("BiGRU")
        nonlin_model.train()

        nonlin_model.model.save(f"{HOME}/dev/models/investing/GRU.model")
        joblib.dump(nonlinear_data.feature_scaler, f"{HOME}/dev/models/investing/feature_scaler.joblib")
        joblib.dump(nonlinear_data.label_scaler, f"{HOME}/dev/models/investing/label_scaler.joblib")
        joblib.dump(fe_service.feature_selector, f"{HOME}/dev/models/investing/feature_selector.joblib")
        joblib.dump(fe_service.news_feature_helper.lda_model, f"{HOME}/dev/models/investing/lda_model.joblib")
        joblib.dump(fe_service.news_feature_helper.emb_scaler, f"{HOME}/dev/models/investing/emb_scaler.joblib")
    
    def predict(self):
        # TODO update forecast mode
        fe_service=FeatureEngineeringService(mode="forecast")
        fe_service.feature_extraction(mode="forecast")
        fe_service.feature_selection(mode="forecast")

        df_inf=pd.DataFrame(inverted_inf_y,columns=["CLC1_forecast"])
        df_inf.index=df_inf_original_price.index
        df_inf.index=df_inf.index.map(lambda x: datetime.datetime.combine(x,datetime.time(22,0)))
        def shift_weekend(x):
            if x.weekday()in [5,6]:
                x=x+datetime.timedelta(days=7-x.weekday())
            return x
        df_inf.index=df_inf.index.map(shift_weekend)
        df_inf.columns=["CLC1"]
        df_inf["h"]=1
        df_inf["type"]="forecast"

        from influxdb_client.client.write_api import SYNCHRONOUS
        client= InfluxDBClient.from_config_file(f"{HOME}/dev/DB/influxdb_config.ini")
        write_api=client.write_api(write_options=SYNCHRONOUS)
        write_api.write(bucket="dummy",record=df_inf, data_frame_measurement_name='WTI',data_frame_tag_columns=["h","type"])
        write_api.close()
        client.close()
