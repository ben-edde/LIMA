import datetime
import logging
import os
import numpy as np
import pandas as pd
import requests

HOME = os.environ['LIMA_HOME']

def main():
    url="http://api.eia.gov/series/"
    price_series={"CLC1":'PET.RCLC1.D',"CLC2":'PET.RCLC2.D',"CLC3":'PET.RCLC3.D',"CLC4":'PET.RCLC4.D'}
    price_list=[]
    for each in price_series:
        params={"api_key":'944802979fc9170aa83bb243d3deff8a','series_id':price_series[each]}
        response=requests.get(url,params=params)
        response_json=response.json()
        df=pd.DataFrame(response_json['series'][0]['data'])
        df.columns=["Date",each]
        df.Date=df.Date.map(lambda x: datetime.datetime.strptime(x,"%Y%m%d"))
        df.set_index("Date",inplace=True)
        price_list.append(df)
    df_price=pd.concat(price_list,axis=1)
    df_price.dropna(inplace=True)
    df_price.reset_index(inplace=True)
    file_name=datetime.date.today().strftime("WTI_4C_EIA_%Y-%m-%d.csv")
    file_destination=f"{HOME}/data/fresh/{file_name}"
    df_price.to_csv(file_destination,index=False)


if __name__=="__main__":
    main()