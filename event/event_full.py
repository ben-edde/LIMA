import pandas as pd
import numpy as np
from openie import StanfordOpenIE
from gensim.models import Word2Vec
from sklearn.model_selection import cross_validate,RepeatedKFold
from sklearn.linear_model import Lasso

def run():
    df_news = pd.read_csv("RedditNews_filtered.csv")
    
    openie_client = StanfordOpenIE()

    event_tuples = []
    for idx, row in df_news.iterrows():
        text = row['News']
        for triple in openie_client.annotate(text):
            triple['Date'] = row['Date']
            event_tuples.append(triple)


    df_events = pd.DataFrame(event_tuples)

    

    event_subject = df_events.subject.apply(lambda x: x.split(" "))
    event_relation = df_events.relation.apply(lambda x: x.split(" "))
    event_object = df_events.object.apply(lambda x: x.split(" "))

    w2v_model = Word2Vec(event_subject, min_count=1)

    w2v_model.build_vocab(event_subject, update=True)
    w2v_model.build_vocab(event_relation, update=True)
    w2v_model.build_vocab(event_object, update=True)

    w2v_model.train(event_subject,
                    total_examples=len(event_subject),
                    epochs=30,
                    report_delay=1)
    w2v_model.train(event_relation,
                    total_examples=len(event_relation),
                    epochs=30,
                    report_delay=1)
    w2v_model.train(event_object,
                    total_examples=len(event_object),
                    epochs=30,
                    report_delay=1)


    # if more than 1 word, only pick 1st one
    df_events.subject = df_events.subject.apply(lambda x: x.split(" ")[0]
                                                if len(x.split(" ")) > 1 else x)
    df_events.relation = df_events.relation.apply(lambda x: x.split(" ")[0]
                                                if len(x.split(" ")) > 1 else x)
    df_events.object = df_events.object.apply(lambda x: x.split(" ")[0]
                                            if len(x.split(" ")) > 1 else x)

    df_events.subject = df_events.subject.apply(lambda x: w2v_model.wv[x])
    df_events.relation = df_events.relation.apply(lambda x: w2v_model.wv[x])
    df_events.object = df_events.object.apply(lambda x: w2v_model.wv[x])
    
    # find daily event representation
    df_events["event_sum"]=df_events.subject+df_events.relation+df_events.object
    df_daily_event=pd.DataFrame(columns=["Date","Event"])
    for date, group in df_events.groupby(["Date"]):
        daily_mean=np.mean(group["event_sum"].to_numpy())
        df_daily_event=df_daily_event.append({"Date":date,"Event":daily_mean},ignore_index=True)

    # read price data
    df_price=pd.read_csv("Cushing_OK_WTI_Spot_Price_FOB.csv")


    # set date as index for both df
    df_price.Date=pd.to_datetime(df_price.Date)
    df_price.set_index("Date",inplace=True)
    df_daily_event.Date=pd.to_datetime(df_daily_event.Date)
    df_daily_event.set_index("Date",inplace=True)

    # merge to ensure consistent shape
    df_Xy=df_daily_event.merge(df_price,left_index=True,right_index=True)

    # X must be flatten here to be in shape of (xxx,yyy) instead of (xxx,). Otherwise, model cannot read X
    X, y =np.array(df_Xy["Event"].to_numpy().reshape(-1).tolist()), df_Xy["Price"].to_numpy().reshape(-1)

    train_ratio=int(len(X)*0.7)
    X_train,X_test =X[:train_ratio],X[train_ratio:]
    y_train,y_test =y[:train_ratio],y[train_ratio:]

    model = Lasso(alpha=1.0)
    model.fit(X_train,y_train)
    model.score(X_test,y_test)

    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)

    scores = cross_validate(model, X, y, scoring=['neg_mean_absolute_error','neg_root_mean_squared_error','neg_mean_absolute_percentage_error'], cv=cv, n_jobs=-1)

    print(
        f"""Test error (cross-validated performance)
        {model.__class__.__name__}:
        MAE = {-scores["test_neg_mean_absolute_error"].mean():.3f}
        RMSE = {-scores["test_neg_root_mean_squared_error"].mean():.3f}
        MAPE = {-scores["test_neg_mean_absolute_percentage_error"].mean():.3f}
        """)


def main():
    pass

if __name__=="__main__":
    main()