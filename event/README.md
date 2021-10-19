# Event

## Tools

* Stanford OpenIE: annotate
* stanza: tag
* nltk: TBC

## Code

* generate_event_tuple_n_convert_w2v.py
  * read news as df
  * generate event tuple in form of (Subject, Relation, Object) with StanfordOpenIE.
  * use event tuples to train Word2Vec //train all 3 types together, may separate later
  * transform event tuple with Word2Vec
  * export as pickle

* aggregate_daily_event.py
  * read event tuple (W2V) and price data as df
  * aggregate event by date that taking mean of all events within one day. Each event is taking as sum its tuple.
  * set date as index for both df then join them to filter each other
  * export result

