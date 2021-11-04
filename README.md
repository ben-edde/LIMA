# OPP

## Components

* data cleaning
  * `->` load raw textual and price data
  * remove stop words
  * remove special char
  * lemmatization/stemming
  * `<-` DataFrame of documents with columns: (timestamp, text), DataFrame of price data
* feature extraction
  * `->` DataFrame of documents 
  * processing
  * `<-` DataFrame of feature series: (timestamp, features) //may be shifted
* feature selection
  * `->` DataFrame of feature series
  * select by methods
  * `<-` DataFrame of selected features series: (timestamp, features)
* split training set and test set
  * `->` DataFrame of selected features
  * `<-` training set and test set
* model training
  * `->` training set, parameters (loaded from cfg files)
  * feed training data into model with parameters
  * `<-` trained model
* evaluation
  * `->` trained model, test set, other info (e.g. methods used in feature engineering and feature selection, choice of model, parameters of model, ...)
  * evaluate model forecasting performance by measurements
  * `<-` a structure of evaluation result and info about the experiment
* keeping record
  * store evaluation result and other info (csv/DB/sth)

## AOB

* sacrifice reusability to decouple files
* avoid changing version of toolings after start of project
* add debug mode to code: test code with small scale data before going full scale right after changes being made
* record as much detail as possible for fallback
* [Tensorflow-Project-Template](https://github.com/MrGemy95/Tensorflow-Project-Template): update project structure before scaling up
* export evaluation result for visualization (perhaps using R)
* consider setup env on cloud
* https://github.com/IDSIA/sacred

## Practice

* cp evaluation functions in `framework/Evaluation.py` for each experiment to avoid import problem
* build a pipeline
  * loading and preprocessing
  * feature extraction
  * feature selection
  * modeling building
* then fit the entire pipeline as a model into `evaluate(model, X, y, cv)`
