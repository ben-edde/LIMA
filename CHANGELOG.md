# Change Log

## 2021-11-23

* [Method] update evaluation: separated evaluation for TS model and ML model
  * TS model: forecast horizon determines test set
  * ML model: use 20% of data as test set by default, leaving handling of horizon to feature preparation

## 2021-11-17

* [Method] update evaluation: test set determined by horizon (multi-point test set)
* [Model] auto ARIMA
* [Method] collection of results
* [Method] added template of experiment
* [Method] integrated evaluation functions, packed split of cv as closure
* [FE] glove embedding
* [Method] update evaluation: added hold-out

## 2021-11-16

* [Model] compare models with fasttext emb

## 2021-11-09

* [Method] update evaluation: single point test set
* [Method] update evaluation: added logging of results

## 2021-11-08

* [FE] fasttext embedding

## 2021-11-07

* [FE] generation of  event tuple
* [FE] convert event tuple to glove

## 2021-11-04

* [Method] placed all evaluation functions together
* [Method] added standard deviation to evaluation results

## 2021-11-02

* [Method] completed time-series shifting
* [Method] together with text cleaning script, data pre-processing has been completed;
* [Method] merged and removed data branch
