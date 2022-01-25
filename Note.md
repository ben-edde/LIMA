# Notes

## TBD

```

Done: use more features (30, 50, 100, ...)
    * more features may or may not help
Done: [FE] update GeoIdex with similarity approach

Done: [FS] Decide number of features
    * 10/20
Done: [FS] try to rely on tree based estimator for interpretable selection judgement
    * no obvious advantage

Model:
* discriminative
    Done [Model] hybrid model: ARIMA + GRU/LSTM for linear and non-linearity

* generative(autoregressive)
    [~] [Model] DeepAR
    [~] [Model] TFT
```

## Experiments

```
[~] Data
    * production
    * Imports/exports
[X] feature extraction methods
    [X] Text: sentiment
        * subjectivity
    [X] Text: topic
        * topic5
    [X] Text: embedding
        [X] scale or not?
            * to linear model: scale emb is good
            * to DL model: no scale emb is good
        [X] how to aggregate? mean of mean? daily mean? min+max?
            * max-mean-min is best for all model
        [X] sequence?
            * sequence is good
    [X] Text: event tuple vs event sequence
        [X] single vs sequence
            * sequence is good
        [X] separated VS packed
    [X] Text: Geopolitical index
        [X] similarity
        [TBC] fuzzy membership
[X] feature selection methods
    [X] Filter:
        * Pearson’s r: r_regression 
        * f_regression
        * VarianceThreshold
        * [O] mutual_info_regression
        * Absolute energy
        * Autocorrelation
        * [O] Entropy
        * Slope
        [~] catch22
    [X] Wrapper
        * [O] RFE
        * [O] RFEcv
        * [O] SelectFromModel (Ridge, Lasso, DT)
        * Sequential Feature Selection (using model like Lasso)
    [X] decomposition
        * PCA
        * NMF
    [X] Embedded 
        * LASSO
        * Random Forest
    [~] Causality-based
        * Granger
        * akelleh/causality
    [~] Relief-based
        * scikit-rebate
    [~] Genetic
        * sklearn-genetic
        * FeatureSelectionGA
[X] model selection
    [~] linear models
        * LinearRegression
        * Ridge
        * SGDRegressor
        * LinearSVR
        * decision tree
        * ARIMAX
    [X] *DL models        
        [X] RNN (+ Bidir)
        [X]  LSTM (+ Bidir)
        [X]  GRU (+ Bidir)
        [X]  ConvLSTM2D (+ Bidir)
        [X]  multimodal  (CNN + GRU)
        [X] hybrid (CNN + AMRA + GRU)
        * XGBOOST
        * NBEATS
        * Prophet
        * DeepAR
        * (TBC) Rocket
        * (TBC) TCN
        * (TBC) Temporal Fusion Transformers
        * (TBC) ES-RNN

```
## Idea

* see word embeddings as image -> sequence of images ?
    * sort them by sim?
    * make alignment like doing so in CV?

## Findings

```
DIFF & Scale:
* features and label must be diff and scaled

DONE: [FS] Use ACF and Granger before shifting
    * ACF can be used to determine lag order to shift
    * Granger seems good for FS, now it is slightly better than RFE(Ridge, 60)
* InfluxDB: use tag

Done: apply VAR (statsmodels) for lag order selection instead of using Granger
    * {'aic': 9, 'bic': 4, 'hqic': 4, 'fpe': 9}

### scaling of data (esp embedding)
* for linear model, scaling embedding is better
* for DL model, the original embedding (w/o) scaling is better
* if the embedding is further aggregated as a single series (i.e. no longer embedding but a series), no big difference of scaling

### emb sequence
* better than single

### scaling
* scaling is fine, but the range should be large enough to represent feature (1-100 is better than 0.1-1) //for y only
* scaling X with (0,1) helps
* 2 options: scaling y smaller then use mse; scaling y larger then use mae or mape
* seems scaling also help feature selection (esp entropy)

### RFE
* good for Lasso (as long as scale b/w 0-1)
* good for SVR (as long as scale b/w 0-1)
* not good for Ridge

### SelectFromModel
* good for Ridge

### loss func

when scaling y: (1,100)
msle (0.0005) ~= log_cosh (0.0002) ~= mape > huber ~= mae > mse >>> cos

### more data

* using more news (world and econ) seems good for linear model but not DL although diff is not obvious.

### model training in CV
* it seems model may not be updated properly in each round and weight sometimes preserved for next round. Need to compile model again.
* alternatively, export model archi as json then load it in each round for safety

### hybrid model
* simply using predicted value of ARMA as features does not help
* but its error does

### embedding
* it seems using event emb is better than simple emb

### lag order
* specific lag order for each type of features slightly better than general lag order for all
* need to determine its influence on feature selection, whether to maintain a portion for each kind of features, or simply selecting from all

### model archi
* seems simple structures models with less layer and less neuron can perform as well as those with more complicated structure
* Bidirectional layer is always good for local scope
* for RNN-based models, internal dropout may not be good
* when using multi-modal, adjusting ratio of neurons from different source matters

### k-fold
* don't do 10 folds, test set too small (<200 pt) for each round. 5-fold seems better (>300 pt as test set)

### training
* smaller batch size (40-50) is better than (>100).
* smaller learning rate seems working well with small batch
* loss func must be paired with scale of y

```
