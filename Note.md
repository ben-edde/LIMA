# Notes

## TBD

```

TODO: rolling window VS growing window
TODO: use more features (30, 50, 100, ...)

TODO: [FE] update GeoIdex with similarity approach
TODO: [FE] apply more technical indicators (esp. rate of change)



TODO: [FS] after shifting, select more features
TODO: [FS] try to rely on tree based estimator for interpretable selection judgements
TODO: [FS] apply stepwise with custom scoring
    * adjusted R^2
    * https://scikit-learn.org/stable/modules/model_evaluation.html#scoring
    * https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SequentialFeatureSelector.html

Model:
* discriminative
    TODO [Model] hybrid model: ARIMA + GRU/LSTM for linear and non-linearity
    TODO [Model] multimodal
        * Embedding: CNN
            * event tuple: 300 * 3
                * 3 channels, each for one of S,R,O // (3,1,300)
                * OR reshape as 2D matrix of (300,3), then refer to handling image with CNN
        * TS: GRU or none
* generative(autoregressive)
    TODO [Model] DeepAR
    TODO [Model] TFT
```

## Experiments

```
[ ] feature extraction methods
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
    [ ] Text: event tuple vs event sequence
        [X] single vs sequence
        [X] separated VS packed
[ ] feature selection methods
    [ ] Filter:
        * Pearson’s r: r_regression 
        * f_regression
        * VarianceThreshold
        * mutual_info_regression
        * Absolute energy
        * Autocorrelation
        * Entropy
        * Slope
        * Negative turning points
        * Wavelet energy
        * Wavelet entropy
        * FFT mean coefficient
    [ ] Wrapper
        * RFE
        * RFEcv
        * SelectFromModel (using model like Lasso)
        * Sequential Feature Selection (using model like Lasso)
    [TBC] Embedded 
        * LASSO
        * Random Forest
[ ] model selection
    [ ] linear models
        * LinearRegression
        * Ridge
        * SGDRegressor
        * LinearSVR
        * decision tree
        * ARIMAX
    [ ] *DL models
        * RNN (+ Bidir)
        * LSTM (+ Bidir)
        * GRU (+ Bidir)
        * XGBOOST
        * NBEATS
        * Prophet
        * multimodal  (CNN + GRU)
        * DeepAR
        * (TBC) Rocket
        * (TBC) TCN
        * (TBC)  Temporal Fusion Transformers

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
```
