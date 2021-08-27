# demo

## TODO

- [x] demo with arima
- [x] sentiment indicator
  - [x] TextBlob
  - [x] cumulative decay
- [x] topic indicator
  - [x] embedding
  - [x] SeaNMF
- [x] integration of multiple series
- [ ] VAR and lag order selection
- [x] data preprocessing
  - [x] shifting series as features
  - [x] split of training set and test set
- [x] feature selection with RFE
- [x] demo with AdaBoostRT

## Remarks

### preparation of feature

- lag determines the number of day to shift per series
  - the features of a series wil include (t-lag, t-lag+1,..., t-2, t-1)
  - (t-0) is labels, only for series which supposed to be predicted
- each shift of a series will then be merged
- day with null value will then be discarded
- number of discarded days will be determined by the largest lag among all series due to alignment
- so there is need to shift this gap for the predicted result

### forecast

> Let testing set be x[a:b] and y[a:b]

- predict y[a:b] using original testing set x[a:b] is (t-0)
- predict y[a:b] using back shifted testing set x[a-1:b-1] is (t+1)
- predict y[a:b] using back shifted testing set x[a-2:b-2] is (t+2)
- predict y[a:b] using back shifted testing set x[a-3:b-3] is (t+3)
