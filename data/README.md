# data

* No script running under data/
* all change should be done outside then cp here
* so as using data files, cp files to working directory
* news/ and prices/ are for raw data files, no change shuld be made
* place all intermediate result in reuse/

## Price data

### WTI Spot price

* file: `Cushing_OK_WTI_Spot_Price_FOB.csv`
* with info above header, need to remove before use
* 1986-01-02 to 2021-09-28
* 9007 rows

## News data

### RedditNews

#### after filtering to align with WTI Spot price

* file: `RedditNews_filtered.csv`
* 2008-06-09 to 2016-07-01
* 50843 rows

