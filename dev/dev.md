# Dev

## scraper

* RedditNews_scaper
    * scrape json from reddit directly
    * need to pass `User-agent` in header or it may get error `Too Many Requests`
    * append to `data/fresh/RedditNews.csv`
* WTI_CNBC_scraper.py
    * scrape 4 contracts previous closing price from CNBC (originally in EST; but change to UTC for simplicity)
    * ~10 mins delay
    * append to `data/fresh/WTI_4C_CNBC.csv`
* cron job
    * need to specify `SHELL` and `BASH_ENV` then activate venv there instead of in script
    * Now set up 7am HKT, which is about 6pm EST (market reopen)
* WTI_EIA_scraper
    * scrape 4 contracts prices and spot price from US GOV EIA API
    * updated weekly
    * as reference

## deployment

* https://github.com/crawles/automl_service

## Docker

```sh
docker run -d -p 8086:8086 --name lima_influxdb influxdb:2.1.1
docker run -d --name=lima_grafana -p 3000:3000 grafana/grafana-enterprise:8.3.3-ubuntu
docker run -d --name lima_mongo -p 27017:27017 mongo
```

## timezone

* for simplicity, use UTC time for all data
* CNBC WTI closing price
    * US Eastern: available since 5pm on nth day
    * UTC: 10pm on n th day
    * local time: 6am on n+1 th day
* Investing.com
    * US Eastern

