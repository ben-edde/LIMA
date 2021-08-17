from selenium import webdriver
import time
import csv
import datetime

time_start_str = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
driver = webdriver.Chrome()
driver.get("https://www.investing.com/search/?q=crude%20oil&tab=news")
time.sleep(1)
csv_file = open(f"headlines_{time_start_str}.csv", "w")
writer = csv.writer(csv_file)
writer.writerow(['time', 'headlines'])

for i in range(1, 2213):
    try:
        newstime = driver.find_element_by_xpath(
            '//*[@id="fullColumn"]/div/div[4]/div[3]/div/div[' + str(i) +
            ']/div/div/time').text
        headlines = driver.find_element_by_xpath(
            '//*[@id="fullColumn"]/div/div[4]/div[3]/div/div[' + str(i) +
            ']/div/a').text
        writer.writerow([newstime, headlines])
    except Exception as e:
        print(e)
    if i % 20 == 0:
        js = "var q=document.documentElement.scrollTop+=100000"
        driver.execute_script(js)
        time.sleep(1)
csv_file.close()
driver.close()
