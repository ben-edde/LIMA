import logging
import os
import numpy as np
import pandas as pd
import re
HOME = os.environ['LIMA_HOME']

df_news = pd.read_csv(f"{HOME}/data/reuse/RedditNews_WTI/RedditNews_2008-06-09_2016-07-01.csv")


geo_pattern={
    0:"(([G,g]eopoli(tical)*).*(risk|concern|tension|uncertaint))|(^(?=.*(United States))(?=.*tension)(?=.*(military|war|geopolitical|coup|guerrilla|warfare))(?=.*((LatinAmerica)|(Central America)|(South America)|Europe|Africa|(Middle East)|(Far East)|Asia)).*$)",
    1:"^(?=.*((tomic war)|(uclear war)|(uclear conflict)|(tomic conflict)|(uclear missile)))(?=.*(fear|threat|risk|peril|menace)).*$",
    2:"(war risk)|(risk.* of war)|(fear of war)|(war fear)|(military threat)|(war threat)|(threat of war)|(^(?=.*((military action)|(military operation)|(military force)))(?=.*(risk|threat)).*$)",
    3:"(errorist threat)|(hreat of terrorism)|(errorism menace)|(enace of terrorism)|(errorist risk)|(error risk)|(isk of [T,t]errorism)|(error threat)",
    4:"(eginning of the war)|(utbreak of the war)|(nset of the war)|(scalation of the war)|(tart of the war)|(^(?=.*(war|militar))(?=.*(ir strike)).*$)|(^(?=.*(war|battle))(?=.*(eavy casualt)).*$)",
    5:"(terrorist acts*)"
}
def count_geoidx(content):
    result=0
    for idx in geo_pattern:
        result+= len(re.findall(geo_pattern[idx],content))
    return result

df_news["GeoIdx"]=df_news.News.apply(count_geoidx).dropna()

df=df_news.groupby("Date").sum()
df.reset_index(inplace=True)
df.to_csv("RedditNews_2008-06-09_2016-07-01_daily_GeoIdx.csv",index=False)