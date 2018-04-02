import requests
from bs4 import BeautifulSoup
import pickle
import numpy as np
import csv
import os


#getting the dst values from using the dst html links
i = 1
tables = []
with open("dst_links.txt") as f:
    for link in f:
        print (i)
        link = link.split("\n")[0]
        f = requests.get(link)
        soup = BeautifulSoup(f.text, 'html.parser')
        data = soup.find_all("pre")

        table = data[0].get_text().split("DAY")[1].split("\n")

        table = [x for x in table if x != '']
        tables.append(table)
        i += 1
        
print (len(tables))

fw = open("dst_tables.pkl", "wb")
pickle.dump(tables, fw)
fw.close() 
#print (tables[0])

#getting the hourly omni data set using the wget command on the omni data html links 
with open("omni_links.txt") as f:
    for link in f:
        os.system("wget " + str(link.split("\n")[0]))
