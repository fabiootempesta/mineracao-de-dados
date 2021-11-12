import pandas as pd
import csv

print("Nome do arquivo .csv dentro da pasta:")
filename=input()

df = pd.read_csv(filename+'.csv', delimiter=',')
data = df.select_dtypes(['number'])

data_zscore = open('zscore-'+filename+'.csv', 'w', newline='')
writer = csv.writer(data_zscore)
writer.writerow(data)

zscore = []
aux_colname = data.columns.values

if(data.columns.size>0):
    for i in range(data[aux_colname[0]].size):
        for j in range(data.columns.size):
            zscore.append((data[aux_colname[j]][i]-data.mean()[j])/(data.mad()[j]))
        writer.writerow(zscore)
        zscore.clear()
