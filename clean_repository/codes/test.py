

import pandas as pd
import os
import numpy as np

os.chdir("/home/jhonathan/Documents/dos_Santos_Genomic_Prediction_2019/raw_data")
df1 = pd.read_csv('Biomass_2016.csv')
df2 = pd.read_csv('Biomass_SF2017.csv')
df3 = pd.read_csv('heights_2016.csv')
df4 = pd.read_csv('heights_SF2017.csv')

df1.DryTonsAcre = df1.DryTonsAcre*2.241699
df2.DryTonsAcre = df2.DryTonsAcre*2.241699

df1 = df1.rename(columns = {"DryTonsAcre": "dby"}) 
df2 = df2.rename(columns = {"DryTonsAcre": "dby"})

mask1 = df1.isnull()
mask2 = df2.isnull()
mask3 = df3.isnull()
mask4 = df4.isnull()

df1 = df1.astype(str)
df2 = df2.astype(str)
df3 = df3.astype(str)
df4 = df4.astype(str)

df1[mask1] = 'NA'
df2[mask2] = 'NA'
df3[mask3] = 'NA'
df4[mask4] = 'NA'

df1 = df1.drop(columns=['Range', 'Row', 'Moisture_%', 'Starch', 'Protein', 'ADF', 'NDF'])
df2 = df2.drop(columns=['range', 'row', 'Moisture_%', 'Starch', 'Protein', 'ADF', 'NDF', 'Date'])
df4 = df4.drop(columns=['range', 'row'])

df1.reset_index(level=0, inplace=True)
df2.reset_index(level=0, inplace=True)
df3.reset_index(level=0, inplace=True)
df4.reset_index(level=0, inplace=True)

df1.head(10)
df2.head(10)
df3.head(10)
df4.head(10)

df1.to_csv("Biomass_2016.csv", index=False)
df2.to_csv("Biomass_SF2017.csv", index=False)
df3.to_csv("heights_2016.csv", index=False)
df4.to_csv("heights_SF2017.csv", index=False)