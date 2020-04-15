#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import glob



"""
Import sym output data while symulation is not ready

This doesn't work yet :D
"""
path = "D:\\samue\\Dropbox\\AFES Datascience\\Ciencia y Vida\\Codigo\\Data Simulada\\"
data = [[]]
lines = []
with open(path+"MCMC_101882.out",'r') as f: 
    lines = f.readlines()
    j = 0
    for i in range(len(lines)-1):        
        if not "-" in lines[i] and i>1:           
            if j ==1:
                data[0].append(lines[i].split()[0])
                j+=1
            elif len(lines[i].split())==2:
                data[j].append(lines[i].split()[0])
                j+=1
            else:             
                data[j].append([lines[i].split()])
            


# get data file names
"""
Import Symulation results
"""
path = "D:\\samue\\Dropbox\\AFES Datascience\\Ciencia y Vida\\Codigo\\Data Simulada Final\\"
filenames = glob.glob(path + "/*.csv")
dfs = []

for filename in filenames:
    dfs.append(pd.read_csv(filename,header=None, names=['beta', 'sigma','gamma','mu','err']))

# Concatenate all data into one DataFrame
dataframe = pd.concat(dfs, ignore_index=True)
dataframe.to_csv(path+"..\\MCMC_20200415.csv")
print("ready")

dftest = pd.read_csv(path+"1.csv",names=['beta', 'sigma','gamma','mu','err'],header=None)