# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 11:24:32 2022

@author: jubos
"""

WorkDir = r'C:\Users\jubos\OneDrive\Documents\Doutorado\Artigo 2 - NasaPower\Clima\Com outliers\ETo\ETo'

# data manipulation
import pandas as pd
import numpy as np

# data viz
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns

# apply styling
plt.style.use("fivethirtyeight")
rcParams['figure.figsize'] = (12,  6)

df = pd.read_csv('Todas_I.csv')

#df.info()

from numpy import array
from permetrics.regression import RegressionMetric

selecao = df.loc[:,['PET', 'AET', 'PET_N', 'AET_N', 'PET_X', 'AET_X']]
selecao.corr()

sns.pairplot(selecao, corner = True, diag_kind = 'kde')

fig, ax = plt.subplots(figsize=(15,7))
sns.heatmap(selecao.corr(),
            annot = True,
            cmap = 'coolwarm',
            mask = np.triu(selecao.corr()))

#NasaPower
x = df.loc[:,['P']]
y = df.loc[:,['P_N']]

y_true = array(x)
y_pred = array(y)

evaluator = RegressionMetric(y_true, y_pred, decimal=5)
print('Chuva')
print(evaluator.RMSE())
print(evaluator.MBE())
print(evaluator.MAE())
print(evaluator.MAPE())
print(evaluator.WI())
print(evaluator.R())
print(evaluator.R2s())

C = evaluator.R() * evaluator.WI()

print(C)

x = df.loc[:,['Tmax']]
y = df.loc[:,['Tmax_N']]

y_true = array(x)
y_pred = array(y)

evaluator = RegressionMetric(y_true, y_pred, decimal=5)
print('Tmaxima')
print(evaluator.RMSE())
print(evaluator.MBE())
print(evaluator.MAE())
print(evaluator.MAPE())
print(evaluator.WI())
print(evaluator.R())
print(evaluator.R2s())

C = evaluator.R() * evaluator.WI()

print(C)

x = df.loc[:,['Tmin']]
y = df.loc[:,['Tmin_N']]

y_true = array(x)
y_pred = array(y)

evaluator = RegressionMetric(y_true, y_pred, decimal=5)
print('Tminima')
print(evaluator.RMSE())
print(evaluator.MBE())
print(evaluator.MAE())
print(evaluator.MAPE())
print(evaluator.WI())
print(evaluator.R())
print(evaluator.R2s())

C = evaluator.R() * evaluator.WI()

print(C)

#Xavier
x = df.loc[:,['P']]
y = df.loc[:,['P_X']]

y_true = array(x)
y_pred = array(y)

evaluator = RegressionMetric(y_true, y_pred, decimal=5)
print('Chuva')
print(evaluator.RMSE())
print(evaluator.MBE())
print(evaluator.MAE())
print(evaluator.MAPE())
print(evaluator.WI())
print(evaluator.R())
print(evaluator.R2s())

C = evaluator.R() * evaluator.WI()

print(C)

x = df.loc[:,['Tmax']]
y = df.loc[:,['Tmax_X']]

y_true = array(x)
y_pred = array(y)

evaluator = RegressionMetric(y_true, y_pred, decimal=5)
print('Tmaxima')
print(evaluator.RMSE())
print(evaluator.MBE())
print(evaluator.MAE())
print(evaluator.MAPE())
print(evaluator.WI())
print(evaluator.R())
print(evaluator.R2s())

C = evaluator.R() * evaluator.WI()

print(C)

x = df.loc[:,['Tmin']]
y = df.loc[:,['Tmin_X']]

y_true = array(x)
y_pred = array(y)

evaluator = RegressionMetric(y_true, y_pred, decimal=5)
print('Tminima')
print(evaluator.RMSE())
print(evaluator.MBE())
print(evaluator.MAE())
print(evaluator.MAPE())
print(evaluator.WI())
print(evaluator.R())
print(evaluator.R2s())

C = evaluator.R() * evaluator.WI()

print(C)

selecao = df.loc[:,['PET', 'AET', 'PET_N', 'AET_N', 'PET_X', 'AET_X']]
selecao.corr()

sns.pairplot(selecao, corner = True, diag_kind = 'kde')

fig, ax = plt.subplots(figsize=(15,7))
sns.heatmap(selecao.corr(),
            annot = True,
            cmap = 'coolwarm',
            mask = np.triu(selecao.corr()))
