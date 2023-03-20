import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

# Veri seti midyat ev fiyatlarına göre ayarlanmış gerçek veri setidir.

#veri setimizi okuyoruz
df = pd.read_csv('midyatev.csv',sep=";")
df

#veri setleriyle besleme yapıyoruz
reg = linear_model.LinearRegression()
reg.fit(df[['alan', 'odasayisi', 'binayasi']], df['fiyat'])

#tahmini değeri soruyoruz
reg.predict([[300,6,2]])

#formülümüzle kontrol sağlıyoruz.
a=reg.intercept_
b1 = reg.coef_[0]
b2 = reg.coef_[1]
b3 = reg.coef_[2]

x1 = 300 #Alan girişi 
x2 = 6 #oda sayısı girişi
x3 = 2 # bina yaşı girişi

y = a + b1*x1 + b2*x2 + b3*x3

y