# -*- coding: utf-8 -*-

import numpy as np
from sklearn import svm
from matplotlib import pyplot as plt
          
data = np.genfromtxt('sample_wear.txt', delimiter=',', skiprows=1)

temperature=data[:,0]
rainfall=data[:,1]
#
outer    = data[:,2]
tops     = data[:,3]
bottoms  = data[:,4]

#outer 分類

#outerなし
no_outer_temp = data[outer == 0.,0]
no_outer_rain = data[outer == 0.,1]
no_outer = np.c_[no_outer_temp,no_outer_rain]

#outerジャケット
jkt_outer_temp = data[outer == 1.,0]
jkt_outer_rain = data[outer == 1.,1]
jkt_outer = np.c_[jkt_outer_temp,jkt_outer_rain]

#outerナイロンパーカー
nyln_outer_temp = data[outer == 2.,0]
nyln_outer_rain = data[outer == 2.,1]
nyln_outer = np.c_[nyln_outer_temp,nyln_outer_rain]

#outerダウンジャケット
dwn_outer_temp = data[outer == 3.,0]
dwn_outer_rain = data[outer == 3.,1]
dwn_outer = np.c_[dwn_outer_temp,dwn_outer_rain]

plt.scatter(no_outer[:, 0], no_outer[:, 1], color='gray')
plt.scatter(jkt_outer[:, 0], jkt_outer[:, 1], color='blue')
plt.scatter(nyln_outer[:, 0], nyln_outer[:, 1], color='green')
plt.scatter(dwn_outer[:, 0], dwn_outer[:, 1], color='red')

#トレーニングデータ（気温、降水確率）
training_data = np.r_[no_outer,jkt_outer,nyln_outer,dwn_outer]
#教師データ（気温、降水確率に対応するアウターの種類）
training_labels = np.r_[
        np.zeros(len(no_outer)), #0:なし
        np.ones(len(jkt_outer)) * 1, #1:ジャケット
        np.ones(len(nyln_outer)) * 2, #2:ナイロンパーカー
        np.ones(len(dwn_outer)) * 3, #3:ダウンジャケット
        ]

#classifier = svm.LinearSVC()
classifier = svm.SVC(gamma=0.001)
classifier.fit(training_data, training_labels)

training_x_min = training_data[:, 0].min() - 1
training_x_max = training_data[:, 0].max() + 1
training_y_min = training_data[:, 1].min() - 1
training_y_max = training_data[:, 1].max() + 1

grid_interval = 0.02
xx, yy = np.meshgrid(
        np.arange(training_x_min, training_x_max, grid_interval),
        np.arange(training_y_min, training_y_max, grid_interval),
        )
# 取りうる値の範囲内の全データを0.02間隔で生成する
#print(xx)
#print(yy)
#print(np.c_[xx.ravel(),yy.ravel()])

#Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
#print(Z)
#zs = classifier.predict([20., 50.])
#print(zs)

#Z = Z.reshape(xx.shape)
#print(Z)

#contourf:塗りつぶし等高線 Zはラベルの配列
#plt.contourf(xx, yy, Z, cmap=plt.cm.bone, alpha=0.2)

plt.title('Outer', fontsize='large')
plt.xlabel('Temperature')
plt.ylabel('RainFall')
plt.autoscale()
plt.grid()
plt.show()

