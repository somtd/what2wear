# -*- coding: utf-8 -*-

import numpy as np
from sklearn import svm
from matplotlib import pyplot as plt
from sklearn import cross_validation

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

#plt.scatter(no_outer[:, 0], no_outer[:, 1], color='gray')
#plt.scatter(jkt_outer[:, 0], jkt_outer[:, 1], color='blue')
#plt.scatter(nyln_outer[:, 0], nyln_outer[:, 1], color='green')
#plt.scatter(dwn_outer[:, 0], dwn_outer[:, 1], color='red')

#トレーニングデータ（気温、降水確率）
training_data = np.r_[no_outer,jkt_outer,nyln_outer,dwn_outer]
#教師データ（気温、降水確率に対応するアウターの種類）
training_labels = np.r_[
        np.zeros(len(no_outer)), #0:なし
        np.ones(len(jkt_outer)) * 1, #1:ジャケット
        np.ones(len(nyln_outer)) * 2, #2:ナイロンパーカー
        np.ones(len(dwn_outer)) * 3, #3:ダウンジャケット
        ]

# K-分割交差検証
kfold = cross_validation.KFold(len(training_data), n_folds=10)
results = np.array([])
for training, test in kfold:
    classifier = svm.SVC(gamma=0.01)
    classifier.fit(training_data[training], training_labels[training])

    answers = classifier.predict(training_data[test])
    iscorrect = answers == training_labels[test]
    results = np.r_[results, iscorrect]

correct = np.sum(results)
N = len(training_data)
percent = (float(correct) / N) * 100
print(percent)






