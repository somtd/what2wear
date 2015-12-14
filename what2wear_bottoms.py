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

#bottoms 分類

#bottoms 短パン
hrf_bottoms_temp = data[bottoms == 0.,0]
hrf_bottoms_rain = data[bottoms == 0.,1]
hrf_bottoms = np.c_[hrf_bottoms_temp,hrf_bottoms_rain]

#bottoms チノパン
chn_bottoms_temp = data[bottoms == 1.,0]
chn_bottoms_rain = data[bottoms == 1.,1]
chn_bottoms = np.c_[chn_bottoms_temp,chn_bottoms_rain]

#bottoms デニム
dnm_bottoms_temp = data[bottoms == 2.,0]
dnm_bottoms_rain = data[bottoms == 2.,1]
dnm_bottoms = np.c_[dnm_bottoms_temp,dnm_bottoms_rain]

plt.scatter(hrf_bottoms[:, 0], hrf_bottoms[:, 1], color='blue')
plt.scatter(chn_bottoms[:, 0], chn_bottoms[:, 1], color='green')
plt.scatter(dnm_bottoms[:, 0], dnm_bottoms[:, 1], color='red')

training_data = np.r_[hrf_bottoms,chn_bottoms,dnm_bottoms]
training_labels = np.r_[
        np.zeros(len(hrf_bottoms)),
        np.ones(len(chn_bottoms)) * 1,
        np.ones(len(dnm_bottoms)) * 2,
        ]

clf = svm.SVC(gamma=0.001)
clf.fit(training_data, training_labels)

training_x_min = training_data[:, 0].min() - 1
training_x_max = training_data[:, 0].max() + 1
training_y_min = training_data[:, 1].min() - 1
training_y_max = training_data[:, 1].max() + 1

grid_interval = 0.02
xx, yy = np.meshgrid(
        np.arange(training_x_min, training_x_max, grid_interval),
        np.arange(training_y_min, training_y_max, grid_interval),
        )

#Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

#Z = Z.reshape(xx.shape)
#plt.contourf(xx, yy, Z, cmap=plt.cm.bone, alpha=0.2)

plt.title('Bottoms', fontsize='large')
plt.xlabel('Temperature')
plt.ylabel('RainFall')
plt.autoscale()
plt.grid()
plt.show()

