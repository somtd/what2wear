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

#tops 分類

#tops Tシャツ
t_tops_temp = data[tops == 0.,0]
t_tops_rain = data[tops == 0.,1]
t_tops = np.c_[t_tops_temp,t_tops_rain]

#tops シャツ
ck_tops_temp = data[tops == 1.,0]
ck_tops_rain = data[tops == 1.,1]
ck_tops = np.c_[ck_tops_temp,ck_tops_rain]

#tops パーカー
pkr_tops_temp = data[tops == 2.,0]
pkr_tops_rain = data[tops == 2.,1]
pkr_tops = np.c_[pkr_tops_temp,pkr_tops_rain]

#tops セーター
swr_tops_temp = data[tops == 3.,0]
swr_tops_rain = data[tops == 3.,1]
swr_tops = np.c_[swr_tops_temp,swr_tops_rain]

plt.scatter(t_tops[:, 0], t_tops[:, 1], color='gray')
plt.scatter(ck_tops[:, 0], ck_tops[:, 1], color='blue')
plt.scatter(pkr_tops[:, 0], pkr_tops[:, 1], color='green')
plt.scatter(swr_tops[:, 0], swr_tops[:, 1], color='red')

training_data = np.r_[t_tops,ck_tops,pkr_tops,swr_tops]
training_labels = np.r_[
        np.zeros(len(t_tops)),
        np.ones(len(ck_tops)) * 1,
        np.ones(len(pkr_tops)) * 2,
        np.ones(len(swr_tops)) * 3,
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

Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.bone, alpha=0.2)

plt.title('Tops', fontsize='large')
plt.xlabel('Temperature')
plt.ylabel('RainFall')
plt.autoscale()
plt.grid()
plt.show()

