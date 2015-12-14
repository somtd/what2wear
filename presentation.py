# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot as plt

data     = np.genfromtxt('sample_wear.txt', delimiter=',', skiprows=1)

temperature=data[:,0]
rainfall=data[:,1]
#
outer    = data[:,2]
tops     = data[:,3]
bottoms  = data[:,4]

print(outer)
print(tops)
print(bottoms)

# http://matplotlib.org/api/markers_api.html
plt.figure(figsize=(12,3))
plt.subplots_adjust(bottom=.2)

plt.subplot(1,3,1)
plt.xlim(-10., 110.)
plt.ylim(-10., 40.)
plt.xlabel("rainfall(%)")
plt.ylabel("temperature")
plt.title("outer", fontsize='small')
for t,marker,c in zip(xrange(4),"ox^D",'bgrc'):
    plt.scatter(data[outer == t,1],
                data[outer == t,0],
                marker=marker,
                c=c)

plt.subplot(1,3,2)
plt.xlim(-10., 110.)
plt.ylim(-10., 40.)
plt.xlabel("rainfall(%)")
plt.ylabel("temperature")
plt.title("tops", fontsize='small')
for t,marker,c in zip(xrange(4),"ox^D",'bgrc'):
    plt.scatter(data[tops == t,1],
                data[tops == t,0],
                marker=marker,
                c=c)

plt.subplot(1,3,3)
plt.xlim(-10., 110.)
plt.ylim(-10., 40.)
plt.xlabel("rainfall(%)")
plt.ylabel("temperature")
plt.title("bottoms", fontsize='small')
for t,marker,c in zip(xrange(3),"ox^",'bgr'):
    plt.scatter(data[tops == t,1],
                data[tops == t,0],
                marker=marker,
                c=c)
plt.show()
plt.savefig('sample_wear.png')

