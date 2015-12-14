from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
import numpy as np

data = load_iris()

features = data['data']
feature_names = data['feature_names']
target = data['target']
target_names = data['target_names']
labels = target_names[target]

print(target)

print(target_names)

for t,marker,c in zip(xrange(3),">ox","rgb"):
    plt.scatter(features[target == t,2],
            features[target == t,0],
            marker=marker,
            c=c)
#plt.show()
