import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

X = np.asarray([[1,1],[2,2],[2,0],[0,0],[1,0],[0,1]])
Y = np.asarray([0,0,0,1,1,1])

fig, ax = plt.subplots()
clf = svm.LinearSVC(C=3).fit(X,Y)

w = clf.coef_[0]
a = -w[0]/w[1]
xx = np.linspace(-5,5)
yy = a * xx - (clf.intercept_[0]) / w[1]

x_min, x_max = X[:,0].min() - 1, X[:,0].max() + 1
y_min, y_max = X[:,1].min() - 1, X[:,1].max() + 1
xx2, yy2 = np.meshgrid(np.arange(x_min, x_max, .2),np.arange(y_min, y_max, .2))
Z = clf.predict(np.c_[xx2.ravel(), yy2.ravel()])
Z = Z.reshape(xx2.shape)
ax.contourf(xx2, yy2, Z, cmap=plt.cm.coolwarm, alpha = 0.3)
ax.scatter(X[:,0], X[:,1] , c = Y, cmap = plt.cm.coolwarm, s=25)
ax.plot(xx, yy)

ax.axis([x_min, x_max, y_min, y_max])
plt.title("SVM")
plt.show()
print(Z)
