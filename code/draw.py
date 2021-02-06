import numpy as np
import mmcv
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

file = './data/0315/solid_0.264_256_score_location.pkl'
file = './data/0315/detbox_init/score_location.pkl'
file = './data/0330-16-256/detbox_init/score_location.pkl'
data = mmcv.load(file)
scores = data['scores']
transforms = np.asarray(data['transforms'])
dx = transforms[:,0]
dy = transforms[:,1]
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(1,1,1)
# print((dx**2+dy**2)**0.5)
cm = plt.cm.get_cmap('Blues')
ax.scatter(dx,dy,s=np.asarray(scores)*80+5, c=scores,cmap=cm)
# ax1 = plt.axes(projection='3d')
# ax1.scatter3D(dx,dy,scores, cmap='Blues')
# ax1.plot3D(dx,dy,scores, 'gray')
# plt.show()
plt.savefig('./colored_points.jpg')

fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(1,1,1)
scores = np.linspace(0,1,11)
x = [1] * 11
ax.scatter(x,scores,s=np.asarray(scores)*80+5, c=scores,cmap=cm)
for i in range(11):
    plt.annotate('{:.1f}'.format(scores[i]), xy = (x[i], scores[i]), xytext = (x[i]+0.1, scores[i]))
plt.savefig('./colored_annos.jpg')
# for x,y in zip(dx,dy):
#     print(np.sqrt(x**2+y**2))
# a = np.reshape(dx, (4, 25))
# print(a[0])
by_distance = np.mean(np.reshape(scores, (4, 25)), axis=1)
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot([5.0, 8.0, 12.0, 16.0], by_distance)
ax.set_xlabel('distance')
ax.set_ylabel('score')
plt.savefig('./distance.jpg')