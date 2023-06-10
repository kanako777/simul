import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches

np.random.seed(19680801)

x = []
y = []

with open("./buspos.txt","r") as fp:
    t=0
    while t<20:
        t+=1
        line = fp.readline()
        poslst = line.split('/')[:-1]
        print(len(poslst))
        for pos in poslst:
            tx,ty = np.array(pos.split(','),dtype=np.float32)
            tx = tx * 1
            ty = ty * 1
            if tx>=0:
                x.append(tx)
                y.append(ty)

N=386
colors = np.random.rand(N)
c=colors*0+0.225
area = (30 * np.random.rand(N))**2  # 0 to 15 point radii

filename = "map.png"
lonmin, lonmax, latmin, latmax = (-400,1400,-100,1100) # just example
image_extent = (lonmin, lonmax, latmin, latmax)
ax = plt.gca()
ax.imshow(plt.imread(filename), extent=image_extent)
rect = patches.Rectangle((0, 0), 1000, 1000, linewidth=1, edgecolor='r', facecolor='none')
ax.add_patch(rect)

plt.scatter(x,y, s=area*0+150, marker=".", color='b', alpha=0.5)
plt.scatter(500,450, s=200, marker="X", color='r', alpha=1)
plt.show()
