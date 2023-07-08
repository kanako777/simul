import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
import cmocean.cm as cmo
import seaborn as sns

cmap = cmo.ice
cmap
#np.random.seed(19680801)

x = []
y = []

map_pos = [[0,0,0,0,0],
           [0,0,0,0,0],
           [0,0,0,0,0],
           [0,0,0,0,0],
           [0,0,0,0,0]]

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
                map_pos[int(tx/200)][int(ty/200)] += 1

# 버스 위치를 파일로 쓰기
str1 = ''
for i in range(len(x)):
    str1 += str(x[i]) + "," + str(y[i]) + "/"
    #str1 = ''.join([str(a) for a in x])

with open("sample.txt", "w+") as f:
  data = f.read()
  f.write(str1)

# 그래프 시작
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 4),
                        gridspec_kw={"wspace":0.1, "hspace":0.1},
                        sharex=True, sharey=True, constrained_layout=True)

filename = "map2.png"
lonmin, lonmax, latmin, latmax = (0,1000,0,1000) # just example

#filename = "map.png"
#lonmin, lonmax, latmin, latmax = (-400,1400,-100,1100) # just example

image_extent = (lonmin, lonmax, latmin, latmax)

ax1.imshow(plt.imread(filename), extent=image_extent)
ax2.imshow(plt.imread(filename), extent=image_extent)
rect = patches.Rectangle((0, 0), 1000, 1000, linewidth=0.5, edgecolor='g', facecolor='none')
ax1.add_patch(rect)
ax1.scatter(500,480, s=150, marker="X", color='r', alpha=1)
ax2.scatter(500,480, s=150, marker="X", color='r', alpha=1)

ax1.scatter(x, y, s=50, alpha=0.5)
sns.kdeplot(x=x, y=y, cmap=plt.cm.Reds, levels=20, ax=ax2)

#ax2.hist2d(x, y, cmap=plt.cm.BuPu, bins=30)


map = []
num = [0, 1, 2, 3, 4]
for i in num:
    for j in num:
        test = patches.Rectangle((j*200, i*200), 200, 200, linewidth=0.5, edgecolor='g', facecolor='none')
        ax2.add_patch(test)


#plt.show()


# titles
titles = ["Matplotlib scatter (alpha=1)", "Matplotlib scatter (alpha=0.1)"]
for ax, title in zip((ax1, ax2), titles):
    ax.set_title(title, fontsize="x-large", pad=16)
    ax.set_aspect(1)

plt.show()


