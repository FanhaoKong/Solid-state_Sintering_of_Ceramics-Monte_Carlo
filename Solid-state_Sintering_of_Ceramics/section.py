import numpy as np
import matplotlib.pyplot as plt

MCsteps=113
count=0
NT=60
gasvalue=-1.0
source=np.load("MCresult/MCS{}.npy".format(MCsteps))
PHAI1=list()
for i in range(NT):
    for j in range(NT):
        for k in range(NT):
            if abs(source[i,j,k,0]-gasvalue)>1e-3:
                PHAI1.append(source[i,j,k,0])
PHAI1=set(PHAI1)
slices=source[:,:,:,0]
#slices=slices[int(len(source)*1/2)]
#slices=slices[int(len(source))-3]
slices=slices[:,:,59]
for i in range(3,NT-3):
     for j in range(3,NT-3):
         if abs(slices[i,j]+1)>1e-3:
             #slices[i,j]+=5
             pass
         else:
             count=count+1
print(count)
print(len(PHAI1))
plt.figure(figsize=(12,12))
plt.pcolormesh(slices,cmap='binary')
ax=plt.gca()
ax.set_aspect(1)
#plt.savefig("/home/kongfh/Desktop/MCS_{}.jpg".format(step),dpi=500)
plt.show()
