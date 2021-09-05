import numpy as np
import matplotlib.pyplot as plt

MCsteps=50
gasvalue=-1.0
source=np.load("F:/M5EVACON/MCresult/MCS{}.npy".format(MCsteps))
NT=len(source)
PHAI1=list()
for i in range(NT):
    for j in range(NT):
        for k in range(NT):
            if abs(source[i,j,k,0]-gasvalue)<1e-3:
                continue
            PHAI1.append(source[i,j,k,0])
PHAI1=set(PHAI1)

sourcecolor=list()
zaxis=list()
yaxis=list()
sourcefill=np.zeros((NT,NT,NT))
for i in range(NT):
    for j in range(NT):
        for k in range(NT):
            if i>10 and i<NT-10 and j>10 and j<NT-10 and k>10 and k<NT-10:
                source[i,j,k]=np.array([gasvalue,gasvalue,gasvalue])
            if source[i,j,k,0]==gasvalue:
                #print(hex(255))
                #print(str(hex(255)))
                value=str(hex(255))[2:]
                #print(value)
                colorstring="#"+value+value+value
            else:
                base1=source[i,j,k,0]/(2*np.pi)
                base2=source[i,j,k,1]/(2*np.pi)
                base3=source[i,j,k,2]/(2*np.pi)
                #value1=str(hex(int(255*(1-base1))))[2:]
                #value2=str(hex(int(255*(1-base2))))[2:]
                #value3=str(hex(int(255*(1-base3))))[2:]
                value1=str(hex(int(255*(base1))))[2:]
                value2=str(hex(int(255*(base2))))[2:]
                value3=str(hex(int(255*(base3))))[2:]
                if len(value1)<2:
                    value1="0"+value1
                if len(value2)<2:
                    value2="0"+value2
                if len(value3)<2:
                    value3="0"+value3
                colorstring="#"+value1+value2+value3
                sourcefill[i,j,k]=1
            zaxis.append(colorstring)
        yaxis.append(zaxis)
        zaxis=list()
    sourcecolor.append(yaxis)
    yaxis=list()
sourcecolor=np.array(sourcecolor)

def explode(colordata):
    size=np.array(colordata.shape)*2
    colordata_e=np.zeros(size-1, dtype=colordata.dtype)
    colordata_e[::2,::2,::2]=colordata
    return colordata_e

facecolors=sourcecolor
edgecolors=facecolors

filled_2=explode(sourcefill)
fcolors_2=explode(facecolors)
ecolors_2=explode(edgecolors)

print(filled_2.shape)
print(fcolors_2.shape)
print(ecolors_2.shape)
x,y,z=np.indices(np.array(filled_2.shape)+1).astype(float)//2
x[0::2,:,:]+=0.00
y[:,0::2,:]+=0.00
z[:,:,0::2]+=0.00
x[1::2,:,:]+=1.00
y[:,1::2,:]+=1.00
z[:,:,1::2]+=1.00
print(x.shape)
print(y.shape)
print(z.shape)
ax=plt.figure().add_subplot(projection='3d')
ax.voxels(x,y,z,filled_2, facecolors=fcolors_2, edgecolors=ecolors_2)
ax.set_title("\n{} MCS, {} orientations".format(MCsteps,len(PHAI1)))
#plt.savefig("F:/M5EVACON/3Dresult/MCS{}.jpg".format(MCsteps),dpi=1000)
plt.savefig("F:/M5EVACON/3Dresult/MCS{}_2.jpg".format(MCsteps),dpi=1000)
#plt.show()
