import numpy as np
import matplotlib.pyplot as plt

#The inverse transformation of Euler angles:
#This is to transform the vectors in localized reference frame into  \
#vectors in RD-TD-ND space.
def eulerinv(vector,euler):
    #vector: expressed in the reference frame of the grain
    #euler: the orientation (three euler angles, array(3)) of the point
    #vector: array(3); euler: array(3).
    #change the vector to a column one:
    vector=np.matrix(vector).T
    #vector: martrix(3,1)
    #Three euler angles: (phai1, PHI, phai2: float64)
    phai1=euler[0]
    PHI=euler[1]
    phai2=euler[2]
    #euler angles transformation matrices: phai1--A, PHAI--B, phai2--C
    #A, B, C: matrix(3,3)
    A=np.matrix([[np.cos(phai1),np.sin(phai1),0.0],[-np.sin(phai1),np.cos(phai1),0.0], \
            [0.0,0.0,1.0]], dtype='float64')
    B=np.matrix([[1.0,0.0,0.0],[0.0,np.cos(PHI),np.sin(PHI)], \
            [0.0,-np.sin(PHI),np.cos(PHI)]],dtype='float64')
    C=np.matrix([[np.cos(phai2),np.sin(phai2),0.0],[-np.sin(phai2),np.cos(phai2),0.0], \
            [0.0,0.0,1.0]], dtype='float64')
    #euler transformation
    #trans: matrix(3,3)
    trans=C*B*A
    invtrans=np.linalg.inv(trans)
    #local vector: vector represented by the local coordinate system of the grain
    #local vector: matrix(3,1)
    specivector=invtrans*vector
    specivector=np.array(specivector.T)[0]
    #localvector: array(3)
    return specivector

MCsteps=80
gasvalue=-1
count=0

source=np.load("F:/M5EVACON/MCresult/MCS{}.npy".format(MCsteps))
#source=np.load("F:/MC1/round2400000.npy")
NT=len(source)

#[0,0,1] vector
VEC=np.array([0.0,0.0,1.0])
r=list()
theta=list()
PHAI1=list()

for i in range(NT):
    for j in range(NT):
        for k in range(NT):
            if abs(source[i,j,k,0]-gasvalue)<1e-3:
                continue
            PHAI1.append(source[i,j,k,0])
            x,y,z=eulerinv(VEC,source[i,j,k])
            rad=np.sqrt(x**2+y**2)
            r.append(rad)
            if x>0 and y>0:
                thetas=np.arctan(y/x)
            if x<0 and y>0:
                thetas=np.arctan(y/x)+np.pi
            if x<0 and y<0:
                thetas=np.arctan(y/x)+np.pi
            if x>0 and y<0:
                thetas=np.arctan(y/x)
            theta.append(thetas)
            count=count+1

PHAI1=set(PHAI1)
print("\n\nLOADING...")
area = 20
colors = r
print(len(PHAI1))

fig = plt.figure()
ax = fig.add_subplot(projection='polar')
c = ax.scatter(theta, r, c=colors, s=area, cmap='hsv', alpha=0.75)
ax.set_title("\n{} MCS, {} orientations".format(MCsteps,len(PHAI1)))
plt.savefig("F:/M5EVACON/polarresult/polar_MCS{}_ori={}.jpg".format(MCsteps,len(PHAI1)),dpi=1000)
