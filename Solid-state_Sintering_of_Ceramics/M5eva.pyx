import numpy as np
import time

#ALL ANGLES SHOULD BE MEASURED IN RADIANS. DEGREES ARE PROHIBITED.
#setup
    #ENERGY
#Boltzmann Constant
kb=1.380649e-23
#Gas constant (J/(mol*K))
R=8.314
#Temperature T0 (Kelvin)
T0=1300.0
#Temperature of the system (Kelvin):
temperature=1600.0
#The surface energy (solid-gas interface energy, sgenergy):
sgenergy=10*kb*T0
#The ratio of GBenergy/surfenergy:
ratio=0.5
#The grain boundary energy (solid-solid interface energy,ssenergy):
ssenergy=ratio*sgenergy

    #MASS TRANSMISSION
    #densification mechanism
#diffusion on grain boundaries: (mechanism 0)
#diffusion from grain boundaries to grain lattice: (mechanism 1)
    #non-densification mechanism
#surface diffusion: (mechanism 2)
#diffusion from surface to grain lattice: (mechanism 3)
#evaporation: (mechanism 4)
#condensation: (mechanism 5)
    #and also the motion of gas (motion of pores)
#gas through grain boundaries: (6)
#gas through lattice (7)
    #activation energy of the mechanisms, array(8), KJ/mol
              #  0    1    2    3    4     5     6    7
Qact=np.array([50.0,50.0,45.0,50.0,40.0,40.0,30.0,100.0])
    #probability factor of these mechanisms: exp(-Q/RT)
    #"*1000" can eliminate the error introduced by float numbers.
PROB=np.exp((-Qact*1000)/(R*temperature))*1000

    #MATERIAL
#oxide ceramics:0  non-oxide ceramics:1
#if it is oxide ceramics, evaporation and condensation should be considered.
mater=0

    #SIMULATION_BOX
#The size of the  simulation box 
NT=60
#The start and end index of the initialization:
starti=0
endi=NT-1
#The initial proportion of gas:
#(The initial location of pores should not be the surface of the specimen)
pgas=0.20
refgas=pgas*(NT**3)/((NT-2)**3)
#How to represent gas (the value inserted in euler angles, no meaning):
gasvalue=-1.0
#The number of orientations in the box:
orient=0

    #MONTE_CARLO_SETUP
#The length of one Monte Carlo step (MCS):
MCS=NT**3
#The number of steps (in the unit of MCS):
rounds=500*MCS
#The round to begin anisotropy (measured in MCS):
beginani=500
'''
#The cutoff setting (Gaussian radical function, three sigma rule):
#sigma, cutoff, intcut and cutoffsq are in the unit of spacestep.
sigma=1.0
cutoff=3.0*sigma
intcut=int(cutoff)
cutoffsq=int(1.0+(sigma*3)**2)
'''
intcut=1
#The radius of the considered range: (whether to swap orientation with a point in this range or not)
radconsider=1
#The total number of points in the considered range:
radpoints=0
for i in range(-radconsider,radconsider+1):
    for j in range(-radconsider,radconsider+1):
        for k in range(-radconsider,radconsider+1):
            if radconsider>1:
                if i**2+j**2+k**2>radconsider**2:
                    continue
            radpoints=radpoints+1

#The transformation of Euler angles:
#This is to transform the vectors in RD-TD-ND reference frame into  \
#vectors in a localized grain reference frame.
def eulerinv(vector,euler):
    #vector: expressed in the reference frame of the specimen
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
    #local vector: vector represented by the local coordinate system of the grain
    #local vector: matrix(3,1)
    localvector=trans*vector
    localvector=np.array(localvector.T)[0]
    #localvector: array(3)
    return localvector

#The anisotropy of grain growth:
def aniso(localvector,MCsteps):
    #localvector: vector represented by the local coordinate system of the grain
    #local vector: array(3)
    #length: the length of "localvector", float64
    length=np.sqrt(localvector[0]**2+localvector[1]**2+localvector[2]**2)
    #reference: Computational Materials Science 71(2013) 25-32
    #n controls the shape of the  periphery, c, a controls c/a
    n=4
    c=4
    a=1
    #sym controls the symmetry of the lattice
    sym=6
    #base controls the extent of anisotropy on XOY plane
    base=0.95
    if MCsteps<beginani:
        #Anisotropy has not yet been activated. 
        coefficient=1.0
    else:
        if abs(localvector[0]/length)<1e-6 and abs(localvector[1]/length)<1e-6 \
            and abs(abs(localvector[2]/length)-1)<1e-6:
            coefficient=(c/2.0)/max(a,(c/2.0))
        else:
            #coordinate transformation from cartesian coord to spherical coord
            #phai: -pi/2--pi/2, float64; theta: 0--pi, float64
            phai=np.arctan(localvector[1]/localvector[0])
            theta=np.arccos(localvector[2]/length)
            #r: float64
            r=(base+(1.0-base)*np.sin(sym*phai))*np.power( \
                ((np.cos(theta)/(c/2))**n+(np.sin(theta)/a)**n), (-1.0/n))
            r=abs(r)
            #coefficient: value of anisotropy, float64
            coefficient=r/max(a,(c/2.0))
    return coefficient

#count the average of anisotropy coefficient:
def anisoaverage():
    n=4; c=4; a=1; base=0.95; sym=6;
    dtheta,dphai=np.pi/180,np.pi/180
    theta,phai=np.mgrid[0:np.pi+dtheta:dtheta, \
                     -np.pi:np.pi+dphai:dphai]
    r=(base+(1.0-base)*np.sin(sym*phai))*np.power( \
        ((np.cos(theta)/(c/2))**n+(np.sin(theta)/a)**n), (-1.0/n))
    r=abs(r)
    suma=0.0; count=0;
    coefficient=r/max(a,(c/2.0))
    for i in range(len(coefficient)):
        for j in range(len(coefficient[0])):
            suma=suma+coefficient[i,j]
            count=count+1
    average=suma/count
    return average
    
#initialization (from cratch, completely random)
def initial1():
    randnum=0.0
    #ORI: orientation of all points, array(NT,NT,NT,3)
    #the orientation of each point is represented by three euler angles. 
    ORI=np.full((NT,NT,NT,3),gasvalue)
    #go through all points in the center part of the simulation box where solid should be put.
    #starti and endi: the start and end index of the initialization:
    for i in range(starti, endi+1):
        for j in range(starti, endi+1):
            for k in range(starti, endi+1):
                if i>starti and i<endi and j>starti and j<endi and \
                    k>starti and k<endi:
                    #The point is not on the surface. It can be either gaseous or solid. 
                    randnum=np.random.rand()
                    if randnum>refgas:
                        #This point is solid. Three euler angle values should not be "gasvalue".
                        #ORI is a 4-dimension array. ORI[i,j,k] has three entries. 
                        ORI[i,j,k]=2*np.pi*np.random.rand(3)
                else:
                    ORI[i,j,k]=2*np.pi*np.random.rand(3)
    #countatm: the amount of ceramics in the atmosphere, which is initially 0. 
    countatm=0
    #ORI: array(NT,NT,NT,3). countatm: int. 
    return ORI,countatm

#initialization (from cratch, limited number of orientations)
def initial2():
    randnum=0.0
    #ORI: orientation of all points, array(NT,NT,NT,3)
    #the orientation of each point is represented by three euler angles. 
    ORI=np.full((NT,NT,NT,3),gasvalue)
    orientation=2*np.pi*np.random.rand(orient,3)
    #go through all points in the center part of the simulation box where solid should be put.
    #starti and endi: the start and end index of the initialization:
    for i in range(starti, endi+1):
        for j in range(starti, endi+1):
            for k in range(starti, endi+1):
                if i>starti and i<endi and j>starti and j<endi and \
                    k>starti and k<endi:
                    #The point is not on the surface. It can be either gaseous or solid. 
                    randnum=np.random.rand()
                    if randnum>refgas:
                        randinteger=np.random.randint(0,orient)
                        #This point is solid. Three euler angle values should not be "gasvalue".
                        #ORI is a 4-dimension array. ORI[i,j,k] has three entries. 
                        ORI[i,j,k]=orientation[randinteger]
                else:
                    randinteger=np.random.randint(0,orient)
                    ORI[i,j,k]=orientation[randinteger]
    #countatm: the amount of ceramics in the atmosphere, which is initially 0. 
    countatm=0
    #ORI: array(NT,NT,NT,3). countatm: int. 
    return ORI,countatm

#MCS: choose the points randomly yet without repetition in a MCS
def MCSmap():
    count=0
    coordmap=np.zeros((NT**3,3),dtype=int)
    for i in range(NT):
        for j in range(NT):
            for k in range(NT):
                coordmap[count]=np.array([i,j,k])
                count=count+1
    np.random.shuffle(coordmap)
    #coordmap: array(NT**3, 3)
    return coordmap

#determine the kind of the point, namely, whether it is gaseous, on the surface, etc. 
#surfcurv: the number of neighboring solid points.
def point(ORI,coord):
    #ORI: the orientation field. coord: the coordinate of a point
    #ORI: array(NT,NT,NT,3). coord: array(3)
    #the kind of points:
    #1: gas in gas (this kind of points should not be chosen in MC steps);
    #2: gas on surface (where condensation can happen);
    #3: solid on surface (where evaporation can happen);
    #4: solid on grain boundary;
    #5: solid in lattice (this kind of points should not be chosen in MC steps)
    #6: solid on both surface and grain boundaries
    #copy the coordinate of the point
    x,y,z=coord
    #pointkind: the kind of the point, represented by an integer.
    #surfcurv: the number of neighboring solid points, integer. 
    #two returned variables of this function. Both of them are integer. 
    pointkind=0
    surfcurv=0
    #assistant variables
    jud=0
    #First, decide whether this point is solid or gaseous. 
    if abs(ORI[x,y,z,0]-gasvalue)<1e-3:
        #the point is gaseous
        #go through its neighboring points
        for i in range(-radconsider,radconsider+1):
            for j in range(-radconsider,radconsider+1):
                for k in range(-radconsider,radconsider+1):
                    if i==0 and j==0 and k==0:
                        #avoid considering the point itself
                        continue
                    if x+i<0 or x+i>NT-1 or y+j<0 or y+j>NT-1 or z+k<0 or z+k>NT-1:
                        #this point is on the edge of the simulation box
                        continue
                    if radconsider>1:
                        if i**2+j**2+k**2>radconsider**2:
                            #not in the range of consideration.
                            continue
                    #count the number of solid neighboring points (no consideration of orientation)
                    if abs(ORI[x+i,y+j,z+k,0]-gasvalue)>1e-3:
                        #this neighboring point is solid
                        surfcurv=surfcurv+1
                        jud=1
        if jud==0:
            #the point is "gas in gas" --1
            pointkind=1
        else:
            #the point is "gas on surface" --2
            pointkind=2
        #recycle the variable "jud"
        jud=0
    else:
        #the point is solid
        #decide whether a solid point is on the edge of the simulation box
        #bond=0: not on the edge. bond=1: on the edge
        bond=0
        #neighgas:
        #neighgas=0: there is no neighboring gaseous points. neighgas=1: there are gas neighboring points.
        neighgas=0
        #count: the number of solid neighboring points with different orientation.
        count=0
        #go through its neighboring point: 
        for i in range(-radconsider, radconsider+1):
            for j in range(-radconsider, radconsider+1):
                for k in range(-radconsider, radconsider+1):
                    if i==0 and j==0 and k==0:
                        #avoid considering itself
                        continue
                    if x+i<0 or x+i>NT-1 or y+j<0 or y+j>NT-1 or z+k<0 or z+k>NT-1:
                        #this solid point is on the edge of the simulation box
                        #bond=1, no matter what its surrounding is, it is a \
                            #"solid on surface --3" point.
                        bond=1
                        continue
                    if radconsider>1:
                        if i**2+j**2+k**2>radconsider**2:
                            #not in the range of consideration.
                            continue
                    #see whether this solid point has neighboring gaseous point. 
                    if abs(ORI[x+i,y+j,z+k,0]-gasvalue)<1e-3:
                        #a neighboring point is gaseous.
                        #the point is "solid on surface" or "solid on surface and grain boundaries"
                        neighgas=1
                    else:
                        surfcurv=surfcurv+1
                        if abs(ORI[x+i,y+j,z+k,0]-ORI[x,y,z,0])>1e-3 or abs(ORI[x+i,y+j,z+k,1]-ORI[x,y,z,1])>1e-3 \
                            or abs(ORI[x+i,y+j,z+k,2]-ORI[x,y,z,2])>1e-3:
                            #a neighboring point is solid but has different orientation
                            count=count+1
        if bond==1:
            #The point is on the edge of the simulation box.
            if count>0:
                #The point has neighboring solid points with different orientation
                #The point is "solid on both surface and grain boundaries" --6
                pointkind=6
            else:
                #The point doesn't have any neighboring solid point with different orientation
                #The point is "solid on surface" --3
                pointkind=3
        else:
            if neighgas==0:
                #This solid point doesn't have any gaseous neighboring points
                if count>0:
                    #The point has solid neighboring point with different orientation
                    #The point is "solid on grain boundary" --4
                    pointkind=4
                    #print("boundary")
                else:
                    #The point doesn't have solid neighboring point with different orientation
                    #The point is "solid in lattice" --5
                    pointkind=5
            else:
                #This solid point has gaseous neighboring points.
                if count==0:
                    #The point doesn't have any solid neighboring point with different orientation
                    #The point is "solid on surface" --3
                    pointkind=3
                else:
                    #The point has solid neighboring point with different orientation.
                    #The point is "solid on both surfacce and grain boundaries" --6
                    pointkind=6
    #pointkind, surfcurv: int
    #print(pointkind, surfcurv)
    return pointkind, surfcurv

#decide the kind of tramsmission mechanism (excluding evaporation and condensation):
#return the type of mechanism and its probability factor
def transmech(coord1,coord2,pointkind1,surfcurv1,pointkind2,surfcurv2):
    #point1: studied point; point2: chosen point
    #the type of transmission
    transmission=0
    #its probability factor
    probability=0.0
    #transmission mechanisms:
        #densification mechanism
    #diffusion on grain boundaries: (mechanism 0)
    #diffusion from grain boundaries to grain lattice: (mechanism 1)
        #non-densification mechanism
    #surface diffusion: (mechanism 2)
    #diffusion from surface to grain lattice: (mechanism 3)
    #evaporation: (mechanism 4)
    #condensation: (mechanism 5)
        #and also the motion of gas (motion of pores)
    #gas through grain boundaries: (6)
    #gas through lattice (7)
    #PROB: array(8), probability factor of these mechanisms: exp(-Q/RT)
    if pointkind1==2 or pointkind2==2:
        #one of the two points is "gas on surface"
        if pointkind1==2:
            #point1 is gaseous, "gas on surface"--2
            if pointkind2==3:
                #point1 is "gas on surface" and point2 is "solid on surface"
                if (surfcurv2+1)/radpoints<0.5:
                    transmission=2
                else:
                    transmission=7
            if pointkind2==6:
                #point2 is "solid on both surface and grain boundaries"
                transmission=6
        else:
            #point2 is gaseous
            if pointkind1==3:
                #point1 is "solid on surface" and point2 is "gas on surface"
                if (surfcurv1+1)/radpoints<0.5:
                    transmission=2
                else:
                    transmission=7
            if pointkind1==6:
                #point1 is "solid on both surface and grain boundaries"
                transmission=6
    elif pointkind1==3 or pointkind2==3:
        #one point is "solid on surface"
        if pointkind1==3 and pointkind2==3:
            #the other point is "solid on surface"
            #transmission: surface diffusion --2
            transmission=2
        elif pointkind1==4 or pointkind2==4:
            #the other point is "solid on grain boundaries"
            #transmission: diffusion on grain boundaries --0
            transmission=0
        elif pointkind1==5 or pointkind2==5:
            #the other point is "solid in lattice"
            #transmission: diffusion from surface to grain lattice --3
            transmission=3
        elif pointkind1==6 or pointkind2==6:
            #the other point is "solid on grain boundaries and surface"
            #transmission: surface diffusion --2
            transmission=2
    elif pointkind1==4 or pointkind2==4:
        #one point is "solid on grain boundaries"
        if pointkind1==4 and pointkind2==4:
            #the other point is "solid on grain boundaries"
            #transmission: diffusion on grain boundaries --0
            transmission=0
        elif pointkind1==5 or pointkind2==5:
            #the other point is "solid in lattce"
            #transmission: diffusion from grain boundaries to lattice --1
            transmission=1
        elif pointkind1==6 or pointkind2==6:
            #the other point is "solid on grain boundaries and surface"
            #transmission: diffusion on grain boundaries --0
            transmission=0
    elif pointkind1==5 or pointkind2==5:
        #one point is in the lattice
        if pointkind1==6 or pointkind2==6:
            #the other point is "solid on grain boundaries and surface"
            #transmission: diffusion from surface to lattice --3
            transmission=3
    elif pointkind1==6 and pointkind2==6:
        #both points are "solid on grain boundaries and surface"
        #transmission: surface diffusion --2
        transmission=2
    #The probability factor of the transmission mechanism: 
    probability=PROB[transmission]
    #Prohibition of transmission     
    if transmission==2 or transmission==6 or transmission==7:
        if pointkind1==2:
            #point1 is gaseous but point2 is solid.
            if coord1[0]==starti or coord1[0]==endi or coord1[1]==starti or coord1[1]==endi \
                or coord1[2]==starti or coord1[2]==endi:
                #point2 is solid.
                if coord2[0]>starti and coord2[0]<endi and coord2[1]>starti and coord2[1]<endi \
                    and coord2[2]>starti and coord2[2]<endi:
                    probability=0.0
            #the studied point is gaseous
            elif (surfcurv1+1)/radpoints<0.5:
                probability=0.0
        if pointkind2==2:
            #point1 is solid but point2 is gaseous. 
            if coord2[0]==starti or coord2[0]==endi or coord2[1]==starti or coord2[1]==endi \
                or coord2[2]==starti or coord2[2]==endi:
                #point1 is solid. 
                if coord1[0]>starti and coord1[0]<endi and coord1[1]>starti and coord1[1]<endi \
                    and coord1[2]>starti and coord1[2]<endi:
                    probability=0.0
            elif (surfcurv2+1)/radpoints<0.5:
                probability=0.0
    #print(transmission, end='   ')
    return transmission,probability

#choose a studied point (at least one of its neighbors has different orientation) and a \
#neighboring point that has different orientation
#which neighboring point is chosen should be decided by the activation energy. 
def choosepoint(ORI, countatm, coordmap, counter):
    #overflow: mechanism for safety.
    overflow=0
    #The shape of ORI: (NT,NT,NT,3)
    #assistant variables
    jud=0
    jud2=0
    #finish: finish choosing the point or not
    finish=0
    #BEGIN THE SELECTION
    while finish==0 and counter<MCS:
        #initialization
        coord1=tuple(np.array([-1,-1,-1]))
        coord2=tuple(np.array([-1,-1,-1]))
        #candidate neighboring points: their index 
        chooseloc=list()
        #the probability of choosing the candidate points (decided by the transmission \
        #mechanism and its activation energy)
        #chooseprob: the probability factor of candidate points
        chooseprob=list()
        #countprob: the accumulation of probability factor
        countprob=list()
        #the transmission mechanisem between the studied point and the candidate points
        choosetrans=list()
        #partition function introduced to help decide the probability of choosing points
        partfunc=0.0
        #the kind of transmission mechanism:
        transmission=0
        #choose the studied point
        while jud==0:
            #begin to choose the studied point ("1")
            #randomly propose a coordinate
            x1,y1,z1=coordmap[counter]
            coord1=tuple(np.array([x1,y1,z1]))
            #the kind of the point and its surroundings(surfcurv):
            pointkind1,surfcurv1=point(ORI,coord1)
            if pointkind1==2 or pointkind1==3 or pointkind1==4 or pointkind1==6:
                #the point is a qualified point. "gas in gas" point (1) or \
                #"solid in lattice" point (5) are not qualified.
                jud=1
                if pointkind1==2 and surfcurv1==1:
                    jud=0
            if jud==0:
                #if the point is unsatisfactory, choose another point. 
                counter=counter+1
            if counter>=MCS:
                finish==1
                overflow=1
                break
        if overflow!=0:
            break
        #the studied point is chosen.
        #record the orientation of the studied point (in three euler angles):
        phai1,PHI,phai2=ORI[x1,y1,z1]
        #recycle the variable "jud"
        jud=0
        #choose a neighboring point with different orientation:
        #go through all its neighboring points
        for i in range(-radconsider, radconsider+1):
            for j in range(-radconsider, radconsider+1):
                for k in range(-radconsider, radconsider+1):
                    if i==0 and j==0 and k==0:
                        #avoid considering itself
                        continue
                    if x1+i<0 or x1+i>NT-1 or y1+j<0 or y1+j>NT-1 or z1+k<0 or z1+k>NT-1:
                        #don't consider "points" outside the simulation box
                        continue
                    if radconsider>1:
                        if i**2+j**2+k**2>radconsider**2:
                            #not in the range of consideration
                            continue
                    #record the orientation of this neighboring point
                    phai1n,PHIn,phai2n=ORI[x1+i,y1+j,z1+k]
                    if abs(phai1-phai1n)>1e-3 or abs(PHI-PHIn)>1e-3 or abs(phai2-phai2n)>1e-3:
                        #the neighboring point has different orientation
                        #the kind and the surroundings of the neighboring point
                        pointkind2,surfcurv2=point(ORI,np.array([x1+i,y1+j,z1+k]))
                        if pointkind2==2 and surfcurv2==1:
                            continue
                        transmit,probability=transmech(coord1, tuple(np.array([x1+i,y1+j,z1+k])), \
                                pointkind1,surfcurv1,pointkind2,surfcurv2)
                        if probability<1e-9:
                            continue
                        #the location of the neighboring point
                        #print(i,j,k)
                        chooseloc.append([x1+i,y1+j,z1+k])
                        #print(i,j,k)
                        #the transmission mechanism
                        choosetrans.append(transmit)
                        #the probability factor
                        chooseprob.append(probability)
                        partfunc=partfunc+probability
                        #the accumulation of probability factor
                        countprob.append(partfunc)
        if len(chooseprob)>0:
            #the point choosing process can be done
            finish=1
            if mater==0:
                #Oxide ceramics. Evaporation and condensation should be considered
                if pointkind1==2 or pointkind1==3 or pointkind1==6:
                    #If the point satisfies some conditions, evaporation or condensation should \
                    #be considered
                    if pointkind1==2 and (surfcurv1+1)/radpoints>0.5 and countatm>0:
                        #condensation
                        transmit=5
                        jud2=1
                    if (pointkind1==3 or pointkind1==6) and (surfcurv1+1)/radpoints<0.5:
                        #evaporation
                        transmit=4
                        jud2=1
                #jud==1: evaporation and condensation will be considered
                if jud2==1:
                    probability=1000*np.exp((-Qact[transmit]*1000)/(R*temperature))
                    chooseprob.append(probability)
                    choosetrans.append(transmit)
                    partfunc=partfunc+probability
                    countprob.append(partfunc)
                    randnum=partfunc*np.random.rand()
                    for i in range(len(countprob)):
                        if countprob[i]>randnum and i!=len(countprob)-1:
                            coord2=tuple(np.array(chooseloc[i]))
                            transmission=choosetrans[i]
                            break
                        if countprob[i]>randnum and i==len(countprob)-1:
                            transmission=choosetrans[i]
                            coord2=tuple(np.array([-1,-1,-1]))
                else:
                    randnum=partfunc*np.random.rand()
                    for i in range(len(countprob)):
                        if countprob[i]>randnum:
                            coord2=tuple(np.array(chooseloc[i]))
                            transmission=choosetrans[i]
                            break
            else:
                #no evaporation or condensation.
                randnum=partfunc*np.random.rand()
                for i in range(len(countprob)):
                    if countprob[i]>randnum:
                        coord2=tuple(np.array(chooseloc[i]))
                        transmission=choosetrans[i]
                        break
        else:
            if mater==0:
                if pointkind1==2 or pointkind1==3 or pointkind1==6:
                    if pointkind1==2 and (surfcurv1+1)/radpoints>0.5 and countatm>0:
                        #only condensation
                        finish=1
                        transmission=5
                        coord2=tuple(np.array([-1,-1,-1]))
                    if (pointkind1==3 or pointkind1==6) and (surfcurv1+1)/radpoints<0.5:
                        #only evaporation
                        finish=1
                        transmission=4
                        coord2=tuple(np.array([-1,-1,-1]))
        if finish==0:
            counter=counter+1
        if counter>=MCS:
            overflow=1
            finish=1
    #print(countprob)
    #print(randnum)
    #print(coord2)
    #print(len(chooseprob))
    return coord1,coord2,transmission,counter,overflow

#energy calculation (only a local calculation to save time, not the total energy)
def calcenergy(ORI,coord1,coord2):
    energy=0.0
    x1,y1,z1=coord1
    x2,y2,z2=coord2
    #energy around coord1
    phai1,PHI,phai2=ORI[coord1]
    #go through all nearby points:
    for i in range(-intcut,intcut+1):
        for j in range(-intcut,intcut+1):
            for k in range(-intcut,intcut+1):
                if i==0 and j==0 and k==0:
                    #avoid considering itself
                    continue
                if x1+i<0 or x1+i>NT-1 or y1+j<0 or y1+j>NT-1 \
                        or z1+k<0 or z1+k>NT-1:
                    #outside the simulation box
                    if abs(phai1-gasvalue)>1e-3:
                        #coord1 is solid:
                        weight=1.0/np.sqrt(i**2+j**2+k**2)
                        energy=energy+weight*sgenergy
                    continue
                phai1n,PHIn,phai2n=ORI[x1+i,y1+j,z1+k]
                if abs(phai1n-phai1)>1e-3 or abs(PHIn-PHI)>1e-3 or abs(phai2n-phai2)>1e-3:
                    #the two points have different orientation
                    weight=1.0/np.sqrt(i**2+j**2+k**2)
                    if abs(phai1n-gasvalue)<1e-3 or abs(phai1-gasvalue)<1e-3:
                        #one of the two points is gaseous -- surface energy: sgenergy
                        energy=energy+weight*sgenergy
                    else:
                        #none of them is gaseous -- grain boundary energy: ssenergy
                        energy=energy+weight*ssenergy
    if abs(coord2[0]+1)>1e-3:
        #no evaporation or condensation:
        #energy around coord2
        phai1,PHI,phai2=ORI[coord2]
        #go through all nearby points:
        for i in range(-intcut,intcut+1):
            for j in range(-intcut,intcut+1):
                for k in range(-intcut,intcut+1):
                    if i==0 and j==0 and k==0:
                        #avoid considering itself
                        continue
                    if x2+i<0 or x2+i>NT-1 or y2+j<0 or y2+j>NT-1 \
                        or z2+k<0 or z2+k>NT-1:
                        #not in the simulation box
                        if abs(phai1-gasvalue)>1e-3:
                            #coord2 is solid:
                            weight=1.0/np.sqrt(i**2+j**2+k**2)
                            energy=energy+weight*sgenergy
                        continue
                    if x2+i==x1 and y2+j==y1 and z2+k==z1:
                        #avoid calculating the interaction energy between point1 and point2 twice
                        continue
                    phai1n,PHIn,phai2n=ORI[x2+i,y2+j,z2+k]
                    if abs(phai1n-phai1)>1e-3 or abs(PHIn-PHI)>1e-3 or abs(phai2n-phai2)>1e-3:
                        weight=1.0/np.sqrt(i**2+j**2+k**2)
                        #the two points have different orientation
                        if abs(phai1n-gasvalue)<1e-3 or abs(phai1-gasvalue)<1e-3:
                            #one of the two points is gaseous --surface energy, sgenergy
                            energy=energy+weight*sgenergy
                        else:
                            #none of the two points is gaseous --grain boundary energy, ssenergy
                            energy=energy+weight*ssenergy
    return energy

#modify the orientation of two points:
def modify(ORI,coord1,coord2,transmission,storage1,storage2,countatm,back):
    #coord1: studied point, array(3). coord2: the chosen neighboring point, array(3).
    #transmission: the type of transmission. 
    #transmission mechanisms:
        #densification mechanism
    #diffusion on grain boundaries: (mechanism 0)
    #diffusion from grain boundaries to grain lattice: (mechanism 1)
        #non-densification mechanism
    #surface diffusion: (mechanism 2)
    #diffusion from surface to grain lattice: (mechanism 3)
    #evaporation: (mechanism 4)
    #condensation: (mechanism 5)
        #and also the motion of gas (motion of pores)
    #gas through grain boundaries: (6)
    #gas through lattice (7)
    juda=0
    if back==0:
        #back=0: make the change in the trial
        #Go forward. Make the change.
        juda=1
        #storage1 and storage2: both are array(3) to store the original state.
        #First, consider the situation of evaporation or condensation. 
        if transmission==4 or transmission==5:
            #evaporation or condensation
            #storage1: store the orientation of the studied point
            storage1=ORI[coord1]
            storage2=np.array([gasvalue,gasvalue,gasvalue])
            if transmission==5:
                #condensation:
                countatm=countatm-1
                #During condensation, the newly formed solid point should have the orientation \
                #of its neighboring solid point.
                chooseloc=list()
                #The coordinate of the studied point.
                x,y,z=coord1
                #Go through all its neighboring points:
                for i in range(-radconsider,radconsider+1):
                    for j in range(-radconsider,radconsider+1):
                        for k in range(-radconsider,radconsider+1):
                            if i==0 and j==0 and k==0:
                                #avoid considering itself
                                continue
                            if x+i<0 or x+i>NT-1 or y+j<0 or y+j>NT-1 or z+k<0 or z+k>NT-1:
                                #out of the simulation box
                                continue
                            if radconsider>1:
                                if i**2+j**2+k**2>radconsider**2:
                                    #not in the range of consideration
                                    continue
                            if abs(ORI[x+i,y+j,z+k,0]-gasvalue)>1e-3:
                                #this neighboring point is not gaseous
                                chooseloc.append([x+i,y+j,z+k])
                #During condensation, choose the orientation of one neighboring solid point.
                randinteger=np.random.randint(0,len(chooseloc))
                xt,yt,zt=chooseloc[randinteger]
                ORI[coord1]=ORI[xt,yt,zt]
            if transmission==4:
                #evaporation
                countatm=countatm+1
                #Change the studied point into a gaseous one. 
                ORI[coord1]=np.array([gasvalue,gasvalue,gasvalue])
        #Consider other transmission mechanisms (not evaporation or condensation)
        else:
            #Store the orientation values of two points before making the change.
            storage1=ORI[coord1]
            storage2=ORI[coord2]
            pointkind1,surfcurv1=point(ORI,coord1)
            pointkind2,surfcurv2=point(ORI,coord2)
            if pointkind1==2 or pointkind2==2:
                phai1m,PHIm,phai2m=ORI[coord1]
                ORI[coord1][0]=ORI[coord2][0]
                ORI[coord1][1]=ORI[coord2][1]
                ORI[coord1][2]=ORI[coord2][2]
                ORI[coord2][0]=phai1m
                ORI[coord2][1]=PHIm
                ORI[coord2][2]=phai2m
            else:
                ORI[coord2][0]=ORI[coord1][0]
                ORI[coord2][1]=ORI[coord1][1]
                ORI[coord2][2]=ORI[coord1][2]
            
    else:
        #back=1: the change is rejected and the original state should be restored.
        juda=0
        if transmission==4:
            #Evaporation is rejected
            countatm=countatm-1
            ORI[coord1]=storage1
        if transmission==5:
            #Condensation is rejected
            countatm=countatm+1
            ORI[coord1]=storage1
        else:
            #Other transmission mechanism is rejected.
            #Then, the orientation values of the two points can be restored directly.
            ORI[coord1]=storage1
            ORI[coord2]=storage2
    back=juda
    #print(back)
    return ORI,coord1,coord2,transmission,storage1,storage2,countatm,back

#probability transition rule in Monte Carlo:
def transition(energybefore, energyafter, studiedcoord, choosecoord, transmission, \
               ORI, MCsteps):
    #energy difference obtained by making the change
    energydiff=energyafter-energybefore
    #print(energydiff/(kb*T0),end='    ')
    if MCsteps<beginani:
        coefficient=1.0
    else:
        if transmission!=4 and transmission!=5:
            #not evaporation or condensation
            if abs(ORI[studiedcoord][0]-gasvalue)>1e-3 and \
                abs(ORI[choosecoord][0]-gasvalue)>1e-3:
                #both points are solid.
                #motion: the proposed motion direction of grain boundary
                motion=np.array(choosecoord)-np.array(studiedcoord)
                #the orientation of the grain, np.array([phai1,PHI,phai2])
                euler=ORI[studiedcoord]
                #the motion direction in the local reference frame of the grain
                localvector=eulerinv(motion,euler)
                #print(localvector, end='     ')
                #anisotropy coefficient
                coefficient=aniso(localvector,MCsteps)/average
                #print(coefficient*average, end='     ')
                #print(average)
            else:
                coefficient=1.0
        else:
            coefficient=1.0
    #the probability of making the change
    probability=0.0
    transition=0
    if energydiff<0:
        probability=coefficient
    else:
        probability=coefficient*np.exp((-energydiff)/(kb*temperature))
    randnum=np.random.rand()
    if randnum<probability:
        transition=1
        #print(energydiff/(kb*T0))
    #print(coefficient, probability, transition)
    return transition

#one step in the Monte Carlo simulation
def change(ORI,countatm,MCsteps,coordmap,counter):
    #Default: make the change (forward: back=0; backward: back=1)
    back=0
    #the coordinate of the studied point and the chosen point. \
    #Transmission mechanism is also included.
    #The counter is also the output of "choosepoint" function. The output counter is \
    #the one pointing to the last qualifed point. If a new point is needed, counter should \
    #be counter+1 nat the end of the "change function"
    studiedcoord,choosecoord,transmission,counter,overflow \
        =choosepoint(ORI,countatm,coordmap,counter)
    if overflow==0:
        #Calculate the initial energy (Before the change is made)
        energy1=calcenergy(ORI,studiedcoord,choosecoord)
        #Make the change
        #First, initialize storage1 and storage2 for the convenience of programming.
        storage1=np.array([gasvalue,gasvalue,gasvalue])
        storage2=np.array([gasvalue,gasvalue,gasvalue])
        #print(back, end='   ')
        ORI,studiedcoord,choosecoord,transmission,storage1,storage2,countatm,back= \
            modify(ORI,studiedcoord,choosecoord,transmission,storage1,storage2,countatm,back)
        #print(back, end='   ')
        #calculate the energy after the change:
        energy2=calcenergy(ORI,studiedcoord,choosecoord)
        #decide whether the change can be made:
        yesorno=transition(energy1,energy2,studiedcoord,choosecoord,transmission,ORI,MCsteps)
        #print("yesorno={}".format(yesorno),end='   ')
        if yesorno==0:
            ORI,studiedcoord,choosecoord,transmission,storage1,storage2,countatm,back= \
                modify(ORI,studiedcoord,choosecoord,transmission,storage1,storage2,countatm,back)
        #print(back)
        counter=counter+1
    return ORI,countatm,counter,overflow

#the main loop of Monte Carlo
def main():
    #Initialize the orientation field and the amount of ceramics in the atmosphere
    ORI,countatm=initial1()
    #Initialize counter
    counter=0
    #Intialize the MCSmap (the order to go through all points in the simulation box)
    coordmap=MCSmap()
    #MCsteps: How many MCS.
    MCsteps=0
    np.save("MCresult/MCS{}.npy".format(MCsteps),ORI)
    for i in range(rounds):
        #overflow
        overflow=0
        #One change (accepted or rejected) in the simulation
        ORI,countatm,counter,overflow=change(ORI,countatm,MCsteps,coordmap,counter)
        #Monitering the process of the program
        print(counter)
        if overflow!=0:
            print("overflow")
            time.sleep(2)
        if counter>=MCS or overflow!=0:
            #One MCS has been completed
            MCsteps=MCsteps+1
            #Save the file of the orientation field
            np.save("MCresult/MCS{}.npy".format(MCsteps),ORI)
            #Initialize the MCSmap again
            coordmap=MCSmap()
            #Reset the counter
            counter=0
            #How many MCS steps have been finished
            print("\n{} MCS is finished\n".format(MCsteps))

average=anisoaverage()
main()







