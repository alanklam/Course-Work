#!/usr/bin/python

# ----------------------------------------------------------------------------------------------- #
# This Python program runs a 2-Dimensional Dislocation molecular dynamics simulation
# Served as the Phys466 Group Project Code
# ----------------------------------------------------------------------------------------------- #

import numpy as np
from numpy import *
from math import *
from scipy.interpolate import griddata
import matplotlib.pylab as pylab
import matplotlib.pyplot as pt
import random, datetime, array, time


tstart = datetime.datetime.now()


# ----------------------------------------------------------------------------------------------- #
# [Initial] Position Functions: place atoms in lattice with dislocation,
# without dislocation, or with multiple dislocations.
# ----------------------------------------------------------------------------------------------- #

## Square Lattice -- Std. Configuration

def InitPosition2D(N,L):  # N is the number of atoms
        # L is the side length of the sim. cell

  position = np.zeros((N,2)) + 0.0
  Ncube = 1
  while(N > (Ncube*Ncube)):
    Ncube += 1
  if(Ncube**2 != N):
    print("SquareInit Warning: Your particle number",N, \
          "is not a perfect square; this may result " \
          "in a lousy initialization")
  rs = float(L)/(Ncube-1)
  roffset = float(L)/2 
  added = 0
  for x in range(0, Ncube):
    for y in range(0, Ncube):
        if(added < N):
          position[added, 0] = rs*x -roffset
          position[added, 1] = rs*y -roffset
          added += 1
  return position


## Square Lattice -- Circle Configuration

def InitPositionCircle(n,a,radius): # n is the number of atoms
          # a is the lattice constant
          # radius is the circle radius

  dislocate = 1
   
  R=[]
  
  for i in range(0,n):
    for j in range(0,n):
        R.append([(0.5+i)*a,(0.5+j)*a]) 
        R.append([(-0.5-i)*a,(0.5+j)*a])     
        R.append([(0.5+i)*a,(-0.5-j)*a])
        R.append([(-0.5-i)*a,(-0.5-j)*a])
  
  added = 0
  
  position=[]
  for m in range(0,len(R)):
     if dislocate:
        x = R[m][0] 
        y = R[m][1]
        r = x**2+y**2
        u = b/(2*pi)*(arctan2(y,x) + x*y/(2*(1-p))/r)
        v = b/(2*pi)*((1-2*p)*log(1/sqrt(r))/(2*(1-p)) + y**2/(2*(1-p)*r)) 
        R[m][0] = x+u 
        R[m][1] = y+v
  
     if (R[m][0]**2 + R[m][1]**2 <=radius**2): 
        position.append([R[m][0],R[m][1]])
        added += 1
  
  N = added   
  
  print("total number atoms in the circle:", N  )

  return position


## Triangular Lattice -- Circle Configuration

def InitPositionTri(n,a,radius):  # n is the number of atoms
          # a is the lattice constant
          # radius is the circle radius

  dislocate=1
  R=[]
  for i in range(0,n):
     for j in range(0,n):       
        R.append([(0.5+i+((j+1)%2)/2)*a,(sqrt(3)/4+j)*a]) 
        R.append([(-0.5-i+((j+1)%2)/2)*a,(sqrt(3)/4+j)*a])     
        R.append([(0.5+i+(j%2)/2)*a,(-sqrt(3)/4-j)*a]) 
        R.append([(-0.5-i+(j%2)/2)*a,(-sqrt(3)/4-j)*a])
  
  added = 0
   
  position=[]
  for m in range(0,len(R)):
     if dislocate:
        x = R[m][0] 
        y = R[m][1]
        r = x**2+y**2
        u = b/(2*pi)*(arctan2(y,x) + x*y/(2*(1-p))/r)
        v = b/(2*pi)*((1-2*p)*log(1/sqrt(r))/(2*(1-p)) + y**2/(2*(1-p)*r)) 
        R[m][0] = x+u 
        R[m][1] = y+v
  
     if (R[m][0]**2 + R[m][1]**2 <= radius**2): 
        position.append([R[m][0],R[m][1]])
        added += 1

  N = added   
  
  print("total number atoms in the circle:", N  )
  
  return position

# ----------------------------------------------------------------------------------------------- #
# InitVelocity Function
# ----------------------------------------------------------------------------------------------- #

def InitVelocity(N,T0,mass):
  initNDIM = 2
  velocity = np.zeros((N,2)) + 0.0
  random.seed(1)
  netP = np.zeros((2,)) + 0.
  netE = 0.0
  
  for n in range(0, N):
     for x in range(0, initNDIM):
        newP = random.random()-0.5
        netP[x] += newP
        netE += newP*newP
        velocity[n, x] = newP
  netP *= 1.0/N
  vscale = sqrt(2*N*T0/(mass*netE))
  
  for n in range(0, N):
     for x in range(0, initNDIM):
        velocity[n, x] = (velocity[n, x] - netP[x]) * vscale
  
  return velocity

# ----------------------------------------------------------------------------------------------- #
# Boundary Condition Functions (PBC, NPBC, External)
# ----------------------------------------------------------------------------------------------- #

def PutInBox(Ri): # Ri is 3D coordinate of the ith particle
  for i in range(2):
     while Ri[i]>L/2:
        Ri[i]+=-L
     while Ri[i]<-L/2:
        Ri[i]+=+L

  return Ri

def Distance(Ri, Rj): # Ri is 3D coordinate of the ith particle
      # Rj is 2D coordinate of the jth particle
  
  global BC
  di=[0.0,0.0]
  for i in range(2):
    # PBC
     if (BC == 1):
        if Ri[i] > Rj[i]:
           di[i] = min(Rj[i] + L - Ri[i], Ri[i] - Rj[i])
        else:
           di[i] = min(Ri[i] + L - Rj[i], Rj[i] - Ri[i])
    # Non-PBC
     else:      
        di[i] = Ri[i] - Rj[i]

  d = sqrt(di[0]**2 + di[1]**2)         

  return d


def Displacement(Ri, Rj): # Ri is 3D coordinate of the ith particle
        # Rj is 3D coordinate of the jth particle

  global BC
  dx = 0.0; dy = 0.0;
  
  # PBC
  if (BC == 1):
     if (Ri[0] - Rj[0] > L/2): 
        dx = Ri[0] - Rj[0] - L
     
     elif (Ri[0] - Rj[0] < -L/2):
        dx = Ri[0] - Rj[0] + L
     
     else:
        dx = Ri[0] - Rj[0]
     
     if (Ri[1] - Rj[1] > L/2):
        dy = Ri[1] - Rj[1] - L
     
     elif (Ri[1] - Rj[1] < -L/2):
        dy = Ri[1] - Rj[1] + L
     
     else:
        dy = Ri[1] - Rj[1]       
  # Non-PBC
  else:
    dx = Ri[0] - Rj[0]
    dy = Ri[1] - Rj[1]
  return [dx, dy]


# ----------------------------------------------------------------------------------------------- #
# Internal Force // (Potential) Energy: solve Newton's EOM for a chosen
# potential function (i.e. Lennard-Jones, Double-Yukawa, etc.)
# ----------------------------------------------------------------------------------------------- #

def InternalForce(i, R):  # i is the particle number 
        # R is the position of all particles

  F = [0.0, 0.0 ]
  for k in range(2):
     for j in range(N):
        if j!=i:
           r = Distance(R[i],R[j])
           dis = Displacement(R[i],R[j])
        
  # Lennard-Jones
        F[k] += ( -4.0 * (-12.0 * dis[k] * r**(-14) * sigma**12 
      + 6.0 * dis[k] * r**(-8) * sigma**6) )

  # Double-Yukawa 

  return F

def InternalForceCut(i, R, nb):   # i is the particle number
          # R is the position of all particles
          # nb is the neighbor list
  global Fshear
  
  F = [0.0, 0.0 ]
  for k in range(2):
     for j in nb:
        r = Distance(R[i],R[j])
        dis = Displacement(R[i],R[j])
        
  #Lennard-Jones
        F[k] += ( -4.0 * (-12.0 * dis[k] * r**(-14) * sigma**12 + 6.0 * dis[k] * r**(-8) * sigma**6) )
  if R[i][1]>=0:
    F[0]+=Fshear
  else:
    F[0]+=-Fshear
    
  return F


def ComputeEnergy(R, V):  # R is the position of all particles (not yet modified)
        # V is the velocity of all particles
  
  global radius, wshell
  totalU  = 0.0
  
  for i in range(len(R)):
     if ( R[i][0]**2 + R[i][1]**2 <= (radius - wshell)**2 ):
        for j in range(i+1, len(R)):
           if ( R[j][0]**2 + R[j][1]**2 <= (radius - wshell)**2 ):
              r = Distance(R[i],R[j])
              totalU += 4.0 * ((sigma/r)**12 - (sigma/r)**6)

  return totalU


def ComputeEnergyCut(R, V, cutoff): # R is the position of all particles
          # V is the velocity of all particles

  global radius, wshell
  totalU  = 0.0

  for i in range(len(R)):
     if ( R[i][0]**2 + R[i][1]**2 <= (radius - wshell)**2 ):
        for j in range(i+1,len(R)):
           if ( R[j][0]**2 + R[j][1]**2 <= (radius - wshell)**2 ):
              r = Distance(R[i],R[j])
              totalU += ( 4.0 * ((sigma/r)**12 - (sigma/r)**6 - (sigma/cutoff)**12 + (sigma/cutoff)**6) )

  return totalU


def rigidbd(R):   # generate list of boundary atoms

  global radius, wshell
  bdlist=[]

  for i in range(len(R)):
     if ( R[i][0]**2 + R[i][1]**2 >= (radius - wshell)**2 ):
        bdlist.append(i)

  return bdlist


def insidebd(R):  # generate list of atoms for dynamics

  global radius, wshell
  uplist=[]
  
  for i in range(len(R)):
     if ( R[i][0]**2 + R[i][1]**2 < (radius-wshell)**2 ):
        uplist.append(i)

  return uplist


def nblist(i, cutoff, R): # neighbor list within cutoff

  nb=[]
  for j in range(len(R)):
     if j!=i:
        r = Distance(R[i], R[j])
        if (r <= cutoff):
           nb.append(j)

  return nb

# ----------------------------------------------------------------------------------------------- #
# Local Strain Calculation
# ----------------------------------------------------------------------------------------------- #



def Local_Strain(R, a):

  eta_s_list=[]
  
  for i in range(0,len(R)):
##      q_ij = np.matrix([0,0])
##      M_i = np.dot(q_ij.T, q_ij)
      M_i=0
      for j in range(0,len(R)):
        if ( Distance(R[i], R[j]) <= 3.5*a ):
          if j != i:
            q_ij = Displacement(R[i],R[j])
            q_ij = np.matrix(q_ij)
            M_i += np.dot(q_ij.T, q_ij)

      Tr_Mi = np.trace(M_i)
      A = M_i - np.eye(2) * Tr_Mi/3.  
      Tr_A = np.trace(A)

      eta_m = 1./6 * (Tr_Mi - 3)
      eta_s = sqrt(1./8 * Tr_A**2)
      eta_s_list.append(eta_s)

##     print R[i]
     #eta_s_list(R[i]) = eta_s

  return eta_s_list

def findcenter(data,x):
   mx=-1
   for i in range(len(x)):
      for j in range(len(x)):
         if data[i][j]>mx and math.isfinite(data[i][j]):
            mx=data[i][j]
            xc=x[i]
            yc=x[j]
   return [xc,yc]  

def Udis(pos,x,y):
  #potential for edge dislocation
  global b, p, shear
  D=shear*b**2/2/pi/(1-p)
  U=np.zeros((len(x),len(y)))
  for i in range(len(x)):
    for j in range(len(y)):
      if (x[i]-pos[0])**2+(y[j]-pos[1])**2 !=0:
        U[i][j]=D/2*( ((x[i]-pos[0])**2-(y[j]-pos[1])**2)/((x[i]-pos[0])**2+(y[j]-pos[1])**2) - log((x[i]-pos[0])**2+(y[j]-pos[1])**2) )
      
  return U

# ----------------------------------------------------------------------------------------------- #
# Verlet time-stepping algorithm
# ----------------------------------------------------------------------------------------------- #

h = 0.01 # time in 10^-13, 0.01 = 1 fs

def VerletNextR(r_t,v_t,a_t):
  r_t_plus_h = [0.0, 0.0 ]
  r_t_plus_h[0] = r_t[0] + v_t[0]*h + 0.5*a_t[0]*h*h
  r_t_plus_h[1] = r_t[1] + v_t[1]*h + 0.5*a_t[1]*h*h

  return r_t_plus_h

def VerletNextV(v_t,a_t,a_t_plus_h):
  v_t_plus_h = [0.0, 0.0]  
  v_t_plus_h[0]=v_t[0]+0.5*(a_t[0]+a_t_plus_h[0])*h
  v_t_plus_h[1]=v_t[1]+0.5*(a_t[1]+a_t_plus_h[1])*h

  return v_t_plus_h


# ----------------------------------------------------------------------------------------------- #
# Main Loop // User Input Section
# ----------------------------------------------------------------------------------------------- #



def main():

  # R, V, and A are the position, velocity, and acceleration of the atoms
  # respectively. nR, nV, and nA are the _next_ positions, velocities, etc.
  global BC
  BC = 0
  global a
  a = 0.361*0.72  #0.5256 Ar, 0.361 Cu ,0.3147 Mo
  global radius
  radius = 15*a
  global wshell
  wshell = 1.5*a
  global sigma
  sigma = 0.227 #0.341 Ar, 0.227 Cu , 0.42 Mo
  global b
  b = a
  global p
  p = 0.34  #0.31 Mo, 0.34 Cu
  global shear  
  shear=48   #shear modulous in GPa 20 Mo, 48 Cu
  global Fshear   #apply uniform shear force in pN
  
  T0 = 0.1
  M = 63.5  # 63.5 Cu , 95.94 Mo, 39.948 Ar
  steps = 1
  outstep = 2
  cutoff = 1.5
  nbtime = 10   # renew the neighbor list every nbtime steps
  
  Latt_Config = 3

  if (Latt_Config == 1):
     R = InitPosition2D(int((radius/a)**2),radius)

  if (Latt_Config == 2):
     R = InitPositionCircle(int(radius/a)+1,a,radius)

  if (Latt_Config == 3):
     R = InitPositionTri(int(radius/a)+1,a,radius)
  
  L = radius
  N = len(R)
  
  bdlist = rigidbd(R)
  uplist = insidebd(R)
  
  
  nb=[]
  for i in range(N):
    nb.append([])
  
  V = np.zeros((N,2)) + 0.1
  A = np.zeros((N,2))

  nR = np.zeros((N,2))
  nV = np.zeros((N,2))
  V = InitVelocity(N, T0, M)
  nR = R.copy()

  count = 0
  
  Energy = np.zeros((steps,))
  DisPos=np.zeros((int(steps/outstep)+1,2))
  countp=0
  name="Cu/Dis_movement_tri_shear"+str(Fshear)+".txt"
  outFile =open(name,"w")
  # Start simulation with fixed boundary
  BC = 0
  for t in range(0,steps):
    Energy[t] = ComputeEnergyCut(R,V,cutoff)
    
    if ( (t+1)%outstep == 0 or t == 0 ):
        #plot the position and local strain of atom every 'outstep' steps
        eta_s_list = Local_Strain(R, a)
        x=np.linspace(-radius+1.5*a,radius-1.5*a,8*int(radius/a))
        y=np.linspace(-radius+1.5*a,radius-1.5*a,8*int(radius/a))
        grid_x, grid_y= np.meshgrid(x,y)
        data1=griddata(R ,eta_s_list,(grid_x,grid_y),method='cubic')
        xc,yc=findcenter(data1.T,x)
        DisPos[countp][0]=xc
        DisPos[countp][1]=yc
        countp+=1
        outFile.write(str(xc)+" "+str(yc)+"\n")
        if ((t+1)%(outstep*200) == 0 or t == 0):
          count += 1
          pylab.figure(count)
         
          for i in uplist:
              pylab.plot(R[i][0],R[i][1],'kx')
      
          for i in bdlist:
              pylab.plot(R[i][0],R[i][1],'r+')
              pylab.xlabel("X")
              pylab.ylabel("Y")

          pt.contourf(grid_x,grid_y,data1,20)
          pt.colorbar()
          pt.scatter(xc,yc)
          pt.title("Shear="+str(Fshear)+"t="+str(t))

##          count+=1
##          pylab.figure(count)
##          for i in uplist:
##              pylab.plot(R[i][0],R[i][1],'kx')
##      
##          for i in bdlist:
##              pylab.plot(R[i][0],R[i][1],'r+')
##              pylab.xlabel("X")
##              pylab.ylabel("Y")         
##          U=Udis([xc,yc],x,y).T
##          pt.contourf(grid_x,grid_y,U,15)
##          pt.colorbar()

    for i in uplist:  
     # rigid boundary, move only the atom inside
        if (t%nbtime == 0):
           nb[i] = nblist(i,cutoff,R)
          
        F = InternalForceCut(i, R, nb[i])
        A[i] = [ F[0]/M, F[1]/M ]  
        nR[i] = VerletNextR( R[i], V[i], A[i] )

        
    for i in uplist:
        nF = InternalForceCut(i, nR ,nb[i])
        nA = [ nF[0]/M, nF[1]/M ]
        nV[i] = VerletNextV( V[i], A[i], nA )

        R = nR.copy()
        V = nV.copy()

    #Andersen thermostat for each time step
    for i in uplist:
      if random.uniform(0,1) <= 0.01:
        V[i][0]=random.gauss(0,math.sqrt(T0/M))
        V[i][1]=random.gauss(0,math.sqrt(T0/M))
     
  # Plot total energy
  outFile.close()
  pylab.figure(count+1)
##  pylab.plot(np.linspace(0,steps,steps),Energy)
  
  dis2=np.square(diff(DisPos[:,0]))+np.square(diff(DisPos[:,1]))
  D=sum(dis2)/len(dis2)/400
  print('Diffusivity =',D,'cm^2/s F=',Fshear)
  
  tend = datetime.datetime.now()
  print('The total simulation time was:',(tend - tstart),'sec.')
##  pylab.show()
  pt.show()



for Fshear in linspace(0,0,1):
  main()


# ----------------------------------------------------------------------------------------------- #
# Plotting and Output Section
# ----------------------------------------------------------------------------------------------- #




