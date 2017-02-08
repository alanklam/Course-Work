#!/usr/bin/env python3

# ------------------------------------------------------------------------
# This python program run a molecular dynamics simulations using Lennard-Jones particles.
# NVT ensemble is assumed. Integrations are done using Verlet scheme.
# ------------------------------------------------------------------------


import numpy, math, random, cmath , pylab


# You may adjust the gas properties here.
# The main routine is at the bottom.
# ------------------------------------------------------------------------

# mass
M = 48.0

# number of Particles
N = 125

# box side length
L = 4.2323167

# scaled temperature
T0 = 2.0


# initial configuration
# ------------------------------------------------------------------------
def InitPositionCubic(N,L):
  position = numpy.zeros((N,3)) + 0.0
  Ncube = 1
  while(N > (Ncube*Ncube*Ncube)):
    Ncube += 1
  if(Ncube**3 != N):
    print("CubicInit Warning: Your particle number",N, \
          "is not a perfect cube; this may result " \
          "in a lousy initialization")
  rs = float(L)/Ncube
  roffset = float(L)/2 - rs/2
  added = 0
  for x in range(0, Ncube):
    for y in range(0, Ncube):
      for z in range(0, Ncube):
        if(added < N):
          position[added, 0] = rs*x - roffset 
          position[added, 1] = rs*y - roffset 
          position[added, 2] = rs*z - roffset 
          added += 1
  return position

def InitVelocity(N,T0,mass):
  initNDIM = 3
  velocity = numpy.zeros((N,3)) + 0.0
  random.seed(1)
  netP = numpy.zeros((3,)) + 0.
  netE = 0.0
  for n in range(0, N):
    for x in range(0, initNDIM):
      newP = random.random()-0.5
      netP[x] += newP
      netE += newP*newP
      velocity[n, x] = newP
  netP *= 1.0/N
  vscale = math.sqrt(3*N*T0/(mass*netE))
  for n in range(0, N):
    for x in range(0, initNDIM):
      velocity[n, x] = (velocity[n, x] - netP[x]) * vscale
  return velocity


# Routines to ensure periodic boundary conditions
# ------------------------------------------------------------------------
def PutInBox(Ri):
  for i in range(3):
      while Ri[i]>L/2:
          Ri[i]=Ri[i]-L
      while Ri[i]<-L/2:
          Ri[i]=Ri[i]+L
  return 

def Distance(Ri,Rj):
  di=[0.0,0.0,0.0]
  for i in range(3):
      if Ri[i]>Rj[i]:
          di[i]=min(Rj[i]+L-Ri[i],Ri[i]-Rj[i])
      else:
          di[i]=min(Ri[i]+L-Rj[i],Rj[i]-Ri[i])
  d=math.sqrt(di[0]**2+di[1]**2+di[2]**2)         
  return d

def Displacement(Ri,Rj): 
  dx = 0.0; dy = 0.0; dz = 0.0
  if Ri[0]-Rj[0]>L/2:
      dx=Ri[0]-Rj[0]-L
  elif Ri[0]-Rj[0]<-L/2:
      dx=Ri[0]-Rj[0]+L
  else:
      dx=Ri[0]-Rj[0]
  if Ri[1]-Rj[1]>L/2:
      dy=Ri[1]-Rj[1]-L
  elif Ri[1]-Rj[1]<-L/2:
      dy=Ri[1]-Rj[1]+L
  else:
      dy=Ri[1]-Rj[1]
  if Ri[2]-Rj[2]>L/2:
      dz=Ri[2]-Rj[2]-L
  elif Ri[2]-Rj[2]<-L/2:
      dz=Ri[2]-Rj[2]+L
  else:
      dz=Ri[2]-Rj[2]
  return [dx,dy,dz]

# The Verlet time-stepping algorithm, 'h' is the time step.
# ------------------------------------------------------------------------
h=0.01
def VerletNextR(r_t,v_t,a_t):
  r_t_plus_h = [0.0, 0.0, 0.0]
  r_t_plus_h[0] = r_t[0] + v_t[0]*h + 0.5*a_t[0]*h*h
  r_t_plus_h[1] = r_t[1] + v_t[1]*h + 0.5*a_t[1]*h*h
  r_t_plus_h[2] = r_t[2] + v_t[2]*h + 0.5*a_t[2]*h*h
  return r_t_plus_h

def VerletNextV(v_t,a_t,a_t_plus_h):
  v_t_plus_h = [0.0, 0.0, 0.0]  
  v_t_plus_h[0]=v_t[0]+0.5*(a_t[0]+a_t_plus_h[0])*h
  v_t_plus_h[1]=v_t[1]+0.5*(a_t[1]+a_t_plus_h[1])*h
  v_t_plus_h[2]=v_t[2]+0.5*(a_t[2]+a_t_plus_h[2])*h
  return v_t_plus_h

def TaylorNextV(v_t,a_t,a_t_plus_h):
  
  v_t_plus_h = [0.0, 0.0, 0.0]
  
  v_t_plus_h[0]=v_t[0]+a_t[0]*h
  
  v_t_plus_h[1]=v_t[1]+a_t[1]*h
  
  v_t_plus_h[2]=v_t[2]+a_t[2]*h
  
  return v_t_plus_h

def V_t(r_t_min_h,r_t_plus_h):
  v_t = [0.0, 0.0, 0.0]  
  v_t[0]=(r_t_plus_h[0]-r_t_min_h[0])/2/h
  v_t[1]=(r_t_plus_h[1]-r_t_min_h[1])/2/h
  v_t[2]=(r_t_plus_h[2]-r_t_min_h[2])/2/h
  return v_t

# Lennard-Jones forces
# ------------------------------------------------------------------------
def InternalForce(i, R):
  F = [0.0, 0.0, 0.0]
  for k in range(3):
    for j in range(N):
      if j!=i:
        r=Distance(R[i],R[j])
        dis=Displacement(R[i],R[j])
        F[k]+=-4.0*(-12.0*dis[k]*r**(-14)+6.0*dis[k]*r**(-8))    
  return F


# Some instantaneous properties of the system
# ------------------------------------------------------------------------

def ComputeEnergy(R, V):

  totalKE = 0.0
  totalU  = 0.0
  for i in range(N):
    totalKE+=M*(V[i][0]**2+V[i][1]**2+V[i][2]**2)/2
    for j in range(i+1,N):
      r=Distance(R[i],R[j])
      totalU+=4.0*(1.0/(r**12)-1.0/(r**6))
 
  totalE = totalKE + totalU
  return totalU, totalKE, totalE

def ComputeTemp(V):
  totalKE = 0.0
  for i in range(N):
    totalKE+=M*(V[i][0]**2+V[i][1]**2+V[i][2]**2)/2
    
  return totalKE*2/3/N , totalKE

def ComputeMomentum(V):
  px=0.0
  py=0.0
  pz=0.0
  for i in range(N):
    px+=M*V[i][0]
    py+=M*V[i][1]
    pz+=M*V[i][2]
  return px , py , pz , math.sqrt(px**2+py**2+pz**2)

def Computeg(R,dr):
  g=numpy.zeros((math.ceil(math.sqrt(3)/2*L/dr)))
  for i in range(N):
    for j in range(i+1,N):
        l=math.floor(Distance(R[i],R[j])/dr)
        g[l]+=1
  for i in range(len(g)):
    g[i]=g[i]*(L**3)/2/3.1415/(N**2)/((i+1)**2)/(dr**3)
  return g

def LegalKVecs(maxK):
  kList=[]
  for i in range(maxK+1):
    for j in range(maxK+1):
      for k in range(maxK+1):
        kList.append([(i)*2*3.14159/L,(j)*2*3.14159/L,(k)*3.14159*2/L])
  return kList

def rhoK(k):
  rhok=0.0
  for i in range(N):
    rhok+=cmath.exp(-(k[0]*R[i][0]+k[1]*R[i][1]+k[2]*R[i][2])*1j)
  return rhok

def Sk(kList):
   skList=[]
   for k in kList:
     skList.append(abs(rhoK(k)*rhoK(numpy.multiply(k,-1))/N))
   return skList

def VV(V0, Vt):
  vv=0.0
  for i in range(N):
    vv+=V0[i][0]*Vt[i][0]+V0[i][1]*Vt[i][1]+V0[i][2]*Vt[i][2]
  return vv/N


# Main Loop.  
# ------------------------------------------------------------------------

# R, V, and A are the position, velocity, and acceleration of the atoms
# respectively. nR, nV, and nA are the _next_ positions, velocities, etc.
# You can adjust the total number of timesteps here. 

if __name__ == '__main__':
  R = numpy.zeros((N,3)) + 0.0
  V = numpy.zeros((N,3)) + 0.1
  A = numpy.zeros((N,3))

  nR = numpy.zeros((N,3))
  nV = numpy.zeros((N,3))
  
  R = InitPositionCubic(N, L)
  V = InitVelocity(N, T0, M)
  for i in range(N):
    PutInBox(R[i])
  steps = 100
  dr=0.01
  gr=0.0
  Nstore=steps-400
  vstore=numpy.zeros((Nstore,N,3))
  Energy=numpy.zeros(steps)
  vcount=0
  kList=LegalKVecs(5)
  skList=numpy.zeros((len(kList),))
  for t in range(0,steps):
    Energy[t]=ComputeEnergy(R,V)[0]
    if t%(steps-1)==0:
      pylab.figure()
      for i in range(N):
        pylab.plot(R[i][0],R[i][1],'x')
      pylab.figure()
      for i in range(N):
        pylab.plot(R[i][0],R[i][2],'x')
    
    for i in range(0,len(R)):
      F    = InternalForce(i, R)
      A[i] = [ F[0]/M, F[1]/M, F[2]/M ]
      
      nR[i] = VerletNextR( R[i], V[i], A[i] )
      PutInBox( nR[i] )
            
    for i in range(0,len(R)):
      nF = InternalForce(i, nR)
      nA = [ nF[0]/M, nF[1]/M, nF[2]/M ]
      
      nV[i] = VerletNextV( V[i], A[i], nA )
      
    R = nR.copy()
    V = nV.copy()
# Andersen thermostat for each time step
    for i in range(N):
      if random.uniform(0,1)<=0.01:
        V[i][0]=random.gauss(0,math.sqrt(T0/M))
        V[i][1]=random.gauss(0,math.sqrt(T0/M))
        V[i][2]=random.gauss(0,math.sqrt(T0/M))
    
      
# time averaging for g(r) and S(k) after t=400, store velocity    
    if t>399:   
       skList+=Sk(kList)
       gr+=Computeg(R,dr)
       vstore[vcount]=V
       vcount+=1

#calculate <vv> and diffusion constant
  D=0.0
  vvt=numpy.zeros((Nstore,))
  c=VV(vstore[0],vstore[0])
  for i in range(Nstore):
    vvt[i]=VV(vstore[0],vstore[i])/c
    D+=vvt[i]
  print(D)

#plotting results of S(k) and g(r) or v-v correlation
  pylab.xlabel("time step")
  pylab.ylabel("V-V correlation")
  pylab.plot(numpy.linspace(0,Nstore-1,num=Nstore),vvt,'+')
  pylab.show()
    
  for i in range(len(skList)):
    skList[i]=skList[i]/Nstore
  for i in range(len(gr)):
    gr[i]=gr[i]/Nstore
  kMagList=[]
  for k in kList:
    kMagList.append(math.sqrt(k[0]**2+k[1]**2+k[2]**2))
  kMagList2=[]
  skList2=[]
  count=0
  for i in range(len(kMagList)):
    if kMagList2.count(kMagList[i])==0:
      n=kMagList.count(kMagList[i])
      kMagList2.append(kMagList[i])
      skList2.append(skList[i])
      for j in range(i+1,len(kMagList)):
        if kMagList[j]==kMagList[i]:
          skList2[count]+=skList[j]
      skList2[count]*=1/n
      count+=1
  pylab.figure(1)
  pylab.xlabel("r")
  pylab.ylabel("g(r)")
  r=numpy.multiply(numpy.linspace(1,len(gr),num=len(gr)),dr)
  pylab.plot(r,gr,'+')
  pylab.figure(2)
  pylab.xlabel("k magnitude")
  pylab.ylabel("Structure Factor")
  pylab.plot(kMagList2,skList2,'+')

  pylab.plot(numpy.linspace(1,len(Energy),num=len(Energy)),Energy)     
  pylab.show()


