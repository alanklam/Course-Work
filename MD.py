#!/usr/bin/env python3

# ------------------------------------------------------------------------
# PHY466/MSE485 Atomic Scale Simulations
# Homework 2: Introduction to Molecular Dynamics
# ------------------------------------------------------------------------


import numpy, math, random, cmath , pylab


# You may adjust the gas properties here.
# The main routine is at the bottom.
# ------------------------------------------------------------------------

# mass
M = 48.0

# number of Particles
N = 64

# box side length
L = 4.0

# temperature
T0 = 0.5

# Everyone will start their gas in the same initial configuration.
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

# Routines to ensure periodic boundary conditions that YOU must write.
# ------------------------------------------------------------------------
def PutInBox(Ri):
  global L
  for i in range(3):
      while Ri[i]>L/2:
          Ri[i]=Ri[i]-L
      while Ri[i]<-L/2:
          Ri[i]=Ri[i]+L
  return Ri

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

def ComputeEnergy(R):
  totalU  = 0.0
  for i in range(N):
    for j in range(i+1,N):
      r=Distance(R[i],R[j])
      totalU+=4.0*(1.0/(r**12)-1.0/(r**6))
 
  return totalU

def Computeg(R,dr):
  g=numpy.zeros((math.ceil(math.sqrt(3)*L/2/dr)))
  for i in range(N):
    for j in range(i+1,N):
        l=math.floor(Distance(R[i],R[j])/dr)
        g[l]+=1
  for i in range(len(g)):
    g[i]=g[i]*(L**3)/2/3.1415/(N**2)/(i+1)**2/(dr**3)
  return g

def LegalKVecs(maxK):
  global L
  kList=[]
  for i in range(maxK+2):
    for j in range(maxK+2):
      for k in range(maxK+2):
        kList.append([(i-1)*2*3.14159/L,(j-1)*2*3.14159/L,(k-1)*3.14159*2/L])
  return kList

def rhoK(k,R):
  global N
  rhok=0.0
  for i in range(N):
    rhok+=cmath.exp(-(k[0]*R[i][0]+k[1]*R[i][1]+k[2]*R[i][2])*1j)
  return rhok

def Sk(kList,R):
   skList=[]
   for k in kList:
     skList.append(abs(rhoK(k,R)*rhoK(numpy.multiply(k,-1),R)/N))
   return skList
  
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

# Some instantaneous properties of the system. YOU must write this.
# ------------------------------------------------------------------------
def CalcPotential(position,i):
  myPotential = 0.0
  for j in range(len(position)):
    if j!=i:
         r=Distance(position[j],position[i])
#         myPotential+=4.0*(1.0/(r**12)-1.0/(r**6))
         myPotential+=r/0.8
  return myPotential

def UpdatePosition(position,i):
  global sigma
  for j in range(3):
     position[i][j]+=random.gauss(0,sigma)
  position[i]=PutInBox(position[i])  #put particle in box
  return position  #return the NEW position

#Main Loop
if __name__ == '__main__':
  passes=200
  eqt=50 
  sigma=0.07
  rejected=0
  gr=0.0
  dr=0.01
  kList=LegalKVecs(6)
  skList=numpy.zeros((len(kList),))
  Positions= InitPositionCubic(N,L)
  for i in range(N):
    Positions[i]= PutInBox(Positions[i])
  Energy=numpy.zeros(int(passes*N))
  count=0
  for i in range(passes):
    for update in range(N):
      Old_positions=Positions.copy() #store old position
      Positions=UpdatePosition(Positions,update)
      deltaE=CalcPotential(Positions,update)-CalcPotential(Old_positions,update)
      A=min(1,math.exp(-deltaE/T0))
      if A<random.uniform(0.0,1.0):
        Positions=Old_positions.copy()
        rejected+=1
      if passes>eqt-1:  
        Energy[count]=ComputeEnergy(Positions)
##        gr+=Computeg(Positions,dr)
##        skList+=Sk(kList,Positions)
        #print((i+1)*update,Energy[count])
        count+=1
    if passes>eqt-1: #calculate g and S only after equilibration
        gr+=Computeg(Positions,dr)
        skList+=Sk(kList,Positions)

#plotting energy, g and S
  for i in range(len(gr)):
    gr[i]*=1/(passes-eqt)
  for i in range(len(skList)):
    skList[i]*=1/(passes-eqt)
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
  r=numpy.multiply(numpy.linspace(1,math.ceil(L/2/dr),num=math.ceil(L/2/dr)),dr)
##  r=numpy.multiply(numpy.linspace(1,len(gr),num=len(gr)),dr)
  pylab.plot(r,gr[0:len(r)])
  pylab.figure(2)  
  pylab.xlabel("MC step")
  pylab.ylabel("Potential Energy")
  print("Average E=",numpy.mean(Energy))
  print("Standard Error in E=",numpy.std(Energy))
  r=numpy.linspace(1,len(Energy),num=len(Energy))
  pylab.plot(r,Energy)
  pylab.figure(3)  
  pylab.xlabel("k magnitude")
  pylab.ylabel("Structure Factor")
  pylab.plot(kMagList2,skList2,'x')
  pylab.show()
  print("Accepted ratio=",1-rejected/passes/N)

  outFile =open("testxy.txt","w")
  for i in range(0,N):
    outFile.write(str(Positions[i][0])+" "+str(Positions[i][1])+" "+str(Positions[i][2])+"\n")
  outFile.close()
  

