# This Python program runs simulations for a 2D Ising model using Monte-Carlo approach
#

import numpy as np , random , math as m

spinsPerSide=20
spins=np.ones((spinsPerSide,spinsPerSide))    # fills new array with 1
# spin coupling strength
J=0.44069 #0.3 0.44069 0.8

def FlipSpin():
  global spins
  x=random.randint(0,spinsPerSide-1)
  y=random.randint(0,spinsPerSide-1)
  nbx=[x,x,(x-1)%spinsPerSide,(x+1)%spinsPerSide]
  nby=[(y+1)%spinsPerSide,(y-1)%spinsPerSide,y,y]
  dV=0;
  for i in range(4):
      dV+=spins[nbx[i]][nby[i]]
  dV*=-2*J*spins[x][y]
  p=min(1,m.exp(dV))
  if p>random.random():
      dM=-2*spins[x][y]/(spinsPerSide**2)
      spins[x][y]*=-1
  else:
      dM=0
  return dM

def Heatbath():
  global spins
  x=random.randint(0,spinsPerSide-1)
  y=random.randint(0,spinsPerSide-1)
  nbx=[x,x,(x-1)%spinsPerSide,(x+1)%spinsPerSide]
  nby=[(y+1)%spinsPerSide,(y-1)%spinsPerSide,y,y]
  sumj=0;
  for i in range(4):
      sumj+=spins[nbx[i]][nby[i]]
  pup=m.exp(J*sumj)
  pdown=m.exp(-J*sumj)
  if pup/(pup+pdown)>random.random():
      dM=(1-spins[x][y])/(spinsPerSide**2)
      spins[x][y]=1
  else:
      dM=(-1-spins[x][y])/(spinsPerSide**2)
      spins[x][y]=-1
  return dM

def BuildCluster(): 
  x=random.randint(0,spinsPerSide-1)
  y=random.randint(0,spinsPerSide-1)
  checkx=[x,x,(x-1)%spinsPerSide,(x+1)%spinsPerSide]
  checky=[(y+1)%spinsPerSide,(y-1)%spinsPerSide,y,y]
  cluster=[]
  cluster.append([x,y])
  i=0
  while i<len(checkx):
      if (not ([checkx[i],checky[i]] in cluster)) and spins[checkx[i]][checky[i]]==spins[x][y]:
        if random.random()<(1-m.exp(-2*J)):
          cluster.append([checkx[i],checky[i]])
          checkx.append(checkx[i])
          checkx.append(checkx[i])
          checkx.append((checkx[i]-1)%spinsPerSide)
          checkx.append((checkx[i]+1)%spinsPerSide)
          checky.append((checky[i]+1)%spinsPerSide)
          checky.append((checky[i]-1)%spinsPerSide)
          checky.append(checky[i])
          checky.append(checky[i])          
      i+=1    
  return  cluster

def FlipSpinsCluster(cluster):
      global spins
      for i in cluster:
        spins[i[0]][i[1]]*=-1
      dM=2*len(cluster)*spins[cluster[0][0]][cluster[0][1]]/(spinsPerSide**2)
      return dM

sweep=1000
##Mt=np.zeros(sweep*spinsPerSide**2+1)
##M=0
##for trial in range(1):
##    for i in range(spinsPerSide):
##        for j in range(spinsPerSide):
##            M+=spins[i][j]
##    M*=1/(spinsPerSide**2)
##    Mt[0]=M.copy()
##    for t in range(sweep*spinsPerSide**2):
##        M+=Heatbath()
##        Mt[t+1]=M
##
##    for i in range(len(Mt)):
##        Mt[i]=Mt[i]**2
##        print(Mt[i])
        
##    print(Mt[sweep*spinsPerSide**2])    
   
Mt=np.zeros(sweep+1)
M=0
for trial in range(10):
    for i in range(spinsPerSide):
        for j in range(spinsPerSide):
            M+=spins[i][j]
    M*=1/(spinsPerSide**2)
    Mt[0]=M.copy()
    for t in range(sweep):
        M+=FlipSpinsCluster(BuildCluster())
        Mt[t+1]=M

    for i in range(len(Mt)):
        Mt[i]=Mt[i]**2
##        print(Mt[i])

    print(Mt[sweep]) 
