#!/usr/bin/env python

# This Python program performs some very basic statistical calculations to calculate
# mean, standard deviation, autocorrelation time and standard error of the input data set

import sys, os
import ReadData 
import numpy
from math import *


if __name__ == '__main__':
  if len(sys.argv) > 1:
    readFilename = os.path.realpath(sys.argv[1])
  else:
    # define input file below
    readFilename = "AtomicScale_HW1_data4.txt"

  myArray=ReadData.loadAscii(readFilename)
#  print("I have read in myArray and the first element is",myArray[0])

def mean(g):
  total=0.0
  num=0
  for i in range(0,len(g)):
    total+=g[i]
    num+=1
  return total/float(num)

def correlation(g,t,m,sd):
    total=0.0;
    num=0;
    for i in range(0,len(g)-t):
        total+=(g[i]-m)*(g[t+i]-m)
        num+=1
    return total/float(num)/sd**2

def stats(g):
  m=mean(g)
  #using numpy.square for simplicity
  st_dev=numpy.sqrt(mean(numpy.square(g))-numpy.square(m))
  i=0
  cutoff=0
  #check for t_cutoff
  while (cutoff==0 and i<=len(g)):
      c=correlation(g,i,m,st_dev)
      if c<=0:
          cutoff=i
      i+=1
  total=0.0
  #summing up terms, range function take care of the -1 in upper limit
  for i in range(1,cutoff):
      total+=correlation(g,i,m,st_dev)
      
  return (m,st_dev,1+2*total,numpy.sqrt((1+2*total)/len(g))*st_dev)

print("The mean, standard deviation, autocorrelation time and standard error are:",stats(myArray))
  
