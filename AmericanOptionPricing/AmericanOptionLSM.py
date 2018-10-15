
#import necessary packages
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import curve_fit
from scipy import special

#definition of help functions
def LaguPoly(x, a0, a1, a2, a3):
    #defintion of weighted Laguerre polynomial up to 3rd order
    #return   a1*np.exp(-x/2) + a2*np.exp(-x/2)*(1-x) + a3*np.exp(-x/2)*(1-2*x+x**2/2)
    return  a0 + a1*(1-x) + a2*(1-2*x+x**2/2) + a3*(1-3*x+3*x**2/2-x**3/6)

def quadra(x, a0, a1, a2):
    #definition of quadratic function
    return a0 + a1*x + a2*x**2

def AmerPut(St,K,r,Ne,T):
    #evaluation of American Put option with Least Square Monte-Carlo (LSM) method
    #take St:stock price, K, r, Ne:number of execrises per year, T:expiration time as inputs
    #output put option price and average early exercise time
    
    #payoff matrix (Ns x Ne*T) stores the (estimated) payoff at each stop for each path
    payoff = K-St
    #initialize payoff as max(K-St,0)
    payoff = np.maximum(payoff,np.zeros(np.shape(St)))
    #cont matrix (Ns x Ne*T) stores in which step each path will stop
    cont = np.ones(np.shape(St)) 
    cont = cont.astype(int)
    
    #loop backward in time to update payoff at each step
    for i in range(int(Ne*T)-2,-1,-1):
        #ITM stores raw indices for in-the-money paths
        ITM = (np.nonzero(payoff[:,i])[0])
        #take St(i) as X, and corresponding payoff at exercising step (h) as Y, 
        #perform least square fit with basis function: y=f(x)
        if i==int(Ne*T)-2:
            h = cont[ITM,i+1]
        else:
            h = np.sum(cont[ITM,i+1:],axis=1)
        X = St[ITM,i]
        Y = payoff[ITM,i+h]*np.exp(-h*r/Ne)

        #throw a warning message if number of in-the-money paths is less than 4
        try:
            popt, pcov = curve_fit(LaguPoly, X, Y)
        except:
            print('Too few data points at time step '+str(i)+'! Estimation of the put price may not be accurate.')
            break
        #checking the plot of curve fitting     
        #if i==48:   
        #    plt.scatter(X,Y,color="red")
        #    plt.scatter(X,LaguPoly(X, *popt))
        #    plt.show()
        
        #calculate expectation payoff (exppay) at current step and compare with payoff if exercise contract now
        exppay = LaguPoly(X, *popt)
        stopnow = (payoff[ITM,i]>exppay)*1
        #cont[ITM,i] = 1-stopnow
        #if decided to stop now, update the cont matrix for step>i to zero
        cont[ITM,i+1:] = cont[ITM,i+1:]*(1-stopnow).reshape(np.size(stopnow),1)
        #update payoff matrix, nonzero only if there is cashflow at current step
        payoff[ITM,i] = np.maximum(payoff[ITM,i],exppay)*stopnow

    #extract exercise time for each path    
    stoptime = np.sum(cont,axis=1)
    #price the option by discounting payoff to present time and avarage over all sample paths
    price = np.sum(np.exp(-(stoptime)*r/Ne)*payoff[range(np.size(stoptime)),stoptime-1])/np.size(stoptime)

    return price , np.mean(stoptime/Ne)

def AmerCall(St,K,r,Ne,T):
    #evaluation of American Put option with Least Square Monte-Carlo (LSM) method
    #take St:stock price, K, r, Ne:number of execrises per year, T:expiration time as inputs
    #output call option price and average early exercise time
    
    #payoff matrix (Ns x Ne*T) stores the (estimated) payoff at each stop for each path
    payoff = St-K
    #initialize payoff as max(St-K,0)
    payoff = np.maximum(payoff,np.zeros(np.shape(St)))
    
    #cont matrix (Ns x Ne*T) stores in which step each path will stop
    cont = np.ones(np.shape(St)) 
    cont = cont.astype(int)
    
    #loop backward in time to update payoff at each step
    for i in range(int(Ne*T)-2,-1,-1):
        #ITM stores raw indices for in-the-money paths
        ITM = (np.nonzero(payoff[:,i])[0])
        #take St(i) as X, and corresponding payoff at exercising step (h) as Y, 
        #perform least square fit with basis function: y=f(x)
        if i==int(Ne*T)-2:
            h = cont[ITM,i+1]
        else:
            h = np.sum(cont[ITM,i+1:],axis=1)
        X = St[ITM,i]
        Y = payoff[ITM,i+h]*np.exp(-h*r/Ne)
        
        #throw a warning message if number of in-the-money paths is less than 4
        try:
            popt, pcov = curve_fit(LaguPoly, X, Y)
        except:
            print('Too few data points at time step '+str(i)+'! Estimation of the call price may not be accurate.')
            break
        #checking the plot of curve fitting     
        #if i==5:   
        #    plt.scatter(X,Y,color="red")
        #    plt.scatter(X,LaguPoly(X, *popt))
        #    plt.show()
        
        #calculate expectation payoff (exppay) at current step and compare with payoff if exercise contract now
        exppay = LaguPoly(X, *popt)
        stopnow = (payoff[ITM,i]>exppay)*1
        #cont[ITM,i] = 1-stopnow
        #if decided to stop now, update the cont matrix for step>i to zero
        cont[ITM,i+1:] = cont[ITM,i+1:]*(1-stopnow).reshape(np.size(stopnow),1)
        #update payoff matrix, nonzero only if there is cashflow at current step
        payoff[ITM,i] = np.maximum(payoff[ITM,i],exppay)*stopnow

    #extract exercise time for each path    
    stoptime = np.sum(cont,axis=1)
    #price the option by discounting payoff to present time and avarage over all sample paths
    price = np.sum(np.exp(-(stoptime)*r/Ne)*payoff[range(np.size(stoptime)),stoptime-1])/np.size(stoptime)
    
    return price , np.mean(stoptime/Ne)

def Phi(d):
    return 0.5 + special.erf(d/np.sqrt(2))/2

def EuroPut(S0,K,r,delta,sigma,T):
    d1=(np.log(S0/K)+(r-delta+sigma**2/2)*T)/sigma/np.sqrt(T)
    d2=d1-sigma*np.sqrt(T)

    return K*np.exp(-r*T)*Phi(-d2)-S0*np.exp(-delta*T)*Phi(-d1)

def EuroCall(S0,K,r,delta,sigma,T):
    d1=(np.log(S0/K)+(r-delta+sigma**2/2)*T)/sigma/np.sqrt(T)
    d2=d1-sigma*np.sqrt(T)

    return S0*np.exp(-delta*T)*Phi(d1)-K*np.exp(-r*T)*Phi(d2)


#comparison with simulation data in the paper (Longstaff and Schwartz 2001)
S= 44 #initial stock price
Ne = 50 #Number exercises in a year
T = 1 #expiration time in year
sigma = 0.2 #Annual volatility
K = 40 #strike price
r = 0.06 #risk free interest rate
Ns = 100000 #number of sample paths, half of which will be generated antithetically
delta = 0.1 #continuously compounded dividend yield

print("S sigma T AmerPut EuroPut Diff PStopTime AmerCall EuroCall Diff CStopTime")  
print("----------------------")
for S in range(36,45,4):
    for sigma in [0.2,0.4]:
        for T in [1,2]:
            u = np.random.normal(0,1,(int(Ns/2),int(Ne*T)))
            GBM=np.exp(u/np.sqrt(Ne)*sigma+(r-delta-sigma**2/2)/Ne)
            GBM = np.append(GBM,np.exp(-u/np.sqrt(Ne)*sigma+(r-delta-sigma**2/2)/Ne),axis=0)
            St=S*np.cumprod(GBM,1)
            [P,Pstop] = AmerPut(St,K,r,Ne,T)
            [C,Cstop] = AmerCall(St,K,r,Ne,T)
            Pe = EuroPut(S,K,r,delta,sigma,T)
            Ce = EuroCall(S,K,r,delta,sigma,T)
            print(str(S),' ',str(sigma),' ',str(T),' ',str(P),' ',str(Pe),' ',str(P-Pe),' ',str(Pstop),' ',                   str(C),' ',str(Ce),' ',str(C-Ce),' ',str(Cstop))  
    print("----------------------")

