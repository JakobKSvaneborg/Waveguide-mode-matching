#Implements the algorithm by Sorolla, Mosig and Mattes for finding roots of Cross product bessel functions #jn(qx)yn(x) - yn(qx)*jn(x) and their derivatives $jn'(qx)yn'(x) - yn'(qx)*jn'(x)

import numpy as np
import scipy.special as sp

def fv(nu,q,x): #Cross product Bessel function of order nu. q for the coaxial cable is the ratio r_outer / r_inner. The zeros of this function determine TM modes
    return sp.jv(nu,q*x)*sp.yv(nu,x) - sp.jv(nu,x)*sp.yv(nu,q*x)


def fvp(nu,q,x): #derivative of fv wrt x
    term1=q*sp.jvp(nu,q*x)*sp.yv(nu,x) + sp.jv(nu,q*x)*sp.yvp(nu,x)
    term2= - sp.jvp(nu,x)*sp.yv(nu,q*x) - q*sp.jv(nu,x)*sp.yvp(nu,q*x)
    result = term1 + term2
    return result

def fvpp(nu,q,x): #second derivative of fv wrt x
    term1=q*q*sp.jvp(nu,q*x,2)*sp.yv(nu,x)+q*sp.jvp(nu,q*x)*sp.yvp(nu,x)
    term2= q*sp.jvp(nu,q*x)*sp.yvp(nu,x) + sp.jv(nu,q*x)*sp.yvp(nu,x,2)
    term3=- sp.jvp(nu,x,2)*sp.yv(nu,q*x) - q*sp.jvp(nu,x)*sp.yvp(nu,q*x)
    term4=- q*sp.jvp(nu,x)*sp.yvp(nu,q*x) - q*q*sp.jv(nu,x)*sp.yvp(nu,q*x,2)
    result=term1+term2+term3+term4
    return result

def fvtilde(nu,q,x): #Cross product of derivatives of Bessel functions of order nu. The zeros of this function determines the TE modes in the coaxial line.
    return sp.jvp(nu,q*x)*sp.yvp(nu,x)-sp.jvp(nu,x)*sp.yvp(nu,q*x)

def fvptilde(nu,q,x): #derivative of fvtilde wrt x
    term1=q*sp.jvp(nu,q*x,2)*sp.yvp(nu,x) + sp.jvp(nu,q*x)*sp.yvp(nu,x,2)
    term2=-sp.jvp(nu,x,2)*sp.yvp(nu,q*x) -q* sp.jvp(nu,x)*sp.yvp(nu,q*x,2)
    return term1+term2

def F0(q,x): #Function to be optimized in optimization algorithm
    return fv(0,q,x)/fvp(0,q,x)

def F0p(q,x): #Derivative of f0 wrt x
    return 1 - fv(0,q,x)*fvpp(0,q,x)/fvp(0,q,x)**2
    

def zeroOrderRoots(q,s_max,eps=1e-12,i_max=1000):
    #This function implements algorithm 1 in the paper, calculating the roots of the Bessel function cross product of order zero
    #returns an np.array of shape (s_max,) containing the roots in ascending order
    #q is the parameter that enters along with x into the functions that we optimize
    #s_max is the number of roots to be found. 
    #eps is the error tolerance
    #i_max is the maximum allowed number of loop iterations
    
    c0s=[] #this list will contain the roots 
    for s in range(s_max):
        i=0
        err=1
        Xi=(s+1)*np.pi/(q-1) #first estimate of the s'th root
        while(err > eps and i <= i_max ):
            #This loop determines the s'th root by the newton-raphson method. 
            #Each iteration updates the value of Xi, which is the current best guess for the root to be found.
            Xinew = Xi - F0(q,Xi)/F0p(q,Xi) 
            err = abs((Xinew - Xi)/Xinew)
            i = i+1
            Xi=Xinew
            if i==i_max:
                print('calcZeroOrderRoots did not converge when trying to find the %d\'th root' %s)
        c0s.append(Xi) #append the found root to the list
    return np.array(c0s) 

def besselCrossRoots(q,nu_max,s_max,eps=1e-12,n_sub=100,i_max=1000):
    #Implementation of algorithm 2 from the paper. Finds roots of higher order Bessel functions up to order nu_max.
    #Returns: np.array of shape (nu_max+1, s_max). The n'th row contains roots of the cross product function of order n (starting with 0). 
    #The number of roots found decreases with increasing order; s_max - nu roots are found for the function of order nu.
    #s_max is the number of zero order roots to find. For the algorithm to work, the condition s_max >= nu_max + 1 must be satisfied.
    #For s_max = nu_max + 1, only one root is found for the Bessel function of order nu_max.
    
    
    c0s=zeroOrderRoots(q,s_max,eps,i_max)
    Cs=np.zeros((nu_max+1,s_max))
    Cs[0]=c0s
    for nu in range(1,nu_max+1):
        for s in range(1,s_max-nu+1):
            s_index=s-1
            i=0
            err=1
            count=0
            epsilon = max(eps,eps*(Cs[nu-1,s_index+1] - Cs[nu-1,s_index]))
            delta = (Cs[nu-1,s_index+1]-Cs[nu-1,s_index])/n_sub
            while(err > eps and count<n_sub):
                Xi = Cs[nu-1,s_index] + epsilon + count*delta
                count=count+1
                while(Xi >= Cs[nu-1,s_index] and Xi<Cs[nu-1,s_index+1] and i<=i_max and err>eps):
                    Xinew = Xi - fv(nu,q,Xi)/fvp(nu,q,Xi)
                    err=abs((Xinew - Xi)/Xinew)
                    i=i+1
                    Xi=Xinew
                    if i==i_max:
                            print('i=i_max reached in calculateBesselCrossRoots for nu = %d, s = %d. Error was %f.' %(nu,s, err))
                        
            Cs[nu,s_index]=Xi
    return Cs
      
    
def besselDerivativeCrossRoots(q,nu_max,s_max,delta=1e-3,n_sub=100,i_max=1000,eps=1e-12):
     #Implementation of algorithm 3 from the paper. Finds roots of higher order derivative Bessel functions up to order nu_max.
    #Returns: np.array of shape (nu_max+1, s_max). The n'th row contains roots of the cross product function of order n (starting with 0). 
    #The number of roots found decreases with increasing order; s_max - nu roots are found for the function of order nu.
    #s_max is the number of zero order roots to find. For the algorithm to work, the condition s_max >= nu_max + 1 must be satisfied.
    #For s_max = nu_max + 1, only one root is found for the Bessel function of order nu_max.
    Cs=np.zeros((nu_max+1,s_max))
    C0s=besselCrossRoots(q,1,s_max+1,eps,n_sub,i_max)
    Cs[0]=C0s[1,:-1] #calculate zero order derivative roots
    C_search = np.concatenate((np.array([delta]),Cs[0,:-1])) #this array contains roots of the previous order (nu-1) and may be searched through to find the new roots
    for nu in range(1,nu_max+1):
        for s in range(1,s_max-nu+1):
            s_index=s-1
            i=0
            err=1
            count=0
            epsilon=max(eps,eps*(C_search[s_index+1]-C_search[s_index]))
            delta = (C_search[s_index+1] - C_search[s_index])/n_sub
            while(err > eps and count < n_sub):
                Xi=C_search[s_index] + epsilon + count*delta
                count= count + 1
                while(err > eps and Xi >= C_search[s_index] and Xi < C_search[s_index+1] and i <= i_max):
                    Xinew = Xi - fvtilde(nu,q,Xi)/fvptilde(nu,q,Xi)
                    err=abs((Xinew-Xi)/Xinew)
                    i=i+1
                    Xi=Xinew
                    if i == i_max:
                        print('i=i_max reached in DerivativeRootsBesselDerivativeCrossRoots for nu = %d, s = %d. Error was %f.' %(nu,s, err))
            Cs[nu,s_index]=Xi
        C_search=Cs[nu] #update search array for the next iteration        
    return Cs
    
    
def get_coaxial_modes(q,numModes,order=None): #q is r_outer / r_inner
    #Function for finding the modes in a coaxial cable with q = r_outer / r_inner. 
    
    #Returns 3 np.arrays of shape (numModes,). 
    #The first array "zeros" contains zeros of the bessel cross functions in ascending order.
    #The second array "zorder" contains the order of the besselX function to which the corresponding zero belongs.
    #The third array "modetype" specifies whether the corresponding zero belongs to a TE mode or a TM mode. It contains 1 if the mode is TE, and 0 if the mode is TM.
    
    #"order" is an optional parameter. When specified, only bessel functions of the specified order are considered.
    
    if order==None:
        nu_max=numModes #in this case we want exactly numModes zeros at the order specified
        s_max=numModes + 1

        #The line below generates a list containing the orders of all the roots found above
        orders=np.repeat(np.arange(0,nu_max+1).reshape(nu_max+1,1),repeats=s_max,axis=1).flatten()
        #print(np.shape(orders))
        #print(np.shape(np.ones(((1+nu_max)*s_max))))
        #print(np.shape(DerivativeRoots(q,nu_max,s_max).flatten()))
        TMzeros=(np.zeros(((1+nu_max)*s_max)),orders,(besselCrossRoots(q,nu_max,s_max)).flatten()) #zeros for TM modes
        TEzeros=(np.ones(((1+nu_max)*s_max)),orders,besselDerivativeCrossRoots(q,nu_max,s_max).flatten()) #zeros for TE modes

        allzeros=np.concatenate((TEzeros,TMzeros),axis=1) 
        index_orders=np.lexsort(allzeros) #this magic function sorts the indices of the modes in the correct way
        allzeros_sorted=allzeros[2,index_orders]
        modetype_sorted=allzeros[0,index_orders]
        zorder_sorted=allzeros[1,index_orders]

        modetype_sorted=modetype_sorted[allzeros_sorted!=0]
        zorder_sorted=zorder_sorted[allzeros_sorted!=0]
        allzeros_sorted=allzeros_sorted[allzeros_sorted!=0]


        zeros=allzeros_sorted[:numModes]
        modetype=modetype_sorted[:numModes]
        zorder=zorder_sorted[:numModes] #stores the order of the modes
        
    else:
        nu_max=order #in this case we want exactly numModes zeros at the order specified
        s_max=numModes + order
        TMzeros=(np.zeros(numModes),besselCrossRoots(q,nu_max,s_max)[order,:numModes]) #zeros for TM modes
        TEzeros=(np.ones(numModes),besselDerivativeCrossRoots(q,nu_max,s_max)[order,:numModes]) #zeros for TE modes
        allzeros=np.concatenate((TEzeros,TMzeros),axis=1) 
        index_orders=np.lexsort(allzeros) #this magic function sorts the indices of the modes in the correct way
        allzeros_sorted=allzeros[1,index_orders]
        modetype_sorted=allzeros[0,index_orders]
        
        zeros=allzeros_sorted[:numModes]
        modetype=modetype_sorted[:numModes]
        zorder=order*np.ones(numModes) #stores the order of the modes
    return zeros, zorder, modetype

print(get_coaxial_modes(q=1.001,numModes=10)) 
