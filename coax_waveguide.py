import numpy as np
import scipy.special as sp
import BesselCrossZeros.calcBesselZeros as besselX
import circular_waveguide as circWG

def coax_psiTM(a,b,m,chi,rho): #returns value of normalized TM potential in a coaxial cable of inner radius b,outer radius m. The potential is of order m and is evaluated at the point rho, where b <= rho <= a.
#chi is the relevant root.
    eps_m = 1 + 1*(m!=0) # 1 if m==0, otherwise 2.
    c=a/b
    func=sp.jv(m,chi*rho/b)*sp.yv(m,chi) - sp.yv(m,chi*rho/b)*sp.jv(m,chi)
    normalization=np.sqrt(np.pi*eps_m)/(2*np.sqrt((sp.jv(m,chi)/sp.jv(m,c*chi))**2-1))
    result=func*normalization
    return result

def coax_phiTE(a,b,m,chi,rho): #returns value of normalized TE potential in a coaxial cable of inner radius b,outer radius m. The potential is of order m and is evaluated at the point rho, where b <= rho <= a.
#chi is the relevant root.
    eps_m = 1 + 1*(m!=0) # 1 if m==0, otherwise 2.
    c=a/b
    func=sp.jv(m,chi*rho/b)*sp.yvp(m,chi) - sp.yv(m,chi*rho/b)*sp.jvp(m,chi)
    normalization = np.sqrt(np.pi*eps_m)/2 * 1/np.sqrt(
    (sp.jvp(m,chi)/sp.jvp(m,c*chi))**2 * (1-(m/(c*chi))**2) - (1-(m/chi)**2))
    result = func*normalization
    return result

class waveguide_coax:
    zeros=0 #list of mode parameters
    order=0
    TE=0
    TM=0
    a=0 #outer radius
    b=0 #inner radius
    T=0 #transverse wavenumber
    numbers=0

    def __init__(self,a,b,numModes,order=None):#a is the outer radius, b the inner radius. numModes is the number of modes to be calculated; order, if specified is the order of the modes considered
        if a<=b: 
            print('Error: outer radius \'a\' must be larger than inner radius \'b\' ' )
        
        self.numModes=numModes
        self.a=a
        self.b=b
        c=a/b
        self.zeros, self.order, self.numbers, self.TE = self.get_coaxial_modes(c,numModes,order)
        self.T=self.zeros*(c-1)/(a-b) #uses standard formula to set transverse wavenumbers - this is not correct for TE(m,1) modes however. This is corrected in the next line
        TEm1_indices=(self.numbers==1)*(self.TE==1)
        self.T[TEm1_indices] = (c+1)/(a+b)*self.zeros[TEm1_indices]
        self.TM=1-self.TE       
    

    def get_coaxial_modes(self,q,numModes,order=None): #q is r_outer / r_inner
    #Function for finding the modes in a coaxial cable with q = r_outer / r_inner. 
    
    #Returns 4 np.arrays of shape (numModes,). 
    #The first array "zeros" contains zeros of the bessel cross functions in ascending order.
    #The second array "zorder" contains the order of the besselX function to which the corresponding zero belongs.
    #The third array "znumber" enumerates the roots of order m
    #The fourth array "modetype" specifies whether the corresponding zero belongs to a TE mode or a TM mode. It contains 1 if the mode is TE, and 0 if the mode is TM.
    
    #"order" is an optional parameter. When specified, only bessel functions of the specified order are considered.
    
        if order==None:
            nu_max=numModes #in this case we want exactly numModes zeros at the order specified
            s_max=numModes + 1

            #The line below generates a list containing the orders of all the roots found above
            orders=np.repeat(np.arange(0,nu_max+1).reshape(nu_max+1,1),repeats=s_max,axis=1).flatten()
            numbers=np.repeat(np.arange(1,s_max+1).reshape(1,s_max),repeats=nu_max+1,axis=0).flatten()
            #print(np.shape(orders))
            #print(np.shape(np.ones(((1+nu_max)*s_max))))
            #print(np.shape(DerivativeRoots(q,nu_max,s_max).flatten()))
            TMzeros=(np.zeros(((1+nu_max)*s_max)),orders,numbers,(besselX.besselCrossRoots(q,nu_max,s_max)).flatten()) #zeros for TM modes
            TEzeros=(np.ones(((1+nu_max)*s_max)),orders,numbers,besselX.besselDerivativeCrossRoots(q,nu_max,s_max).flatten()) #zeros for TE modes

            allzeros=np.concatenate((TEzeros,TMzeros),axis=1) 
            index_orders=np.lexsort(allzeros) #this magic function sorts the indices of the modes in the correct way
            allzeros_sorted=allzeros[3,index_orders]
            modetype_sorted=allzeros[0,index_orders]
            zorder_sorted=allzeros[1,index_orders]
            znumber_sorted=allzeros[2,index_orders]

            modetype_sorted=modetype_sorted[allzeros_sorted!=0]
            zorder_sorted=zorder_sorted[allzeros_sorted!=0]
            znumber_sorted=znumber_sorted[allzeros_sorted!=0]
            allzeros_sorted=allzeros_sorted[allzeros_sorted!=0]


            zeros=allzeros_sorted[:numModes]
            modetype=modetype_sorted[:numModes]
            znumber=znumber_sorted[:numModes]
            zorder=zorder_sorted[:numModes] #stores the order of the modes

        else:
            nu_max=order #in this case we want exactly numModes zeros at the order specified
            s_max=numModes + order
            numbers=np.arange(1,numModes+1)
            TMzeros=(np.zeros(numModes),numbers,besselX.besselCrossRoots(q,nu_max,s_max)[order,:numModes]) #zeros for TM modes
            TEzeros=(np.ones(numModes),numbers,besselX.besselDerivativeCrossRoots(q,nu_max,s_max)[order,:numModes]) #zeros for TE modes
            allzeros=np.concatenate((TEzeros,TMzeros),axis=1) 
            index_orders=np.lexsort(allzeros) #this magic function sorts the indices of the modes in the correct way
            allzeros_sorted=allzeros[2,index_orders]
            znumbers_sorted=allzeros[1,index_orders]
            modetype_sorted=allzeros[0,index_orders]

            zeros=allzeros_sorted[:numModes]
            modetype=modetype_sorted[:numModes]
            zorder=order*np.ones(numModes) #stores the order of the modes
            znumber=znumbers_sorted[:numModes]
        return zeros, zorder, znumber, modetype
    
def Coax_Circ_CouplingMatrix(WG_coax,WG_circ):
    xi,wi=np.polynomial.legendre.leggauss(20)
    
    