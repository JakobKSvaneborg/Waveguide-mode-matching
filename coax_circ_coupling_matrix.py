import numpy as np
import scipy.special as sp
import BesselCrossZeros.calcBesselZeros as besselX
import circular_waveguide as circWG
import scipy.constants

def coax_phiTM(a,b,m,chi,rho): #returns value of normalized TM potential in a coaxial cable of inner radius b,outer radius m. The potential is of order m and is evaluated at the point rho, where b <= rho <= a.
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

def coax_phiTEM(a,b,rho):
    return np.log(rho/b)/np.sqrt(2*np.pi*np.log(a/b))

def coax_phiTEM_p(a,b,rho):
    return 1/(rho*np.sqrt(2*np.pi*np.log(a/b)))

def circ_phiTE(r,m,chi,rho):
    eps_m = 1 + 1*(m!=0) # 1 if m==0, otherwise 2.
    normalization=np.sqrt(eps_m/np.pi)*1/(sp.jv(m,chi)*np.sqrt(chi**2-m**2))
    result=normalization*sp.jv(m,chi*rho/r)
    return result

def circ_phiTM(r,m,chi,rho):
    eps_m = 1 + 1*(m!=0) # 1 if m==0, otherwise 2.
    normalization=np.sqrt(eps_m/np.pi)*1/(sp.jv(m+1,chi)*chi)
    result=normalization*sp.jv(m,chi*rho/r)
    return result

class waveguide_coax:
    zeros=0 #list of mode parameters
    orders=0
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
        zeros, zorder, numbers, TE = self.get_coaxial_modes(c,numModes,order)
        TM=1-TE      
        if order==None or order==0: #manually adds the TEM mode as the first mode, and removes the last one calculate to keep the total number of modes correct
            self.zeros=np.concatenate((np.array([0]),zeros[:-1]))
            self.orders=np.concatenate((np.array([0]),zorder[:-1]))
            self.numbers=np.concatenate((np.array([0]),numbers[:-1]))
            self.TE=np.concatenate((np.array([0]),TE[:-1]))
            self.TM=np.concatenate((np.array([0]),TM[:-1]))
        else:
            self.zeros=zeros
            self.orders=zorder
            self.numbers=numbers
            self.TE=TE
            self.TM=TM
            
        
        self.T=self.zeros*(c-1)/(a-b) #uses standard formula to set transverse wavenumbers - this is not correct for TE(m,1) modes however. This is corrected in the next line
        TEm1_indices=(self.numbers==1)*(self.TE==1)
        self.T[TEm1_indices] = (c+1)/(a+b)*self.zeros[TEm1_indices]
        
    

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
    
def Coax_Circ_CouplingMatrix(coax,circ):
    a=coax.a
    b=coax.b
    r=circ.r
    if(r != a):
        print('error: radius of circular waveguide must equal outer radius of coaxial waveguide')
    
    L=coax.numModes
    K=circ.numModes
    C=np.zeros((L,K))
    xi,wi=intba(a,b) #gauss-legendre evaluation points and weights for the appropriate integrals. the function scales these appropriately for our purpose.
    
    for l in range(L):
        for k in range(K):
            m=coax.orders[l]
            if m != circ.order[k]:
                continue #skip to next step if the orders of the two modes are different; there wont be any coupling between them
            a_phi=np.pi*(1+1*(m==0)) #result of integration over phi; 2pi if m=0, otherwise pi.
            chi_coax=coax.zeros[l]
            chi_circ=circ.zeros[k]

            if chi_coax==0: #TEM mode
                if circ.TE[k]:
                    #C[l,k]=circ.T[k]**2*a_phi*np.sum(wi*xi*circ_phiTE(r,m,chi_circ,xi)*coax_phiTEM(a,b,xi))
                        C[l,k]=0
                if circ.TM[k]:
                    C[l,k]= - b*a_phi*circ_phiTM(r,m,chi_circ,b)*coax_phiTEM_p(a,b,b) #here there is no integration over r
                continue #goes to next loop iteration if the coax mode was TEM

            #in this part of the loop, the coaxial mode cannot be a TEM mode

            if coax.TE[l] and circ.TE[k]: #TE-TE case
                C[l,k]=a_phi*coax.T[l]**2*np.sum(wi*xi*circ_phiTE(r,m,chi_circ,xi)*
                                                coax_phiTE(a,b,m,chi_coax,xi))
                continue

            if coax.TM[l] and circ.TM[k]: #TM-TM case
                C[l,k]=a_phi*circ.T[k]**2*np.sum(wi*xi*circ_phiTM(r,m,chi_circ,xi)*
                                                coax_phiTM(a,b,m,chi_coax,xi))
                continue

            if coax.TM[l] and circ.TE[k]:
                C[l,k]=0
                continue

            if coax.TE[l] and circ.TM[k]:
                #this formula is implemented under the assumption that the coaxial mode has cosine polarization. the induced circular mode will then have sine polarization. I think the sign changes if the two polarizations were switched.
                C[l,k]= b*(m)*a_phi * coax_phiTE(a,b,m,chi_coax,b)*circ_phiTM(r,m,chi_circ,b)

            #that should be all
                
    return C
        
        
def intba(a,b,N=20): #gauss-legendre weights and points for integration between x=b and x=a
    xi,wi=np.polynomial.legendre.leggauss(N)  #for numerical integration: temp.
    yi=(xi+1)/2 #shifts the xi from the interval [-1,1] to [0,1]
    xi=b+yi*(a-b) #these are the evaluation point for integration
    wi=wi*(a-b)/2 # new weights for the new integration range
    return xi,wi
    

def computeSMatrix(coax,circ,frequency,C12=0): #frequency in Hz. C12 is passed as argument to avoid repeat calculations
    k=2*np.pi*frequency/299792458 # free space wavenumber
    #Impedance and admittance matrices
    eps0=scipy.constants.epsilon_0
    mu0=scipy.constants.mu_0
    Z0=np.sqrt(mu0/eps0) #normalization
    if np.all(C12==0):
        C12=Coax_Circ_CouplingMatrix(coax,circ)
    
    
    #formulas: ZTE/Z0 = jk0 / gamma ; ZTM/Z0 = gamma / jk0
    
    #propagation constants
    gamma1=np.sqrt(coax.T**2-k**2,dtype=complex)
    gamma2=np.sqrt(circ.T**2-k**2,dtype=complex)
    #gamma2=np.sqrt(wg1.T**2-k**2,dtype=complex) #THESE ARE NOW SWITCHED
    #gamma1=np.sqrt(wg2.T**2-k**2,dtype=complex) #THESE ARE NOW SWITCHED
    Z1=Z0*(coax.TE*1j*k/gamma1 + coax.TM*gamma1/(1j*k))
    Z2=Z0*(circ.TE*1j*k/gamma2 + circ.TM*gamma2/(1j*k))
    Z1[(coax.TE==0)*(coax.TM==0)]=Z0 #set impedance of TEM mode
    Y1=np.diag(1/Z1)
    Y2=np.diag(1/Z2) #not actually necessary
    Z1=np.diag(Z1)
    Z2=np.diag(Z2)
    C21=C12.T    
    #my derivation - in the circular case, this works only when r1 < r2
    A=np.concatenate((np.concatenate((-C21@np.sqrt(Z1),np.sqrt(Z2)),axis=1),np.concatenate((np.sqrt(Y1),C12@np.sqrt(Y2)),axis=1)),axis=0)
    B=np.concatenate((np.concatenate((C21@np.sqrt(Z1),-np.sqrt(Z2)),axis=1),np.concatenate((np.sqrt(Y1),C12@np.sqrt(Y2)),axis=1)),axis=0)
    
    
    #Michael's derivation - one step back from final matrix - this works only when r1>r2
    #A=np.concatenate((np.concatenate((np.sqrt(Z1),-C12@np.sqrt(Z2)),axis=1),np.concatenate((C21@np.sqrt(Y1),np.sqrt(Y2)),axis=1)),axis=0)
    #B=np.concatenate((np.concatenate((-np.sqrt(Z1),C12@np.sqrt(Z2)),axis=1),np.concatenate((C21@np.sqrt(Y1),np.sqrt(Y2)),axis=1)),axis=0)
    
    S=np.linalg.inv(B)@A
    return S
    
def dB(S):
    return 20*np.log10(np.abs(S))