import numpy as np
#import matplotlib.pyplot as plt
from scipy import special as sp
import scipy.constants
class waveguide:
    zeros=0 #list of mode parameters
    order=0
    TE=0
    TM=0
    r=0 #radius
    T=0 #transverse wavenumber
    numModes=0
    def __init__(self,r,numModes,order=None): #finds first numModes modes. if order is specified, only modes of that order are included
        self.numModes=numModes
        self.r=r
        
        self.zeros, self.order, self.TE = self.get_modes(numModes,order)
        self.T=self.zeros/r
        self.TM=1-self.TE        
        
    def get_modes(self,numModes,order=None): #unifying function for getting modes whether or not order is specified
        if order==None:
            zeros, order, TE = self.get_Bessel_zeros(numModes)
        else:
            zeros, order, TE = self.get_jn_zeros(order,numModes)
        return zeros, order, TE
    
    @staticmethod
    def get_Bessel_zeros(N): #returns first N zeros of bessel functions and derivatives, their order and modetype
        zeros, zorder, znumber, modetype = sp.jnjnp_zeros(N+1)
        zeros=zeros[1:] #remove first element of array since this is the root at x=0 which is unwanted
        zorder = zorder[1:]
        modetype = modetype[1:]
        return zeros,zorder,modetype
    
    @staticmethod
    def get_jn_zeros(n,N): #returns first N zeros of jn and jnp in ascending order, for fixed value of n.
        TEzeros=(np.ones(N),sp.jnp_zeros(n,N)) #generate zeros of TE modes
        TMzeros=(np.zeros(N),sp.jn_zeros(n,N)) #generate zeros of TM modes
        allzeros=np.concatenate((TEzeros,TMzeros),axis=1) 
        index_orders=np.lexsort(allzeros) #this magic function sorts the indices of the modes in the correct way
        allzeros_sorted=allzeros[1,index_orders]
        modetype_sorted=allzeros[0,index_orders]
        
        zeros=allzeros_sorted[:N]
        modetype=modetype_sorted[:N]
        zorder=n*np.ones(N)
        
        return zeros,zorder,modetype

def computeNormalizationFactors(mode): #mode is a modeClass object
    scales=np.zeros(mode.numModes)
    for count in range(mode.numModes):
        m=mode.order[count]
        TE=mode.TE[count]
        besselzero=mode.zeros[count]
        epsilon_m=2 #correct scaling factor taking into account the polarizations
        if m==0:
            epsilon_m=1
        if TE==1:
            scale=np.sqrt(epsilon_m/np.pi)/(np.sqrt(besselzero**2-m**2)*sp.jv(m,besselzero))
        if TE==0:
            scale=np.sqrt(epsilon_m/np.pi)/(besselzero*sp.jv(m+1,besselzero))
        scales[count]=scale
    return scales

def computeCouplingMatrix(WG1,WG2):
    r1=WG1.r
    r2=WG2.r
    rmin=min(r1,r2)
    rmax=max(r1,r2)
    
    if r1==r2:
        M=np.zeros((WG1.numModes,WG2.numModes))
        np.fill_diagonal(M,1)
        return M
        
    if r1>r2:
        smallWG=WG2
        largeWG=WG1
    else:
        smallWG=WG1
        largeWG=WG2
        
    L=smallWG.numModes
    K=largeWG.numModes
    C=np.zeros((L,K)) #coupling matrix from small WG to large WG
    
    
    #Calculate Gauss-Legendre quadrature points for numerical integration.
    #xi are sample points in the interval [-1,1], wi are the weights.
    #xi,wi = np.polynomial.legendre.leggauss(40) #40 evaluation points - arbitrarily chosen
    #yi=(xi+1)/2 #goes from 0 to 1. yi = rho / rmin.
    for l in range(0,L):
        for k in range(0,K):
            n = smallWG.order[l] #finds the order of the Bessel functions corresponding to the angular dependence, i.e. the cos(n*phi)
            if n != largeWG.order[k]: #if the two modes have different orders they are orthogonal; no need for calculation
                continue #skip to next step
                   
            a_phi=np.pi
            if n==0:
                a_phi=2*np.pi
                
            chi_np=smallWG.zeros[l]
            chi_nq=largeWG.zeros[k]
            
            scales1=computeNormalizationFactors(smallWG) #compute normalization coefficients in small WG      
            scales2=computeNormalizationFactors(largeWG) # -||- in large WG
 
            if smallWG.TE[l] and largeWG.TE[k]: #this is the TE-TE case
                #prefactor=(chi_np**2)/2*scales1[l]*scales2[k]*a_phi
                #f_TETE=(sp.jv(n,smallWG.zeros[l]*yi)*sp.jv(n,largeWG.zeros[k]*yi*rmin/rmax)*yi)
                #C[l,k]=prefactor*np.sum(f_TETE*wi)
                prefactor=(chi_np**2)*scales1[l]*scales2[k]*a_phi
                alpha=chi_np
                beta=chi_nq*rmin/rmax
                f_TETE=(beta*sp.jv(n,alpha)*sp.jvp(n,beta))/(-beta**2+alpha**2)
                C[l,k]=prefactor*f_TETE

            elif smallWG.TE[l] and largeWG.TM[k]: #TE-TM case
                C[l,k]=  - n*scales1[l]*scales2[k]*a_phi*sp.jv(n,smallWG.zeros[l])*sp.jv(n,largeWG.zeros[k]*rmin/rmax)

            elif smallWG.TM[l] and largeWG.TM[k]: #TM-TM case
                #prefactor=(chi_nq*rmin/rmax)**2 /2 *scales1[l]*scales2[k]*a_phi #prefactor in front of integral
                #f_TMTM=(sp.jv(n,smallWG.zeros[l]*yi)*sp.jv(n,largeWG.zeros[k]*yi*rmin/rmax)*yi) #integrand evaluated at LG quadrature points
                #C[l,k]=prefactor*np.sum(f_TMTM*wi) 
                prefactor=(chi_nq*rmin/rmax)**2 *scales1[l]*scales2[k]*a_phi #prefactor in front of integral
                alpha=chi_np
                beta=chi_nq*rmin/rmax
                f_TMTM=alpha*sp.jv(n,beta)*sp.jvp(n,alpha)/(beta**2-alpha**2)
                C[l,k]=prefactor*f_TMTM
    if r1>r2:
        C = C.T
    return C


def computeSMatrix(wg1,wg2,frequency,C12=0): #frequency in Hz. C12 is passed as argument to avoid repeat calculations
    k=2*np.pi*frequency/299792458 # free space wavenumber
    #Impedance and admittance matrices
    eps0=scipy.constants.epsilon_0
    mu0=scipy.constants.mu_0
    Z0=np.sqrt(mu0/eps0) #normalization
    
    if np.all(C12==0):
        C12=computeCouplingMatrix(wg1,wg2)
    
    
    #formulas: ZTE/Z0 = jk0 / gamma ; ZTM/Z0 = gamma / jk0
    
    #propagation constants
    gamma1=np.sqrt(wg1.T**2-k**2,dtype=complex)
    gamma2=np.sqrt(wg2.T**2-k**2,dtype=complex)
    #gamma2=np.sqrt(wg1.T**2-k**2,dtype=complex) #THESE ARE NOW SWITCHED
    #gamma1=np.sqrt(wg2.T**2-k**2,dtype=complex) #THESE ARE NOW SWITCHED
    Z1=Z0*(wg1.TE*1j*k/gamma1 + wg1.TM*gamma1/(1j*k))
    Z2=Z0*(wg2.TE*1j*k/gamma2 + wg2.TM*gamma2/(1j*k))
    Y1=np.diag(1/Z1)
    Y2=np.diag(1/Z2) #not actually necessary
    Z1=np.diag(Z1)
    Z2=np.diag(Z2)
    C21=C12.T
    
    #derivation 3
    #A=np.concatenate((np.concatenate((T,-IK),axis=1),np.concatenate((IL,T.T),axis=1)),axis=0)
    #B=np.concatenate((np.concatenate((-T,IK),axis=1),np.concatenate((IL,T.T),axis=1)),axis=0)
    #S=np.linalg.inv(B)@A
    
    if wg1.r<wg2.r:
        #my derivation - this works only when r1 < r2
        A=np.concatenate((np.concatenate((-C21@np.sqrt(Z1),np.sqrt(Z2)),axis=1),np.concatenate((np.sqrt(Y1),C12@np.sqrt(Y2)),axis=1)),axis=0)
        B=np.concatenate((np.concatenate((C21@np.sqrt(Z1),-np.sqrt(Z2)),axis=1),np.concatenate((np.sqrt(Y1),C12@np.sqrt(Y2)),axis=1)),axis=0)
        S=np.linalg.inv(B)@A
    else:
        #Michael's derivation - one step back from final matrix - this works only when r1>r2
        A=np.concatenate((np.concatenate((np.sqrt(Z1),-C12@np.sqrt(Z2)),axis=1),np.concatenate((C21@np.sqrt(Y1),np.sqrt(Y2)),axis=1)),axis=0)
        B=np.concatenate((np.concatenate((-np.sqrt(Z1),C12@np.sqrt(Z2)),axis=1),np.concatenate((C21@np.sqrt(Y1),np.sqrt(Y2)),axis=1)),axis=0)
        S=np.linalg.inv(B)@A
    
    
    
    #Michael's symmetric derivation using an infinitely thin aperture
    #A=np.concatenate((np.concatenate((np.sqrt(Z1),-C12@np.sqrt(Z2)),axis=1),np.concatenate((np.sqrt(Y1),C12@np.sqrt(Y2)),axis=1)),axis=0)
    #B=np.concatenate((np.concatenate((-np.sqrt(Z1),C12@np.sqrt(Z2)),axis=1),np.concatenate((np.sqrt(Y1),C12@np.sqrt(Y2)),axis=1)),axis=0)
    #S=np.linalg.inv(B)@A
    
    

    #Michael's derivation
    #T=np.sqrt(Y1)@(C12@np.sqrt(Z2))
    #IL=np.identity(wg1.numModes)
    #IK=np.identity(wg2.numModes)
    #A=np.concatenate((np.concatenate((-IL,T),axis=1),np.concatenate((T.T,IK),axis=1)),axis=0)
    #B=np.concatenate((np.concatenate((IL,-T),axis=1),np.concatenate((T.T,IK),axis=1)),axis=0)
    #S=np.dot(np.linalg.inv(B),A)
    
    #Below; test code - interchanging waveguide indices 1 and 2 
    #T=np.sqrt(Y2)@((C12.T)@np.sqrt(Z1))
    #IL=np.identity(wg2.numModes)
    #IK=np.identity(wg1.numModes)
    #A=np.concatenate((np.concatenate((-IL,T),axis=1),np.concatenate((T.T,IK),axis=1)),axis=0)
    #B=np.concatenate((np.concatenate((IL,-T),axis=1),np.concatenate((T.T,IK),axis=1)),axis=0)
    #S=np.dot(np.linalg.inv(B),A)
    
    return S

def dB(S):
    return 20*np.log10(np.abs(S))
    
def selectNumModes(r1,r2,L,order=None): 
    WG=waveguide(r=1,numModes=1)#dummy waveguide to enable use of waveguide class functions
    chimax=WG.get_modes(L,order)[0][-1] #calculates largest Bessel root in WG1.
    chi2max = r2/r1*chimax #desired value of max bessel root in waveguide 2
    N=L
    zeros=WG.get_modes(N,order)[0]
    while chi2max > np.max(zeros):
        N=int(2*N*chi2max/np.max(zeros)) #increase number of modes to be calculated 
        zeros=WG.get_modes(N,order)[0]
        if N > 10000:
            print('error - couldn\'t find enough modes')
            break
    #now chi2max is smaller than np.max(zeros)
    K=np.where(chi2max<=zeros)[0][0] #K is the index of the smallest zero which is larger than the requirement
    
    return K
    