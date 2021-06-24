import numpy as np
import scipy.special as sp
#import matplotlib.pyplot as plt
#%matplotlib notebook
import BesselCrossZeros.calcBesselZeros as besselX
#import circular_waveguide as circWG
import scipy.constants

class waveguide_coax:
    zeros=0 #list of mode parameters
    orders=0
    TE=0
    TM=0
    a=0 #outer radius
    b=0 #inner radius
    T=0 #transverse wavenumber
    numbers=0
    numModes=0
    order_specified=False
    WG_type='coaxial'

    def __init__(self,a,b,numModes,order=None):#a is the outer radius, b the inner radius. numModes is the number of modes to be calculated; order, if specified is the order of the modes considered
        if a<=b: 
            print('Error: outer radius \'a\' must be larger than inner radius \'b\' ' )
        
        self.numModes=numModes
        self.a=a
        self.b=b
        if order != None:
            self.order_specified=True
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
        
    def __str__(self):
        Line1="Coaxial waveguide\n"
        Line2="Outer radius: %.2f mm\n"%(1000*self.a)
        Line3="Inner radius: %.2f mm\n"%(1000*self.b)
        Line4="Number of modes calculated: %d\n"%len(self.zeros)
        return Line1 + Line2 + Line3 + Line4
        
    
    

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
    
    def phiTM(self,modeNumber,rho,increase_m=0): #returns value of normalized TM potential in a coaxial cable of inner radius b,outer radius m. The potential is of order m and is evaluated at the point rho, where b <= rho <= a.
    #chi is the relevant root.
    #increase_m is a value added to m when evaluating the functions; it is used for analytic integration of the modes
        a=self.a
        b=self.b
        m=self.orders[modeNumber]
        chi=self.zeros[modeNumber]
        eps_m = 1 + 1*(m!=0) # 1 if m==0, otherwise 2.
        c=a/b
        func=sp.jv(m,chi*rho/b)*sp.yv(m,chi) - sp.yv(m,chi*rho/b)*sp.jv(m,chi)
        normalization=np.sqrt(np.pi*eps_m)/(2*np.sqrt((sp.jv(m,chi)/sp.jv(m,c*chi))**2-1))
        if increase_m:
            mNew=m+increase_m   
            func=sp.jv(mNew,chi*rho/b)*sp.yv(m,chi) - sp.yv(mNew,chi*rho/b)*sp.jv(m,chi)
        result=func*normalization
        return result
    
    
    def phiTM_dr(self,modeNumber,rho,increase_m=0): #returns value of normalized TM potential in a coaxial cable of inner radius b,outer radius m. The potential is of order m and is evaluated at the point rho, where b <= rho <= a.
    #chi is the relevant root.
    #increase_m is a value added to m when evaluating the functions; it is used for analytic integration of the modes
        a=self.a
        b=self.b
        m=self.orders[modeNumber]
        chi=self.zeros[modeNumber]
        eps_m = 1 + 1*(m!=0) # 1 if m==0, otherwise 2.
        c=a/b
        func=chi/b*sp.jvp(m,chi*rho/b)*sp.yv(m,chi) - chi/b*sp.yvp(m,chi*rho/b)*sp.jv(m,chi)
        normalization=np.sqrt(np.pi*eps_m)/(2*np.sqrt((sp.jv(m,chi)/sp.jv(m,c*chi))**2-1))
        result=func*normalization
        return result

    def phiTE(self,modeNumber,rho,increase_m=0): #returns value of normalized TE potential in a coaxial cable of inner radius b,outer radius m. The potential is of order m and is evaluated at the point rho, where b <= rho <= a.
    #chi is the relevant root.
        a=self.a
        b=self.b
        m=self.orders[modeNumber]
        chi=self.zeros[modeNumber]
        eps_m = 1 + 1*(m!=0) # 1 if m==0, otherwise 2.
        c=a/b
        func=sp.jv(m,chi*rho/b)*sp.yvp(m,chi) - sp.yv(m,chi*rho/b)*sp.jvp(m,chi)
        normalization = np.sqrt(np.pi*eps_m)/2 * 1/np.sqrt(
        (sp.jvp(m,chi)/sp.jvp(m,c*chi))**2 * (1-(m/(c*chi))**2) - (1-(m/chi)**2))
        
        if increase_m:
            mNew=m+increase_m    
            func=sp.jv(mNew,chi*rho/b)*sp.yvp(m,chi) - sp.yv(mNew,chi*rho/b)*sp.jvp(m,chi)
        
        result = func*normalization
        return result
    
    def phiTE_dr(self,modeNumber,rho,increase_m=0): #returns value of normalized TE potential in a coaxial cable of inner radius b,outer radius m. The potential is of order m and is evaluated at the point rho, where b <= rho <= a.
        a=self.a
        b=self.b
        m=self.orders[modeNumber]
        chi=self.zeros[modeNumber]
        eps_m = 1 + 1*(m!=0) # 1 if m==0, otherwise 2.
        c=a/b
        func=chi/b*sp.jvp(m,chi*rho/b)*sp.yvp(m,chi) - chi/b* (sp.yvp(m,chi*rho/b)*sp.jvp(m,chi))
        normalization = np.sqrt(np.pi*eps_m)/2 * 1/np.sqrt(
        (sp.jvp(m,chi)/sp.jvp(m,c*chi))**2 * (1-(m/(c*chi))**2) - (1-(m/chi)**2))
        result = func*normalization
        return result

    def phiTEM(self,rho):
        a=self.a
        b=self.b
        return np.log(rho/b)/np.sqrt(2*np.pi*np.log(a/b))

    def phiTEM_dr(self,rho):
        a=self.a
        b=self.b
        return 1/(rho*np.sqrt(2*np.pi*np.log(a/b)))
    
    
class waveguide_circ:
    zeros=0 #list of mode parameters
    orders=0
    TE=0
    TM=0
    r=0 #radius
    T=0 #transverse wavenumber
    numModes=0
    order_specified=False
    WG_type='circular'
    
    def __init__(self,r,numModes,order=None): #finds first numModes modes. if order is specified, only modes of that order are included
        self.numModes=numModes
        self.r=r
        if order != None:
            self.order_specified=True
        self.zeros, self.orders, self.TE = self.get_modes(numModes,order)
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
    
    def phiTE(self,modeNumber,rho,increase_m=0):
        r=self.r
        chi=self.zeros[modeNumber]
        m=self.orders[modeNumber]
        eps_m = 1 + 1*(m!=0) # 1 if m==0, otherwise 2.
        normalization=np.sqrt(eps_m/np.pi)*1/(sp.jv(m,chi)*np.sqrt(chi**2-m**2))
        func=sp.jv(m,chi*rho/r)
        if increase_m:
            mNew=m+increase_m    
            func=sp.jv(mNew,chi*rho/r)
        result=normalization*func
        return result

    def phiTE_dr(self,modeNumber,rho,increase_m=0):
        r=self.r
        chi=self.zeros[modeNumber]
        m=self.orders[modeNumber]
        eps_m = 1 + 1*(m!=0) # 1 if m==0, otherwise 2.
        normalization=np.sqrt(eps_m/np.pi)*1/(sp.jv(m,chi)*np.sqrt(chi**2-m**2))
        result=chi/r*normalization*sp.jvp(m,chi*rho/r)
        return result
    
    def phiTM(self,modeNumber,rho,increase_m=0):
        r=self.r
        chi=self.zeros[modeNumber]
        m=self.orders[modeNumber]
        eps_m = 1 + 1*(m!=0) # 1 if m==0, otherwise 2.
        normalization=np.sqrt(eps_m/np.pi)*1/(sp.jv(m+1,chi)*chi)
        func=sp.jv(m,chi*rho/r)
        if increase_m:
            mNew=m+increase_m    
            func=sp.jv(mNew,chi*rho/r)
        result=normalization*func
        return result
    
    def phiTM_dr(self,modeNumber,rho):
        r=self.r
        chi=self.zeros[modeNumber]
        m=self.orders[modeNumber]
        eps_m = 1 + 1*(m!=0) # 1 if m==0, otherwise 2.
        normalization=np.sqrt(eps_m/np.pi)*1/(sp.jv(m+1,chi)*chi)
        result=chi/r*normalization*sp.jvp(m,chi*rho/r)
        return result   

def computeNormalizationFactors(mode): #mode is a modeClass object
    scales=np.zeros(mode.numModes)
    for count in range(mode.numModes):
        m=mode.orders[count]
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


def Coax_Circ_CouplingMatrix(coax,circ):
    a=coax.a
    b=coax.b
    R=min(a,r)
    
    L=coax.numModes
    K=circ.numModes
    C=np.zeros((L,K))
    if r<b: #no coupling can happen here
        return C 
    
    for l in range(L):
        for k in range(K):
            m=coax.orders[l]
            if m != circ.orders[k]:
                continue #skip to next step if the orders of the two modes are different; there wont be any coupling between them
            a_phi=np.pi*(1+1*(m==0)) #result of integration over phi; 2pi if m=0, otherwise pi.
            chi_coax=coax.zeros[l]
            chi_circ=circ.zeros[k]

            if chi_coax==0: #TEM mode
                if circ.TE[k]:
                    C[l,k]=0
                if circ.TM[k]:
                    #C[l,k]= - b*a_phi*circ.phiTM(k,b)*coax.phiTEM_dr(b) #here there is no integration over r
                    inner_integral = - b*a_phi*circ.phiTM(k,b)*coax.phiTEM_dr(b) #here there is no integration over r
                    outer_integral= R*a_phi*circ.phiTM(k,R)*coax.phiTEM_dr(R)
                    C[l,k]=inner_integral+outer_integral
                    print('New TEM-TM coefficient: %f' %outer_integral)
                continue #goes to next loop iteration if the coax mode was TEM

            #in this part of the loop, the coaxial mode cannot be a TEM mode

            if coax.TE[l] and circ.TE[k]: #TE-TE case
                #oldformula=a_phi*coax.T[l]**2*np.sum(wi*xi*circ.phiTE(k,xi)*coax.phiTE(l,xi))
                vol_prefactor=a_phi*coax.T[l]**2/((chi_coax/b)**2-(chi_circ/r)**2)
                vol_integral1=a*(chi_coax/b*coax.phiTE(l,a,increase_m=1)*circ.phiTE(k,a) - chi_circ/r*coax.phiTE(l,a)*circ.phiTE(k,a,increase_m=1))
                vol_integral2=b*(chi_coax/b*coax.phiTE(l,b,increase_m=1)*circ.phiTE(k,b) - chi_circ/r*coax.phiTE(l,b)*circ.phiTE(k,b,increase_m=1))
                vol_integral=(vol_integral1-vol_integral2)*vol_prefactor
                surf_integral = R*a_phi*circ.phiTE(k,R)*coax.phiTE_dr(l,R)
                print('TE surface integral: %f' %surf_integral)
                C[l,k]=vol_integral+surf_integral
                #print('new formula difference: %f' %(C[l,k]-oldformula))
                continue

            if coax.TM[l] and circ.TM[k]: #TM-TM case
                vol_prefactor=a_phi*circ.T[k]**2/((chi_coax/b)**2-(chi_circ/r)**2)
                vol_integral1=a*(chi_coax/b*coax.phiTM(l,a,increase_m=1)*circ.phiTM(k,a) - chi_circ/r*coax.phiTM(l,a)*circ.phiTM(k,a,increase_m=1))
                vol_integral2=b*(chi_coax/b*coax.phiTM(l,b,increase_m=1)*circ.phiTM(k,b) - chi_circ/r*coax.phiTM(l,b)*circ.phiTM(k,b,increase_m=1))
                vol_integral=vol_prefactor*(vol_integral1-vol_integral2)
                surf_integral = a_phi*R*coax.phiTM(l,R)*circ.phiTM_dr(k,R)
                print('TM surface integral: %f' %surf_integral)
                C[l,k]=vol_integral+surf_integral
                continue

            if coax.TM[l] and circ.TE[k]:
                #maybe wrong sign?
                C[l,k]=- a_phi*R*m*coax.phiTM(l,R)*circ.phiTE(k,R)
                print('TM-TE coefficient: %f' %C[l,k])
                continue

            if coax.TE[l] and circ.TM[k]:
                #this formula is implemented under the assumption that the coaxial mode has cosine polarization. the induced circular mode will then have sine polarization. I think the sign changes if the two polarizations were switched.
                inner_integral= b*(m)*a_phi * coax.phiTE(l,b)*circ.phiTM(k,b)
                outer_integral= -R*(m)*a_phi * coax.phiTE(l,R)*circ.phiTM(k,R)
                C[l,k]=inner_integral+outer_integral
                print('New TE-TM coefficient: %f' %outer_integral)

            #that should be all
                
    return C
    

    
def Coax_Coax_CouplingMatrix(coax1,coax2):
    a1=coax1.a
    a2=coax2.a
    b1=coax1.b
    b2=coax2.b
    a=min(a1,a2)
    b=max(b1,b2)
    
    L=coax1.numModes
    K=coax2.numModes
    C=np.zeros((L,K))
    if b>a: #no coupling can happen here
        return C 
    
    for l in range(L):
        for k in range(K):
            m=coax1.orders[l]
            if m != coax2.orders[k]:
                continue #skip to next step if the orders of the two modes are different; there wont be any coupling between them
            a_phi=np.pi*(1+1*(m==0)) #result of integration over phi; 2pi if m=0, otherwise pi.
            chi_1=coax1.zeros[l]
            chi_2=coax2.zeros[k]

            if chi_1==0: #TEM mode
                if coax2.TE[k]:
                    C[l,k]=0
                elif coax2.TM[k]:
                    #C[l,k]= - b*a_phi*circ.phiTM(k,b)*coax.phiTEM_dr(b) #here there is no integration over r
                    inner_integral = - b*a_phi*coax2.phiTM(k,b)*coax1.phiTEM_dr(b) #here there is no integration over r
                    outer_integral= a*a_phi*coax2.phiTM(k,a)*coax1.phiTEM_dr(a)
                    C[l,k]=inner_integral+outer_integral
                else: #TEM-TEM
                    inner_integral = - b*a_phi*coax2.phiTEM(b)*coax1.phiTEM_dr(b) #here there is no integration over r
                    outer_integral= a*a_phi*coax2.phiTEM(a)*coax1.phiTEM_dr(a)
                    C[l,k]=inner_integral+outer_integral
                    
                continue #goes to next loop iteration if the coax mode was TEM
            
            if chi_2==0: #second one is TEM and first one isn't
                if coax1.TE[l]:
                    C[l,k]=0
                elif coax1.TM[l]:
                    inner_integral = - b*a_phi*coax1.phiTM(l,b)*coax2.phiTEM_dr(b) #here there is no integration over r
                    outer_integral= a*a_phi*coax1.phiTM(l,a)*coax2.phiTEM_dr(a)
                    C[l,k]=inner_integral+outer_integral
                
                continue
                        
                
                
            #in this part of the loop, neither mode is TEM

            if coax1.TE[l] and coax2.TE[k]: #TE-TE case
                x1=chi_1/b1 #following notation in that paper
                x2=chi_2/b2
                if a1==a2 and b1==b2: #test if waveguides are identical to avoid division by zero in the following
                    if chi_1==chi_2:
                        C[l,k]=1 #if we are at the same root, the coupling is one. otherwise it is zero, and we should just continue
                    continue
                vol_prefactor=a_phi*coax1.T[l]**2/(x1**2-x2**2)
                vol_integral1=a*(x1*coax1.phiTE(l,a,increase_m=1)*coax2.phiTE(k,a) - x2*coax1.phiTE(l,a)*coax2.phiTE(k,a,increase_m=1))
                vol_integral2=b*(x1*coax1.phiTE(l,b,increase_m=1)*coax2.phiTE(k,b) - x2*coax1.phiTE(l,b)*coax2.phiTE(k,b,increase_m=1))
                vol_integral=(vol_integral1-vol_integral2)*vol_prefactor
                
                surf_integral1 =a*a_phi*coax1.phiTE_dr(l,a)*coax2.phiTE(k,a)
                surf_integral2 =-b*a_phi*coax1.phiTE_dr(l,b)*coax2.phiTE(k,b)
                surf_integral=surf_integral1+surf_integral2
        
                C[l,k]=vol_integral+surf_integral
                #print('new formula difference: %f' %(C[l,k]-oldformula))
                continue

            if coax1.TM[l] and coax2.TM[k]: #TM-TM case
                x1=chi_1/b1 #following notation in that paper
                x2=chi_2/b2
                if a1==a2 and b1==b2: #test if waveguides are identical to avoid division by zero in the following
                    if chi_1==chi_2:
                        C[l,k]=1 #if we are at the same root, the coupling is one. otherwise it is zero, and we should just continue
                    continue
                vol_prefactor=a_phi*coax2.T[k]**2/(x1**2-x2**2)
                vol_integral1=a*(x1*coax1.phiTM(l,a,increase_m=1)*coax2.phiTM(k,a) - x2*coax1.phiTM(l,a)*coax2.phiTM(k,a,increase_m=1))
                vol_integral2=b*(x1*coax1.phiTM(l,b,increase_m=1)*coax2.phiTM(k,b) - x2*coax1.phiTM(l,b)*coax2.phiTM(k,b,increase_m=1))
                vol_integral=(vol_integral1-vol_integral2)*vol_prefactor
                
                surf_integral1 =a*a_phi*coax1.phiTM(l,a)*coax2.phiTM_dr(k,a)
                surf_integral2 =-b*a_phi*coax1.phiTM(l,b)*coax2.phiTM_dr(k,b)
                surf_integral=surf_integral1+surf_integral2
                
                C[l,k]=vol_integral+surf_integral
                continue

            if coax1.TM[l] and coax2.TE[k]:
                outer_integral = - a_phi*a*m*coax1.phiTM(l,a)*coax2.phiTE(k,a)
                inner_integral =  a_phi*b*m*coax1.phiTM(l,b)*coax2.phiTE(k,b)
                C[l,k] = inner_integral + outer_integral 
                continue

            if coax1.TE[l] and coax2.TM[k]:
                #this formula is implemented under the assumption that the coaxial mode has cosine polarization. the induced circular mode will then have sine polarization. I think the sign changes if the two polarizations were switched.
                inner_integral= -b*(m)*a_phi * coax1.phiTE(l,b)*coax2.phiTM(k,b)
                outer_integral= a*(m)*a_phi * coax1.phiTE(l,a)*coax2.phiTM(k,a)
                C[l,k]=inner_integral+outer_integral

            #that should be all
                
    return C
    
    
    
    
    
    
    
    
def Circ_Circ_CouplingMatrix(WG1,WG2):
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
            n = smallWG.orders[l] #finds the order of the Bessel functions corresponding to the angular dependence, i.e. the cos(n*phi)
            if n != largeWG.orders[k]: #if the two modes have different orders they are orthogonal; no need for calculation
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
    Z2[(circ.TE==0)*(circ.TM==0)]=Z0 #set impedance of TEM mode
    Y1=np.diag(1/Z1)
    Y2=np.diag(1/Z2) #not actually necessary
    Z1=np.diag(Z1)
    Z2=np.diag(Z2)
    C21=C12.T    
    

    
    
    #my derivation - in the circular case, this works only when r1 < r2
    #A=np.concatenate((np.concatenate((-C21@np.sqrt(Z1),np.sqrt(Z2)),axis=1),np.concatenate((np.sqrt(Y1),C12@np.sqrt(Y2)),axis=1)),axis=0)
    #B=np.concatenate((np.concatenate((C21@np.sqrt(Z1),-np.sqrt(Z2)),axis=1),np.concatenate((np.sqrt(Y1),C12@np.sqrt(Y2)),axis=1)),axis=0)
    
    
    #Michael's derivation - one step back from final matrix - this works only when r1>r2
    #A=np.concatenate((np.concatenate((np.sqrt(Z1),-C12@np.sqrt(Z2)),axis=1),np.concatenate((C21@np.sqrt(Y1),np.sqrt(Y2)),axis=1)),axis=0)
    #B=np.concatenate((np.concatenate((-np.sqrt(Z1),C12@np.sqrt(Z2)),axis=1),np.concatenate((C21@np.sqrt(Y1),np.sqrt(Y2)),axis=1)),axis=0)
    
    S=np.linalg.inv(B)@A
    return S


def computeSMatrix2(wg1,wg2,frequency,Cs=None): #frequency in Hz. 
    #Cs is passed as argument to avoid repeat calculations. it consists of a list of 4 matrices
    k=2*np.pi*frequency/299792458 # free space wavenumber
    #Impedance and admittance matrices
    eps0=scipy.constants.epsilon_0
    mu0=scipy.constants.mu_0
    Z0=np.sqrt(mu0/eps0) #normalization
    if np.all(Cs==None):
        Cs = get_coupling_matrix(wg1,wg2)

    
    CU1=Cs[0]
    CU2=Cs[1]
    CJ1=Cs[2]
    CJ2=Cs[3]
        
    
    
    #formulas: ZTE/Z0 = jk0 / gamma ; ZTM/Z0 = gamma / jk0
    
    #propagation constants
    gamma1=np.sqrt(wg1.T**2-k**2,dtype=complex)
    gamma2=np.sqrt(wg2.T**2-k**2,dtype=complex)
    
    Z1=Z0*(wg1.TE*1j*k/gamma1 + wg1.TM*gamma1/(1j*k))
    Z2=Z0*(wg2.TE*1j*k/gamma2 + wg2.TM*gamma2/(1j*k))
    Z1[(wg1.TE==0)*(wg1.TM==0)]=Z0 #set impedance of TEM mode
    Z2[(wg2.TE==0)*(wg2.TM==0)]=Z0 #set impedance of TEM mode
    Y1=np.diag(1/Z1)
    Y2=np.diag(1/Z2) #not actually necessary
    Z1=np.diag(Z1)
    Z2=np.diag(Z2)
    
    
    #attempt at most general formula
    A=np.concatenate((np.concatenate((CU1@np.sqrt(Z1),-CU2@np.sqrt(Z2)),axis=1),np.concatenate((CJ1@np.sqrt(Y1),CJ2@np.sqrt(Y2)),axis=1)),axis=0)
    B=np.concatenate((np.concatenate((-CU1@np.sqrt(Z1),CU2@np.sqrt(Z2)),axis=1),np.concatenate((CJ1@np.sqrt(Y1),CJ2@np.sqrt(Y2)),axis=1)),axis=0)
    
    S=np.linalg.inv(B)@A
    
    return S

def dB(S):
    return 20*np.log10(np.abs(S))

def get_coupling_matrix(wg1,wg2): #function to return all four general coupling matrices
    #These are in order: CU1, CU2, CJ1, CJ2. 
    
    #check if waveguide orders are specified
    
    numModesU=max(wg1.numModes,wg2.numModes) #integral over largest cross section should contain the largest number of modes
    numModesJ=min(wg1.numModes,wg2.numModes)
    
    if wg1.order_specified:
            order1 = int(wg1.orders[0])
    else:
        order1=None
    if wg2.order_specified:
        order2=int(wg2.orders[0])
    else:
        order2=None
    
    
    if wg1.WG_type == 'circular' and wg2.WG_type == 'circular':
        #calculate circ circ
        #simplest case. the coupling matrix function integrates over the small cross section. 
        r1=wg1.r
        r2=wg2.r
        ru=max(r1,r2)
        rj=min(r1,r2)
        
        if order1==order2:
        #here wgU and wgJ can be the same, so this is calculated separately to save time
            wgU1=waveguide_circ(ru,numModesU,order1)
            wgU2=wgU1
            wgJ1=waveguide_circ(rj,numModesJ,order1)
            wgJ2=wgJ1
        else:
            wgU1=waveguide_circ(ru,numModesU,order1)
            wgU2=waveguide_circ(ru,numModesU,order2)
            wgJ1=waveguide_circ(rj,numModesJ,order1)
            wgJ2=waveguide_circ(rj,numModesJ,order2)
        
        CU1=Circ_Circ_CouplingMatrix(wgU1,wg1)
        CU2=Circ_Circ_CouplingMatrix(wgU2,wg2)
        CJ1=Circ_Circ_CouplingMatrix(wgJ1,wg1)
        CJ2=Circ_Circ_CouplingMatrix(wgJ2,wg2)
        
        
        
    elif wg1.WG_type == 'circular' and wg2.WG_type == 'coaxial':
        r=wg1.r
        a=wg2.a
        
        bj=wg2.b
        ru=max(r,a)
        aj=min(r,a)
        
        if order1==order2:
        #here wgU and wgJ can be the same, so this is calculated separately to save time
            wgU1=waveguide_circ(ru,numModesU,order1)
            wgU2=wgU1
            wgJ1=waveguide_coax(aj,bj,numModesJ,order1)
            wgJ2=wgJ1
        else:
            wgU1=waveguide_circ(ru,numModesU,order1)
            wgU2=waveguide_circ(ru,numModesU,order2)
            wgJ1=waveguide_coax(aj,bj,numModesJ,order1)
            wgJ2=waveguide_coax(aj,bj,numModesJ,order2)
        
        CU1=Circ_Circ_CouplingMatrix(wgU1,wg1)
        CU2=Coax_Circ_CouplingMatrix(wg2,wgU2).T #the function takes the coaxial line as the first argument; so we compute it like this, and then transpose the matrix to switch them
        CJ1=Coax_Circ_CouplingMatrix(wgJ1,wg1)
        CJ2=Coax_Coax_CouplingMatrix(wgJ2,wg2)
        
    elif wg1.WG_type == 'coaxial' and wg2.WG_type == 'circular':
        r=wg2.r
        a=wg1.a
        bj=wg1.b
        ru=max(r,a)
        aj=min(r,a)
        
        if order2==order1:
        #here wgU and wgJ can be the same, so this is calculated separately to save time
            wgU2=waveguide_circ(ru,numModesU,order2)
            wgU1=wgU2
            wgJ2=waveguide_coax(aj,bj,numModesJ,order2)
            wgJ1=wgJ2
        else:
            wgU2=waveguide_circ(ru,numModesU,order2)
            wgU1=waveguide_circ(ru,numModesU,order1)
            wgJ2=waveguide_coax(aj,bj,numModesJ,order2)
            wgJ1=waveguide_coax(aj,bj,numModesJ,order1)
        
        CU2=Circ_Circ_CouplingMatrix(wgU2,wg2)
        CU1=Coax_Circ_CouplingMatrix(wg1,wgU1).T
        CJ2=Coax_Circ_CouplingMatrix(wgJ2,wg2)
        CJ1=Coax_Coax_CouplingMatrix(wgJ1,wg1)
        
    elif wg1.WG_type == 'coaxial' and wg2.WG_type == 'coaxial':
        
        a1=wg1.a
        b1=wg1.b
        a2=wg2.a
        b2=wg2.b
        au=max(a1,a2) #outer radius of union coax
        bu=min(b1,b2)
        aj=min(a1,a2) #outer radius of intersection coax
        bj=max(b1,b2)
        
        if order1==order2:
        #here wgU and wgJ can be the same, so this is calculated separately to save time
            wgU1=waveguide_coax(au,bu,numModesU,order1)
            wgU2=wgU1
            wgJ1=waveguide_coax(aj,bj,numModesJ,order1)
            wgJ2=wgJ1
        else:
            wgU1=waveguide_coax(au,bu,numModesU,order1)
            wgU2=waveguide_coax(au,bu,numModesU,order2)
            wgJ1=waveguide_coax(aj,bj,numModesJ,order1)
            wgJ2=waveguide_coax(aj,bj,numModesJ,order2)
        
        CU1=Coax_Coax_CouplingMatrix(wgU1,wg1)
        CU2=Coax_Coax_CouplingMatrix(wgU2,wg2)
        CJ1=Coax_Coax_CouplingMatrix(wgJ1,wg1)
        CJ2=Coax_Coax_CouplingMatrix(wgJ2,wg2)
        
        
    else: 
        print('Unknown waveguide type')
        return None
    
    return [CU1,CU2,CJ1,CJ2]
    

def get_S_parameters(wg1,wg2,fstart,fend,Nsteps,mode1,mode2):
    fs=np.linspace(fstart,fend,Nsteps)
    L=wg1.numModes
    K=wg2.numModes
    Cs=get_coupling_matrix(wg1,wg2)
    S11=np.zeros(Nsteps,dtype=complex)
    S21=np.zeros(Nsteps,dtype=complex)
    S22=np.zeros(Nsteps,dtype=complex)
    S12=np.zeros(Nsteps,dtype=complex)
    for count,f in enumerate(fs):
        S=computeSMatrix2(wg1,wg2,f,Cs)
        S11[count]=S[mode1,mode1]
        S21[count]=S[L+mode2,mode1]
        S12[count]=S[mode1,L+mode2]
        S22[count]=S[L+mode2,L+mode2]  


    return [S11,S12,S21,S22]  