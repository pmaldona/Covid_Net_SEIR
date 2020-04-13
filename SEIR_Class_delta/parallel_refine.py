
import class_SEIR as S
import param_finder as p
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
from multiprocessing import Process


#Import Data

So = pd.read_excel("../Data/poblacion_Inicial_S_stgo.xlsx", header=None).to_numpy()
S0 = So[:,0]
Eo = pd.read_excel("../Data/poblacion_Inicial_E_stgo.xlsx", header=None).to_numpy()
E0 = Eo[:,0]
Io = pd.read_excel("../Data/poblacion_Inicial_I_stgo.xlsx", header=None).to_numpy()
I0 = Io[:,0]
Ro = pd.read_excel("../Data/poblacion_Inicial_R_stgo.xlsx", header=None).to_numpy()
R0 = Ro[:,0]

P = pd.read_excel("../Data/connectivity_stgo2.xlsx", header=None).to_numpy()

Ir=pd.read_excel("../Data/Simulacion-400dias-I.xlsx", header=None).to_numpy()

#Init variables
tr=np.arange(Ir.shape[1])
h=0.1

# Parameter range
b_r=[0.1,0.3] #0.1
s_r=[0.05,0.15] #0.1
g_r=[0.05,0.15] #0.1
mu_r=[1,3] #2


# Strategy functions
def alpha(t):
    return(np.ones([34,34])-np.eye(34))

def eta(t):
    return(np.ones(34))

# Object init
test = S.SEIR(P,eta,alpha,S0,E0,I0,R0,min(tr),max(tr),h,b_r,g_r,s_r,mu_r)

#parms=p.pso_opt(test,Ir,tr) # 

#mesh test
def thread_function(name):

    params = p.pso_opt(test,Ir,tr,omega=0.5, phip=0.5, phig=0.5,swarmsize=60,maxiter=30)
    df = pd.DataFrame(params)
    df.to_csv(str(name)+".csv", index=False)


for i in range(12):
    mp = Process(target=thread_function, args=(i,))
    mp.start()


#
## Run integr
#test.integr_RK4(test.t0,test.T,test.h,test.beta,test.sigma,test.gamma,test.mu,True,True)
#
#test.S[1,idx]-S_su1[1,:]
#
#
#
#
#S_su1 = pd.read_excel("../Data/Simulacion-400dias-S.xlsx", header=None).to_numpy()
#E_su1 = pd.read_excel("../Data/Simulacion-400dias-E.xlsx", header=None).to_numpy()
#I_su1 = pd.read_excel("../Data/Simulacion-400dias-I.xlsx", header=None).to_numpy()
#R_su1 = pd.read_excel("../Data/Simulacion-400dias-R.xlsx", header=None).to_numpy()
#
#idx=np.searchsorted(test.t,tr)
#
#S_dif=S_su1-test.S[:,idx]
#E_dif=E_su1-test.E[:,idx]
#I_dif=I_su1-test.I[:,idx]
#R_dif=R_su1-test.R[:,idx]
#
#np.amax(S_dif)
#np.amax(E_dif)
#np.amax(I_dif)
#np.amax(R_dif)
#
#plt.figure()
#plt.plot(test.t[idx],S_dif[1,:],label='Susceptible')
#plt.plot(test.t[idx],E_dif[1,:],label='Exposed')
#plt.plot(test.t[idx],I_dif[1,:],label='Infected simulation')
#plt.plot(test.t[idx],R_dif[1,:],label='Removed')
#plt.xlabel('Days')
#plt.ylabel('Population')
#plt.title('COVID-19 Model')
#plt.legend(loc=0)
#plt.show()
#
#plt.figure()
#plt.plot(test.t[idx],test.S[1,idx],label='Susceptible')
#plt.plot(test.t[idx],test.E[1,idx],label='Exposed')
#plt.plot(test.t[idx],test.I[1,idx],label='Infected simulation')
#plt.plot(test.t[idx],test.R[1,idx],label='Removed')
#plt.xlabel('Days')
#plt.ylabel('Population')
#plt.title('COVID-19 Model')
#plt.legend(loc=0)
#plt.show()
#
#
#plt.figure()
#plt.plot(test.t[idx],S_su1[1,:],label='Susceptible')
#plt.plot(test.t[idx],E_su1[1,:],label='Exposed')
#plt.plot(test.t[idx],I_su1[1,:],label='Infected simulation')
#plt.plot(test.t[idx],R_su1[1,:],label='Removed')
#plt.xlabel('Days')
#plt.ylabel('Population')
#plt.title('COVID-19 Model')
#plt.legend(loc=0)
#plt.show()
#
#
##run mesh
#
##run mesh
#
#
#
## Report results
