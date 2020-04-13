
import class_SEIR as 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


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
b_r=[0.01,0.5] #0.1
s_r=[0.01,0.5] #0.1
g_r=[0.01,0.5] #0.1
mu_r=[0.5,4] #2


# Strategy functions
def alpha(t):
    return(np.ones([34,34])-np.eye(34))

def eta(t):
    return(np.ones(34))

# Object init
test = SEIR(P,eta,alpha,S0,I0,E0,R0,min(tr),max(tr),h,b_r,g_r,s_r,mu_r)


# Run integr
x=test.integr_RK4(test.t0,test.T,test.h,test.beta,test.sigma,test.gamma,test.mu,False,True)

#run mesh

parms=mesh(test,Ir,tr,5,5,0.025,20)

# Report results
plt.figure()
plt.plot(test.t,test.S[0,:],label='Susceptible')
plt.plot(test.t,test.E[0,:],label='Exposed')
plt.plot(test.t,test.I[0,:],label='Infected simulation')
plt.plot(test.t,test.R[0,:],label='Removed')
plt.xlabel('Days')
plt.ylabel('Population')
plt.title('COVID-19 Model')
plt.legend(loc=0)
plt.show()
