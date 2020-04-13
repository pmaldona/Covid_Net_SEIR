#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 

def Matriz_Movilidad_alpha(P,k):
    dim=P.shape
    if k is not 0:
        Alpha=np.zeros((dim[0],dim[1]))
        for i in range(dim[0]):
            if i==k:
                Alpha[i,:]=1
                for j in range(dim[1]):
                    if j==1:
                        Alpha[:,j]=1
        return np.eye(dim[0],dim[1]) + Alpha.dot(P)
    else:
        return np.eye(dim[0],dim[1]) + np.eye(dim[0],dim[1]).dot(P)
    
def Matriz_Movilidad(P):
    dim=P.shape
    return np.eye(dim[0],dim[1]) + np.eye(dim[0],dim[1]).dot(P)

