#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from numpy import linalg as LA
import pygmo as pg
import Single_dist_ref_SEIR as SDSEIR
import pandas as pd


"""
    Testing pygmo applied into SEIR Class
"""
# To do: 
#  - Revisar una mejor construccion del objeto SEIRModel, usaria el objeto seir que hicimos al principio
class SEIRModel:
    def __init__(self,Ir,tr,S0,E0,I0,R0,h,mov,qp,movefunct,bounds):
        self.Ir = Ir
        self.tr = tr
        self.S0 = S0
        self.E0 = E0
        self.I0 = I0
        self.R0 = R0
        self.h = h
        self.mov = mov
        self.qp = qp
        self.movefunct = movefunct
        self.bounds = bounds
    def fitness(self,x):        
        self.E0=x[3]*I0
        sol=pd.DataFrame(SDSEIR.intger(self.S0,self.E0,self.I0,self.R0,min(tr),max(tr),self.h,x[0],x[1],x[2],self.mov,self.qp,tr[-1],self.movfunct))
        idx=np.searchsorted(sol.t,self.tr)
        res = LA.norm(self.Ir-sol.I[idx])        
        return(res)

    def get_bounds(self):
        return(self.bounds)

    def set_bounds(self,bounds):
        self.bounds = bounds
        return(self.bounds)




class sphere_function:
    def __init__(self, dim):
         self.dim = dim  
    def fitness(self,x): 
        return [sum(x*x)] 
    def get_bounds(self): 
        return ([-1] * self.dim, [1] * self .dim)
    def get_name(self):
        return "Sphere Function"  
    def get_extra_info(self):
        return "\tDimensions: " + str(self.dim)

algo = pg.algorithm(pg.pso(gen = 20))
pop = pg.population(prob,20)
pop = algo.evolve(pop)
print(pop.champion_f)