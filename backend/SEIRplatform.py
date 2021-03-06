#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
from pathlib import Path
home = str(Path.home())                                                                                                                                                                                 
sys.path.insert(1, home+'/Covid_Net_SEIR/SEIRHVD/') 
sys.path.insert(1, home+'/Covid_Net_SEIR/SEIRHVDB/') 
sys.path.insert(1, home+'/Covid_Net_SEIR/SEIRUNI/') 
sys.path.insert(1, home+'/Covid_Net_SEIR/SEIRMULTI/') 
sys.path.insert(1, home+'/Covid_Net_SEIR/SEIR_Class_beta/')
from functions_SEIRHDV import SEIRHVD_DA
from class_SEIR import SEIR
from  SEIRrefiner import SEIRrefiner 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from timeit import default_timer as timer
import Single_dist_ref_SEIR as SDSEIR
import Multi_dist_ref_SEIR as MDSEIR
from datetime import datetime

import logging
import json

from flask import Flask
from flask import jsonify
from flask import request
from flask import send_file
from flask_cors import CORS


import dill as pickle


app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

"""
# --------------------------- #
#      SEIR Functions         #   
# --------------------------- #
"""
def alpha(t,alpha=1):
    return(alpha*np.ones([t,t]))

def eta(t,eta=1):
    return(eta*np.ones(t))


"""
# ------------------------- #
#     GUI communication     #
# ------------------------- #
"""

@app.route('/health_check', methods=['GET'])
def health_check():
    '''
    health_check
    '''
    app.logger.info("health_check")
    response = {'status': 'OK'}
    return jsonify(response), 200

@app.route('/zeus_input', methods=['POST'])
def zeus_input():
    '''
    http://192.168.2.223:5003/zeus_input?campo=1
    '''
    try: 
        campo1 = request.form.get('campo1')#request.args.get('campo1')
        
        campo2 = request.form.get('campo2')# request.args.get('campo2')
        campo3 = request.form.get('campo3')

        result = int(campo1)*int(campo2)
        print(result) 
        response = {'status': 'OK','result' : result,'campo3': str(type(campo3))}

        return jsonify(response), 200
    except:
        response = {"error": "la cagaste :D"}
        return response, 200    

@app.route('/refineuni', methods=['POST'])
def refineuni():
    '''
    # ----------------------------------------------------------- #
    #     Parameters input when the simulate button is pressed    #
    # ----------------------------------------------------------- #
    '''
    print("Refine")
    try: 
        # Manage input parameters
        if request.form:
            print("I have form data")
            state = request.form.get('state') #State (Region)
            comuna = request.form.get('comuna') # District 
            qp = int(request.form.get('qp')) # Quarantine period
            mov = float(request.form.get('mov')) # Quarantine movilty
            tsim = int(request.form.get('tSim')) # Simulation time
            movfunct = str(request.form.get('movfunct'))    

        if request.json:
            print("I have json")        
            state = request.json['state']
            comuna = request.json['comuna']
            qp = int(request.json['qp'])
            mov = float(request.json['mov']) # Quarantine movilty
            tsim = int(request.json['tSim'])
            movfunct = str(request.json['movfunct'])
        
        #tsim = 100
        #mov = 0.2          
        if qp == -1:
            qp = tsim
             
        # ---------------------------- #
        #        Get Refine Data       #
        # ---------------------------- #
        path = '../Data/unirefine/'
        
        # Get data for different quarantine periods
        # Import data, parameters and initdate

        #parameters = pd.read_csv(path+'parameters_qp0.csv',index_col=0)
        #initdate = pd.read_csv(path+'initdate_qp0.csv',index_col=0)
        parameters = pd.read_csv(path+'parameters.csv',index_col=0)
        initdate = pd.read_csv(path+'initdate.csv',index_col=0)        

        ## Find data for paramters given
        parameters = parameters[comuna]
        initdate = initdate[comuna][0]
        results = SDSEIR.simulate(state,comuna,parameters[0],parameters[1],parameters[2],parameters[3],qp = qp,mov = mov,tsim = tsim, movfunct=movfunct)
        S = results['S'].tolist()
        E = results['E'].tolist()
        I = results['I'].tolist()
        R = results['R'].tolist()
        t = list(results['t'])

        # Round results:
        S = [round(i) for i in S]
        E = [round(i) for i in E]
        I = [round(i) for i in I]
        R = [round(i) for i in R]        
        
        Ti = 1/parameters[1]
        Tr = 1/parameters[2]

        #print(results)
        print("done simulating")
        print('beta,sigma,gama,mu')
        print(parameters)

        response = {'status': 'OK','S':S,'E':E,'I':I,'R':R,'t':t, 'Ti':Ti,'Tr':Tr,'beta':parameters[0],'r0':parameters[3],'initdate':initdate,'I_peak':max(I),'R_total':(max(R))}
        #print(response)
        return jsonify(response), 200

    except Exception as e: 
        print(e)
        response = {"error": str(e)}
        return response, 200


@app.route('/refineuni_test', methods=['POST'])
def refineuni_test():
    '''
    # ----------------------------------------------------------- #
    #     Parameters input when the simulate button is pressed    #
    # ----------------------------------------------------------- #
    '''
    print("Refine")
    try: 
        # Manage input parameters
        if request.form:
            print("I have form data")
            state = request.form.get('state') #State (Region)
            comuna = request.form.get('comuna') # District 
            qp = int(request.form.get('qp')) # Quarantine period
            mov = float(request.form.get('mov')) # Quarantine movilty
            tsim = int(request.form.get('tSim')) # Simulation time
            movfunct = str(request.form.get('movfunct'))    

        if request.json:
            print("I have json")        
            state = request.json['state']
            comuna = request.json['comuna']
            qp = int(request.json['qp'])
            mov = float(request.json['mov']) # Quarantine movilty
            tsim = int(request.json['tSim'])
            movfunct = str(request.json['movfunct'])
        
        #tsim = 100
        #mov = 0.2          
        if qp == -1:
            qp = tsim
             
        # ---------------------------- #
        #        Get Refine Data       #
        # ---------------------------- #
        path = '../Data/unirefine/'
        
        # Get data for different quarantine periods
        # Import data, parameters and initdate

        #parameters = pd.read_csv(path+'parameters_qp0.csv',index_col=0)
        #initdate = pd.read_csv(path+'initdate_qp0.csv',index_col=0)
        parameters = pd.read_csv(path+'parameters.csv',index_col=0)
        initdate = pd.read_csv(path+'initdate.csv',index_col=0)        

        ## Find data for paramters given
        parameters = parameters[comuna]
        initdate = initdate[comuna][0]
        results = SDSEIR.simulate(state,comuna,parameters[0],parameters[1],parameters[2],parameters[3],qp = qp,mov = mov,tsim = tsim, movfunct=movfunct)
        S = results['S'].tolist()
        E = results['E'].tolist()
        I = results['I'].tolist()
        R = results['R'].tolist()
        t = list(results['t'])

        # Round results:
        S = [round(i) for i in S]
        E = [round(i) for i in E]
        I = [round(i) for i in I]
        R = [round(i) for i in R]        
        
        Ti = 1/parameters[1]
        Tr = 1/parameters[2]

        #print(results)
        print("done simulating")
        print('beta,sigma,gama,mu')
        print(parameters)

        response = {'status': 'OK','S':S,'E':E,'I':I,'R':R,'t':t, 'Ti':Ti,'Tr':Tr,'beta':parameters[0],'r0':parameters[3],'initdate':initdate,'I_peak':max(I),'R_total':(max(R))}
        #print(response)
        return jsonify(response), 200

    except Exception as e: 
        print(e)
        response = {"error": str(e)}
        return response, 200


@app.route('/simulateuni', methods=['POST'])
def simulateuni():
    '''
    # ----------------------------------------------------------- #
    #     Parameters input when the simulate button is pressed    #
    # ----------------------------------------------------------- #
    '''
    # Manage input parameters
    try: 
        if request.form:
            print("I have form data")
            state = str(request.form.get('state'))
            comuna = str(request.form.get('comuna'))
            qp = int(request.form.get('qp'))
            beta = float(request.form.get('beta'))
            Ti = int(request.form.get('Ti')) #sigma-1
            Tr = int(request.form.get('Tr')) #gamma-1
            r0 = float(request.form.get('r0'))    
            mov = float(request.form.get('mov')) #Quarantine movilty
            tsim = int(request.form.get('tSim')) 
            movfunct = str(request.form.get('movfunct'))                   

        if request.json:
            print("I have json")        
            state = str(request.json['state'])
            comuna = str(request.json['comuna'])
            qp = int(request.json['qp'])
            beta = float(request.json['beta'])
            Ti = int(request.json['Ti']) #sigma-1
            Tr = int(request.json['Tr']) #gamma-1
            r0 = float(request.json['r0'])
            mov = float(request.json['mov'])
            tsim = int(request.json['tSim'])
            movfunct = str(request.json['movfunct'])
                     
        #mov = 0.2#request.form.get('aten')
        #tsim = 100
        #tci = request.form.get('tci')

        #print('Inputs')
        #print(state,comuna,qp,beta,Ti,Tr,r0,mov,tsim,movfunct)
        #state, comuna, beta, Ti,Tr,r0,Q
        if Ti==0:
            sigma = 1
        else:
            sigma = 1/Ti

        if Tr==0:
            gamma = 1
        else:
            gamma = 1/Tr
        #{"S":S,"E":E,"I":I,"R":R,"t":t}
        print('state,comuna,beta,sigma,gamma,r0,qp,mov,tsim')
        print(state,comuna,beta,sigma,gamma,r0,qp,mov,tsim)
        results = SDSEIR.simulate(state,comuna,beta,sigma,gamma,r0,qp,mov,tsim,movfunct=movfunct)
        S = results['S'].tolist()
        E = results['E'].tolist()
        I = results['I'].tolist()
        R = results['R'].tolist()
        t = list(results['t'])
        init_date = results['init_date']

        # Round results:
        S = [round(i) for i in S]
        E = [round(i) for i in E]
        I = [round(i) for i in I]
        R = [round(i) for i in R]
        a = pd.DataFrame(results)
        a.to_csv('lastrequest.csv')

        response = {'status': 'OK','S':S,'E':E,'I':I,'R':R,'t':t,'init_date':init_date,'I_peak':max(I),'R_total':(max(R))}
        print(response.keys())
        return response, 200
    except Exception as e:
        print(e)
        response = {"error": str(e)}
        return response, 200

@app.route('/SEIR2refine', methods=['GET'])
def SEIR2refine():
    '''
    # ----------------------------------------------------------- #
    #           Parameters Refine for intercomunal model          #
    # ----------------------------------------------------------- #
    '''
    print("Refine")
    try: 
        # Manage input parameters
        if request.form:
            print("I have form data")
            state = request.form.get('state') #State (Region)
            comuna = request.form.get('comuna') # District 
            qp = int(request.form.get('qp')) # Quarantine period
            mov = float(request.form.get('mov')) # Quarantine movilty
            tsim = int(request.form.get('tSim')) # Simulation time
            movfunct = str(request.form.get('movfunct'))    

        if request.json:
            print("I have json")        
            state = request.json['state']
            comuna = request.json['comuna']
            qp = int(request.json['qp'])
            mov = float(request.json['mov']) # Quarantine movilty
            tsim = int(request.json['tSim'])
            movfunct = str(request.json['movfunct'])
        
        #tsim = 100
        #mov = 0.2          
        if qp == -1:
            qp = tsim
             
        # ---------------------------- #
        #        Get Refine Data       #
        # ---------------------------- #
        path = '../Data/unirefine/'
        
        # Get data for different quarantine periods
        # Import data, parameters and initdate

        #parameters = pd.read_csv(path+'parameters_qp0.csv',index_col=0)
        #init_date = pd.read_csv(path+'initdate_qp0.csv',index_col=0)
        parameters = pd.read_csv(path+'parameters.csv',index_col=0)
        init_date = pd.read_csv(path+'initdate.csv',index_col=0)        

        ## Find data for paramters given
        parameters = parameters[comuna]
        init_date = init_date[comuna][0]
        results = SDSEIR.simulate(state,comuna,parameters[0],parameters[1],parameters[2],parameters[3],qp = qp,mov = mov,tsim = tsim, movfunct=movfunct)
        S = results['S'].tolist()
        E = results['E'].tolist()
        I = results['I'].tolist()
        R = results['R'].tolist()
        t = list(results['t'])
        
        Ti = 1/parameters[1]
        Tr = 1/parameters[2]

        # Round results:
        S = [round(i) for i in S]
        E = [round(i) for i in E]
        I = [round(i) for i in I]
        R = [round(i) for i in R]

        #print(results)
        print("done simulating")    

        response = {'status': 'OK','S':S,'E':E,'I':I,'R':R,'t':t, 'Ti':Ti,'Tr':Tr,'beta':parameters[0],'r0':parameters[3],'init_date':init_date,'I_peak':max(I),'R_total':(max(R))}
        #print(response)
        return jsonify(response), 200

    except Exception as e: 
        print(e)
        response = {"error": str(e)}
        return response, 200


@app.route('/SEIR2simulate', methods=['GET'])
def SEIR2simulate():
    '''
    # ----------------------------------------------------------- #
    #             Multidistricts SEIR Model simulation            #
    # ----------------------------------------------------------- #
    '''
    
    # ------------------------ #
    #      Input parameters    #
    # ------------------------ #
    try: 
        if request.form:
            print("I have form data")
            state = str(request.form.get('state'))
            urbancenter = str(request.form.get('urbancenter'))
            qp = int(request.form.get('qp'))
            beta = float(request.form.get('beta'))
            Ti = int(request.form.get('Ti')) #sigma-1
            Tr = int(request.form.get('Tr')) #gamma-1
            r0 = float(request.form.get('r0'))    
            mov = float(request.form.get('mov')) #Quarantine movilty
            tsim = int(request.form.get('tSim')) 
            movfunct = str(request.form.get('movfunct'))                   

        if request.json:
            print("I have json")        
            state = str(request.json['state'])
            urbancenter = str(request.json['urbancenter'])
            qp = int(request.json['qp'])
            beta = float(request.json['beta'])
            Ti = int(request.json['Ti']) #sigma-1
            Tr = int(request.json['Tr']) #gamma-1
            r0 = float(request.json['r0'])
            mov = float(request.json['mov'])
            tsim = int(request.json['tSim'])
            movfunct = str(request.json['movfunct'])

        # Get Comunas from OD 
        # falta identificar centros urbanos     
        comunas = []
        if qp ==-1:
            qp = tsim

        if Ti==0:
            sigma = 1
        else:
            sigma = 1/Ti

        if Tr==0:
            gamma = 1
        else:
            gamma = 1/Tr    
        tci = None
        #params = beta sigma gamma mu
        params = [beta,sigma,gamma,r0] 
        
        # inputs: state, centrourbano, beta, sigma, gamma, mu,qp, mov,tsim, tci, movfunct 
        # outputs: {'S':sim.S,'E':sim.E,'I':sim.I,'R':sim.R,'tout':sim.t}
        results = MDSEIR.simulate_multi(state,comunas,params,qp,mov,tsim,tci,movfunct)
        S = results['S'].tolist()
        E = results['E'].tolist()
        I = results['I'].tolist()
        R = results['R'].tolist()
        t = list(results['tout'])

        # Round results:
        S = [round(i) for i in S]
        E = [round(i) for i in E]
        I = [round(i) for i in I]
        R = [round(i) for i in R]
                
        init_date = results['init_date']
        response = {'status': 'OK','S':S,'E':E,'I':I,'R':R,'t':t,'init_date':init_date,'I_peak':max(I),'R_total':(max(R))}
        return jsonify(response), 200

    # Solve
    #integr(self,t0,T,h,E0init=False)
    except Exception as e:
        print(e)
        response = {'status': 'OK',"error": str(e)}
        return response, 200

'''
# ----------------------------------------------- #
#                  SEIRHVD                        #
# ----------------------------------------------- #
'''

@app.route('/SEIRHVDsimulate', methods=['GET'])
def SEIRHVDsimulate():
    '''
    # ----------------------------------------------- #
    #             SEIRHVD Model simulation            #
    # ----------------------------------------------- #
    # Params needed:
    #   - state = region por cut
    #   - beta
    #   - mu
    #   - ScaleFactor
    #   - SeroPrevFactor
    #   - expinfection
    #   - initdate
    #   - tasas (no todavia)
    #   - Objeto de cuarentena 
    #  

    tstate = '13'

    # Parámetros de tiempo
    initdate = datetime(2020,5,15)


    # Ajuste por ventiladores
    # Parametros del modelo
    beta = 0.117 #0.25#0.19 0.135
    mu = 0.6 #2.6 0.6
    ScaleFactor = 1.9 #4.8
    SeroPrevFactor = 0.5 # Sero Prevalence Factor. Permite ajustar la cantidad de gente que entra en la dinamica
    expinfection = 1 # Proporcion en la que contagian los expuestos

    tsim = 500

    simulation = SEIRHVD_DA(beta,mu,ScaleFactor=ScaleFactor,SeroPrevFactor=SeroPrevFactor,expinfection=expinfection,initdate = initdate, tsim = tsim,tstate=tstate)
    simulation.importdata()
    simulation.escenarios()
    simulation.simulate()

    '''
    
    # ------------------------ #
    #      Input parameters    #
    # ------------------------ #
    try: 
        if request.form:
            print("I have form data")
            state = str(request.form.get('state')) # Region            
            beta = float(request.form.get('beta'))
            mu = float(request.form.get('mu')) #mu 
            tsim = int(request.form.get('tSim'))  # Simulation time
            initdate = str(request.form.get('initdate'))  # Simulation time
            ScaleFactor = float(request.form.get('ScaleFactor')) #Scale Factor
            SeroPrevFactor = float(request.form.get('SeroPrevFactor')) #Sero prevalence factor

            qp = int(request.form.get('qp')) # Quarantine period            
            min_mov = float(request.form.get('min_mov')) #Quarantine remament movlity
            max_mov = float(request.form.get('max_mov')) #Quarantine maximum movilty            
            movfunct = str(request.form.get('movfunct')) # Quarantine Type
            qit = int(request.form.get('qit')) #Quarantine initial time
            qft = int(request.form.get('qit')) #Quarantine initial time

        if request.json:
            print("I have json")        
            state = str(request.json['state'])
            beta = float(request.json['beta'])
            mu = float(request.json['mu'])           
            tsim = int(request.json['tSim'])
            initdate = str(request.json['initdate'])
            ScaleFactor = float(request.json['ScaleFactor'])
            SeroPrevFactor = float(request.json['SeroPrevFactor'])

            qp = int(request.json['qp'])                        
            min_mov = float(request.json['min_mov'])
            max_mov = float(request.json['max_mov'])
            movfunct = str(request.json['movfunct'])
            qit = int(request.json['qit'])
            qft = int(request.json['qft'])

        params = [beta,mu,state,tsim,initdate,ScaleFactor,SeroPrevFactor]
        initdate = datetime.strptime(initdate, "%Y/%m/%d")# pej 2020/04/03 
        expinfection = 1
        
        simulation = SEIRHVD_DA(beta,mu,ScaleFactor=ScaleFactor,SeroPrevFactor=SeroPrevFactor,expinfection=expinfection,initdate = initdate, tsim = tsim,tstate=state)
        simulation.importdata()
        simulation.escenarios()
        simulation.simulate()
        
        # Reduce data to send it as json
        #poplist = ['inputarray','Hcmodel','Vcmodel','initdate','Br','Br_dates', 'Br_tr', 'ED_RM_df','ED_RM', 'ED_RM_dates', 'ED_RM_ac', 'ED_tr', 'Ir', 'Ir_dates', 'tr', 'Hr', 'Vr', 'Vr_tot', 'Hr_tot', 'sochimi_dates', 'sochimi_tr']
        #
        #for key in simulation.__dict__: 
        #    if 'pandas' in str(type(simulation.__dict__[key])): 
        #        print(type(simulation.__dict__[key])) 
        #        print(key) 
        #        poplist.append(key) 
        #for i in poplist:
        #    simulation.__dict__.pop(i)
#
        #simulation.initdate = simulation.initdate.strftime('%Y/%m/%d')      
        # Extract Results



        a = pickle.dumps(vars(simulation))  
        # Send results
        #response = {'status': 'OK','S':S,'E':E,'I':I,'R':R,'t':t,'init_date':init_date,'I_peak':max(I),'R_total':(max(R))}
        #return jsonify(response), 200
        return send_file(a), 200

    # Solve
    #integr(self,t0,T,h,E0init=False)
    except Exception as e:
        print(e)
        response = {'status': 'OK',"error": str(e)}
        return response, 200


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5003, debug=True)
else:
    # setup logging using gunicorn logger
    formatter = logging.Formatter(
        '[%(asctime)s.%(msecs)03d] [%(name)s] [%(levelname)s] - %(message)s',
        '%d-%m-%Y %H:%M:%S'
    )
    gunicorn_logger = logging.getLogger('gunicorn.error')
    app.logger.handlers = gunicorn_logger.handlers
    app.logger.handlers[0].setFormatter(formatter)
    app.logger.setLevel(logging.DEBUG)



