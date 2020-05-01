#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from class_SEIR import SEIR
from  SEIRrefiner import SEIRrefiner 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from timeit import default_timer as timer
import Single_dist_ref_SEIR as SDSEIR
import datetime

import logging
import json

from flask import Flask
from flask import jsonify
from flask import request
from flask_cors import CORS

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

        parameters = pd.read_csv(path+'parameters_qp0.csv',index_col=0)
        initdate = pd.read_csv(path+'initdate_qp0.csv',index_col=0)

        ## Find data for paramters given
        parameters = parameters[comuna]
        initdate = initdate[comuna][0]
        results = SDSEIR.simulate(state,comuna,parameters[0],parameters[1],parameters[2],parameters[3],qp = qp,mov = mov,tsim = tsim, movfunct=movfunct)
        S = results['S'].tolist()
        E = results['E'].tolist()
        I = results['I'].tolist()
        R = results['R'].tolist()
        t = list(results['t'])
        
        Ti = 1/parameters[1]
        Tr = 1/parameters[2]

        #print(results)
        print("done simulating")    

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
        if qp ==-1:
            qp = tsim

        print('Inputs')
        print(state,comuna,qp,beta,Ti,Tr,r0,mov,tsim,movfunct)
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
        results = SDSEIR.simulate(state,comuna,beta,sigma,gamma,r0,qp,mov,tsim,movfunct=movfunct)
        S = results['S'].tolist()
        E = results['E'].tolist()
        I = results['I'].tolist()
        R = results['R'].tolist()
        t = list(results['t'])
        init_date = results['init_date']
        response = {'status': 'OK','S':S,'E':E,'I':I,'R':R,'t':t,'init_date':init_date,'I_peak':max(I),'R_total':(max(R))}
        print(response.keys())
        return response, 200
    except Exception as e:
        print(e)
        response = {"error": str(e)}
        return response, 200

@app.route('/refine', methods=['GET'])
def refine():
    '''
    # ----------------------------------------------------------- #
    #     Parameters input when the simulate button is pressed    #
    # ----------------------------------------------------------- #
    '''
    # Manage input parameters
    params = json.loads(request.args.get('parameters'))
    print(params)
    
    # default value
    # sigma = 0.2
    # gamma = 0.07
    
    #diccionario = REFINEuni(params.state,params.cut,Qdt
    #output  IR, Tr, params=[beta,sigma,gamma,mu],err,sim={S,E,I,R,t}(dic),initdate (datetime)    


    # Get data from DB
    #Sr, Er, Ir, Rr, P = getdata(params.comuna)    
    # Build and run seir object
    # Eta es el periodo de cuarentena
    #tr=np.arange(Ir.shape[1])
    #h=0.1       
    # El T no aparece =S     
    response = {'status': 'OK','function':'refine parameters uni','params':params}
    return jsonify(response), 200


@app.route('/simulate', methods=['GET'])
def simulate():
    '''
    # ----------------------------------------------------------- #
    #     Parameters input when the simulate button is pressed    #
    # ----------------------------------------------------------- #
    '''
    # Manage input parameters
    params = json.loads(request.args.get('parameters'))
    print(params)
    
    # default value
    # sigma = 0.2
    # gamma = 0.07        
    # Get data from DB
    #Sr, Er, Ir, Rr, P = getdata(params.comuna)
    
    # Build and run seir object
    #tr=np.arange(Ir.shape[1])
    #h=0.1
    #seir = SEIR(P,params.eta,params.alpha,Sr[0],Er[0],Ir[0],Rr[0],params.beta,params.sigma,params.gamma,params.mu)
    #seir.integr([tr[0], tr[-1], h)
    # El T no aparece =S 
    
    response = {'status': 'OK','function':'simulate multi','params':params}
    return jsonify(response), 200

    # Solve
    #integr(self,t0,T,h,E0init=False)




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



