# Modelo SEIRHVD

## Parámetros Generales


## Parámetros del Modelo


## Creación de Escenarios


inputarray = np.array([tsim,max_mov,rem_mov,qp,iqt,fqt,movfunct])

* tsim: Tiempo de simulación
* max_mov: Movilidad máxima durante tiempo sin cuarentena
* rem_mov: Movilidad remanente durante tiempo de cuarentena
* qp: Periodo de Cuarentena para cuarentenas alternantes - qp días con cuarentena, luego qp días sin cuarentena
* iqt: Día de inicio de cuarentena (desde el inicio de la simulación)
* fqt: Día de fin de cuarentena (desde el inicio de la simulación)
* movfunct: Función de movilidad
    * 0: Cuarentena total durante el período comprendido entre iqt y fqt
    * 1: Cuarentena alternante tipo onda Cuadrada con período qp
    * 2: Cuarnetena tipo diente de cierra con período qp

Ej: 

Escenarios:
Realistas
0) Cuarentena total 14 movmax = 0.85 mov_rem = 0.6
1) Cuarentena total 14 movmax = 0.85 mov_rem = 0.7
2) Cuarentena total 14 movmax = 0.85 mov_rem = 0.8
3) Cuarentena total 21 movmax = 0.85 mov_rem = 0.6
4) Cuarentena total 21 movmax = 0.85 mov_rem = 0.7
5) Cuarentena total 21 movmax = 0.85 mov_rem = 0.8
6) Cuarentena total de 60 días, cuarentena hiperdinámica de 14 días movmax = 0.85 mov_rem = 0.7
7) Cuarentena total de 60 días, cuarentena hiperdinámica de 21 días movmax = 0.85 mov_rem = 0.7

Optimistas
8) Cuarentena total 14 movmax = 0.55 mov_rem = 0.2
9) Cuarentena total 14 movmax = 0.55 mov_rem = 0.3
10) Cuarentena total 14 movmax = 0.55 mov_rem = 0.4
11) Cuarentena total 21 movmax = 0.55 mov_rem = 0.2
12) Cuarentena total 21 movmax = 0.55 mov_rem = 0.3
13) Cuarentena total 21 movmax = 0.55 mov_rem = 0.4
14) Cuarentena total de 60 días, cuarentena hiperdinámica de 14 días movmax = 0.55 mov_rem = 0.3
15) Cuarentena total de 60 días, cuarentena hiperdinámica de 21 días movmax = 0.55 mov_rem = 0.3
""" 



## Parámetros Encontrados
# Optimos 13-04
#beta = 0.115
#mu = 0.85
#fI = 1.6 



### Reproducen bien datos de Muertos


### Reproducen bien datos de Camas
#### Optimos fit de camas 13-04
    beta = 0.117#0.25#0.19 0.135
    mu = 0.6#2.6 0.6
    fI = 1.9 #4.8
    SeroPrevFactor = 0.5


# Variables:
    # initial params
    self.initdate = None
    self.tsim = None
    self.May15 = None
    self.tstate = None

    # Parameters
    self.beta = None
    self.mu = None
    self.ScaleFactor = None
    self.SeroPrevFactor = None
    self.expinfection = None

    # Quarantine Scenarios
    self.inputarray  = None 
    self.numescenarios  = None 

    # Data
    # Sochimi
    self.sochimi  = None 
    self.Hr  = None 
    self.Vr  = None  
    self.Vr_tot  = None 
    self.Hr_tot  = None  
    self.sochimi_dates  = None 
    self.sochimi_tr  = None 
    # Infected minsal
    self.Ir  = None 
    self.Ir_dates  = None 
    self.tr  = None 
    # Death minsal
    self.Br  = None 
    self.Br_dates  = None 
    self.Br_tr  = None 
    # Death excess
    self.ED_RM  = None 
    self.ED_RM_dates  = None 
    self.ED_tr  = None 
    self.ED_RM_ac  = None 
    # Death deis 


    # Sim and Auxvar:
    self.sims = None  
    self.Vcmodel = None  
    self.Hcmodel = None  
    self.sims = None  
    self.Vmax = None  
    self.Hmax = None  
    self.T  = None 
    self.S  = None
    self.H_sum  = None
    self.H_bed  = None
    self.H_vent  = None
    self.Iac  = None
    self.I  = None
    self.I_act  = None
    self.I_as  = None
    self.I_mi  = None
    self.I_se  = None
    self.I_cr  = None
    self.I_sum  = None
    self.E  = None
    self.E_as  = None
    self.E_sy  = None
    self.B  = None
    self.D  = None
    self.R  = None
    self.V  = None
    self.t  = None
    self.dt  = None
    self.idx  = None
    self.H_crin  = None
    self.H_in  = None
    self.H_out  = None
    self.H_sum  = None
    self.H_tot  = None
    self.CH  = None
    self.CV  = None
    self.ACH  = None
    self.ACV  = None
    self.peakindex  = None
    self.peak  = None
    self.peak_t  = None
    self.peak_date  = None
    self.population  = None
    self.infectedsusc  = None
    self.infectedpop  = None
    self.err_bed  = None
    self.err_vent  = None
    self.err_Iactives  = None
    self.H_colapsedate  = None
    self.V_colapsedate  = None