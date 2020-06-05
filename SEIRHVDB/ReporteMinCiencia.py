#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SEIRHVDB Reporte MinCiencia 25/05/2020
author: Samuel Ropert
"""




# -------------------------------------- #
#      Graficos Reporte MinCiencia       #
# -------------------------------------- #


# Utilities
#plt.xlim((0, 30))
#plt.ylim((0, 15))
#plt.savefig('/home/samuel/Covid_Net_SEIR/SEIRHVDB/Plots/ReporteMinCiencia25-05/Letalidad500D.png', dpi=300)
#plot(ylabel='Letalidad (%)',xlabel='dias')

# --------------------------- #
#   Letalidad a 500 dias      #
# --------------------------- #
# *Grafico 1

# Realista
#i=0
#plt.plot(t[i],100*B[i]/Iac[i],color='red',linestyle='dotted',linewidth = 3.0)
#i=1
#plt.plot(t[i],100*B[i]/Iac[i],color='red',linestyle='solid',linewidth = 3.0)
#i=2
#plt.plot(t[i],100*B[i]/Iac[i],color='red',linestyle='dotted',linewidth = 3.0)
#i=3
#plt.plot(t[i],100*B[i]/Iac[i],color='blue',linestyle='dotted',linewidth = 3.0)
i=4
plt.plot(t[i],100*B[i]/Iac[i],color='blue',linestyle='solid',linewidth = 3.0)
#i=5
#plt.plot(t[i],100*B[i]/Iac[i],color='blue',linestyle='dotted',linewidth = 3.0)
i=6
plt.plot(t[i],100*B[i]/Iac[i],color='lime',linestyle='solid',linewidth = 3.0)
i=7
plt.plot(t[i],100*B[i]/Iac[i],color='black',linestyle='solid',linewidth = 3.0)

plt.ylim((0, 15))
plt.title('Letalidad')
plt.xlabel('Dias desde el 15 de Mayo')
plt.ylabel('Letalidad(%)')
plt.savefig('/home/samuel/Covid_Net_SEIR/SEIRHVDB/Plots/ReporteMinCiencia25-05/Letalidad500DRealista.png', dpi=300)
plot(ylabel='Letalidad (%)',xlabel='dias', title = 'Letalidad' )

# Optimista
#i=8
#plt.plot(t[i],100*B[i]/Iac[i],color='red',linestyle='dashed',linewidth = 3.0)
#i=9
#plt.plot(t[i],100*B[i]/Iac[i],color='red',linestyle='solid',linewidth = 3.0)
#i=10
#plt.plot(t[i],100*B[i]/Iac[i],color='red',linestyle='dotted',linewidth = 3.0)
#i=11
#plt.plot(t[i],100*B[i]/Iac[i],color='blue',linestyle='dashed',linewidth = 3.0)
i=12
plt.plot(t[i],100*B[i]/Iac[i],color='blue',linestyle='solid',linewidth = 3.0)
#i=13
#plt.plot(t[i],100*B[i]/Iac[i],color='blue',linestyle='dotted',linewidth = 3.0)
i=14
plt.plot(t[i],100*B[i]/Iac[i],color='lime',linestyle='solid',linewidth = 3.0)
i=15
plt.plot(t[i],100*B[i]/Iac[i],color='black',linestyle='solid',linewidth = 3.0)

plt.ylim((0, 15))
plt.title('Letalidad')
plt.xlabel('Dias desde el 15 de Mayo')
plt.ylabel('Letalidad(%)')
plt.savefig('/home/samuel/Covid_Net_SEIR/SEIRHVDB/Plots/ReporteMinCiencia25-05/Letalidad500DOptimista.png', dpi=300)
plot(ylabel='Letalidad (%)', xlabel = 'Días desde el 15 de Mayo', title = 'Letalidad' )



# --------------------------- #
#    Letalidad a 30 dias      #
# --------------------------- #
# Grafico 2
# 70 y 30
time = 30
endD = [np.searchsorted(t[i],range(time)) for i in range(len(input))]

#i=0
#plt.plot(t[i][:endD[i][-1]],100*B[i][:endD[i][-1]]/Iac[i][:endD[i][-1]], color='red',linestyle='dotted',linewidth = 3.0)
#i=1
#plt.plot(t[i][:endD[i][-1]],100*B[i][:endD[i][-1]]/Iac[i][:endD[i][-1]], color='red',linestyle='solid',linewidth = 3.0)
#i=2
#plt.plot(t[i][:endD[i][-1]],100*B[i][:endD[i][-1]]/Iac[i][:endD[i][-1]], color='red',linestyle='dotted',linewidth = 3.0)
#i=3
#plt.plot(t[i][:endD[i][-1]],100*B[i][:endD[i][-1]]/Iac[i][:endD[i][-1]], color='blue',linestyle='dotted',linewidth = 3.0)
i=4
plt.plot(t[i][:endD[i][-1]],100*B[i][:endD[i][-1]]/Iac[i][:endD[i][-1]], color='blue',linestyle='solid',linewidth = 3.0)
#i=5
#plt.plot(t[i][:endD[i][-1]],100*B[i][:endD[i][-1]]/Iac[i][:endD[i][-1]], color='blue',linestyle='dotted',linewidth = 3.0)
i=6
plt.plot(t[i][:endD[i][-1]],100*B[i][:endD[i][-1]]/Iac[i][:endD[i][-1]], color='lime',linestyle='solid',linewidth = 3.0)
i=7
plt.plot(t[i][:endD[i][-1]],100*B[i][:endD[i][-1]]/Iac[i][:endD[i][-1]], color='black',linestyle='solid',linewidth = 3.0)

plt.ylim((0, 8))
plt.title('Letalidad')
plt.xlabel('Dias desde el 15 de Mayo')
plt.ylabel('Letalidad(%)')
plt.savefig('/home/samuel/Covid_Net_SEIR/SEIRHVDB/Plots/ReporteMinCiencia25-05/Letalidad30DRealista.png', dpi=300)
plot(ylabel='Letalidad (%)')

#i=8
#plt.plot(t[i][:endD[i][-1]],100*B[i][:endD[i][-1]]/Iac[i][:endD[i][-1]], color='red',linestyle='dashed',linewidth = 3.0)
#i=9
#plt.plot(t[i][:endD[i][-1]],100*B[i][:endD[i][-1]]/Iac[i][:endD[i][-1]], color='red',linestyle='solid',linewidth = 3.0)
#i=10
#plt.plot(t[i][:endD[i][-1]],100*B[i][:endD[i][-1]]/Iac[i][:endD[i][-1]], color='red',linestyle='dotted',linewidth = 3.0)
#i=11
#plt.plot(t[i][:endD[i][-1]],100*B[i][:endD[i][-1]]/Iac[i][:endD[i][-1]], color='blue',linestyle='dashed',linewidth = 3.0)
i=12
plt.plot(t[i][:endD[i][-1]],100*B[i][:endD[i][-1]]/Iac[i][:endD[i][-1]], color='blue',linestyle='solid',linewidth = 3.0)
#i=13
#plt.plot(t[i][:endD[i][-1]],100*B[i][:endD[i][-1]]/Iac[i][:endD[i][-1]], color='blue',linestyle='dotted',linewidth = 3.0)
i=14
plt.plot(t[i][:endD[i][-1]],100*B[i][:endD[i][-1]]/Iac[i][:endD[i][-1]], color='lime',linestyle='solid',linewidth = 3.0)
i=15
plt.plot(t[i][:endD[i][-1]],100*B[i][:endD[i][-1]]/Iac[i][:endD[i][-1]], color='black',linestyle='solid',linewidth = 3.0)
plt.ylim((0, 8))

plt.title('Letalidad')
plt.xlabel('Dias desde el 15 de Mayo')
plt.ylabel('Letalidad(%)')
plt.savefig('/home/samuel/Covid_Net_SEIR/SEIRHVDB/Plots/ReporteMinCiencia25-05/Letalidad30DOptimista.png', dpi=300)
plot(ylabel='Letalidad (%)')


# --------------------------- #
#   Curva Epidemica 500D      #
# --------------------------- #
# Grafico 3:
# Infectados Activos a 500 dias

I = [sims[i][0].I_as+sims[i][0].I_cr + sims[i][0].I_mi + sims[i][0].I_se + sims[i][0].H_in+sims[i][0].H_cr+sims[i][0].H_out+sims[i][0].V for i in range(len(input))] 
I_act = [sims[i][0].I_mi + sims[i][0].I_cr + sims[i][0].I_se + sims[i][0].H_in+sims[i][0].H_cr+sims[i][0].H_out+sims[i][0].V for i in range(len(input))] 

# Buscar maximo
maxI = max([max(I_act[i]) for i in range(len(input))])

# Realista
#i=0
#plt.plot(t[i], I_act[i]/maxI,color='red',linestyle='dotted',linewidth = 3.0)
#i=1
#plt.plot(t[i], I_act[i]/maxI,color='red',linestyle='solid',linewidth = 3.0)
#i=2
#plt.plot(t[i], I_act[i]/maxI,color='red',linestyle='dotted',linewidth = 3.0)
#i=3
#plt.plot(t[i], I_act[i]/maxI,color='blue',linestyle='dotted',linewidth = 3.0)
i=4
plt.plot(t[i], I_act[i]/maxI,color='blue',linestyle='solid',linewidth = 3.0)
#i=5
#plt.plot(t[i], I_act[i]/maxI,color='blue',linestyle='dotted',linewidth = 3.0)
i=6
plt.plot(t[i], I_act[i]/maxI,color='lime',linestyle='solid',linewidth = 3.0)
i=7
plt.plot(t[i], I_act[i]/maxI,color='black',linestyle='solid',linewidth = 3.0)

plt.title('Curva Epidemica')
plt.xlabel('Dias desde el 15 de Mayo')
plt.ylabel('Infectados')
plt.ylim((0, 1.1))
plt.savefig('/home/samuel/Covid_Net_SEIR/SEIRHVDB/Plots/ReporteMinCiencia25-05/CurvaEpidemicaRealista.png', dpi=300)
plot(ylabel='Infectados Activos')

# Optimista
#i=8
#plt.plot(t[i], I_act[i]/maxI,color='red',linestyle='dashed',linewidth = 3.0)
#i=9
#plt.plot(t[i], I_act[i]/maxI,color='red',linestyle='solid',linewidth = 3.0)
#i=10
#plt.plot(t[i], I_act[i]/maxI,color='red',linestyle='dotted',linewidth = 3.0)
#i=11
#plt.plot(t[i], I_act[i]/maxI,color='blue',linestyle='dashed',linewidth = 3.0)
i=12
plt.plot(t[i], I_act[i]/maxI,color='blue',linestyle='solid',linewidth = 3.0)
#i=13
#plt.plot(t[i], I_act[i]/maxI,color='blue',linestyle='dotted',linewidth = 3.0)
i=14
plt.plot(t[i], I_act[i]/maxI,color='lime',linestyle='solid',linewidth = 3.0)
i=15
plt.plot(t[i], I_act[i]/maxI,color='black',linestyle='solid',linewidth = 3.0)

plt.title('Curva Epidemica')
plt.xlabel('Dias desde el 15 de Mayo')
plt.ylabel('Infectados')
plt.ylim((0, 1.1))
plt.savefig('/home/samuel/Covid_Net_SEIR/SEIRHVDB/Plots/ReporteMinCiencia25-05/CurvaEpidemicaOptimista.png', dpi=300)
plot(ylabel='Infectados Activos')


# ----------------------------- #
#   Muertos Acumulados 500D     #
# ----------------------------- #
# Grafico 4

# Buscar maximo
maxB = max([max(B[i]) for i in range(len(input))])

i=0
plt.plot(t[i], B[i]/maxB,color='red',linestyle='dotted',linewidth = 3.0)
i=1
plt.plot(t[i], B[i]/maxB,color='red',linestyle='solid',linewidth = 3.0)
i=2
plt.plot(t[i], B[i]/maxB,color='red',linestyle='dotted',linewidth = 3.0)
i=3
plt.plot(t[i], B[i]/maxB,color='blue',linestyle='dotted',linewidth = 3.0)
i=4
plt.plot(t[i], B[i]/maxB,color='blue',linestyle='solid',linewidth = 3.0)
i=5
plt.plot(t[i], B[i]/maxB,color='blue',linestyle='dotted',linewidth = 3.0)
i=6
#plt.plot(t[i], B[i]/maxB,color='lime',linestyle='solid',linewidth = 3.0)
i=7
#plt.plot(t[i], B[i]/maxB,color='black',linestyle='solid',linewidth = 3.0)


plt.title('Fallecidos Acumulados')
plt.xlabel('Dias desde el 15 de Mayo')
plt.ylabel('Fallecidos')
plt.ylim((0, 1.1))
plt.savefig('/home/samuel/Covid_Net_SEIR/SEIRHVDB/Plots/ReporteMinCiencia25-05/MuertosAcumuladosRealista.png', dpi=300)
plot(ylabel='Fallecidos')

i=8
plt.plot(t[i], B[i]/maxB,color='red',linestyle='dashed',linewidth = 3.0)
i=9
plt.plot(t[i], B[i]/maxB,color='red',linestyle='solid',linewidth = 3.0)
i=10
plt.plot(t[i], B[i]/maxB,color='red',linestyle='dotted',linewidth = 3.0)
i=11
plt.plot(t[i], B[i]/maxB,color='blue',linestyle='dashed',linewidth = 3.0)
i=12
plt.plot(t[i], B[i]/maxB,color='blue',linestyle='solid',linewidth = 3.0)
i=13
plt.plot(t[i], B[i]/maxB,color='blue',linestyle='dotted',linewidth = 3.0)
i=14
plt.plot(t[i], B[i]/maxB,color='lime',linestyle='solid',linewidth = 3.0)
i=15
plt.plot(t[i], B[i]/maxB,color='black',linestyle='solid',linewidth = 3.0)


plt.title('Fallecidos Acumulados')
plt.xlabel('Dias desde el 15 de Mayo')
plt.ylabel('Fallecidos')
plt.ylim((0, 1.1))
plt.savefig('/home/samuel/Covid_Net_SEIR/SEIRHVDB/Plots/ReporteMinCiencia25-05/MuertosAcumuladosOptimista.png', dpi=300)
plot(ylabel='Fallecidos')


# ----------------------------- #
#       Camas Requeridas        #
# ----------------------------- #
# Grafico 5

time = 30
endD = [np.searchsorted(t[i],range(time)) for i in range(len(input))]

CH = [sims[i][0].CH for i in range(len(input))]
CV = [sims[i][0].CV for i in range(len(input))]
ACH = [sims[i][0].ACH for i in range(len(input))]
ACV = [sims[i][0].ACV for i in range(len(input))]


i=0
plt.plot(t[i][:endD[i][-1]],CH[i][:endD[i][-1]],color='red',linestyle='dotted',linewidth = 3.0)
plt.plot(t[i][:endD[i][-1]],CV[i][:endD[i][-1]],color='lime',linestyle='dotted',linewidth = 3.0)
i=1
plt.plot(t[i][:endD[i][-1]],CH[i][:endD[i][-1]],color='red',linestyle='solid',linewidth = 3.0)
plt.plot(t[i][:endD[i][-1]],CV[i][:endD[i][-1]],color='lime',linestyle='solid',linewidth = 3.0)
i=2
plt.plot(t[i][:endD[i][-1]],CH[i][:endD[i][-1]], color='red',linestyle='dotted',linewidth = 3.0)
plt.plot(t[i][:endD[i][-1]],CV[i][:endD[i][-1]], color='lime',linestyle='dotted',linewidth = 3.0)
i=3
plt.plot(t[i][:endD[i][-1]],CH[i][:endD[i][-1]], color='blue',linestyle='dotted',linewidth = 3.0)
plt.plot(t[i][:endD[i][-1]],CV[i][:endD[i][-1]], color='lime',linestyle='dotted',linewidth = 3.0)
i=4
plt.plot(t[i][:endD[i][-1]],CH[i][:endD[i][-1]], color='blue',linestyle='solid',linewidth = 3.0)
plt.plot(t[i][:endD[i][-1]],CV[i][:endD[i][-1]], color='lime',linestyle='solid',linewidth = 3.0)
i=5
plt.plot(t[i][:endD[i][-1]],CH[i][:endD[i][-1]], color='blue',linestyle='dotted',linewidth = 3.0)
plt.plot(t[i][:endD[i][-1]],CV[i][:endD[i][-1]], color='lime',linestyle='dotted',linewidth = 3.0)

plt.title('Camas Requeridas')
plt.xlabel('Dias desde el 15 de Mayo')
plt.ylabel('Camas')
plot(ylabel='camas',xlabel = 'Días desde el 15 de Mayo')

i=8
plt.plot(t[i][:endD[i][-1]],CH[i][:endD[i][-1]], color='red',linestyle='dotted',linewidth = 3.0)
plt.plot(t[i][:endD[i][-1]],CV[i][:endD[i][-1]], color='lime',linestyle='dotted',linewidth = 3.0)
i=9
plt.plot(t[i][:endD[i][-1]],CH[i][:endD[i][-1]], color='red',linestyle='solid',linewidth = 3.0)
plt.plot(t[i][:endD[i][-1]],CV[i][:endD[i][-1]], color='lime',linestyle='solid',linewidth = 3.0)
i=10
plt.plot(t[i][:endD[i][-1]],CH[i][:endD[i][-1]], color='red',linestyle='dotted',linewidth = 3.0)
plt.plot(t[i][:endD[i][-1]],CV[i][:endD[i][-1]], color='lime',linestyle='dotted',linewidth = 3.0)
i=11
plt.plot(t[i][:endD[i][-1]],CH[i][:endD[i][-1]], color='blue',linestyle='dotted',linewidth = 3.0)
plt.plot(t[i][:endD[i][-1]],CV[i][:endD[i][-1]], color='lime',linestyle='dotted',linewidth = 3.0)
i=12
plt.plot(t[i][:endD[i][-1]],CH[i][:endD[i][-1]], color='blue',linestyle='solid',linewidth = 3.0)
plt.plot(t[i][:endD[i][-1]],CV[i][:endD[i][-1]], color='lime',linestyle='solid',linewidth = 3.0)
i=13
plt.plot(t[i][:endD[i][-1]],CH[i][:endD[i][-1]], color='blue',linestyle='dotted',linewidth = 3.0)
plt.plot(t[i][:endD[i][-1]],CV[i][:endD[i][-1]], color='lime',linestyle='dotted',linewidth = 3.0)

plt.title('Camas Requeridas')
plt.xlabel('Dias desde el 15 de Mayo')
plt.ylabel('Camas')
plot(ylabel='camas',xlabel = 'Días desde el 15 de Mayo')
plot(title='Camas Requeridas')


# ----------------------------------------- #
#    Camas Reales vs simuladas a 30 dias    #
# ----------------------------------------- #
# Grafico 6

i = 0 # 70% movilidad
days = 30

H_crin=[sims[i][0].H_cr for i in range(len(input))] 
H_in=[sims[i][0].H_in for i in range(len(input))] 
H_out=[sims[i][0].H_out for i in range(len(input))] 

datestxt = data['Fechas'][32:38]
#dates = list(range(15,15+len(datestxt)))
#datessim = list(range(len(datestxt)+14))
dates = list(range(len(datestxt)))
datessim = list(range(days))


Hr_bed_hoy = Hr_bed[32:32+len(datestxt)]
Hr_vent_hoy = Hr_vent[32:32+len(datestxt)]


H_bed_hoy = H_bed[i][idx[i][0:days]] 
H_in_hoy = H_in[i][idx[i][0:days]] 
H_out_hoy = H_out[i][idx[i][0:days]] 
H_crin_hoy = H_crin[i][idx[i][0:days]] 
H_vent_hoy = H_vent[i][idx[i][0:days]] 

err_bed = LA.norm(Hr_bed_hoy-H_bed_hoy[0:len(Hr_bed_hoy)])/LA.norm(Hr_bed_hoy)
err_vent = LA.norm(Hr_vent_hoy-H_vent_hoy[0:len(Hr_vent_hoy)])/LA.norm(Hr_vent_hoy)

plt.plot([], [], ' ', label='err_camas: '+str(round(100*err_bed,2))+'%')
plt.plot([], [], ' ', label='err_VMI: '+str(round(100*err_vent,2))+'%')
plt.plot(datessim,H_bed_hoy,label='sim Inter/Inte',color='red',linewidth = 3.0)
plt.plot(datessim,H_vent_hoy,label='sim VMI',color='blue',linewidth = 3.0)
plt.scatter(dates,Hr_bed_hoy,label='real Inter/Inten',color='red')
plt.scatter(dates,Hr_vent_hoy,label='real VMI',color='blue')

plt.title('Uso de Camas')
plt.xlabel('Dias desde el 15 de Mayo')
plt.ylabel('Camas')
plt.savefig('/home/samuel/Covid_Net_SEIR/SEIRHVDB/Plots/ReporteMinCiencia25-05/CamasRealvsSim.png', dpi=300)
plot(ylabel='Camas',xlabel='Dias desde el 15 de Mayo')


# ---------------------------------------------- #
#    Fallecidos acumulados reales vs simulados   #
# ---------------------------------------------- #
# Grafico 7
days = 30
datessim = list(range(days))
Br_hoy = Br[32:]
dates = list(range(len(Br_hoy)))

Bdata = []
namelist = ['Dead14Dmin','Dead14Dmed','Dead14Dmax','Dead21Dmin','Dead21Dmed','Dead21Dmax']
for i in range(3):        
    B_hoy = [B[i][0]]
    for j in range(1,len(idx[i])-1):
        if idx[i][j-1]== idx[i][j]:
            B_hoy.extend([(B[i][idx[i][j-1]]+B[i][idx[i][j+1]])/2])
        else:        
            B_hoy.extend([B[i][idx[i][j]]])
    #B_hoy.extend([B[i][idx[i][-1]]])
    Bdata.append(B_hoy)



i=0
plt.plot(datessim,Bdata[i],linewidth = 3.0,linestyle = 'dotted')
err = LA.norm(Br_hoy-Bdata[i][0:len(Br_hoy)])/LA.norm(Br_hoy)
plt.plot([], [], ' ', label='err min: '+str(round(100*err,2))+'%')
i=1
plt.plot(datessim,Bdata[i],linewidth = 3.0,linestyle = 'solid')
err = LA.norm(Br_hoy-Bdata[i][0:len(Br_hoy)])/LA.norm(Br_hoy)
plt.plot([], [], ' ', label='err med: '+str(round(100*err,2))+'%')
i=2
plt.plot(datessim,Bdata[i],linewidth = 3.0,linestyle = 'dotted')
err = LA.norm(Br_hoy-Bdata[i][0:len(Br_hoy)])/LA.norm(Br_hoy)
plt.plot([], [], ' ', label='err max: '+str(round(100*err,2))+'%')
#plt.plot(dates,Br_hoy,label='Fallecidos Reales')

plt.scatter(dates,Br_hoy,color = 'red')
#plt.plot([], [], ' ', label='err: '+str(round(100*err,2))+'%')

plt.title('Fallecidos Acumulados')
plt.xlabel('Dias desde el 15 de Mayo')
plt.ylabel('Fallecidos')
plt.savefig('/home/samuel/Covid_Net_SEIR/SEIRHVDB/Plots/ReporteMinCiencia25-05/FallecidosAcumRealvsSim.png', dpi=300)
plot(ylabel='Fallecidos',xlabel='Dias desde el 15 de Mayo',title='Fallecidos Acumulados')



# ---------------------------------------------- #
#    Infectados acumulados reales vs simulados   #
# ---------------------------------------------- #
# Grafico 8
days = 30
datessim = list(range(days))
Irac_hoy = Irac[32:]
dates = list(range(len(Irac_hoy)))
Iacdata = []
for i in range(3):        
    Iac_hoy = [Iac[i][0]]
    for j in range(1,len(idx[i])-1):
        if idx[i][j-1]== idx[i][j]:
            Iac_hoy.extend([(Iac[i][idx[i][j-1]]+Iac[i][idx[i][j+1]])/2])
        else:        
            Iac_hoy.extend([Iac[i][idx[i][j]]])
    #Iac_hoy.extend([Iac[i][idx[i][-1]]])
    Iacdata.append(Iac_hoy)

i = 0
plt.plot(datessim,Iacdata[i],linewidth = 3.0,linestyle = 'dotted')
err = LA.norm(Irac_hoy-Iacdata[i][0:len(Irac_hoy)])/LA.norm(Irac_hoy)
plt.plot([], [], ' ', label='err min: '+str(round(100*err,2))+'%')
i = 1
plt.plot(datessim,Iacdata[i],linewidth = 3.0,linestyle = 'solid')
err = LA.norm(Irac_hoy-Iacdata[i][0:len(Irac_hoy)])/LA.norm(Irac_hoy)
plt.plot([], [], ' ', label='err med: '+str(round(100*err,2))+'%')
i = 2
plt.plot(datessim,Iacdata[i],linewidth = 3.0,linestyle = 'dotted')
err = LA.norm(Irac_hoy-Iacdata[i][0:len(Irac_hoy)])/LA.norm(Irac_hoy)
plt.plot([], [], ' ', label='err max: '+str(round(100*err,2))+'%')


plt.scatter(dates,Irac_hoy,color = 'red')


plt.title('Infectados Acumulados')
plt.xlabel('Dias desde el 15 de Mayo')
plt.ylabel('Infectados')
plt.savefig('/home/samuel/Covid_Net_SEIR/SEIRHVDB/Plots/ReporteMinCiencia25-05/InfectadosAcumuladosRealvsSim.png', dpi=300)

plot(ylabel='Infectados',xlabel='Dias desde el 15 de Mayo', title= 'Infectados Acumulados')





# ----------- #
#    Tabla    #
# ----------- #
#15 al 45 de mayo 


# ----------- #
#     Time    #
# ----------- #
from datetime import date
from datetime import timedelta
base = date(2020,5,15)
index = [(base+timedelta(days=i)).strftime("%d/%m/%Y") for i in range(31)]
dates = list(range(15,32))  
dates.extend(list(range(1,16)))
dates = [str(i) for i in dates]
days = len(dates)
idx = [np.searchsorted(t[i],range(days+1)) for i in range(len(input))]

# --------------------------- #
#    Letalidad a 30 dias      #
# --------------------------- #
Letalidad = pd.DataFrame(dict(Let14Dmin=100*B[0][idx[0]]/Iac[0][idx[0]],Let14Dmed=100*B[1][idx[1]]/Iac[1][idx[1]],Let14Dmax=100*B[2][idx[2]]/Iac[2][idx[2]],Let21Dmin=100*B[3][idx[3]]/Iac[3][idx[3]],Let21Dmed=100*B[4][idx[4]]/Iac[4][idx[4]],Let21Dmax=100*B[5][idx[5]]/Iac[5][idx[5]]))


# --------------------------- #
#    Fallecidos acumulados    #
# --------------------------- #

# Interpolacion para datos faltantes
Bdata = dict()
namelist = ['Dead14Dmin','Dead14Dmed','Dead14Dmax','Dead21Dmin','Dead21Dmed','Dead21Dmax']
for i in range(len(namelist)):        
    B_hoy = [B[i][0]]
    for j in range(1,len(idx[i])-1):
        if idx[i][j-1]== idx[i][j]:
            B_hoy.extend([(B[i][idx[i][j-1]]+B[i][idx[i][j+1]])/2])
        else:        
            B_hoy.extend([B[i][idx[i][j]]])
    B_hoy.extend([B[i][idx[i][-1]]])
    Bdata[namelist[i]]=B_hoy

Bdata = pd.DataFrame(Bdata)


# ---------------------------------------------- #
#    Infectados acumulados reales vs simulados   #
# ---------------------------------------------- #
# Interpolacion para datos faltantes
Iacdata = dict()
namelist = ['Infectados14Dmin','Infectados14Dmed','Infectados14Dmax','Infectados21Dmin','Infectados21Dmed','Infectados21Dmax']
for i in range(len(namelist)):        
    Iac_hoy = [Iac[i][0]]
    for j in range(1,len(idx[i])-1):
        if idx[i][j-1]== idx[i][j]:
            Iac_hoy.extend([(Iac[i][idx[i][j-1]]+Iac[i][idx[i][j+1]])/2])
        else:        
            Iac_hoy.extend([Iac[i][idx[i][j]]])
    Iac_hoy.extend([Iac[i][idx[i][-1]]])
    Iacdata[namelist[i]]=Iac_hoy

Iacdata = pd.DataFrame(Iacdata)

# ------------------------- #
#     Create Data Frame     #
# ------------------------- #

index = pd.DataFrame(dict(dates=index))
datosminciencia = pd.concat([index,Iacdata,Bdata,Letalidad], axis=1, sort=False) 
datosminciencia = datosminciencia.set_index('dates')
datosminciencia.to_excel('/home/samuel/Covid_Net_SEIR/SEIRHVDB/Plots/ReporteMinCiencia25-05/datos-02.xls') 

