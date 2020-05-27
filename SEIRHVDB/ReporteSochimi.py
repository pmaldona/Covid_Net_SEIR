#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reporte Sochimi
"""

def plot(title='',xlabel='dias',ylabel='Personas'):
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(loc=0)
    plt.show()


ax2 = ax.secondary_xaxis("top", functions=(f,g)) 
datestxt = data['Fechas'][32:38]


# ---------------------------------- #
#    Camas adicionales Requeridas    #
# ---------------------------------- #


dates = list(range(15,32))
dates.extend(list(range(1,14)))
dates = [str(i) for i in dates]
days = len(dates)
endD = [np.searchsorted(t[i],range(time)) for i in range(len(input))]
idx = [np.searchsorted(t[i],range(days+1)) for i in range(len(input))]


CH = [sims[i][0].CH for i in range(len(input))]
CV = [sims[i][0].CV for i in range(len(input))]


CH_hoy = [CH[i][idx[i][0:(days)]] for i in range(len(input))]
#plt.plot(dates,CH_hoy,color='red',linestyle='dotted',linewidth = 3.0)

H_bed_hoy = H_bed[i][idx[i][0:days]] 
H_in_hoy = H_in[i][idx[i][0:days]] 
H_out_hoy = H_out[i][idx[i][0:days]] 
H_crin_hoy = H_crin[i][idx[i][0:days]] 
H_vent_hoy = H_vent[i][idx[i][0:days]] 




# 14 dias de cuarentena inicial
i=0
plt.plot(t[i][:endD[i][-1]],CH[i][:endD[i][-1]],color='red',linestyle='dotted',linewidth = 3.0)
plt.plot(t[i][:endD[i][-1]],CV[i][:endD[i][-1]],color='blue',linestyle='dotted',linewidth = 3.0)
i=1
plt.plot(t[i][:endD[i][-1]],CH[i][:endD[i][-1]],color='red',linestyle='solid',linewidth = 3.0)
plt.plot(t[i][:endD[i][-1]],CV[i][:endD[i][-1]],color='blue',linestyle='solid',linewidth = 3.0)
i=2
plt.plot(t[i][:endD[i][-1]],CH[i][:endD[i][-1]], color='red',linestyle='dotted',linewidth = 3.0)
plt.plot(t[i][:endD[i][-1]],CV[i][:endD[i][-1]], color='blue',linestyle='dotted',linewidth = 3.0)
#plot(title = 'Camas adicionales 14D',ylabel='Camas',xlabel = 'Días desde el 15 de Mayo')

plt.xlim((0, 30))
plt.ylim((0, 2000))
plt.savefig('/home/samuel/Covid_Net_SEIR/SEIRHVDB/Plots/ReporteSochimi25-05/CamasAdicionales14D.png', dpi=300)
plot(title = 'Camas adicionales 14D',ylabel='Camas',xlabel = 'Días desde el 15 de Mayo')

# 21 dias de cuarentena inicial
i=3
plt.plot(t[i][:endD[i][-1]],CH[i][:endD[i][-1]], color='red',linestyle='dotted',linewidth = 3.0)
plt.plot(t[i][:endD[i][-1]],CV[i][:endD[i][-1]], color='blue',linestyle='dotted',linewidth = 3.0)
i=4
plt.plot(t[i][:endD[i][-1]],CH[i][:endD[i][-1]], color='red',linestyle='solid',linewidth = 3.0)
plt.plot(t[i][:endD[i][-1]],CV[i][:endD[i][-1]], color='blue',linestyle='solid',linewidth = 3.0)
i=5
plt.plot(t[i][:endD[i][-1]],CH[i][:endD[i][-1]], color='red',linestyle='dotted',linewidth = 3.0)
plt.plot(t[i][:endD[i][-1]],CV[i][:endD[i][-1]], color='blue',linestyle='dotted',linewidth = 3.0)
#plot(title = 'Camas adicionales 21D',ylabel='Camas',xlabel = 'Días desde el 15 de Mayo')
plt.xlim((0, 30))
plt.ylim((0, 2000))
plt.savefig('/home/samuel/Covid_Net_SEIR/SEIRHVDB/Plots/ReporteSochimi25-05/CamasAdicionales21D.png', dpi=300)
plot(title = 'Camas adicionales 14D',ylabel='Camas',xlabel = 'Días desde el 15 de Mayo')


# ------------------ #
#    Uso de Camas    #
# ------------------ #


dates = list(range(15,32))  
dates.extend(list(range(1,14)))
dates = [str(i) for i in dates]
days = len(dates)
idx = [np.searchsorted(t[i],range(days+1)) for i in range(len(input))]
datessim = list(range(days))

H_bed_hoy = [H_bed[i][idx[i][0:days]] for i in range(len(input))]
H_in_hoy = [H_in[i][idx[i][0:days]] for i in range(len(input))]
H_out_hoy = [H_out[i][idx[i][0:days]] for i in range(len(input))]
H_crin_hoy = [H_crin[i][idx[i][0:days]] for i in range(len(input))]
H_vent_hoy = [H_vent[i][idx[i][0:days]] for i in range(len(input))]


# 14 dias de cuarentena inicial
i=0
plt.plot(datessim,H_bed_hoy[i],color='red',linestyle='dotted',linewidth = 3.0)
plt.plot(datessim,H_vent_hoy[i],color='blue',linestyle='dotted',linewidth = 3.0)
i=1
plt.plot(datessim,H_bed_hoy[i],color='red',linestyle='solid',linewidth = 3.0)
plt.plot(datessim,H_vent_hoy[i],color='blue',linestyle='solid',linewidth = 3.0)
i=2
plt.plot(datessim,H_bed_hoy[i],color='red',linestyle='dotted',linewidth = 3.0)
plt.plot(datessim,H_vent_hoy[i],color='blue',linestyle='dotted',linewidth = 3.0)

plt.ylim((600, 2800))
plt.savefig('/home/samuel/Covid_Net_SEIR/SEIRHVDB/Plots/ReporteSochimi25-05/UsoCamas14D.png', dpi=300)
plot(title = 'Uso de camas 14D',ylabel='Camas',xlabel = 'Días desde el 15 de Mayo')

# 21 dias de cuarentena inicial
i=3
plt.plot(datessim,H_bed_hoy[i],color='red',linestyle='dotted',linewidth = 3.0)
plt.plot(datessim,H_vent_hoy[i],label='VMI',color='blue',linestyle='dotted',linewidth = 3.0)
i=4
plt.plot(datessim,H_bed_hoy[i],color='red',linestyle='solid',linewidth = 3.0)
plt.plot(datessim,H_vent_hoy[i],label='VMI',color='blue',linestyle='solid',linewidth = 3.0)
i=5
plt.plot(datessim,H_bed_hoy[i],color='red',linestyle='dotted',linewidth = 3.0)
plt.plot(datessim,H_vent_hoy[i],color='blue',linestyle='dotted',linewidth = 3.0)

plt.ylim((600, 2800))
plt.savefig('/home/samuel/Covid_Net_SEIR/SEIRHVDB/Plots/ReporteSochimi25-05/UsoCamas21D.png', dpi=300)
plot(title = 'Uso de camas 21D',ylabel='Camas',xlabel = 'Días desde el 15 de Mayo')




# ------------------------------ #
#    Necesidad total de Camas    #
# ------------------------------ #


dates = list(range(15,32))  
dates.extend(list(range(1,14)))
dates = [str(i) for i in dates]
days = len(dates)
idx = [np.searchsorted(t[i],range(days+1)) for i in range(len(input))]
datessim = list(range(days))

H_bed_hoy = [H_bed[i][idx[i][0:days]] for i in range(len(input))]
H_vent_hoy = [H_vent[i][idx[i][0:days]] for i in range(len(input))]

CH = [sims[i][0].CH for i in range(len(input))]
CV = [sims[i][0].CV for i in range(len(input))]

CH_hoy = [CH[i][idx[i][0:days]] for i in range(len(input))]
CV_hoy = [CV[i][idx[i][0:days]] for i in range(len(input))]

totalbed = [CH_hoy[i] + H_bed_hoy[i] for i in range(len(input))]
totalvmi= [CV_hoy[i] + H_vent_hoy[i] for i in range(len(input))]
# 14 dias de cuarentena inicial
i=0
plt.plot(datessim,totalbed[i],color='red',linestyle='dotted',linewidth = 3.0)
plt.plot(datessim,totalvmi[i],color='blue',linestyle='dotted',linewidth = 3.0)
i=1
plt.plot(datessim,totalbed[i],color='red',linestyle='solid',linewidth = 3.0)
plt.plot(datessim,totalvmi[i],color='blue',linestyle='solid',linewidth = 3.0)
i=2
plt.plot(datessim,totalbed[i],color='red',linestyle='dotted',linewidth = 3.0)
plt.plot(datessim,totalvmi[i],color='blue',linestyle='dotted',linewidth = 3.0)

plt.ylim((0, 6000))
plt.savefig('/home/samuel/Covid_Net_SEIR/SEIRHVDB/Plots/ReporteSochimi25-05/NecTotalCamas14D.png', dpi=300)
plot(title = 'Necesidad total de camas 14D',ylabel='Camas',xlabel = 'Días desde el 15 de Mayo')

# 21 dias de cuarentena inicial
i=3
plt.plot(datessim,totalbed[i],color='red',linestyle='dotted',linewidth = 3.0)
plt.plot(datessim,totalvmi[i],color='blue',linestyle='dotted',linewidth = 3.0)
i=4
plt.plot(datessim,totalbed[i],color='red',linestyle='solid',linewidth = 3.0)
plt.plot(datessim,totalvmi[i],color='blue',linestyle='solid',linewidth = 3.0)
i=5
plt.plot(datessim,totalbed[i],color='red',linestyle='dotted',linewidth = 3.0)
plt.plot(datessim,totalvmi[i],color='blue',linestyle='dotted',linewidth = 3.0)

plt.ylim((0, 6000))
plt.savefig('/home/samuel/Covid_Net_SEIR/SEIRHVDB/Plots/ReporteSochimi25-05/NecTotalCamas21D.png', dpi=300)
plot(title = 'Necesidad total de camas 21D',ylabel='Camas',xlabel = 'Días desde el 15 de Mayo')







# ---------------------------------------- #
#    Camas Reales vs simuladas a 30 dias   #
# ---------------------------------------- #
#i = 0 # 60% movilidad
i = 1 # 70% movilidad
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
plt.plot(datessim,H_bed_hoy,label='sim Inter/Inte',color='red')
plt.plot(datessim,H_vent_hoy,label='sim VMI',color='blue')
plt.scatter(dates,Hr_bed_hoy,label='real Inter/Inten',color='red')
plt.scatter(dates,Hr_vent_hoy,label='real VMI',color='blue')


#plt.savefig('/home/samuel/Covid_Net_SEIR/SEIRHVDB/Plots/ReporteSochimi25-05/CamasRealvsSim.png', dpi=300)
plot(ylabel='Camas',xlabel='Dias desde el 15 de Mayo')



# ----------- #
#    Tabla    #
# ----------- #
#15 al 45 de mayo 

# Uso de camas
# Camas Adicionales
# Necesidad Total de Camas

tsoch = pd.DataFrame()



# ----------- #
#     Time    #
# ----------- #
from datetime import date
from datetime import timedelta
base = date(2020,5,15)
index = [(base+timedelta(days=i)).strftime("%d/%m/%Y") for i in range(30)]
dates = list(range(15,32))  
dates.extend(list(range(1,14)))
dates = [str(i) for i in dates]
days = len(dates)
idx = [np.searchsorted(t[i],range(days+1)) for i in range(len(input))]


# ------------------ #
#    Uso de Camas    #
# ------------------ #

H_bed_hoy = [H_bed[i][idx[i][0:days]] for i in range(len(input))]
H_vent_hoy = [H_vent[i][idx[i][0:days]] for i in range(len(input))]

Hbed = pd.DataFrame(dict(UsoCamas14Dmin=H_bed_hoy[0],UsoCamas14Dmed=H_bed_hoy[1],UsoCamas14Dmax=H_bed_hoy[2],UsoCamas21Dmin=H_bed_hoy[3],UsoCamas21Dmed=H_bed_hoy[4],UsoCamas21Dmax=H_bed_hoy[5]))
Hvent = pd.DataFrame(dict(UsoVMI14Dmin=H_vent_hoy[0],UsoVMI14Dmed=H_vent_hoy[1],UsoVMI14Dmax=H_vent_hoy[2],UsoVMI21Dmin=H_vent_hoy[3],UsoVMI21Dmed=H_vent_hoy[4],UsoVMI21Dmax=H_vent_hoy[5]))


# ---------------------------------- #
#    Camas adicionales Requeridas    #
# ---------------------------------- #

CH = [sims[i][0].CH for i in range(len(input))]
CV = [sims[i][0].CV for i in range(len(input))]
CH_hoy = [CH[i][idx[i][0:days]] for i in range(len(input))]
CV_hoy = [CV[i][idx[i][0:days]] for i in range(len(input))]


CH_d = pd.DataFrame(dict(CamaAd14Dmin=CH_hoy[0],CamaAd14Dmed=CH_hoy[1],CamaAd14Dax=CH_hoy[2],CamaAd21Dmin=CH_hoy[3],CamaAd21Dmed=CH_hoy[4],CamaAd21Dmax=CH_hoy[5]))
CV_d = pd.DataFrame(dict(VMIAd14Dmin=CV_hoy[0],VMIAd14Dmed=CV_hoy[1],VMIAd14Dmax=CV_hoy[2],VMIAd21Dmin=CV_hoy[3],VMIAd21Dmed=CV_hoy[4],VMIAd21Dmax=CV_hoy[5]))

# ------------------------------ #
#    Necesidad total de Camas    #
# ------------------------------ #

H_bed_hoy = [H_bed[i][idx[i][0:days]] for i in range(len(input))]
H_vent_hoy = [H_vent[i][idx[i][0:days]] for i in range(len(input))]
CH = [sims[i][0].CH for i in range(len(input))]
CV = [sims[i][0].CV for i in range(len(input))]

CH_hoy = [CH[i][idx[i][0:days]] for i in range(len(input))]
CV_hoy = [CV[i][idx[i][0:days]] for i in range(len(input))]

totalbed = [CH_hoy[i] + H_bed_hoy[i] for i in range(len(input))]
totalvmi= [CV_hoy[i] + H_vent_hoy[i] for i in range(len(input))]

totbed_d = pd.DataFrame(dict(TotCamas14Dmin=totalbed[0],TotCamas14Dmed=totalbed[1],TotCamas14Dmax=totalbed[2],TotCamas21Dmin=totalbed[3],TotCamas21Dmed=totalbed[4],TotCamas21Dmax=totalbed[5]))
totvmi_d = pd.DataFrame(dict(TotVMI14Dmin=totalvmi[0],TotVMI14Dmed=totalvmi[1],TotVMI14Dmax=totalvmi[2],TotVMI21Dmin=totalvmi[3],TotVMI21Dmed=totalvmi[4],TotVMI21Dmax=totalvmi[5]))




# ------------------------- #
#     Create Data Frame     #
# ------------------------- #

index = pd.DataFrame(dict(dates=index))
datosochimi = pd.concat([index,Hbed,Hvent,CH_d,CV_d,totbed_d,totvmi_d], axis=1, sort=False) 
datosochimi = datosochimi.set_index('dates')
datosochimi.to_excel('/home/samuel/Covid_Net_SEIR/SEIRHVDB/Plots/ReporteSochimi25-05/datos.xls') 