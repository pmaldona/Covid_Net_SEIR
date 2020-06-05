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


# To Do
# Tirar las camas al 30 de Junio - 46 dias 



# ---------------------------------- #
#    Camas adicionales Requeridas    #
# ---------------------------------- #


dates = list(range(15,32))
dates.extend(list(range(1,31)))
dates = [str(i) for i in dates]
days = len(dates)
endD = [np.searchsorted(t[i],range(time)) for i in range(len(input))]
idx = [np.searchsorted(t[i],range(days+1)) for i in range(len(input))]
endD = idx

CH = [sims[i][0].CH for i in range(len(input))]
CV = [sims[i][0].CV for i in range(len(input))]


CH_hoy = [CH[i][idx[i][0:(days)]] for i in range(len(input))]
#plt.plot(dates,CH_hoy,color='red',linestyle='dotted',linewidth = 3.0)

H_bed_hoy = H_bed[i][idx[i][0:days]] 
H_in_hoy = H_in[i][idx[i][0:days]] 
H_out_hoy = H_out[i][idx[i][0:days]] 
H_crin_hoy = H_crin[i][idx[i][0:days]] 
H_vent_hoy = H_vent[i][idx[i][0:days]] 


i=0
plt.plot(t[i][idx[i]],CH[i][idx[i]],color='red',linestyle='dotted',linewidth = 3.0)
plt.plot(t[i][idx[i]],CV[i][idx[i]],color='blue',linestyle='dotted',linewidth = 3.0)
i=1
plt.plot(t[i][idx[i]],CH[i][idx[i]],color='red',linestyle='solid',linewidth = 3.0)
plt.plot(t[i][idx[i]],CV[i][idx[i]],color='blue',linestyle='solid',linewidth = 3.0)
i=2
plt.plot(t[i][idx[i]],CH[i][idx[i]], color='red',linestyle='dotted',linewidth = 3.0)
plt.plot(t[i][idx[i]],CV[i][idx[i]], color='blue',linestyle='dotted',linewidth = 3.0)
#plot(title = 'Camas adicionales 14D',ylabel='Camas',xlabel = 'Días desde el 15 de Mayo')

#plt.xlim((0, 30))
#plt.ylim((0, 2000))
plt.savefig('/home/samuel/Covid_Net_SEIR/SEIRHVDB/Plots/ReporteSochimi25-05/CamasAdicionales14D.png', dpi=300)
plot(title = 'Camas adicionales 14D',ylabel='Camas',xlabel = 'Días desde el 15 de Mayo')



# 21 dias de cuarentena inicial
i=3
plt.plot(t[i][:idx[i][-1]],CH[i][:idx[i][-1]], color='red',linestyle='dotted',linewidth = 3.0)
plt.plot(t[i][:idx[i][-1]],CV[i][:idx[i][-1]], color='blue',linestyle='dotted',linewidth = 3.0)
i=4
plt.plot(t[i][:idx[i][-1]],CH[i][:idx[i][-1]], color='red',linestyle='solid',linewidth = 3.0)
plt.plot(t[i][:idx[i][-1]],CV[i][:idx[i][-1]], color='blue',linestyle='solid',linewidth = 3.0)
i=5
plt.plot(t[i][:idx[i][-1]],CH[i][:idx[i][-1]], color='red',linestyle='dotted',linewidth = 3.0)
plt.plot(t[i][:idx[i][-1]],CV[i][:idx[i][-1]], color='blue',linestyle='dotted',linewidth = 3.0)
#plot(title = 'Camas adicionales 21D',ylabel='Camas',xlabel = 'Días desde el 15 de Mayo')
plt.xlim((0, 45))
#plt.ylim((0, 2000))
plt.savefig('/home/samuel/Covid_Net_SEIR/SEIRHVDB/Plots/ReporteSochimi25-05/CamasAdicionales21D.png', dpi=300)
plot(title = 'Camas adicionales 14D',ylabel='Camas',xlabel = 'Días desde el 15 de Mayo')


# ------------------ #
#    Uso de Camas    #
# ------------------ #


dates = list(range(15,32))  
dates.extend(list(range(1,31)))
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
plt.plot(t[i][:idx[i][-1]],H_bed[i][:idx[i][-1]],color='red',linestyle='dotted',linewidth = 3.0)
plt.plot(t[i][:idx[i][-1]],H_vent[i][:idx[i][-1]],color='blue',linestyle='dotted',linewidth = 3.0)
i=1
plt.plot(t[i][:idx[i][-1]],H_bed[i][:idx[i][-1]],color='red',linestyle='solid',linewidth = 3.0)
plt.plot(datessim,H_vent_hoy[i],color='blue',linestyle='solid',linewidth = 3.0)
i=2
plt.plot(t[i][:idx[i][-1]],H_bed[i][:idx[i][-1]],color='red',linestyle='dotted',linewidth = 3.0)
plt.plot(datessim,H_vent_hoy[i],color='blue',linestyle='dotted',linewidth = 3.0)

#plt.ylim((600, 2800))
plt.savefig('/home/samuel/Covid_Net_SEIR/SEIRHVDB/Plots/ReporteSochimi25-05/UsoCamas14D.png', dpi=300)
plot(title = 'Uso de camas 14D',ylabel='Camas',xlabel = 'Días desde el 15 de Mayo')

# 21 dias de cuarentena inicial
#i=3
#plt.plot(t[i][:idx[i][-1]],H_bed[i][:idx[i][-1]],color='red',linestyle='dotted',linewidth = 3.0)
#plt.plot(t[i][:idx[i][-1]],H_vent[i][:idx[i][-1]],color='blue',linestyle='dotted',linewidth = 3.0)
i=4
plt.plot(t[i][:idx[i][-1]],H_bed[i][:idx[i][-1]],color='red',linestyle='solid',linewidth = 3.0)
plt.plot(t[i][:idx[i][-1]],H_vent[i][:idx[i][-1]],color='blue',linestyle='solid',linewidth = 3.0)
i=5
plt.plot(t[i][:idx[i][-1]],H_bed[i][:idx[i][-1]],color='red',linestyle='dotted',linewidth = 3.0)
plt.plot(t[i][:idx[i][-1]],H_vent[i][:idx[i][-1]],color='blue',linestyle='dotted',linewidth = 3.0)

#plt.ylim((600, 2800))
plt.xlim(0,45)
plt.savefig('/home/samuel/Covid_Net_SEIR/SEIRHVDB/Plots/ReporteSochimi25-05/UsoCamas21D.png', dpi=300)
plot(title = 'Uso de camas 21D',ylabel='Camas',xlabel = 'Días desde el 15 de Mayo')




# ------------------------------ #
#    Necesidad total de Camas    #
# ------------------------------ #


dates = list(range(15,32))  
dates.extend(list(range(1,31)))
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

totalbed = [CH_hoy[i] + CH[i] for i in range(len(input))]
totalvmi= [CV_hoy[i] + CV[i] for i in range(len(input))]

totalbed = [H_bed[i] + CH[i] for i in range(len(input))]
totalvmi= [H_vent[i] + CV[i] for i in range(len(input))]



# 21 dias de cuarentena inicial
i=3
plt.plot(t[i][:idx[i][-1]],totalbed[i][:idx[i][-1]],color='red',linestyle='dotted',linewidth = 3.0)
plt.plot(t[i][:idx[i][-1]],totalvmi[i][:idx[i][-1]],color='blue',linestyle='dotted',linewidth = 3.0)
i=4
plt.plot(t[i][:idx[i][-1]],totalbed[i][:idx[i][-1]],color='red',linestyle='solid',linewidth = 3.0)
plt.plot(t[i][:idx[i][-1]],totalvmi[i][:idx[i][-1]],color='blue',linestyle='solid',linewidth = 3.0)
i=5
plt.plot(t[i][:idx[i][-1]],totalbed[i][:idx[i][-1]],color='red',linestyle='dotted',linewidth = 3.0)
plt.plot(t[i][:idx[i][-1]],totalvmi[i][:idx[i][-1]],color='blue',linestyle='dotted',linewidth = 3.0)

#plt.ylim((0, 6000))
plt.xlim(0,45)
plt.savefig('/home/samuel/Covid_Net_SEIR/SEIRHVDB/Plots/ReporteSochimi25-05/NecTotalCamas21D.png', dpi=300)
plot(title = 'Necesidad total de camas 21D',ylabel='Camas',xlabel = 'Días desde el 15 de Mayo')


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

#plt.ylim((0, 6000))
plt.savefig('/home/samuel/Covid_Net_SEIR/SEIRHVDB/Plots/ReporteSochimi25-05/NecTotalCamas14D.png', dpi=300)
plot(title = 'Necesidad total de camas 14D',ylabel='Camas',xlabel = 'Días desde el 15 de Mayo')




# --------------------------------------------- #
#    Camas Reales vs simuladas al 30 de Junio   #
# --------------------------------------------- #
#i = 0 # 60% movilidad
i = 1 # 70% movilidad
days = 46

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


H_bed_hoy = [H_bed[i][idx[i][0:days]] for i in range(len(input))] 
H_in_hoy = [H_in[i][idx[i][0:days]] for i in range(len(input))] 
H_out_hoy = [H_out[i][idx[i][0:days]] for i in range(len(input))] 
H_crin_hoy = [H_crin[i][idx[i][0:days]] for i in range(len(input))] 
H_vent_hoy = [H_vent[i][idx[i][0:days]] for i in range(len(input))] 

i = 4
i=4
plt.plot(t[i][:idx[i][-1]],H_bed[i][:idx[i][-1]],color='red',linestyle='solid',linewidth = 3.0)
plt.plot(t[i][:idx[i][-1]],H_vent[i][:idx[i][-1]],color='blue',linestyle='solid',linewidth = 3.0)

err_bed = LA.norm(Hr_bed_hoy-H_bed_hoy[i][0:len(Hr_bed_hoy)])/LA.norm(Hr_bed_hoy)
err_vent = LA.norm(Hr_vent_hoy-H_vent_hoy[i][0:len(Hr_vent_hoy)])/LA.norm(Hr_vent_hoy)
plt.plot([], [], ' ', label='err_camas: '+str(round(100*err_bed,2))+'%')
plt.plot([], [], ' ', label='err_VMI: '+str(round(100*err_vent,2))+'%')
#plt.plot(datessim,H_bed_hoy[i],label='sim Inter/Inte',color='red')
#plt.plot(datessim,H_vent_hoy[i],label='sim VMI',color='blue')
plt.scatter(dates,Hr_bed_hoy,label='real Inter/Inten',color='red')
plt.scatter(dates,Hr_vent_hoy,label='real VMI',color='blue')

plt.xlim(0,45)
plt.savefig('/home/samuel/Covid_Net_SEIR/SEIRHVDB/Plots/ReporteSochimi25-05/CamasRealvsSim.png', dpi=300)
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
index = [(base+timedelta(days=i)).strftime("%d/%m/%Y") for i in range(47)]
dates = list(range(15,32))  
dates.extend(list(range(1,31)))
dates = [str(i) for i in dates]
days = len(dates)
idx = [np.searchsorted(t[i],range(days+1)) for i in range(len(inputarray))]


# ------------------ #
#    Uso de Camas    #
# ------------------ #

#H_bed_hoy = [H_bed[i][idx[i][0:days]] for i in range(len(input))]
UsoCamas = dict()
namelist = ['UsoCamas21Dmin','UsoCamas21Dmed','UsoCamas21Dmax']
for i in [3,4,5]:        
    UsoCamas_hoy = [H_bed[i][0]]
    for j in range(1,len(idx[i])-1):
        if idx[i][j-1]== idx[i][j]:
            UsoCamas_hoy.extend([(H_bed[i][idx[i][j-1]]+H_bed[i][idx[i][j+1]])/2])
        else:        
            UsoCamas_hoy.extend([H_bed[i][idx[i][j]]])
    UsoCamas_hoy.extend([H_bed[i][idx[i][-1]]])
    UsoCamas[namelist[i-3]]=UsoCamas_hoy

Hbed = pd.DataFrame(UsoCamas)

#H_vent_hoy = [H_vent[i][idx[i][0:days]] for i in range(len(input))]
namelist = ['UsoVMI21Dmin','UsoVMI21Dmed','UsoVMI21Dmax']
UsoVMI = dict()
for i in [3,4,5]:        
    UsoVMI_hoy = [H_vent[i][0]]
    for j in range(1,len(idx[i])-1):
        if idx[i][j-1]== idx[i][j]:
            UsoVMI_hoy.extend([(H_vent[i][idx[i][j-1]]+H_vent[i][idx[i][j+1]])/2])
        else:        
            UsoVMI_hoy.extend([H_vent[i][idx[i][j]]])
    UsoVMI_hoy.extend([H_vent[i][idx[i][-1]]])
    UsoVMI[namelist[i-3]]=UsoVMI_hoy

Hvent = pd.DataFrame(UsoVMI)

#Hbed = pd.DataFrame(dict(UsoCamas14Dmin=H_bed_hoy[0],UsoCamas14Dmed=H_bed_hoy[1],UsoCamas14Dmax=H_bed_hoy[2],UsoCamas21Dmin=H_bed_hoy[3],UsoCamas21Dmed=H_bed_hoy[4],UsoCamas21Dmax=H_bed_hoy[5]))
#Hvent = pd.DataFrame(dict(UsoVMI14Dmin=H_vent_hoy[0],UsoVMI14Dmed=H_vent_hoy[1],UsoVMI14Dmax=H_vent_hoy[2],UsoVMI21Dmin=H_vent_hoy[3],UsoVMI21Dmed=H_vent_hoy[4],UsoVMI21Dmax=H_vent_hoy[5]))

#Hbed = pd.DataFrame(dict(UsoCamas21Dmin=H_bed_hoy[3],UsoCamas21Dmed=H_bed_hoy[4],UsoCamas21Dmax=H_bed_hoy[5]))
#Hvent = pd.DataFrame(dict(UsoVMI21Dmin=H_vent_hoy[3],UsoVMI21Dmed=H_vent_hoy[4],UsoVMI21Dmax=H_vent_hoy[5]))


# ---------------------------------- #
#    Camas adicionales Requeridas    #
# ---------------------------------- #

CH = [sims[i][0].CH for i in range(len(input))]
CV = [sims[i][0].CV for i in range(len(input))]
#CH_hoy = [CH[i][idx[i][0:days]] for i in range(len(input))]
#CV_hoy = [CV[i][idx[i][0:days]] for i in range(len(input))]

CH_d = dict()
namelist = ['CamaAd21Dmin','CamaAd21Dmed','CamaAd21Dmax']
for i in [3,4,5]:        
    CH_hoy = [CH[i][0]]
    for j in range(1,len(idx[i])-1):
        if idx[i][j-1]== idx[i][j]:
            CH_hoy.extend([(CH[i][idx[i][j-1]]+CH[i][idx[i][j+1]])/2])
        else:        
            CH_hoy.extend([CH[i][idx[i][j]]])
    CH_hoy.extend([CH[i][idx[i][-1]]])
    CH_d[namelist[i-3]]=CH_hoy

CH_d = pd.DataFrame(CH_d)


CV_d = dict()
namelist = ['VMIAd21Dmin','VMIAd21Dmed','VMIAd21Dmax']
for i in [3,4,5]:        
    CV_hoy = [CV[i][0]]
    for j in range(1,len(idx[i])-1):
        if idx[i][j-1]== idx[i][j]:
            CV_hoy.extend([(CV[i][idx[i][j-1]]+CV[i][idx[i][j+1]])/2])
        else:        
            CV_hoy.extend([CV[i][idx[i][j]]])
    CV_hoy.extend([CV[i][idx[i][-1]]])
    CV_d[namelist[i-3]]=CV_hoy

CV_d = pd.DataFrame(CV_d)


#CH_d = pd.DataFrame(dict(CamaAd21Dmin=CH_hoy[3],CamaAd21Dmed=CH_hoy[4],CamaAd21Dmax=CH_hoy[5]))
#CV_d = pd.DataFrame(dict(VMIAd21Dmin=CV_hoy[3],VMIAd21Dmed=CV_hoy[4],VMIAd21Dmax=CV_hoy[5]))

#CH_d = pd.DataFrame(dict(CamaAd14Dmin=CH_hoy[0],CamaAd14Dmed=CH_hoy[1],CamaAd14Dax=CH_hoy[2],CamaAd21Dmin=CH_hoy[3],CamaAd21Dmed=CH_hoy[4],CamaAd21Dmax=CH_hoy[5]))
#CV_d = pd.DataFrame(dict(VMIAd14Dmin=CV_hoy[0],VMIAd14Dmed=CV_hoy[1],VMIAd14Dmax=CV_hoy[2],VMIAd21Dmin=CV_hoy[3],VMIAd21Dmed=CV_hoy[4],VMIAd21Dmax=CV_hoy[5]))

# ------------------------------ #
#    Necesidad total de Camas    #
# ------------------------------ #
namelistUsoC = ['UsoCamas21Dmin','UsoCamas21Dmed','UsoCamas21Dmax']
namelistUsoV = ['UsoVMI21Dmin','UsoVMI21Dmed','UsoVMI21Dmax']

namelistCH = ['CamaAd21Dmin','CamaAd21Dmed','CamaAd21Dmax']
namelistCV = ['VMIAd21Dmin','VMIAd21Dmed','VMIAd21Dmax']

namelistcamas = ['TotCamas21Dmin','TotCamas21Dmed','TotCamas21Dmax']
namelistvmi = ['TotVMI21Dmin','TotVMI21Dmed','TotVMI21Dmax']

totbed_d = pd.DataFrame()
totvmi_d = pd.DataFrame()
for i in range(len(namelistCH)):
    totbed_d[namelistcamas[i]] = CH_d[namelistCH[i]] + Hbed[namelistUsoC[i]]
    totvmi_d[namelistvmi[i]] = CV_d[namelistCV[i]] + Hvent[namelistUsoV[i]]



#CV_d['VMIAd21Dmin']+CH_d['CamaAd21Dmin'] 

#H_bed_hoy = [H_bed[i][idx[i][0:days]] for i in range(len(input))]
#H_vent_hoy = [H_vent[i][idx[i][0:days]] for i in range(len(input))]
#CH = [sims[i][0].CH for i in range(len(input))]
#CV = [sims[i][0].CV for i in range(len(input))]

#CH_hoy = [CH[i][idx[i][0:days]] for i in range(len(input))]
#CV_hoy = [CV[i][idx[i][0:days]] for i in range(len(input))]

#totalbed = [CH_hoy[i] + H_bed_hoy[i] for i in range(len(input))]
#totalvmi= [CV_hoy[i] + H_vent_hoy[i] for i in range(len(input))]

#totbed_d = pd.DataFrame(dict(TotCamas21Dmin=totalbed[3],TotCamas21Dmed=totalbed[4],TotCamas21Dmax=totalbed[5]))
#totvmi_d = pd.DataFrame(dict(TotVMI21Dmin=totalvmi[3],TotVMI21Dmed=totalvmi[4],TotVMI21Dmax=totalvmi[5]))

#totbed_d = pd.DataFrame(dict(TotCamas14Dmin=totalbed[0],TotCamas14Dmed=totalbed[1],TotCamas14Dmax=totalbed[2],TotCamas21Dmin=totalbed[3],TotCamas21Dmed=totalbed[4],TotCamas21Dmax=totalbed[5]))
#totvmi_d = pd.DataFrame(dict(TotVMI14Dmin=totalvmi[0],TotVMI14Dmed=totalvmi[1],TotVMI14Dmax=totalvmi[2],TotVMI21Dmin=totalvmi[3],TotVMI21Dmed=totalvmi[4],TotVMI21Dmax=totalvmi[5]))




# ------------------------- #
#     Create Data Frame     #
# ------------------------- #

index = pd.DataFrame(dict(dates=index))
datosochimi = pd.concat([index,Hbed,Hvent,CH_d,CV_d,totbed_d,totvmi_d], axis=1, sort=False) 
datosochimi = datosochimi.set_index('dates')
datosochimi.to_excel('/home/samuel/Covid_Net_SEIR/SEIRHVDB/Plots/ReporteSochimi25-05/datosReporteSochimi.xls') 