#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import bempp.api
from bempp.api.external import fenics
import numpy as np 
import time
from Codigos_BEM_y_FEM_Generalizadas import *

# In[ ]:

#Main data
ks = 0.125
es = 80.
em = 4.
ei = (es+em)/2 
ki = ks*np.sqrt(es/(es+em))
ResB=70
ResF=120
Tol = 1e-6
KP = [0,0.05,0.1,0.25,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9,10,11,12,13,14,15,16,18,20,22,24,25,26,28,30,35,40,50,60,70,80,90,100,120,150,200,300,400,'N']

LG = ['R0','R1','R2','R3','R4'] 
LGB = ['5','65']
LGF = ['','T11','T11A0013']
PQR = 'Sphere/PQR/Sphere5Q3.pqr' 
print(KP)

# In[ ]:

start = time.time()
f1=open("Datos_Global_TanH_Esfera58_Prueba.txt","w")
for j in range(len(LG)):
    file1 = LG[j]
    MS1 = 'Sphere/Mallas_S/Sphere5'+file1+'.off' 
    MS2 = 'Sphere/Mallas_S/Sphere8'+file1+'.off' 
              
    # Case BEM/BEM/BEM
    print(j,file1,'BBB')
    Sol2,T2,T2G,I2 = Caso_BEMBEMBEM(PQR,MS1,MS2,es,ei,em,ks,ki,Tol,ResB,SF=False,Asb='dn') 
    f1.write(str(Sol2)+" "+str(T2)+" "+str(T2G)+" "+str(I2)+" "+str(LG[j])+" "+str(ei)+" BEMBEMBEM \n")     
    
    # Case BEM/BEM
    for i in range(len(LGB)):
        file0 = LGB[i]
        print(j,i,file1,file0,'BB')
        MS11 = 'Sphere/Mallas_S/Sphere'+file0+file1+'.off'
        Sol1,T1,T1G,I1 = Caso_BEMBEM(PQR,MS11,es,em,ks,Tol,ResB,'NR',SF=False,Asb='dn')      
        f1.write(str(Sol1)+" "+str(T1)+" "+str(T1G)+" "+str(I1)+" "+str(LG[j])+" "+str(LGB[i])+" BEMBEMNR \n") 
        
    # Case BEM/FEM/BEM    
    for i in range(len(KP)):    
        kp = KP[i]
        if file1=='R4':
            a=0
        else:
            a=1
        for l in range(len(LGF)-a):
            file2 = LGF[l] 
            print(j,i,l,file1,file2,kp,'BFB')
            MVG = 'Sphere/Mallas_V/outputTetMeshSphere58'+file1+file2+'.xml'
            FF =  'Sphere/Lista_A/Alfa_Sphere58'+file1+file2+'.txt'        
            Sol3,T3,T3G,I3 = Caso_BEMFEMBEM(PQR,MVG,es,ei,em,ks,ki,kp,Tol,ResF,'VTH','dn',FF) 
            f1.write(str(Sol3)+" "+str(T3)+" "+str(T3G)+" "+str(I3)+" "+str(LG[j])+" "+str(KP[i])+" BEMFEMBEM"+str(LGF[l])+" \n")     
f1.close()
end = time.time()
curr_time = (end - start)   
print("Total program time: {:5.2f} [s]".format(curr_time))

# In[ ]:

# In[ ]:

