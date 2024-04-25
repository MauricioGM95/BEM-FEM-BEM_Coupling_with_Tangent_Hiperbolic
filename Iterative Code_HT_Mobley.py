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
ks = 0.
es = 80.
em = 1.
ei = (es+em)/2 
ki = ks*np.sqrt(es/(es+em))
ResB=70
ResF=120
Tol = 1e-6

with open("Moleculas_Mobley.txt","r") as f: 
    lines = f.readlines()
for line in lines:
    line = line.split()
    LG.append(line[0])
print(LG)

KP = [0,1,2,'N'] #   ,2.5,2.75,3,3.25,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,9,10,11,12,13,14,15,20,25,30,40,50,60,80,100,200,400,'N']

# In[ ]:

start = time.time()
f1=open("Datos_Moleculas_de_Mobley_TanH_D8_Prueba.txt","w")
for i in range(len(LG)):
    file = LG[i]
    PQR = 'Mobley/PQR/'+file+'.pqr'
    MS1 = 'Mobley/Mallas_S/'+file+'D8.off'
    MS2 = 'Mobley/Mallas_S/'+file+'R15D8.off'  
    
    # Case BEM/BEM
    print(i,file,'BB')
    Sol1,T1,T1G,I1 = Caso_BEMBEM(PQR,MS1,es,em,ks,Tol, ResB, 'NR',SF=False, Asb='dn')
    f1.write(str(Sol1)+" "+str(T1)+" "+str(T1G)+" "+str(I1)+" "+str(LG[i])+" "+str(0)+" BEMBEMNR \n") 
    
    print(i,file,'BB15')
    Sol2,T2,T2G,I2 = Caso_BEMBEM(PQR,MS2,es,em,ks,Tol, ResB,'NR', SF=False, Asb='dn')
    f1.write(str(Sol2)+" "+str(T2)+" "+str(T2G)+" "+str(I2)+" "+str(LG[i])+" "+str(1.5)+" BEMBEMNR_R15 \n")    
    
    # Case BEM/BEM/BEM
    print(i,file,ei,'BBB')
    Sol4,T4,T4G,I4 = Caso_BEMBEMBEM(PQR,MS1,MS2,es,ei,em,ks,ki,Tol,ResB,SF=False,Asb='dn') 
    f1.write(str(Sol4)+" "+str(T4)+" "+str(T4G)+" "+str(I4)+" "+str(LG[i])+" "+str(ei)+" BEMBEMBEM \n")    
 
    for j in range(len(KP)): 
        kp = KP[j]
        FF =  'Mobley/Lista_A/Alfa_'+file+'D8.txt'
        MVG = 'Mobley/Mallas_V/outputTetMesh'+file+'D8.xml' 
        
        # Case BEM/FEM/BEM
        print(i,j,file,kp,'BFB')
        Sol3,T3,T3G,I3 = Caso_BEMFEMBEM(PQR,MVG,es,ei,em,ks,ki,kp,Tol,ResF,'VTH','dn',FF)        
        f1.write(str(Sol3)+" "+str(T3)+" "+str(T3G)+" "+str(I3)+" "+str(LG[i])+" "+str(KP[j])+" BEMFEMBEM \n")           
f1.close()
end = time.time()
curr_time = (end - start)   
print("Total program time: {:5.2f} [s]".format(curr_time))

# In[ ]:


# In[ ]:

