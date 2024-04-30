#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import bempp.api
from bempp.api.external import fenics
import numpy as np 
import time
import trimesh
from Codigos_BEM_y_FEM_Generalizadas import *

# In[ ]:

#Main data
ks = 0.125
es = 80.
em = 2.
Tol = 1e-5
ei = (es+em)/2 
ki = ks*np.sqrt(es/(es+em))
ResB=70
ResF=300
KP =[3]

LG =['L1SE0','1SE0','1SE0-L']  #Path: Mallas_S2, Mallas_V2, Lista_A2, PQR2
LG1 = ['dn','fmm','fmm']
#LG = ['1brsAmberP1','1brsAmberP2','1brsAmber']  #Path: Mallas_S, Mallas_V, Lista_A, PQR
#LG1 = ['fmm','fmm','fmm'] 
print(LG)

# In[ ]:

start = time.time()
f1=open("Datos_Moleculas_de_Union_TanH_D8_1SE0_k125_BBB.txt","w")
for i in range(len(KP)):
    kp = KP[i] 
    for j in range(len(LG)):  
        file = LG[j]
        Ab = LG1[j]
        PQR = 'Binding_Energy/PQR2/'+file+'.pqr'
        MS1 = 'Binding_Energy/Mallas_S2/'+file+'D8.off'
        MS0 = 'Binding_Energy/Mallas_S2/'+file+'R15D8.off' 
        FF =  'Binding_Energy/Lista_A2/Alfa_'+file+'D8.txt' 
        MVG = 'Binding_Energy/Mallas_V2/outputTetMesh'+file+'D8.xml'  
        print(j,i,file,em,Ab)       

        VV00,FF00 = read_off(MS1) 
        meshSP = trimesh.Trimesh(vertices = VV00, faces= FF00) 
        mesh_split = meshSP.split()
        print("Found %i meshes"%len(mesh_split))
        
        if len(mesh_split)==1:
            print('Sin Cavidad')
            Sol1,T1,T1G,I1 = Caso_BEMBEM(PQR,MS1,es,em,ks,Tol,ResB,'NR',SF=False, Asb=Ab)
            f1.write(str(Sol1)+" "+str(T1)+" "+str(T1G)+" "+str(I1)+" "+str(LG[j])+" "+str(KP[i])+" BEMBEMNR \n") 
        
            Sol2,T2,T2G,I2 = Caso_BEMBEM(PQR,MS0,es,em,ks,Tol,ResB,'NR',SF=False,Asb=Ab)
            f1.write(str(Sol2)+" "+str(T2)+" "+str(T2G)+" "+str(I2)+" "+str(LG[j])+" "+str(KP[i])+" BEMBEMNR_R15 \n")    
            
            Sol12,T12,T12G,I12 = Caso_BEMFEMBEM(PQR,MVG,es,ei,em,ks,ki,kp,Tol,ResF,'VTH',Ab,FF) 
            f1.write(str(Sol12)+" "+str(T12)+" "+str(T12G)+" "+str(I12)+" "+str(LG[j])+" "+str(KP[i])+" BEMFEMBEM \n")                     
        else:
            print('Cavidad')     
            Sol1,T1,T1G,I1 = Caso_BEMBEM_Cavidades(PQR,MS1,MS1,es,em,ks,Tol,ResB,SF=False,Asb=Ab) 
            f1.write(str(Sol1)+" "+str(T1)+" "+str(T1G)+" "+str(I1)+" "+str(LG[j])+" "+str(KP[i])+" BEMBEMBEMNR_C \n")            
            
            Sol2,T2,T2G,I2 = Caso_BEMBEM_Cavidades(PQR,MS0,MS1,es,em,ks,Tol,ResB,SF=False,Asb=Ab)  
            f1.write(str(Sol2)+" "+str(T2)+" "+str(T2G)+" "+str(I2)+" "+str(LG[j])+" "+str(KP[i])+" BEMBEMBEMNR_R15_C \n") 
            
            Sol13,T13,T13G,I13 = Caso_BEMFEMBEM_Cavidades(PQR,MVG,MS1,es,ei,em,ks,ki,kp,Tol,ResF,'VTH',Ab,FF)
            f1.write(str(Sol13)+" "+str(T13)+" "+str(T13G)+" "+str(I13)+" "+str(LG[j])+" "+str(KP[i])+" BEMBEMFEMBEM_C \n") 
        
f1.close()
end = time.time()
curr_time = (end - start)   
print("Total program time: {:5.2f} [s]".format(curr_time))


# In[ ]:





# In[ ]:




