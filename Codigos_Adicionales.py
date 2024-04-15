"""
Thank you to the following github link for the information how to create the surface mesh.
https://github.com/SDSearch/bem_electrostatics
"""
import parmed.amber
import numpy as np 
from pathlib import Path
import sys
sys.path.append('/home/mauricioguerrero/Software/bem_electrostatics-master/')
import bem_electrostatics
import pygamer
import trimesh
import os 
      
def MallaSuperficial(file,de,pr,f,FE,FS,f0):
    # file: file name in text format
    # FE: address where the pqr is located. Ex: 'Sphere/PQR/'
    # FS: address where the mesh is saved. Ex: 'Sphere/Mallas_S/'
    # Choice of mesh creation
    if f=='n':
        fe= "nanoshaper" 
    elif f=='m':
        fe= "msms"    
    protein = bem_electrostatics.solute(
                solute_file_path="/home/mauricioguerrero/"+FE+file+".pqr", #pqr to generate the mesh
                save_mesh_build_files = True,  
                mesh_build_files_dir = "/home/mauricioguerrero/"+FS, # folder to save the mesh
                mesh_density = de, # density(msms) or length(nanoshaper)
                mesh_probe_radius = pr, # test sphere radio
                mesh_generator = fe, # method to create the mesh
                print_times = True)
    V = protein.mesh.number_of_vertices  # Vertices
    E = protein.mesh.number_of_elements  # Elements
    os.remove("/home/mauricioguerrero/"+FS+file+".face")
    os.remove("/home/mauricioguerrero/"+FS+file+".vert")
    os.remove("/home/mauricioguerrero/"+FS+file+".xyzr")
    os.rename("/home/mauricioguerrero/"+FS+file+".off","/home/mauricioguerrero/"+FS+file+f0+".off")
    return V,E

def MallaVolumetrica(file,file1_O,file2_O,file3_O,F,Tm,Vol,FS,FSV):
    # file: name of the final file in text format
    # FS: address where the surface mesh is located. Ex: 'Sphere/Mallas_S/'
    # FSV: address where the volumetric mesh is saved. Ex: 'Sphere/Mallas_V/'
    # file1_O,file2_O,file3_O: Inner, outer and intermediate surface mesh respectively.
    # Tm: The numerical value allows you to choose the minimum length of the tetrahedron to generate the vertices of the mesh, with Tm>=1.0
    # F: Choice of volumetric mesh model 1M, 2M or 3M
    # Vol: Maximum volume of the creation of the tetrahedron. If not considered, place ''.
    
    # Call the surface meshes from off files to generate the volumetric mesh
    mesh0 = pygamer.readOFF("/home/mauricioguerrero/"+FS+file1_O) #Surface inner mesh
    mesh_split = mesh0.splitSurfaces() #If the surface mesh has gaps, with trimesh you can obtain the mesh without gaps
    print("Found %i meshes in 1"%len(mesh_split))
    
    mesh1 = mesh_split[0]  #Inner surface mesh
    mesh1.correctNormals() # Correct the normal vector of the meshes
    gInfo = mesh1.getRoot() # Mesh information 1
    
    if F=='1M': 
        gInfo.ishole = False         
        meshes = [mesh1] # Build a list of meshes for the TetGen function
        L =[mesh1.nVertices,mesh1.nFaces]
    else:
        gInfo.ishole = True       
        mesh2 = pygamer.readOFF("/home/mauricioguerrero/"+FS+file2_O) #Outer surface mesh
        mesh2.correctNormals()   
        print("Found %i meshes in 2"%len(mesh2.splitSurfaces()))
        gInfo = mesh2.getRoot() # Mesh information 2
        gInfo.ishole = False
        L =[mesh1.nVertices,mesh1.nFaces,mesh2.nVertices,mesh2.nFaces]
        if F=='2M':
            meshes = [mesh1,mesh2] 
        elif F=='3M': 
            mesh3 = pygamer.readOFF("/home/mauricioguerrero/"+FS+file3_O)  #Inner surface mesh (Only if necessary)
            print("Found %i meshes in 3"%len(mesh3.splitSurfaces()))
            meshes = [mesh1,mesh3,mesh2]
    
    #Tetrahedralize the list of surface meshes to generate the volumetric mesh with TetGen.
    tetmesh = pygamer.makeTetMesh(meshes, '-pq'+str(Tm)+'a'+str(Vol)+'YAO2/3')  
    pygamer.writeDolfin("/home/mauricioguerrero/"+FSV+"outputTetMesh"+file+".xml", tetmesh)
    V = tetmesh.nVertices
    return V,L


# Case to create the .pqr of the Mobley molecules
def PQR_Moleculas_Pequeñas(file,d,FP,FC,FS):  
    # file: file name in text format
    #FP: address where the prmcrd folder is located. Ex: 'Mobley/prmcrd/'
    #FC: address where the charged_mol2files folder is located. Ex: 'Mobley/charged_mol2files/'
    #FS: address where the pqr will be saved. Ex: 'Mobley/PQR/'
    #d: additional distance for the radius to create the mesh with exclusion  radius
    
    #Extraction of the radius information from '.prmtop'
    mol_param = parmed.amber.AmberParm(FP+ file + '.prmtop')

    N_atom = mol_param.ptr('NATOM')
    atom_type = mol_param.parm_data['ATOM_TYPE_INDEX']
    atom_radius = np.zeros(N_atom)
    atom_depth = np.zeros(N_atom)

    for i in range(N_atom):
        atom_radius[i] = mol_param.LJ_radius[atom_type[i]-1]
        atom_depth[i] = mol_param.LJ_depth[atom_type[i]-1]

    #Extraction of the information of the charges of '.mol2'
    atom_charges = []
    with open(FC + file +".mol2","r") as f:
        lines = f.readlines()
    for line in lines:
        line = line.split()
        if len(line)==9:
            atom_charges.append(float(line[8]))
        
    #Extraction of the position information of the atoms '.prmtop'
    atom_posX = []
    atom_posY = []
    atom_posZ = []
    with open(FP+ file +".crd","r") as f:
        lines = f.readlines()
    for line in lines:
        line = line.split()
        if len(line)==6:        
            atom_posX.append(float(line[0]))
            atom_posY.append(float(line[1]))
            atom_posZ.append(float(line[2]))
            atom_posX.append(float(line[3]))
            atom_posY.append(float(line[4]))
            atom_posZ.append(float(line[5]))
        elif len(line)==3:        
            atom_posX.append(float(line[0]))
            atom_posY.append(float(line[1]))
            atom_posZ.append(float(line[2]))
            
    #Creation of the PQR
    if d==0:
        f0=".pqr"
    else:
        if len(str(d))<3:
            f0="R"+str(d)+".pqr"
        else:
            f0="R"+str(int(d*10))+".pqr"
    
    FF=open(FS+file +f0,"w")
    for i in range(len(atom_radius)):   
        x="{:.3f}".format(atom_posX[i])
        y="{:.3f}".format(atom_posY[i])
        z="{:.3f}".format(atom_posZ[i])
        r="{:.3f}".format(atom_radius[i]+d)
        q="{:.4f}".format(atom_charges[i])
        if len(x)==5:
            x=" "+x
        if len(y)==5:
            y=" "+y
        if len(z)==5:
            z=" "+z
        if len(q)==6:
            q=" "+q
        if len(r)==5:
            r=" "+r
        FF.write("ATOM      1  C1  TMP  1      "+x+"  "+y+"  "+z+"  "+q+" "+r+"\n")  #To correctly read the information, you must have this writing configuration.
    FF.close()
    return d

#Modification of the .pqr of large proteins (type 1A3N) extracted from PDB2PQR, so that the information can be read correctly
def NuevoPQR(file,d,FE,FS):      
    # file: file name in text format
    # FE: address where the pqr is located. Ex: 'Binding_Energy/PQR/'
    # FS: address where the modified pqr is saved. Ex: 'Binding_Energy/Mallas_S/'
    # d: additional distance for the radius to create the mesh with exclusion radius
    Tex = []
    posX = []
    posY = []
    posZ = []
    posQ = []
    posR = []            
    with open(FE+file+"Base.pqr","r") as f:
        lines = f.readlines()
        Tex.append(lines)
    for line in lines:
        line = line.split()
        if len(line)==10:    
            posX.append(line[5])
            posY.append(line[6])
            posZ.append(line[7])
            posQ.append(line[8])
            posR.append(line[9]) 
            
    #Creation of the PQR
    if d==0:
        f0=".pqr"
    else:
        f0="R"+str(d)+".pqr"
    
    FF=open(FS+file +f0,"w")
    for i in range(len(posX)):   
        t=Tex[0][i][0:28]  
        x=str(posX[i])
        y=str(posY[i])
        z=str(posZ[i])
        q="{:.4f}".format(float(posQ[i]))
        r="{:.4f}".format(float(posR[i])+d)
        if len(x)==5:
            x="  "+x
        if len(y)==5:
            y="  "+y
        if len(z)==5:
            z="  "+z
        if len(x)==6:
            x=" "+x
        if len(y)==6:
            y=" "+y
        if len(z)==6:
            z=" "+z        
        if len(q)==6:
            q=" "+q
        if len(r)==6:
            r=" "+r
        FF.write(t+x+" "+y+" "+z+"  "+q+" "+r+"\n")  
    FF.close()
    return d

# Case .pqr for large molecules like 1SE0 and 1BBZ
def PQR_Moleculas_de_Union(file,d,FE,FS):  
    # file: file name in text format
    # FE: address where the prmtop and pdb files are located. Ex: 'Binding_Energy/top/'
    # FS: address where the modified pqr is saved. Ex: 'Binding_Energy/PQR2/'
    # d: additional distance for the radius to create the mesh with exclusion radius
    if file=='1SE0-L':
        N1='PROA'
        N2='PROB'
        fileL='L1SE0'
        file1='1SE0'
        k=3
        k1=1
        k2=0
    elif file=='1BBZ-L':
        N1='SH3D'
        N2='PPRO'   
        fileL='L1BBZ'
        file1='1BBZ'
        k=0
        k1=0
        k2=1
    
    #Case, protein and protein+ligand data
    C=[]
    Tex = []
    posX = []
    posY = []
    posZ = []
    posQ = []        
    #Position, protein and ligand
    with open(FE+file+".pdb","r") as f:
        lines = f.readlines() 
        Tex.append(lines)
    for line in lines:
        line = line.split()
        if line[0]== 'ATOM':
            if line[11] == N1: 
                G=0
                posX.append(line[6])
                posY.append(line[7])
                posZ.append(line[8])
            elif line[11] == N2:
                posX.append(line[6])
                posY.append(line[7])
                posZ.append(line[8])    
                if G==0:
                    C.append(int(line[1])-1)
                    G=1
            else:       
                C.append(int(line[1])-1)
                break
      
    #Radio and Charge, protein and ligand   
    mol_param = parmed.amber.AmberParm(FE+ file + '.prmtop')
    N_atom = C[1] 
    atom_type = mol_param.parm_data['ATOM_TYPE_INDEX']
    atom_charge = mol_param.parm_data['CHARGE']
    posR = np.zeros(N_atom)
    for i in range(N_atom):
        posR[i] = mol_param.LJ_radius[atom_type[i]-1]          
        posQ.append(atom_charge[i])
 
    #Ligand
    CL=[]
    TexL = []
    posXL = []
    posYL = []
    posZL = []
    posQL = []      
    #Position of the ligand
    with open(FE+fileL+".pdb","r") as f:
        lines = f.readlines() 
        TexL.append(lines)
    for line in lines:
        line = line.split()
        if line[0]== 'ATOM':
            if line[11-k1] == N2: 
                posXL.append(line[6-k1])
                posYL.append(line[7-k1])
                posZL.append(line[8-k1])
            else:       
                CL.append(int(line[1])-1)
                break
    
    #Radius and Charge of the ligand  
    mol_paramL = parmed.amber.AmberParm(FE+ fileL + '.prmtop')
    N_atomL = CL[0] 
    atom_typeL = mol_paramL.parm_data['ATOM_TYPE_INDEX']
    atom_chargeL = mol_paramL.parm_data['CHARGE']
    posRL = np.zeros(N_atomL)
    for i in range(N_atomL):
        posRL[i] = mol_paramL.LJ_radius[atom_typeL[i]-1]          
        posQL.append(atom_chargeL[i])

    #Creation of the PQR
    if d==0:
        f0=".pqr"
    else:
        f0="R"+str(d)+".pqr"
    
    FF0=open(FS+file+f0,"w")  #Complex
    FF1=open(FS+file1+f0,"w")  #Protein
    FF2=open(FS+fileL+f0,"w")  #Ligand
    for i in range(len(posX)):  
        t=Tex[0][i+k][0:20]+" "+Tex[0][i+k][22:28]   
        x=str(posX[i])
        y=str(posY[i])
        z=str(posZ[i])
        q="{:.4f}".format(posQ[i])
        r="{:.4f}".format(float(posR[i])+d)
        if len(x)==5:
            x="  "+x
        if len(y)==5:
            y="  "+y
        if len(z)==5:
            z="  "+z
        if len(x)==6:
            x=" "+x
        if len(y)==6:
            y=" "+y
        if len(z)==6:
            z=" "+z        
        if len(q)==6:
            q=" "+q
        if len(r)==6:
            r=" "+r
        if i<C[0]:  
            FF1.write(t+x+" "+y+" "+z+"  "+q+" "+r+"\n")      
        FF0.write(t+x+" "+y+" "+z+"  "+q+" "+r+"\n")  
    FF0.close()
    FF1.close()
    
    for i in range(len(posXL)):  
        t=TexL[0][i+k2][0:20]+" "+TexL[0][i+k2][22:28]   
        x=str(posXL[i])
        y=str(posYL[i])
        z=str(posZL[i])
        q="{:.4f}".format(posQL[i])
        r="{:.4f}".format(float(posRL[i])+d)
        if len(x)==5:
            x="  "+x
        if len(y)==5:
            y="  "+y
        if len(z)==5:
            z="  "+z
        if len(x)==6:
            x=" "+x
        if len(y)==6:
            y=" "+y
        if len(z)==6:
            z=" "+z        
        if len(q)==6:
            q=" "+q
        if len(r)==6:
            r=" "+r
        FF2.write(t+x+" "+y+" "+z+"  "+q+" "+r+"\n")
    FF2.close()
    
    return d