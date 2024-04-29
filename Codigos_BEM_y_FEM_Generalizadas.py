#Library import section.
import bempp.api
import dolfin
from dolfin import Mesh
import numpy as np 
import numba as nb
import trimesh
import time
import psutil
from readoff import *
from readpqr import *
from preconditioners import *

################################################## Class Section ###################################################

# Class to show the GMRES iterations
class gmres_counter(object):
        def __init__(self, disp=True):
            self._disp = disp
            self.niter = 0
        def __call__(self, rk=None):
            self.niter += 1
            if (self.niter /50) == (self.niter // 50):
                print(self.niter,str(rk))                                    
            #if self._disp:
            #    print('iter %3i\trk = %s' % (self.niter, str(rk)))   
                
#Creation of the Coulomb potential in the form of a class for 3-Term FEM/BEM.
class Fun_ucfem(dolfin.UserExpression):  #Dolphin function 
    def __init__(self,PC,Q,em,**kwargs):
        super().__init__(**kwargs)
        self.PC = PC
        self.Q = Q
        self.em = em
    def eval(self, v, x):
        PC = self.PC
        Q = self.Q
        em = self.em
        v[0]= (1/(4.*np.pi*em))  * np.sum( Q / np.linalg.norm( x - PC, axis=1))
    def value_shape(self):
        return () 
    
#Definition of the variable functions of the intermediate domain.
#Case ei by variable by Hyperbolic Tangent. 
class Fun_ei_Tangente_Hiperbolica(dolfin.UserExpression):  
    def __init__(self,kp,L_Alfa,**kwargs):
        super().__init__(**kwargs)
        self.L_Alfa = L_Alfa
        self.kp = kp                
    def eval_cell(self, v, x, ufc_cell):
        L_Alfa = self.L_Alfa
        kp = self.kp 
        alfa = L_Alfa[ufc_cell.index]
        if kp=='N':  #Optional for the Hyperbolic Tangent function with kp equal to infinity.
            if alfa<0.5:
                v[0] = 1
            else:
                v[0] = 0        
        else:
            v[0] = 0.5-0.5*np.tanh(kp*(alfa-0.5)) #Hyperbolic tangent function for kp other than infinity.         
    def value_shape(self):
        return ()  
    
#Case ei for Linear variable.
class Fun_ei_Lineal(dolfin.UserExpression):  
    def __init__(self,L_Alfa,**kwargs):
        super().__init__(**kwargs)
        self.L_Alfa = L_Alfa     
    def eval_cell(self, v, x, ufc_cell):
        L_Alfa = self.L_Alfa
        v[0] = L_Alfa[ufc_cell.index]  #The calculated alpha corresponds to the same linear interpolation between the two surface meshes.
    def value_shape(self):
        return ()     
    
################################################## Function section ###################################################

#Coulomb energy 
def ECoulomb(PQR,em): 
    PC,Q,R = readpqr(PQR)
    C0=0.5*332.064*(1/em)
    EC=0
    for i in range(len(PC)):
        for j in range(len(PC)):    
            if i!=j:
                EC=EC+C0*Q[i]*Q[j]/(np.linalg.norm(PC[i]-PC[j], axis=0))
    return EC

#Coulomb potential
def PCoulomb(PQR,em,PC0): 
    PC,Q,R = readpqr(PQR)
    C0=1/(4.*np.pi*em)
    PCmb=0
    for i in range(len(PC)):
        D=np.linalg.norm(PC0-PC[i], axis=0)
        if D!=0:
            PCmb=PCmb+C0*Q[i]/D
    return PCmb

#Function to obtain the alpha parameter of the text and save it in a list.
def Lista_Alfa(File): 
    L_Alfa = []
    with open(File,"r") as f:  
        lines = f.readlines()
    for line in lines:
        line = line.split()
        L_Alfa.append(float(line[0]))
    return L_Alfa

#Creation of text files to calculate the Alpha value in each centroid of the tetrahedron of the volumetric mesh for BFB.
@nb.njit(fastmath=True)
def Linarg(x,P,c):
    d = np.zeros(len(P))
    for i in range(len(P)):
        p0 = P[i]
        x0 = x[0]-p0[0]
        if abs(x0)<c:
            y0 = x[1]-p0[1]
            if abs(y0)<c:
                z0 = x[2]-p0[2]
                if abs(z0)<c:
                    s = x0**2+y0**2+z0**2     
                else:
                    s=c**2
            else:
                s=c**2
        else:
            s=c**2
        d[i] = np.sqrt(s)
    return d   

#Creation of text files for the alpha parameter
def Alfa_BFB(MeshV,File,t):
    #File: path where it is saved of text files
    start = time.time()
    start01 = time.time() 
    
    #Creation of the volumetric mesh.
    mesh = Mesh(MeshV)  
    
    #Generate global functional spaces of the potential in Fem and its derivative in Bem
    from bempp.api.external import fenics
    fenics_space = dolfin.FunctionSpace(mesh, "CG", 1)  #Electrostatic potential at the interface and domain of the solute.
    trace_space, trace_matrix = \
        fenics.fenics_to_bempp_trace_data(fenics_space) #Global trace space to work in BEM and FEM simultaneously.
    
    #Code to identify vertices and faces of the inner and outer mesh.
    faces_0 = trace_space.grid.elements
    vertices_0 = trace_space.grid.vertices
    meshSP = trimesh.Trimesh(vertices = vertices_0.transpose(), faces= faces_0.transpose())
    mesh_split = meshSP.split()
    vertices_Ref = len(mesh_split[0].vertices)
    faces_Ref = len(mesh_split[0].faces)

    #Obtaining the inner surface mesh.
    faces_1 = faces_0.transpose()[:faces_Ref]
    vertices_1 = vertices_0.transpose()[:vertices_Ref]
    grid1 = bempp.api.grid.grid.Grid(vertices_1.transpose(), faces_1.transpose())

    #Obtaining the outer surface mesh.
    faces_2 = faces_0.transpose()[faces_Ref:]
    vertices_2 = vertices_0.transpose()[vertices_Ref:]
    grid2 = bempp.api.grid.grid.Grid(vertices_2.transpose(), (faces_2-len(vertices_1)).transpose())
 
    print("FEM dofs: {0}".format(mesh.num_vertices()))
    
    #Main data
    C0 = mesh.cells()
    D0 = mesh.coordinates()
    Pa = grid1.centroids
    Pb = grid2.centroids
    end01 = time.time()
    curr_time01 = (end01 - start01) 
    print("Previous time: {:5.2f} [s]".format(curr_time01))
    
    #Creation of the text file
    start01 = time.time() 
    f0=open(File,"w") # File to save the alpha text in BFB 
    print('Number of tetrahedra in the BFB mesh: {:7.0f}'.format(len(C0)))
    for j in range(len(C0)):
        if ((j/100000)==(j//100000)) and (j!=0):  
            end01 = time.time()
            curr_time01 = (end01 - start01) 
            print(len(C0),j,"Total time per cycle: {:5.2f} [s]".format(curr_time01))
            start01 = time.time() 
        X0 = C0[j]  
        x1 = D0[X0[0]]
        x2 = D0[X0[1]]
        x3 = D0[X0[2]]
        x4 = D0[X0[3]]
        x = 0.25*(x1+x2+x3+x4)
        da = Linarg(x,Pa,c=3.5) 
        db = Linarg(x,Pb,c=3.5)  
        ia = np.argsort(da)[:t]
        ib = np.argsort(db)[:t] 
        alfa0 = 0    
        for i in range(t):
            if da[ia[0]]<db[ib[0]]:
                iaa = ia[i]
                ibb = ib[0]
            else:
                ibb = ib[i]
                iaa = ia[0]
            N = np.dot((Pa[iaa]-x),(Pb[ibb]-Pa[iaa]))
            D = np.linalg.norm(Pb[ibb]-Pa[iaa], axis=0)**2
            alfa0 = alfa0-N/D 
        alfa = alfa0/t   
        f0.write(str(alfa)+" \n")
    f0.close()
    end01 = time.time()
    curr_time01 = (end01 - start01) 
    print(len(C0),"Total time per cycle: {:5.2f} [s]".format(curr_time01))
    end = time.time()
    curr_time = (end - start)   
    print("Total time: {:5.2f} [s]".format(curr_time))   
    return t   

##########################################################################################################################

def Caso_BEMBEM(PQR,Mesh1,es,em,ks,Tol,Res,Md,SF,Asb):       
    start = time.time()
    #Specific model choice for BEM/BEM
    if Md=='NR':
        print("Case BEM/BEM(NonRegularized)")
    elif Md=='R':
        print("Case BEM/BEM(Regularized)")
    elif Md=='3T':
        print("Case BEM/BEM(3Terms)")
        
    # Choice of border operator assemblies   
    if Asb == 'fmm':
        Assemble = 'fmm' 
    elif Asb == 'dn':
        Assemble = 'default_nonlocal' 
        
    #Data on the position in space, charge and radii of the atoms of the solute.
    PC,Q,R = readpqr(PQR) 
      
    #Generate the surface mesh of the solute
    vertices_0,faces_0 = read_off(Mesh1) 
    #In case the mesh has small gaps, with trimesh the information of the original mesh without the gaps is obtained.
    meshSP = trimesh.Trimesh(vertices = vertices_0, faces= faces_0) 
    mesh_split = meshSP.split()
    print("Found %i meshes"%len(mesh_split)) #1 mesh means no cavity.

    vertices_1 = mesh_split[0].vertices 
    faces_1 = mesh_split[0].faces   
    grid = bempp.api.grid.grid.Grid(vertices_1.transpose(), faces_1.transpose()) #Creation of the surface mesh.    
    
    #Generate functional spaces of the potential and its derivative
    dirichl_space = bempp.api.function_space(grid, "P", 1)   #Electrostatic potential at the interface.
    neumann_space = bempp.api.function_space(grid, "P", 1)   #Derived from the electrostatic potential at the interface.
    print("DS dofs: {0}".format(dirichl_space.global_dof_count))
    print("NS dofs: {0}".format(neumann_space.global_dof_count))  
    
    #Generate the boundary operators
    #Domain of the solute Ωm.
    IL = bempp.api.operators.boundary.sparse.identity(dirichl_space, dirichl_space, dirichl_space) #1
    KL = bempp.api.operators.boundary.laplace.double_layer(dirichl_space, dirichl_space, dirichl_space, assembler=Assemble) #K
    VL = bempp.api.operators.boundary.laplace.single_layer(neumann_space, dirichl_space, dirichl_space, assembler=Assemble) #V
    #Domain of the solvent Ωs.
    IH = bempp.api.operators.boundary.sparse.identity(dirichl_space, neumann_space, neumann_space) #1
    if ks==0:
        KH = bempp.api.operators.boundary.laplace.double_layer(dirichl_space, neumann_space, neumann_space, assembler=Assemble) #K
        VH = bempp.api.operators.boundary.laplace.single_layer(neumann_space, neumann_space, neumann_space, assembler=Assemble) #V
    else:
        KH = bempp.api.operators.boundary.modified_helmholtz.double_layer(dirichl_space, neumann_space, neumann_space, ks, assembler=Assemble) #K
        VH = bempp.api.operators.boundary.modified_helmholtz.single_layer(neumann_space, neumann_space, neumann_space, ks, assembler=Assemble) #V
        
    #Creation of Coulomb potential function and its derivative
    @bempp.api.complex_callable(jit=False)
    def U_c(x, n, domain_index, result):
        result[:] = (1 /(4*np.pi*em))  * np.sum( Q / np.linalg.norm( x - PC, axis=1))
    U_c = bempp.api.GridFunction(dirichl_space, fun=U_c)

    @bempp.api.complex_callable(jit=False)
    def dU_c(x, n, domain_index, result):
        result[:] = -(1/(4.*np.pi*em))   * np.sum( np.dot( x - PC , n)  * Q / (np.linalg.norm( x - PC , axis=1)**3) )
    dU_c = bempp.api.GridFunction(neumann_space, fun=dU_c)
      
    #Construction of the right vector
    if Md=='NR':
        if SF==False:
            # Rhs in Ωm.
            rhs_M = (U_c).projections(dirichl_space)
            # Rhs in Ωs.
            rhs_S = np.zeros(neumann_space.global_dof_count)
            # The combination of Rhs.
            rhs = np.concatenate([rhs_M, rhs_S])
        else:    
            rhs = [I1*U_c, 0*I2*U_c]
            
    elif Md=='R':
        if SF==False:
            # Rhs in Ωm
            rhs_M = np.zeros(dirichl_space.global_dof_count)
            # Rhs in Ωs
            rhs_S = ((KH - 0.5*IH)*U_c).projections(neumann_space) - (em/es)*(VH*dU_c).projections(neumann_space)
            # The combination of Rhs.
            rhs = np.concatenate([rhs_M, rhs_S])
        else:
            rhs = [0*IL*U_c, (KH - 0.5*IH)*U_c-(em/es)*(VH*dU_c)]

    elif Md=='3T':
        #Generate the boundary operators
        #Domain for harmonic potential
        KL0 = bempp.api.operators.boundary.laplace.double_layer(dirichl_space, neumann_space, neumann_space) #K
        VL0 = bempp.api.operators.boundary.laplace.single_layer(neumann_space, neumann_space, neumann_space) #V
        
        #Calculation of the derivative of the harmonic potential with GMRES
        U_h = -U_c
        rhs0 = ((0.5*IH + KL0)*U_h).projections(neumann_space)
        Auh = VL0.weak_form()
        
        count_iterations = gmres_counter()
        from scipy.sparse.linalg import gmres
        Sol, info = gmres(Auh, rhs0, M=None,callback=count_iterations,tol=Tol)
        print("Number of GMRES iterations of dU_h: {0}".format(count_iterations.niter))

        dU_h = bempp.api.GridFunction(neumann_space, coefficients=Sol)            
        dirichlet_uh_fun = U_h
        neumann_duh_fun = dU_h
        
        if SF==False:
            # Rhs in Ωm
            rhs_M = np.zeros(dirichl_space.global_dof_count)
            # Rhs in Ωs
            rhs_S = -(em/es)*(VH*dU_c).projections(neumann_space) - (em/es)*(VH*dU_h).projections(neumann_space)
            # The combination of Rhs.
            rhs = np.concatenate([rhs_M, rhs_S])
        else:
            rhs = [0*IL*U_c, -(em/es)*(VH*dU_c)-(em/es)*(VH*dU_h)]
    
    #Construction left 2x2 matrix
    if SF==False:
        #Position of the 2x2 matrix.
        blocks = [[None,None],[None,None]] 
        blocks[0][0] = (0.5*IL + KL).weak_form() 
        blocks[0][1] = -VL.weak_form()            
        blocks[1][0] = (0.5*IH - KH).weak_form()  
        blocks[1][1] = (em/es)*VH.weak_form()        
        blocked = bempp.api.assembly.blocked_operator.BlockedDiscreteOperator(np.array(blocks))   
        #Block diagonal preconditioner for BEM.
        P = BlockDiagonal_2x2(dirichl_space, neumann_space, blocks, es, em, ks) 
    else:   
        #Position of the 2x2 matrix.
        blocks = bempp.api.BlockedOperator(2,2)    
        blocks[0,0] = (0.5*IL + KL)  
        blocks[0,1] = -VL            
        blocks[1,0] = (0.5*IH - KH) 
        blocks[1,1] = (em/es)*VH     

    #The solution of the matrix equation Ax=B is solved
    #Iteration counter
    count_iterations = gmres_counter()
   
    # Solution by GMRES.
    from scipy.sparse.linalg import gmres
    if SF==False:
        start1 = time.time()
        soln, info = gmres(blocked, rhs, M=P, callback=count_iterations,tol=Tol, restart=Res)  
        end1 = time.time() 
        
        # Time to solve the equation.
        curr_time1 = (end1 - start1)
        print("Number of GMRES iterations: {0}".format(count_iterations.niter))
        print("Total time in GMRES: {:5.2f} [s]".format(curr_time1))
        TF=count_iterations.niter
        
        #Calculate the entire global domain of the potential from the solution of the calculated edge
        # Solution for Dirichlet data.
        soln_u = soln[:dirichl_space.global_dof_count]
        dirichlet_fun = bempp.api.GridFunction(dirichl_space, coefficients=soln_u)    
        # Solution for Neumann data.
        soln_du = soln[dirichl_space.global_dof_count:] 
        neumann_fun = bempp.api.GridFunction(neumann_space, coefficients=soln_du)
        
    else:
        start1 = time.time()
        soln, info, res, it_count = bempp.api.linalg.gmres(blocks, rhs, return_residuals=True, return_iteration_count=True, use_strong_form=True, tol=Tol, restart=Res)
        end1 = time.time()

        # Time to solve the equation.
        curr_time1 = (end1 - start1)
        print("Number of GMRES iterations: {0}".format(it_count))
        print("Total time in GMRES: {:5.2f} [s]".format(curr_time1))
        TF=it_count

        #Calculate the entire global domain of the potential from the solution of the calculated edge
        dirichlet_fun = soln[0]
        neumann_fun = soln[1]
 
    #Result of the total solvation energy.
    VF = bempp.api.operators.potential.laplace.single_layer(neumann_space, np.transpose(PC)) 
    KF = bempp.api.operators.potential.laplace.double_layer(dirichl_space, np.transpose(PC))
    if Md=='3T':
        uF = VF*(neumann_fun+ neumann_duh_fun) - KF*(dirichlet_fun+dirichlet_uh_fun)
    else:
        uF = VF*neumann_fun - KF*dirichlet_fun
    E_Solv = 0.5*4.*np.pi*332.064*np.sum(Q*uF).real 
    print('Solvation Energy in BEM/BEM: {:7.6f} [kCal/mol]'.format(E_Solv) )

    #Total time.
    end = time.time()
    curr_time = (end - start)
    print("Total time: {:5.2f} [s]".format(curr_time))

    return E_Solv, curr_time, curr_time1, TF 
    
###########################################################################################################################    
    
def Caso_BEMBEMBEM(PQR,Mesh1,Mesh2,es,ei,em,ks,ki,Tol,Res,SF,Asb):            
    start = time.time()  
    print("Case BEM/BEM/BEM")
    
    # Choice of border operator assemblies   
    if Asb == 'fmm':
        Assemble = 'fmm' 
    elif Asb == 'dn':
        Assemble = 'default_nonlocal' 
        
    #Data on the position in space, charge and radii of the atoms of the solute.
    PC,Q,R = readpqr(PQR) 
      
    #Generate the surface meshes of the molecule.
    vertices_0,faces_0 = read_off(Mesh1) 
    #In case the mesh has small gaps, with trimesh the information of the original mesh without the gaps is obtained.
    meshSP = trimesh.Trimesh(vertices = vertices_0, faces= faces_0) 
    mesh_split = meshSP.split()
    print("Found %i meshes"%len(mesh_split)) #1 mesh means no cavity.

    vertices_1 = mesh_split[0].vertices 
    faces_1 = mesh_split[0].faces   
    grid1 = bempp.api.grid.grid.Grid(vertices_1.transpose(), faces_1.transpose()) #Creation of the inner surface mesh.

    vertices_2 ,faces_2 =read_off(Mesh2) 
    grid2 = bempp.api.grid.grid.Grid(vertices_2.transpose(), faces_2.transpose()) #Creation of the outer surface mesh.
        
    #Generate functional spaces of the potential and its derivative.
    dirichl_space1 = bempp.api.function_space(grid1, "P", 1)  #Electrostatic potential at the inner interface.
    neumann_space1 = bempp.api.function_space(grid1, "P", 1)  #Derived from the electrostatic potential at the inner interface. 
    dirichl_space2 = bempp.api.function_space(grid2, "P", 1)  #Electrostatic potential at the outer interface.
    neumann_space2 = bempp.api.function_space(grid2, "P", 1)  #Derived from the electrostatic potential at the outer interface. 
    print("DS1 dofs: {0}".format(dirichl_space1.global_dof_count))
    print("NS1 dofs: {0}".format(neumann_space1.global_dof_count))
    print("DS2 dofs: {0}".format(dirichl_space2.global_dof_count))
    print("NS2 dofs: {0}".format(neumann_space2.global_dof_count))
    
    #Generate the boundary operators.
    #Identity operators.
    I1d = bempp.api.operators.boundary.sparse.identity(dirichl_space1, dirichl_space1, dirichl_space1) # 1
    I1n = bempp.api.operators.boundary.sparse.identity(dirichl_space1, neumann_space1, neumann_space1) # 1
    I2d = bempp.api.operators.boundary.sparse.identity(dirichl_space2, dirichl_space2, dirichl_space2) # 1
    I2n = bempp.api.operators.boundary.sparse.identity(dirichl_space2, neumann_space2, neumann_space2) # 1
    #Domain of the solute Ωm.
    KLaa = bempp.api.operators.boundary.laplace.double_layer(dirichl_space1, dirichl_space1, dirichl_space1, assembler=Assemble) #K
    VLaa = bempp.api.operators.boundary.laplace.single_layer(neumann_space1, dirichl_space1, dirichl_space1, assembler=Assemble) #V
    Z1ba = bempp.api.ZeroBoundaryOperator(dirichl_space2, dirichl_space1, dirichl_space1) #0
    Z2ba = bempp.api.ZeroBoundaryOperator(neumann_space2, dirichl_space1, dirichl_space1) #0
    #Intermediate domain Ωi at the inner interface.
    if ki==0:
        KIaa = bempp.api.operators.boundary.laplace.double_layer(dirichl_space1, neumann_space1, neumann_space1, assembler=Assemble) #K
        VIaa = bempp.api.operators.boundary.laplace.single_layer(neumann_space1, neumann_space1, neumann_space1, assembler=Assemble) #V  
        KIba = bempp.api.operators.boundary.laplace.double_layer(dirichl_space2, neumann_space1, neumann_space1, assembler=Assemble) #K
        VIba = bempp.api.operators.boundary.laplace.single_layer(neumann_space2, neumann_space1, neumann_space1, assembler=Assemble) #V
    else:
        KIaa = bempp.api.operators.boundary.modified_helmholtz.double_layer(dirichl_space1, neumann_space1, neumann_space1, ki, assembler=Assemble) #K
        VIaa = bempp.api.operators.boundary.modified_helmholtz.single_layer(neumann_space1, neumann_space1, neumann_space1, ki, assembler=Assemble) #V  
        KIba = bempp.api.operators.boundary.modified_helmholtz.double_layer(dirichl_space2, neumann_space1, neumann_space1, ki, assembler=Assemble) #K
        VIba = bempp.api.operators.boundary.modified_helmholtz.single_layer(neumann_space2, neumann_space1, neumann_space1, ki, assembler=Assemble) #V
    #Intermediate domain Ωi at the outer interface.
    if ki==0:
        KIab = bempp.api.operators.boundary.laplace.double_layer(dirichl_space1, dirichl_space2, dirichl_space2, assembler=Assemble) #K
        VIab = bempp.api.operators.boundary.laplace.single_layer(neumann_space1, dirichl_space2, dirichl_space2, assembler=Assemble) #V
        KIbb = bempp.api.operators.boundary.laplace.double_layer(dirichl_space2, dirichl_space2, dirichl_space2, assembler=Assemble) #K
        VIbb = bempp.api.operators.boundary.laplace.single_layer(neumann_space2, dirichl_space2, dirichl_space2, assembler=Assemble) #V
    else:
        KIab = bempp.api.operators.boundary.modified_helmholtz.double_layer(dirichl_space1, dirichl_space2, dirichl_space2, ki,  assembler=Assemble) #K
        VIab = bempp.api.operators.boundary.modified_helmholtz.single_layer(neumann_space1, dirichl_space2, dirichl_space2, ki, assembler=Assemble) #V
        KIbb = bempp.api.operators.boundary.modified_helmholtz.double_layer(dirichl_space2, dirichl_space2, dirichl_space2, ki, assembler=Assemble) #K
        VIbb = bempp.api.operators.boundary.modified_helmholtz.single_layer(neumann_space2, dirichl_space2, dirichl_space2, ki, assembler=Assemble) #V     
    #Domain of the solvent Ωs.
    Z1ab = bempp.api.ZeroBoundaryOperator(dirichl_space1, neumann_space2, neumann_space2) #0
    Z2ab = bempp.api.ZeroBoundaryOperator(neumann_space1, neumann_space2, neumann_space2) #0
    if ks==0:
        KHbb = bempp.api.operators.boundary.laplace.double_layer(dirichl_space2, neumann_space2, neumann_space2, assembler=Assemble) #K
        VHbb = bempp.api.operators.boundary.laplace.single_layer(neumann_space2, neumann_space2, neumann_space2, assembler=Assemble) #V
    else:
        KHbb = bempp.api.operators.boundary.modified_helmholtz.double_layer(dirichl_space2, neumann_space2, neumann_space2, ks, assembler=Assemble) #K
        VHbb = bempp.api.operators.boundary.modified_helmholtz.single_layer(neumann_space2, neumann_space2, neumann_space2, ks, assembler=Assemble) #V
    
    #Creation of Coulomb potential function.
    @bempp.api.complex_callable(jit=False)
    def U_c(x, n, domain_index, result):
        result[:] = (1 / (4.*np.pi*em))  * np.sum( Q / np.linalg.norm( x - PC, axis=1))
    Uc1 = bempp.api.GridFunction(dirichl_space1, fun=U_c)  
    
    #Construction of the right vector.
    if SF==False:
        # Rhs in Ωm.
        rhs_M = (Uc1).projections(dirichl_space1) 
        # Rhs in Ωi at inner interface.
        rhs_I1 = np.zeros(neumann_space1.global_dof_count) 
        # Rhs in Ωi at outer interface.
        rhs_I2 = np.zeros(dirichl_space2.global_dof_count) 
        # Rhs in Ωs.
        rhs_S = np.zeros(neumann_space2.global_dof_count) 
        # The combination of Rhs.
        rhs = np.concatenate([rhs_M, rhs_I1, rhs_I2, rhs_S])
    else:
        Uc2 = bempp.api.GridFunction(dirichl_space2, fun=U_c) 
        rhs = [I1d*Uc1, 0*I1n*Uc1, 0*I2d*Uc2, 0*I2n*Uc2] 
       
    #Construction left 4x4 matrix
    if SF==False:
        #Position of the 4x4 matrix.
        blocks = [[None,None,None,None],[None,None,None,None],[None,None,None,None],[None,None,None,None]] 
        blocks[0][0] = (0.5*I1d+KLaa).weak_form()   
        blocks[0][1] = -VLaa.weak_form()           
        blocks[0][2] = Z1ba.weak_form()            
        blocks[0][3] = Z2ba.weak_form()    
    
        blocks[1][0] = (0.5*I1n-KIaa).weak_form()  
        blocks[1][1] = (em/ei)*VIaa.weak_form()    
        blocks[1][2] = KIba.weak_form()            
        blocks[1][3] = -VIba.weak_form()    
    
        blocks[2][0] = -KIab.weak_form()           
        blocks[2][1] = (em/ei)*VIab.weak_form()    
        blocks[2][2] = (0.5*I2d+KIbb).weak_form()  
        blocks[2][3] = -VIbb.weak_form()    
    
        blocks[3][0] = Z1ab.weak_form()            
        blocks[3][1] = Z2ab.weak_form()            
        blocks[3][2] = (0.5*I2n-KHbb).weak_form()  
        blocks[3][3] = (ei/es)*VHbb.weak_form()    
        blocked = bempp.api.assembly.blocked_operator.BlockedDiscreteOperator(np.array(blocks)) 
        #Block diagonal preconditioner for BEM.
        P = BlockDiagonal_4x4(dirichl_space1, neumann_space1, dirichl_space2, neumann_space2, blocks, es,ei,em,ks,ki)
    else:
        blocks = bempp.api.BlockedOperator(4,4)   
        #Position of the 4x4 matrix.
        blocks[0,0] = (0.5*I1d+KLaa)  
        blocks[0,1] = -VLaa          
        blocks[0,2] = Z1ba           
        blocks[0,3] = Z2ba       
    
        blocks[1,0] = (0.5*I1n-KIaa) 
        blocks[1,1] = (em/ei)*VIaa   
        blocks[1,2] = KIba           
        blocks[1,3] = -VIba     
    
        blocks[2,0] = -KIab          
        blocks[2,1] = (em/ei)*VIab   
        blocks[2,2] = (0.5*I2d+KIbb) 
        blocks[2,3] = -VIbb      
    
        blocks[3,0] = Z1ab           
        blocks[3,1] = Z2ab           
        blocks[3,2] = (0.5*I2n-KHbb) 
        blocks[3,3] = (ei/es)*VHbb   
       
    #The solution of the matrix equation Ax=B is solved
    #Iteration counter.
    count_iterations = gmres_counter()
        
    # Solution by GMRES.
    from scipy.sparse.linalg import gmres
    if SF==False:
        start1 = time.time()
        soln, info = gmres(blocked, rhs, M=P, callback=count_iterations,tol=Tol, restart=Res)  
        end1 = time.time() 
        
        # Time to solve the equation.
        curr_time1 = (end1 - start1)
        print("Number of GMRES iterations: {0}".format(count_iterations.niter))
        print("Total time in GMRES: {:5.2f} [s]".format(curr_time1))           
        TF=count_iterations.niter 
        
        #Calculate the entire global domain of the potential from the solution of the edges of both interfaces.
        soln_u1  = soln[:dirichl_space1.global_dof_count]
        soln_du1 = soln[dirichl_space1.global_dof_count : dirichl_space1.global_dof_count + neumann_space1.global_dof_count]
        soln_u2  = soln[dirichl_space1.global_dof_count + neumann_space1.global_dof_count : dirichl_space1.global_dof_count + neumann_space1.global_dof_count + dirichl_space2.global_dof_count]
        soln_du2 = soln[dirichl_space1.global_dof_count + neumann_space1.global_dof_count + dirichl_space2.global_dof_count:]  
        
        # Solution for Dirichlet data at inner surface.
        dirichlet_fun1 = bempp.api.GridFunction(dirichl_space1, coefficients=soln_u1)
        # Solution for Neumann data at inner surface.
        neumann_fun1 = bempp.api.GridFunction(neumann_space1, coefficients=soln_du1)
        # Solution for Dirichlet data at outer surface.
        dirichlet_fun2 = bempp.api.GridFunction(dirichl_space2, coefficients=soln_u2)
        # Solution for Neumann data at outer surface.
        neumann_fun2 = bempp.api.GridFunction(neumann_space2, coefficients=soln_du2)               
    else:
        start1 = time.time()
        soln, info, res, it_count = bempp.api.linalg.gmres(blocks, rhs, return_residuals=True, return_iteration_count=True, use_strong_form=True,tol=Tol, restart=Res)  
        end1 = time.time() 

        # Time to solve the equation.
        curr_time1 = (end1 - start1)
        print("Number of GMRES iterations: {0}".format(it_count))
        print("Total time in GMRES: {:5.2f} [s]".format(curr_time1))   
        TF=it_count
    
        #Calculate the entire global domain of the potential from the solution of the edges of both interfaces.
        dirichlet_fun1 = soln[0] 
        neumann_fun1 = soln[1] 
        dirichlet_fun2 = soln[2] 
        neumann_fun2 = soln[3]  
 
    #Result of the total solvation energy.
    VF1 = bempp.api.operators.potential.laplace.single_layer(neumann_space1, np.transpose(PC)) 
    KF1 = bempp.api.operators.potential.laplace.double_layer(dirichl_space1, np.transpose(PC))
    uF = VF1*neumann_fun1 - KF1*dirichlet_fun1 
    E_Solv = 0.5*4.*np.pi*332.064*np.sum(Q*uF).real 
    print('Solvation Energy in BEM/BEM/BEM: {:7.6f} [kCal/mol]'.format(E_Solv) )

    #Total time.
    end = time.time()
    curr_time = (end - start)
    print("Total time: {:5.2f} [s]".format(curr_time))  
    
    return E_Solv,curr_time, curr_time1, TF
    
########################################################################################################################################
    
def Caso_BEMFEMBEM(PQR,MeshV,es,ei,em,ks,ki,kp,Tol,Res,Va,Asb,FileA):
    start = time.time()         
    # Choice of the permittivity model to be worked on
    if Va=='C':
        print("Case BEM/FEM/BEM(Constant)")
    elif Va=='VL':
        print("Case BEM/FEM/BEM(Linear_Variable)")
    elif Va=='VTH':
        print("Case BEM/FEM/BEM(Variable_Tangent_Hyperbolic)")  
  
    # Choice of border operator assemblies     
    if Asb == 'fmm':
        Assemble = 'fmm' 
    elif Asb == 'dn':
        Assemble = 'default_nonlocal' 
  
    #Data on the position in space, charge and radii of the atoms of the solute.
    PC,Q,R = readpqr(PQR) 
    
    #Creation of the volumetric mesh. 
    mesh = Mesh(MeshV)  
    
    #Generate global functional spaces of the potential in Fem and its derivative in Bem
    from bempp.api.external import fenics
    fenics_space = dolfin.FunctionSpace(mesh, "CG", 1)  #Electrostatic potential at the interface and domain of the solute.
    trace_space, trace_matrix = \
        fenics.fenics_to_bempp_trace_data(fenics_space) #Global trace space to work in BEM and FEM simultaneously.
    
    
    #Process to separate the trace_space for the case of the inner mesh and the outer mesh individually
    #Code to identify vertices and faces of the inner and outer mesh.
    faces_0 = trace_space.grid.elements
    vertices_0 = trace_space.grid.vertices
    meshSP = trimesh.Trimesh(vertices = vertices_0.transpose(), faces= faces_0.transpose())
    mesh_split = meshSP.split()
    print("Found %i meshes"%len(mesh_split))
    vertices_Ref = len(mesh_split[0].vertices)
    faces_Ref = len(mesh_split[0].faces)

    #Obtaining the inner surface mesh.
    faces_1 = faces_0.transpose()[:faces_Ref]
    vertices_1 = vertices_0.transpose()[:vertices_Ref]
    grid1 = bempp.api.grid.grid.Grid(vertices_1.transpose(), faces_1.transpose())
    bempp_space1 = bempp.api.function_space(grid1, "P", 1) # Derived from the electrostatic potential at the inner interface.
    trace_space1 = bempp.api.function_space(grid1, "P", 1) # Trace space to work at inner BEM and FEM simultaneously.

    #Obtaining the outer surface mesh.
    faces_2 = faces_0.transpose()[faces_Ref:]
    vertices_2 = vertices_0.transpose()[vertices_Ref:]
    grid2 = bempp.api.grid.grid.Grid(vertices_2.transpose(), (faces_2-len(vertices_1)).transpose())
    bempp_space2 = bempp.api.function_space(grid2, "P", 1) # Derived from the electrostatic potential at the upper interface.
    trace_space2 = bempp.api.function_space(grid2, "P", 1) # Trace space to work at outer BEM and FEM simultaneously.

    #Element visualization.
    print("FEM dofs: {0}".format(mesh.num_vertices()))
    print("BEM1 dofs: {0}".format(bempp_space1.global_dof_count))
    print("BEM2 dofs: {0}".format(bempp_space2.global_dof_count))
    print("Tra1 dofs: {0}".format(trace_space1.global_dof_count))
    print("Tra2 dofs: {0}".format(trace_space2.global_dof_count))
    print("TraL dofs: {0}".format(trace_space.global_dof_count))
    
    #Process to separate the trace_matrix for the case of the inner mesh and the outer mesh individually
    Nodos = np.zeros(trace_space.global_dof_count)
    Lista_Vertices = []

    #Procedure to locate the vertices of the lower trace in the global trace.
    for i in range(len(trace_space1.grid.vertices.T)):
        valores = np.linalg.norm(trace_space1.grid.vertices[:, i] - trace_space.grid.vertices.T,axis= 1)
        index = np.argmin(valores)
        Lista_Vertices.append(index)
    
    Nodos[Lista_Vertices] = 1
    trace_matrix1 = trace_matrix[Nodos.astype(bool)]
    trace_matrix2 = trace_matrix[np.logical_not(Nodos)]
    
    #Generate the boundary operators
    #Identity operators.
    I1 = bempp.api.operators.boundary.sparse.identity(trace_space1, bempp_space1, bempp_space1) # 1
    I2 = bempp.api.operators.boundary.sparse.identity(trace_space2, bempp_space2, bempp_space2) # 1
    mass1 = bempp.api.operators.boundary.sparse.identity(bempp_space1, trace_space1, trace_space1) # 1
    mass2 = bempp.api.operators.boundary.sparse.identity(bempp_space2, trace_space2, trace_space2) # 1

    #Domain of solute Ωm in BEM.
    K1 = bempp.api.operators.boundary.laplace.double_layer(trace_space1, bempp_space1, bempp_space1, assembler=Assemble) #K
    V1 = bempp.api.operators.boundary.laplace.single_layer(bempp_space1, bempp_space1, bempp_space1, assembler=Assemble) #V
    Z1 = bempp.api.ZeroBoundaryOperator(bempp_space2, bempp_space1, bempp_space1) #0

    #Domain of solvent Ωs in BEM.
    Z2 = bempp.api.ZeroBoundaryOperator(bempp_space1, bempp_space2, bempp_space2) #0
    if ks==0:
        K2 = bempp.api.operators.boundary.laplace.double_layer(trace_space2, bempp_space2, bempp_space2, assembler=Assemble) #K
        V2 = bempp.api.operators.boundary.laplace.single_layer(bempp_space2, bempp_space2, bempp_space2, assembler=Assemble) #V  
    else:
        K2 = bempp.api.operators.boundary.modified_helmholtz.double_layer(trace_space2, bempp_space2, bempp_space2, ks, assembler=Assemble) #K
        V2 = bempp.api.operators.boundary.modified_helmholtz.single_layer(bempp_space2, bempp_space2, bempp_space2, ks, assembler=Assemble) #V
      
    #Define Dolfin functional space
    u = dolfin.TrialFunction(fenics_space)
    v = dolfin.TestFunction(fenics_space)
        
    #Creation of Coulomb potential function
    @bempp.api.complex_callable(jit=False)
    def U_c(x, n, domain_index, result):
        result[:] = (1 / (4.*np.pi*em))  * np.sum( Q / np.linalg.norm( x - PC, axis=1))
    Uca = bempp.api.GridFunction(bempp_space1, fun=U_c)
        
    #Construction of the right vector
    # Rhs in Ωm in BEM.
    rhs_bem1 = (Uca).projections(bempp_space1)
    # Rhs in Ωi in FEM.
    rhs_fem =  np.zeros(mesh.num_vertices()) 
    # Rhs in Ωs in BEM.
    rhs_bem2 = np.zeros(bempp_space2.global_dof_count) 
    # The combination of rhs.
    rhs = np.concatenate([rhs_bem1, rhs_fem, rhs_bem2])
    
    #Choice of variable function
    if Va=='C':  #Constant Case.
        EI = ei
        K  = ei*ki**2
    elif Va=='VL':  #Linear variable case.
        L_Alfa = Lista_Alfa(FileA) 
        S = Fun_ei_Lineal(L_Alfa,degree=0)
        EI = em+(es-em)*S 
        K  = S*(es*ks**2)
    elif Va=='VTH': #Case of variable by Hyperbolic Tangent.
        L_Alfa = Lista_Alfa(FileA) 
        S = Fun_ei_Tangente_Hiperbolica(kp,L_Alfa,degree=0)
        EI = es+(em-es)*S
        K = (1-S)*(es*ks**2)

    #Construction left 3x3 matrix
    from bempp.api.external.fenics import FenicsOperator
    from scipy.sparse.linalg import LinearOperator
    from bempp.api.assembly.blocked_operator import BlockedDiscreteOperator
    blocks = [[None,None,None],[None,None,None],[None,None,None]]

    trace_op1 = LinearOperator(trace_matrix1.shape, lambda x:trace_matrix1*x)
    trace_op2 = LinearOperator(trace_matrix2.shape, lambda x:trace_matrix2*x)
    A = FenicsOperator((EI*dolfin.inner(dolfin.nabla_grad(u),dolfin.nabla_grad(v))+ K*u*v) * dolfin.dx) 

    #Position of the 3x3 matrix.
    blocks[0][0] = V1.weak_form()                    
    blocks[0][1] = (0.5*I1-K1).weak_form()*trace_op1  
    blocks[0][2] = Z1.weak_form()     

    blocks[1][0] = -trace_matrix1.T*em*mass1.weak_form().A   
    blocks[1][1] =  A.weak_form()                      
    blocks[1][2] = -trace_matrix2.T*es*mass2.weak_form().A  

    blocks[2][0] = Z2.weak_form()                     
    blocks[2][1] = (0.5*I2-K2).weak_form()*trace_op2  
    blocks[2][2] = V2.weak_form()                     
    blocked = bempp.api.assembly.blocked_operator.BlockedDiscreteOperator(np.array(blocks))  
    
    #Creation of the Mass Matrix preconditioner for BEM/FEM/BEM
    from bempp.api.assembly.discrete_boundary_operator import InverseSparseDiscreteBoundaryOperator
    from scipy.sparse.linalg import LinearOperator
    from scipy.sparse import diags

    P2 = diags(1./(blocked[1,1].A).diagonal())
    P1 = InverseSparseDiscreteBoundaryOperator(
        bempp.api.operators.boundary.sparse.identity(
            bempp_space1, bempp_space1, bempp_space1).weak_form())
    P3 = InverseSparseDiscreteBoundaryOperator(
        bempp.api.operators.boundary.sparse.identity(
            bempp_space2, bempp_space2, bempp_space2).weak_form())

    def apply_prec(x):
        """Apply the block diagonal preconditioner"""
        m1 = P1.shape[0]
        m2 = P2.shape[0]
        m3 = P3.shape[0]
        n1 = P1.shape[1]
        n2 = P2.shape[1]
        n3 = P3.shape[1]
 
        res1 = P1.dot(x[:n1])
        res2 = P2.dot(x[n1: n1+n2])
        res3 = P3.dot(x[n1+n2:])
        return np.concatenate([res1, res2, res3])

    p_shape = (P1.shape[0] + P2.shape[0] + P3.shape[0], P1.shape[1] + P2.shape[1] + P3.shape[1])
    P = LinearOperator(p_shape, apply_prec, dtype=np.dtype('complex128'))
    
    #The solution of the matrix equation Ax=B is solved
    count_iterations = gmres_counter()  
    
    # Solution by GMRES.
    from scipy.sparse.linalg import gmres
    start1 = time.time()
    soln, info = gmres(blocked, rhs, M=P, callback=count_iterations,tol=Tol, maxiter=3000, restart=Res)  
    end1 = time.time() 

    # Time to solve the equation.
    curr_time1 = (end1 - start1)
    print("Number of GMRES iterations: {0}".format(count_iterations.niter))
    print("Total time in GMRES: {:5.2f} [s]".format(curr_time1))   

    soln_bem1 = soln[:bempp_space1.global_dof_count] 
    soln_fem  = soln[bempp_space1.global_dof_count : bempp_space1.global_dof_count + mesh.num_vertices()]
    soln_bem2 = soln[bempp_space1.global_dof_count + mesh.num_vertices():]
  
    # Calculate the solution of the real potential in the FEM domain in the intermediate region.
    u = dolfin.Function(fenics_space)
    u.vector()[:] = np.ascontiguousarray(np.real(soln_fem)) 
    # Solution for Dirichlet data in the inner interface. 
    dirichlet_data1 = trace_matrix1 * soln_fem
    dirichlet_fun1 = bempp.api.GridFunction(trace_space1, coefficients=dirichlet_data1)
    # Solution for Neumann data in the inner interface.
    neumann_fun1 = bempp.api.GridFunction(bempp_space1, coefficients=soln_bem1)
    # Solution for Dirichlet data in the outer interface.
    dirichlet_data2 = trace_matrix2 * soln_fem
    dirichlet_fun2 = bempp.api.GridFunction(trace_space2, coefficients=dirichlet_data2)
    # Solution for Neumann data in the outer interface.
    neumann_fun2 = bempp.api.GridFunction(bempp_space2, coefficients=soln_bem2)
    
    #Result of the total solvation energy.
    VF1 = bempp.api.operators.potential.laplace.single_layer(bempp_space1, np.transpose(PC)) 
    KF1 = bempp.api.operators.potential.laplace.double_layer(trace_space1, np.transpose(PC))
    uF = -VF1*neumann_fun1 + KF1*dirichlet_fun1 
    E_Solv = 0.5*4.*np.pi*332.064*np.sum(Q*uF).real 
    print('Solvation Energy in BEM/FEM/BEM: {:7.6f} [kCal/mol]'.format(E_Solv) )

    #Total time.
    end = time.time()
    curr_time = (end - start)
    print("Total time: {:5.2f} [s]".format(curr_time))
    
    return E_Solv,curr_time, curr_time1, count_iterations.niter

########################################################### Code for cavities #################################################  
    
def Caso_BEMBEM_Cavidades(PQR,Mesh1,Mesh_C,es,em,ks,Tol,Res,SF,Asb):            
    start = time.time()  
    print("Case BEM/BEM for cavities")
    
    # Choice of border operator assemblies     
    if Asb == 'fmm':
        Assemble = 'fmm' 
    elif Asb == 'dn':
        Assemble = 'default_nonlocal' 
        
    #Data on the position in space, charge and radii of the atoms of the solute.
    PC,Q,R = readpqr(PQR) 
      
    #Generate the surface meshes of the molecule.    
    vertices_0,faces_0 = read_off(Mesh1) 
    #In case the mesh has small gaps, with trimesh the information of the original mesh without the gaps is obtained.
    meshSP = trimesh.Trimesh(vertices = vertices_0, faces= faces_0 ) 
    mesh_split = meshSP.split()
    print("Found %i meshes"%len(mesh_split))  #1 mesh means no cavity.

    vertices_2 = mesh_split[0].vertices 
    faces_2  = mesh_split[0].faces 
    grid2 = bempp.api.grid.grid.Grid(vertices_2.transpose(), faces_2.transpose()) #Creation of the surface mesh of the solute-solvent interface.

    if len(Mesh_C)==0:
        #Sort surface meshes of the cavities by number of vertices.
        LMS = []
        for i in range(len(mesh_split)):
            LMS.append(len(mesh_split[i].vertices))
        IMS = np.argsort(LMS)[::-1]  
        #Unite the information from the individual meshes of the cavities into a single surface mesh. 
        MS = mesh_split[IMS[1]]
        if len(mesh_split)>2: 
            for i in range(len(mesh_split)-2):
                MS = MS + mesh_split[IMS[i+2]]
        vertices_1 = MS.vertices 
        faces_1 = MS.faces  
    else:    
        vertices_1,faces_1 =read_off(Mesh_C)  #Optional cavity mesh if in an '.off' file.
    grid1 = bempp.api.grid.grid.Grid(vertices_1.transpose(), faces_1.transpose()) #Creation of the surface mesh with cavities.    
   
    #Generate functional spaces of the potential and its derivative for the domain Ωm and Ωi
    dirichl_space1 = bempp.api.function_space(grid1, "P", 1)  #Electrostatic potential at the interface of cavity.
    neumann_space1 = bempp.api.function_space(grid1, "P", 1)  #Derived from the electrostatic potential at the interface of cavity.
    dirichl_space2 = bempp.api.function_space(grid2, "P", 1)  #Electrostatic potential at the solute-solvent interface.
    neumann_space2 = bempp.api.function_space(grid2, "P", 1)  #Derived from the electrostatic potential at the solute-solvent interface.
    
    print("DS1 dofs: {0}".format(dirichl_space1.global_dof_count))
    print("NS1 dofs: {0}".format(neumann_space1.global_dof_count))
    print("DS2 dofs: {0}".format(dirichl_space2.global_dof_count))
    print("NS2 dofs: {0}".format(neumann_space2.global_dof_count))
    
    #Generate the boundary operators
    #Identity operators.
    I1d = bempp.api.operators.boundary.sparse.identity(dirichl_space1, dirichl_space1, dirichl_space1) # 1
    I1n = bempp.api.operators.boundary.sparse.identity(dirichl_space1, neumann_space1, neumann_space1) # 1
    I2d = bempp.api.operators.boundary.sparse.identity(dirichl_space2, dirichl_space2, dirichl_space2) # 1
    I2n = bempp.api.operators.boundary.sparse.identity(dirichl_space2, neumann_space2, neumann_space2) # 1

    #Domain of the cavity.
    if ks==0:
        KHaa = bempp.api.operators.boundary.laplace.double_layer(dirichl_space1, dirichl_space1, dirichl_space1, assembler=Assemble) #K
        VHaa = bempp.api.operators.boundary.laplace.single_layer(neumann_space1, dirichl_space1, dirichl_space1, assembler=Assemble) #V
    else:
        KHaa = bempp.api.operators.boundary.modified_helmholtz.double_layer(dirichl_space1, dirichl_space1, dirichl_space1, ks, assembler=Assemble) #K
        VHaa = bempp.api.operators.boundary.modified_helmholtz.single_layer(neumann_space1, dirichl_space1, dirichl_space1, ks, assembler=Assemble) #V
    Z1ba = bempp.api.ZeroBoundaryOperator(dirichl_space2, dirichl_space1, dirichl_space1) #0
    Z2ba = bempp.api.ZeroBoundaryOperator(neumann_space2, dirichl_space1, dirichl_space1) #0

    #Domain of the solute Ωm at the interface of cavity.
    KLaa = bempp.api.operators.boundary.laplace.double_layer(dirichl_space1, neumann_space1, neumann_space1, assembler=Assemble) #K
    VLaa = bempp.api.operators.boundary.laplace.single_layer(neumann_space1, neumann_space1, neumann_space1, assembler=Assemble) #V
    KLba = bempp.api.operators.boundary.laplace.double_layer(dirichl_space2, neumann_space1, neumann_space1, assembler=Assemble) #K
    VLba = bempp.api.operators.boundary.laplace.single_layer(neumann_space2, neumann_space1, neumann_space1, assembler=Assemble) #V

    #Domain of the solute Ωm at the solute-solvent interface.
    KLab = bempp.api.operators.boundary.laplace.double_layer(dirichl_space1, dirichl_space2, dirichl_space2, assembler=Assemble) #K
    VLab = bempp.api.operators.boundary.laplace.single_layer(neumann_space1, dirichl_space2, dirichl_space2, assembler=Assemble) #V
    KLbb = bempp.api.operators.boundary.laplace.double_layer(dirichl_space2, dirichl_space2, dirichl_space2, assembler=Assemble) #K
    VLbb = bempp.api.operators.boundary.laplace.single_layer(neumann_space2, dirichl_space2, dirichl_space2, assembler=Assemble) #V

    #Domain of the solvent Ωs.
    Z1ab = bempp.api.ZeroBoundaryOperator(dirichl_space1, neumann_space2, neumann_space2) #0
    Z2ab = bempp.api.ZeroBoundaryOperator(neumann_space1, neumann_space2, neumann_space2) #0
    if ks==0:
        KHbb = bempp.api.operators.boundary.laplace.double_layer(dirichl_space2, neumann_space2, neumann_space2, assembler=Assemble) #K
        VHbb = bempp.api.operators.boundary.laplace.single_layer(neumann_space2, neumann_space2, neumann_space2, assembler=Assemble) #V
    else:
        KHbb = bempp.api.operators.boundary.modified_helmholtz.double_layer(dirichl_space2, neumann_space2, neumann_space2, ks, assembler=Assemble) #K
        VHbb = bempp.api.operators.boundary.modified_helmholtz.single_layer(neumann_space2, neumann_space2, neumann_space2, ks, assembler=Assemble) #V
 
    #Creation of Coulomb potential function
    @bempp.api.complex_callable(jit=False)
    def U_c(x, n, domain_index, result):
        result[:] = (1 / (4.*np.pi*em))  * np.sum( Q / np.linalg.norm( x - PC, axis=1))
    Uc1 = bempp.api.GridFunction(neumann_space1, fun=U_c)  
    Uc2 = bempp.api.GridFunction(dirichl_space2, fun=U_c)   

    #Construction of the right vector.
    if SF==False:
        # Rhs in cavity.
        rhs_S0 = np.zeros(dirichl_space1.global_dof_count) #0   
        # Rhs in Ωm at the interface of cavity.
        rhs_M1 = (Uc1).projections(neumann_space1) # uc    
        # Rhs in Ωm at the solute-solvent interface.
        rhs_M2 = (Uc2).projections(dirichl_space2) # uc    
        # Rhs in Ωs
        rhs_S = np.zeros(neumann_space2.global_dof_count) #0
        # The combination of Rhs.
        rhs = np.concatenate([rhs_S0, rhs_M1, rhs_M2, rhs_S])
    else:
        Uc0 = bempp.api.GridFunction(dirichl_space1, fun=U_c) 
        rhs = [0*I1d*Uc0, I1n*Uc0, I2d*Uc2, 0*I2n*Uc2] 
        
    #Construction left 4x4 matrix
    if SF==False:
        #Position of the 4x4 matrix.
        blocks = [[None,None,None,None],[None,None,None,None],[None,None,None,None],[None,None,None,None]] 
        blocks[0][0] = (0.5*I1d-KHaa).weak_form()  
        blocks[0][1] = VHaa.weak_form()            
        blocks[0][2] = Z1ba.weak_form()            
        blocks[0][3] = Z2ba.weak_form()       
    
        blocks[1][0] = (0.5*I1n+KLaa).weak_form()  
        blocks[1][1] = -(es/em)*VLaa.weak_form()   
        blocks[1][2] = KLba.weak_form()            
        blocks[1][3] = -VLba.weak_form()   
    
        blocks[2][0] = KLab.weak_form()           
        blocks[2][1] = -(es/em)*VLab.weak_form()   
        blocks[2][2] = (0.5*I2d+KLbb).weak_form()  
        blocks[2][3] = -VLbb.weak_form()   
    
        blocks[3][0] = Z1ab.weak_form()           
        blocks[3][1] = Z2ab.weak_form()            
        blocks[3][2] = (0.5*I2n-KHbb).weak_form()  
        blocks[3][3] = (em/es)*VHbb.weak_form()    
        blocked = bempp.api.assembly.blocked_operator.BlockedDiscreteOperator(np.array(blocks))     
        #Block diagonal preconditioner for BEM.
        P = BlockDiagonal_4x4_Cavidad(dirichl_space1, neumann_space1, dirichl_space2, neumann_space2, blocks, es,em,ks)
    else:
        #Position of the 4x4 matrix.
        blocks = bempp.api.BlockedOperator(4,4)  
        blocks[0,0] = (0.5*I1d-KHaa) 
        blocks[0,1] = VHaa           
        blocks[0,2] = Z1ba           
        blocks[0,3] = Z2ba         
    
        blocks[1,0] = (0.5*I1n+KLaa) 
        blocks[1,1] = -(es/em)*VLaa  
        blocks[1,2] = KLba           
        blocks[1,3] = -VLba   
    
        blocks[2,0] = KLab           
        blocks[2,1] = -(es/em)*VLab 
        blocks[2,2] = (0.5*I2d+KLbb) 
        blocks[2,3] = -VLbb         
    
        blocks[3,0] = Z1ab           
        blocks[3,1] = Z2ab           
        blocks[3,2] = (0.5*I2n-KHbb) 
        blocks[3,3] = (em/es)*VHbb   
    
    #The solution of the matrix equation Ax=B is solved
    #Iteration counter
    count_iterations = gmres_counter()
          
    # Solution by GMRES.
    from scipy.sparse.linalg import gmres
    if SF==False:
        start1 = time.time()
        soln, info = gmres(blocked, rhs, M=P, callback=count_iterations,tol=Tol, restart=Res)  
        end1 = time.time() 
        
        # Time to solve the equation.
        curr_time1 = (end1 - start1)
        print("Number of GMRES iterations: {0}".format(count_iterations.niter))
        print("Total time in GMRES: {:5.2f} [s]".format(curr_time1))
        TF=count_iterations.niter
        
        #Calculate the entire global domain of the potential from the solution of the edges of both interfaces.
        soln_u1 = soln[:dirichl_space1.global_dof_count]
        soln_du1 = soln[dirichl_space1.global_dof_count : dirichl_space1.global_dof_count + neumann_space1.global_dof_count]
        soln_u2 =  soln[dirichl_space1.global_dof_count + neumann_space1.global_dof_count : dirichl_space1.global_dof_count + neumann_space1.global_dof_count + dirichl_space2.global_dof_count]
        soln_du2 = soln[dirichl_space1.global_dof_count + neumann_space1.global_dof_count + dirichl_space2.global_dof_count:]
        # Solution to the function with Dirichlet in the cavity interface.
        dirichlet_fun1 = bempp.api.GridFunction(dirichl_space1, coefficients=soln_u1)
        # Solution of the function with Neumann at the cavity interface.
        neumann_fun1 = bempp.api.GridFunction(neumann_space1, coefficients=soln_du1)
        # Solution of the function with Dirichlet at the solute-solvent interface.
        dirichlet_fun2 = bempp.api.GridFunction(dirichl_space2, coefficients=soln_u2)
        # Solution of the function with Neumann at the solute-solvent interface.
        neumann_fun2 = bempp.api.GridFunction(neumann_space2, coefficients=soln_du2)    
    else:
        start1 = time.time()
        soln, info, res, it_count = bempp.api.linalg.gmres(blocks, rhs, return_residuals=True, return_iteration_count=True, use_strong_form=True,tol=Tol, restart=Res)
        end1 = time.time() 
    
        # Time to solve the equation.
        curr_time1 = (end1 - start1)
        print("Number of GMRES iterations: {0}".format(it_count))
        print("Total time in GMRES: {:5.2f} [s]".format(curr_time1))
        TF=it_count
 
        #Calculate the entire global domain of the potential from the solution of the edges of both interfaces.
        dirichlet_fun1 = soln[0] 
        neumann_fun1 = soln[1] 
        dirichlet_fun2 = soln[2] 
        neumann_fun2 = soln[3]   
    
    #Result of the total solvation energy.
    VF1 = bempp.api.operators.potential.laplace.single_layer(neumann_space1, np.transpose(PC)) 
    KF1 = bempp.api.operators.potential.laplace.double_layer(dirichl_space1, np.transpose(PC))
    VF2 = bempp.api.operators.potential.laplace.single_layer(neumann_space2, np.transpose(PC)) 
    KF2 = bempp.api.operators.potential.laplace.double_layer(dirichl_space2, np.transpose(PC))
    uF = VF1*neumann_fun1*(es/em) - KF1*dirichlet_fun1 + VF2*neumann_fun2 - KF2*dirichlet_fun2 
    E_Solv = 0.5*4.*np.pi*332.064*np.sum(Q*uF).real 
    print('Solvation Energy in BEM/BEM for cavities: {:7.6f} [kCal/mol]'.format(E_Solv) )

    #Total time.
    end = time.time()
    curr_time = (end - start)
    print("Total time: {:5.2f} [s]".format(curr_time))    

    return E_Solv,curr_time, curr_time1, TF

###############################################################################################################

def Caso_BEMFEMBEM_Cavidades(PQR,MeshV,Mesh_C,es,ei,em,ks,ki,kp,Tol,Res,Va,Asb,FileA):
    start = time.time()         
    # Choice of the permittivity model to be worked on
    if Va=='C':
        print("Case BEM/FEM/BEM(Constant) for cavities")
    elif Va=='VL':
        print("Case BEM/FEM/BEM(Linear_Variable) for cavities")
    elif Va=='VTH':
        print("Case BEM/FEM/BEM(Variable_Tangent_Hyperbolic) for cavities")  
  
    # Choice of border operator assemblies     
    if Asb == 'fmm':
        Assemble = 'fmm' 
    elif Asb == 'dn':
        Assemble = 'default_nonlocal' 
  
    #Data on the position in space, charge and radii of the atoms of the solute.
    PC,Q,R = readpqr(PQR) 
    
    #Creation of the volumetric mesh. 
    mesh = Mesh(MeshV)  
   
    #Creation of the cavity mesh.
    vertices_C,faces_C  = read_off(Mesh_C) 
    #In case the mesh has small gaps, with trimesh the information of the original mesh without the gaps is obtained.
    meshSP = trimesh.Trimesh(vertices = vertices_C, faces= faces_C) 
    mesh_split = meshSP.split()
    print("Found %i meshes"%len(mesh_split)) 

    if len(mesh_split)>=2:
        #Sort surface meshes of the cavities by number of vertices.
        LMS = []
        for i in range(len(mesh_split)):
            LMS.append(len(mesh_split[i].vertices))
        IMS = np.argsort(LMS)[::-1]  
        #Unite the information from the individual meshes of the cavities into a single surface mesh. 
        MS = mesh_split[IMS[1]]
        if len(mesh_split)>2: 
            for i in range(len(mesh_split)-2):
                MS = MS + mesh_split[IMS[i+2]]
        vertices_3 = MS.vertices 
        faces_3 = MS.faces  
    elif len(mesh_split)==1:   #Optional cavity mesh if in an '.off' file.
        vertices_3 = mesh_split[0].vertices 
        faces_3 = mesh_split[0].faces 
    grid3 = bempp.api.grid.grid.Grid(vertices_3.transpose(), faces_3.transpose()) #Creation of the surface mesh with cavities.

    #Generate functional spaces of the potential and its derivative in the cavity.
    bempp_space0 = bempp.api.function_space(grid3, "P", 1)  #Electrostatic potential at the inner interface.
    bempp_space3 = bempp.api.function_space(grid3, "P", 1)  #Derived from the electrostatic potential at the inner interface.
    
    #Generate global functional spaces of the potential in Fem and its derivative in Bem
    from bempp.api.external import fenics
    fenics_space = dolfin.FunctionSpace(mesh, "CG", 1)  #Electrostatic potential at the interface and domain of the solute.
    trace_space, trace_matrix = \
        fenics.fenics_to_bempp_trace_data(fenics_space) #Global trace space to work in BEM and FEM simultaneously.
    
    #Code to identify vertices and faces of the inner and outer mesh.
    faces_0 = trace_space.grid.elements
    vertices_0 = trace_space.grid.vertices
    meshSP = trimesh.Trimesh(vertices = vertices_0.transpose(), faces= faces_0.transpose())
    mesh_split = meshSP.split()
    print("Found %i meshes"%len(mesh_split))
    vertices_Ref = len(mesh_split[0].vertices)
    faces_Ref = len(mesh_split[0].faces)

    #Obtaining the inner surface mesh.
    faces_1 = faces_0.transpose()[:faces_Ref]
    vertices_1 = vertices_0.transpose()[:vertices_Ref]
    grid1 = bempp.api.grid.grid.Grid(vertices_1.transpose(), faces_1.transpose())
    bempp_space1 = bempp.api.function_space(grid1, "P", 1) # Derived from the electrostatic potential at the inner interface.
    trace_space1 = bempp.api.function_space(grid1, "P", 1) # Trace space to work at inner BEM and FEM simultaneously.

    #Obtaining the outer surface mesh.
    faces_2 = faces_0.transpose()[faces_Ref:]
    vertices_2 = vertices_0.transpose()[vertices_Ref:]
    grid2 = bempp.api.grid.grid.Grid(vertices_2.transpose(), (faces_2-len(vertices_1)).transpose())
    bempp_space2 = bempp.api.function_space(grid2, "P", 1) # Derived from the electrostatic potential at the upper interface.
    trace_space2 = bempp.api.function_space(grid2, "P", 1) # Trace space to work at outer BEM and FEM simultaneously.

    #Element visualization.
    print("FEM dofs: {0}".format(mesh.num_vertices()))
    print("BEM1 dofs: {0}".format(bempp_space1.global_dof_count))
    print("BEM2 dofs: {0}".format(bempp_space2.global_dof_count))
    print("Tra1 dofs: {0}".format(trace_space1.global_dof_count))
    print("Tra2 dofs: {0}".format(trace_space2.global_dof_count))
    print("TraL dofs: {0}".format(trace_space.global_dof_count))
    print("BEM0 dofs: {0}".format(bempp_space0.global_dof_count))
    print("BEM3 dofs: {0}".format(bempp_space3.global_dof_count))
        
    #Process to separate the trace_space for the case of the inner mesh and the outer mesh individually
    Nodos = np.zeros(trace_space.global_dof_count)
    Lista_Vertices = []

    #Procedure to locate the vertices of the lower trace in the global trace.
    for i in range(len(trace_space1.grid.vertices.T)):
        valores = np.linalg.norm(trace_space1.grid.vertices[:, i] - trace_space.grid.vertices.T,axis= 1)
        index = np.argmin(valores)
        Lista_Vertices.append(index)

    Nodos[Lista_Vertices] = 1
    trace_matrix1 = trace_matrix[Nodos.astype(bool)]
    trace_matrix2 = trace_matrix[np.logical_not(Nodos)]    
    
    #Generate the boundary operators
    #Identity operators.
    I3 = bempp.api.operators.boundary.sparse.identity(bempp_space0, bempp_space3, bempp_space3) # 1
    I0 = bempp.api.operators.boundary.sparse.identity(bempp_space0, bempp_space0, bempp_space0) # 1
    I1 = bempp.api.operators.boundary.sparse.identity(trace_space1, bempp_space1, bempp_space1) # 1
    I2 = bempp.api.operators.boundary.sparse.identity(trace_space2, bempp_space2, bempp_space2) # 1

    #Domain in the cavity in BEM.
    if ks==0:
        V3 = bempp.api.operators.boundary.laplace.single_layer(bempp_space3, bempp_space3, bempp_space3, assembler=Assemble) #V 
        K3 = bempp.api.operators.boundary.laplace.double_layer(bempp_space0, bempp_space3, bempp_space3, assembler=Assemble) #K 
    else:
        V3 = bempp.api.operators.boundary.modified_helmholtz.single_layer(bempp_space3, bempp_space3, bempp_space3, ks, assembler=Assemble) #V
        K3 = bempp.api.operators.boundary.modified_helmholtz.double_layer(bempp_space0, bempp_space3, bempp_space3, ks, assembler=Assemble) #K
    Z3 = bempp.api.ZeroBoundaryOperator(bempp_space1, bempp_space3, bempp_space3) #0
    Z31 = bempp.api.ZeroBoundaryOperator(trace_space1, bempp_space3, bempp_space3) #0
    Z32 = bempp.api.ZeroBoundaryOperator(bempp_space2, bempp_space3, bempp_space3) #0

    #Domain of the solute Ωm in the inner mesh of BEM.
    V0 = bempp.api.operators.boundary.laplace.single_layer(bempp_space3, bempp_space0, bempp_space0, assembler=Assemble) #V 
    K0 = bempp.api.operators.boundary.laplace.double_layer(bempp_space0, bempp_space0, bempp_space0, assembler=Assemble) #K 
    V01 = bempp.api.operators.boundary.laplace.single_layer(bempp_space1, bempp_space0, bempp_space0, assembler=Assemble) #V 
    K01 = bempp.api.operators.boundary.laplace.double_layer(trace_space1, bempp_space0, bempp_space0, assembler=Assemble) #K 
    Z0 = bempp.api.ZeroBoundaryOperator(bempp_space2, bempp_space0, bempp_space0) #0

    #Domain of the solute Ωm in the outer mesh of BEM.
    V10 = bempp.api.operators.boundary.laplace.single_layer(bempp_space3, bempp_space1, bempp_space1, assembler=Assemble) #V 
    K10 = bempp.api.operators.boundary.laplace.double_layer(bempp_space0, bempp_space1, bempp_space1, assembler=Assemble) #K 
    V1 = bempp.api.operators.boundary.laplace.single_layer(bempp_space1, bempp_space1, bempp_space1, assembler=Assemble) #V
    K1 = bempp.api.operators.boundary.laplace.double_layer(trace_space1, bempp_space1, bempp_space1, assembler=Assemble) #K
    Z1 = bempp.api.ZeroBoundaryOperator(bempp_space2, bempp_space1, bempp_space1) #0

    #Intermediate domain in FEM.
    ZF1 = bempp.api.ZeroBoundaryOperator(bempp_space3, trace_space1, trace_space1) #0
    ZF0 = bempp.api.ZeroBoundaryOperator(bempp_space0, trace_space1, trace_space1) #0
    mass1 = bempp.api.operators.boundary.sparse.identity(bempp_space1, trace_space1, trace_space1) # 1
    mass2 = bempp.api.operators.boundary.sparse.identity(bempp_space2, trace_space2, trace_space2) # 1

    #Domain of the solvent Ωs in BEM.
    Z22 = bempp.api.ZeroBoundaryOperator(bempp_space3, bempp_space2, bempp_space2) #0
    Z21 = bempp.api.ZeroBoundaryOperator(bempp_space0, bempp_space2, bempp_space2) #0
    Z2 = bempp.api.ZeroBoundaryOperator(bempp_space1, bempp_space2, bempp_space2) #0
    if ks==0:
        K2 = bempp.api.operators.boundary.laplace.double_layer(trace_space2, bempp_space2, bempp_space2, assembler=Assemble) #K
        V2 = bempp.api.operators.boundary.laplace.single_layer(bempp_space2, bempp_space2, bempp_space2, assembler=Assemble) #V  
    else:
        K2 = bempp.api.operators.boundary.modified_helmholtz.double_layer(trace_space2, bempp_space2, bempp_space2, ks, assembler=Assemble) #K
        V2 = bempp.api.operators.boundary.modified_helmholtz.single_layer(bempp_space2, bempp_space2, bempp_space2, ks, assembler=Assemble) #V
    
    #Define Dolfin functional space
    u = dolfin.TrialFunction(fenics_space)
    v = dolfin.TestFunction(fenics_space)
          
    #Creation of Coulomb potential function
    @bempp.api.complex_callable(jit=False)
    def U_c(x, n, domain_index, result):
        result[:] = (1 / (4.*np.pi*em))  * np.sum( Q / np.linalg.norm( x - PC, axis=1))
    Uc0 = bempp.api.GridFunction(bempp_space0, fun=U_c)
    Uc1 = bempp.api.GridFunction(bempp_space1, fun=U_c)

    #Construction of the right vector
    # Rhs in Ωs0 cavity in BEM.
    rhs_bem3 = np.zeros(bempp_space3.global_dof_count) 
    # Rhs in Ωm inner mesh in BEM.
    rhs_bem0 = (Uc0).projections(bempp_space0)
    # Rhs in Ωm outer mesh in BEM.
    rhs_bem1 = (Uc1).projections(bempp_space1)
    # Rhs in Ωi in FEM.
    rhs_fem =  np.zeros(mesh.num_vertices()) 
    # Rhs in Ωs in BEM.
    rhs_bem2 = np.zeros(bempp_space2.global_dof_count) 
    # The combination of rhs.
    rhs = np.concatenate([rhs_bem3, rhs_bem0, rhs_bem1, rhs_fem, rhs_bem2])
    
    #Choice of variable function
    if Va=='C':  #Constant Case.
        EI = ei
        K  = ei*ki**2
    elif Va=='VL':  #Linear variable case.
        L_Alfa = Lista_Alfa(FileA) 
        S = Fun_ei_Lineal(L_Alfa,degree=0)
        EI = em+(es-em)*S 
        K  = S*(es*ks**2)
    elif Va=='VTH': #Case of variable by Hyperbolic Tangent.
        L_Alfa = Lista_Alfa(FileA) 
        S = Fun_ei_Tangente_Hiperbolica(kp,L_Alfa,degree=0)
        EI = es+(em-es)*S
        K = (1-S)*(es*ks**2)
    
    #Construction left 5x5 matrix
    from bempp.api.external.fenics import FenicsOperator
    from scipy.sparse.linalg import LinearOperator
    from bempp.api.assembly.blocked_operator import BlockedDiscreteOperator
    blocks = [[None,None,None,None,None],[None,None,None,None,None],[None,None,None,None,None],[None,None,None,None,None],[None,None,None,None,None]]

    trace_op1 = LinearOperator(trace_matrix1.shape, lambda x:trace_matrix1*x)
    trace_op2 = LinearOperator(trace_matrix2.shape, lambda x:trace_matrix2*x)
    A = FenicsOperator((EI*dolfin.inner(dolfin.nabla_grad(u),dolfin.nabla_grad(v))+ K*u*v) * dolfin.dx)

    #Position of the 5x5 matrix.
    blocks[0][0] = V3.weak_form()*(em/es)                 
    blocks[0][1] = (0.5*I3-K3).weak_form()                
    blocks[0][2] = Z3.weak_form()                         
    blocks[0][3] = Z31.weak_form()*trace_op1              
    blocks[0][4] = Z32.weak_form()                        

    blocks[1][0] = -V0.weak_form()                       
    blocks[1][1] = (0.5*I0+K0).weak_form()               
    blocks[1][2] = V01.weak_form()                      
    blocks[1][3] = -K01.weak_form()*trace_op1            
    blocks[1][4] = Z0.weak_form()                        

    blocks[2][0] = -V10.weak_form()                     
    blocks[2][1] = K10.weak_form()                      
    blocks[2][2] = V1.weak_form()                       
    blocks[2][3] = (0.5*I1-K1).weak_form()*trace_op1    
    blocks[2][4] = Z1.weak_form()                       

    blocks[3][0] = trace_matrix1.T*ZF1.weak_form().A               
    blocks[3][1] = trace_matrix1.T*ZF0.weak_form().A               
    blocks[3][2] = -trace_matrix1.T *em*mass1.weak_form().A        
    blocks[3][3] =  A.weak_form()                                  
    blocks[3][4] = -trace_matrix2.T *es*mass2.weak_form().A        

    blocks[4][0] = Z22.weak_form()                         
    blocks[4][1] = Z21.weak_form()                         
    blocks[4][2] = Z2.weak_form()                          
    blocks[4][3] = (0.5*I2-K2).weak_form()*trace_op2       
    blocks[4][4] = V2.weak_form()                          
    blocked = bempp.api.assembly.blocked_operator.BlockedDiscreteOperator(np.array(blocks))  
    
    #Creation of the Mass Matrix preconditioner for BEM/FEM/BEM with cavities
    from bempp.api.assembly.discrete_boundary_operator import InverseSparseDiscreteBoundaryOperator
    from scipy.sparse.linalg import LinearOperator
    from scipy.sparse import diags

    P1 = InverseSparseDiscreteBoundaryOperator(
        bempp.api.operators.boundary.sparse.identity(
            bempp_space3, bempp_space3, bempp_space3).weak_form())

    P2 = InverseSparseDiscreteBoundaryOperator(
        bempp.api.operators.boundary.sparse.identity(
            bempp_space0, bempp_space0, bempp_space0).weak_form())

    P3 = InverseSparseDiscreteBoundaryOperator(
        bempp.api.operators.boundary.sparse.identity(
            bempp_space1, bempp_space1, bempp_space1).weak_form())

    P4 = diags(1./(blocked[3,3].A).diagonal())

    P5 = InverseSparseDiscreteBoundaryOperator(
        bempp.api.operators.boundary.sparse.identity(
            bempp_space2, bempp_space2, bempp_space2).weak_form())

    def apply_prec(x):
        """Apply the block diagonal preconditioner"""
        m1 = P1.shape[0]
        m2 = P2.shape[0]
        m3 = P3.shape[0]
        m4 = P4.shape[0]
        m5 = P5.shape[0]
        n1 = P1.shape[1]
        n2 = P2.shape[1]
        n3 = P3.shape[1]
        n4 = P4.shape[1]
        n5 = P5.shape[1]

        res1 = P1.dot(x[:n1])
        res2 = P2.dot(x[n1: n1+n2])
        res3 = P3.dot(x[n1+n2:  n1+n2+n3])
        res4 = P4.dot(x[n1+n2+n3: n1+n2+n3+n4])
        res5 = P5.dot(x[n1+n2+n3+n4:])
        return np.concatenate([res1, res2, res3, res4, res5])

    p_shape = (P1.shape[0] + P2.shape[0] + P3.shape[0]+ P4.shape[0] + P5.shape[0], P1.shape[1] + P2.shape[1] + P3.shape[1]+ P4.shape[1] + P5.shape[1])
    P = LinearOperator(p_shape, apply_prec, dtype=np.dtype('complex128'))
    
    #The solution of the matrix equation Ax=B is solved
    count_iterations = gmres_counter()  
    
    # Solution by GMRES.
    from scipy.sparse.linalg import gmres
    start1 = time.time()
    soln, info = gmres(blocked, rhs, M=P, callback=count_iterations,tol=Tol, maxiter=3000, restart=Res)  
    end1 = time.time() 

    # Time to solve the equation.
    curr_time1 = (end1 - start1)
    print("Number of GMRES iterations: {0}".format(count_iterations.niter))
    print("Total time in GMRES: {:5.2f} [s]".format(curr_time1))  
    
    soln_bem3 = soln[:bempp_space3.global_dof_count]
    soln_bem0 = soln[bempp_space3.global_dof_count : bempp_space3.global_dof_count + bempp_space0.global_dof_count]
    soln_bem1 = soln[bempp_space3.global_dof_count + bempp_space0.global_dof_count : bempp_space3.global_dof_count + bempp_space0.global_dof_count + bempp_space1.global_dof_count]
    soln_fem  = soln[bempp_space3.global_dof_count + bempp_space0.global_dof_count + bempp_space1.global_dof_count : bempp_space3.global_dof_count + bempp_space0.global_dof_count + bempp_space1.global_dof_count + mesh.num_vertices()]
    soln_bem2 = soln[bempp_space3.global_dof_count + bempp_space0.global_dof_count + bempp_space1.global_dof_count + mesh.num_vertices():]

    # Calculate the solution of the real potential in the FEM domain in the intermediate region.
    u = dolfin.Function(fenics_space)
    u.vector()[:] = np.ascontiguousarray(np.real(soln_fem)) 

    # Solution for Dirichlet data in the inner interface.
    dirichlet_data1 = trace_matrix1 * soln_fem
    dirichlet_fun1 = bempp.api.GridFunction(trace_space1, coefficients=dirichlet_data1)
    # Solution for Neumann data in the inner interface.
    neumann_fun1 = bempp.api.GridFunction(bempp_space1, coefficients=soln_bem1)

    # Solution for Dirichlet data in the outer interface.
    dirichlet_data2 = trace_matrix2 * soln_fem
    dirichlet_fun2 = bempp.api.GridFunction(trace_space2, coefficients=dirichlet_data2)
    # Solution for Neumann data in the outer interface.
    neumann_fun2 = bempp.api.GridFunction(bempp_space2, coefficients=soln_bem2)

    # Solution for Dirichlet data in the cavity.
    dirichlet_fun0 = bempp.api.GridFunction(bempp_space0, coefficients=soln_bem0)
    # Solution for Neumann data in the cavity.
    neumann_fun0 = bempp.api.GridFunction(bempp_space3, coefficients=soln_bem3)
    
    #Result of the total solvation energy.
    VF0 = bempp.api.operators.potential.laplace.single_layer(bempp_space3, np.transpose(PC)) 
    KF0 = bempp.api.operators.potential.laplace.double_layer(bempp_space0, np.transpose(PC))
    VF1 = bempp.api.operators.potential.laplace.single_layer(bempp_space1, np.transpose(PC)) 
    KF1 = bempp.api.operators.potential.laplace.double_layer(trace_space1, np.transpose(PC))
    uF = VF0*neumann_fun0 - KF0*dirichlet_fun0 + KF1*dirichlet_fun1 - VF1*neumann_fun1 
    E_Solv = 0.5*4.*np.pi*332.064*np.sum(Q*uF).real 
    print('Solvation Energy in BEM/FEM/BEM for cavities: {:7.6f} [kCal/mol]'.format(E_Solv) )

    #Total time.
    end = time.time()
    curr_time = (end - start)
    print("Total time: {:5.2f} [s]".format(curr_time))
    
    return E_Solv,curr_time, curr_time1, count_iterations.niter
