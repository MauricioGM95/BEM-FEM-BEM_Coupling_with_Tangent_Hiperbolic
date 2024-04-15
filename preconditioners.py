"""
Thank you to the following github link for the information to use and modify this .py file
https://github.com/barbagroup/bempp_exafmm_paper/blob/master/repro-pack/bempp_pbs/preconditioners.py
"""

"""
Implementation of different preconditioners. Bempp-cl uses mass-matrix preconditioner by default.
"""

#from .preprocess import PARAMS
import bempp.api
from bempp.api import ZeroBoundaryOperator
from bempp.api.operators.boundary import sparse, laplace, modified_helmholtz
from scipy.sparse import diags, bmat, block_diag
from scipy.sparse.linalg import aslinearoperator


def BlockDiagonal_2x2(dirichl_space, neumann_space, A,es,em,ks):

    ILD = sparse.identity(dirichl_space, dirichl_space, dirichl_space).weak_form().A.diagonal() 
    KLD = laplace.double_layer(dirichl_space, dirichl_space, dirichl_space, assembler="only_diagonal_part").weak_form().A 
    VLD = laplace.single_layer(neumann_space, dirichl_space, dirichl_space, assembler="only_diagonal_part").weak_form().A 
    IHD = sparse.identity(dirichl_space, neumann_space, neumann_space).weak_form().A.diagonal() 
    if ks==0:
        KHD = laplace.double_layer(dirichl_space, neumann_space, neumann_space, assembler="only_diagonal_part").weak_form().A 
        VHD = laplace.single_layer(neumann_space, neumann_space, neumann_space, assembler="only_diagonal_part").weak_form().A 
    else:
        KHD = modified_helmholtz.double_layer(dirichl_space, neumann_space, neumann_space, ks, assembler="only_diagonal_part").weak_form().A 
        VHD = modified_helmholtz.single_layer(neumann_space, neumann_space, neumann_space, ks, assembler="only_diagonal_part").weak_form().A 
    
    D11 = (0.5*ILD + KLD)  
    D12 = -VLD           
    D21 = (0.5*IHD - KHD) 
    D22 = (em/es)*VHD        
    
    DA = 1/(D11*D22-D21*D12)
    DI11 = D22*DA
    DI12 = -D12*DA
    DI21 = -D21*DA
    DI22 = D11*DA
    
    block_mat_precond = bmat([[diags(DI11), diags(DI12)],
                              [diags(DI21), diags(DI22)]]).tocsr()
    
    return aslinearoperator(block_mat_precond)

def BlockDiagonal_4x4(dirichl_space1, neumann_space1, dirichl_space2, neumann_space2, A,es,ei,em,ks,ki):
    #Identity.
    I1d = sparse.identity(dirichl_space1, dirichl_space1, dirichl_space1).weak_form().A.diagonal()  # 1
    I1n = sparse.identity(dirichl_space1, neumann_space1, neumann_space1).weak_form().A.diagonal()  # 1
    I2d = sparse.identity(dirichl_space2, dirichl_space2, dirichl_space2).weak_form().A.diagonal()  # 1
    I2n = sparse.identity(dirichl_space2, neumann_space2, neumann_space2).weak_form().A.diagonal()  # 1
    #Domain of the solute Ωm.
    KLaa = laplace.double_layer(dirichl_space1, dirichl_space1, dirichl_space1, assembler="only_diagonal_part").weak_form().A  #K
    VLaa = laplace.single_layer(neumann_space1, dirichl_space1, dirichl_space1, assembler="only_diagonal_part").weak_form().A  #V     
    Z1ba = ZeroBoundaryOperator(dirichl_space2, dirichl_space1, dirichl_space1).weak_form().A  #0
    Z2ba = ZeroBoundaryOperator(neumann_space2, dirichl_space1, dirichl_space1).weak_form().A  #0   
    #Intermediate domain Ωi at the inner interface.
    if ki==0:
        KIaa = laplace.double_layer(dirichl_space1, neumann_space1, neumann_space1, assembler="only_diagonal_part").weak_form().A  #K
        VIaa = laplace.single_layer(neumann_space1, neumann_space1, neumann_space1, assembler="only_diagonal_part").weak_form().A  #V
    else:
        KIaa = modified_helmholtz.double_layer(dirichl_space1, neumann_space1, neumann_space1, ki, assembler="only_diagonal_part").weak_form().A  #K
        VIaa = modified_helmholtz.single_layer(neumann_space1, neumann_space1, neumann_space1, ki, assembler="only_diagonal_part").weak_form().A  #V       
    Z3ba = ZeroBoundaryOperator(dirichl_space2, neumann_space1, neumann_space1).weak_form().A  #0
    Z4ba = ZeroBoundaryOperator(neumann_space2, neumann_space1, neumann_space1).weak_form().A  #0
    #Intermediate domain Ωi at the outer interface.
    Z3ab = ZeroBoundaryOperator(dirichl_space1, dirichl_space2, dirichl_space2).weak_form().A  #0
    Z4ab = ZeroBoundaryOperator(neumann_space1, dirichl_space2, dirichl_space2).weak_form().A  #0
    if ki==0:
        KIbb = laplace.double_layer(dirichl_space2, dirichl_space2, dirichl_space2, assembler="only_diagonal_part").weak_form().A  #K
        VIbb = laplace.single_layer(neumann_space2, dirichl_space2, dirichl_space2, assembler="only_diagonal_part").weak_form().A  #V
    else:
        KIbb = modified_helmholtz.double_layer(dirichl_space2, dirichl_space2, dirichl_space2, ki, assembler="only_diagonal_part").weak_form().A  #K
        VIbb = modified_helmholtz.single_layer(neumann_space2, dirichl_space2, dirichl_space2, ki, assembler="only_diagonal_part").weak_form().A  #V       
    #Domain of the solvent Ωs.
    Z1ab = ZeroBoundaryOperator(dirichl_space1, neumann_space2, neumann_space2).weak_form().A  #0
    Z2ab = ZeroBoundaryOperator(neumann_space1, neumann_space2, neumann_space2).weak_form().A  #0
    if ks==0:
        KHbb = laplace.double_layer(dirichl_space2, neumann_space2, neumann_space2, assembler="only_diagonal_part").weak_form().A  #K
        VHbb = laplace.single_layer(neumann_space2, neumann_space2, neumann_space2, assembler="only_diagonal_part").weak_form().A  #V
    else:
        KHbb = modified_helmholtz.double_layer(dirichl_space2, neumann_space2, neumann_space2, ks, assembler="only_diagonal_part").weak_form().A  #K
        VHbb = modified_helmholtz.single_layer(neumann_space2, neumann_space2, neumann_space2, ks, assembler="only_diagonal_part").weak_form().A  #V
        
    D11 = (0.5*I1d+KLaa)  
    D12 = -VLaa          
    D21 = (0.5*I1n-KIaa) 
    D22 = (em/ei)*VIaa              
    D33 = (0.5*I2d+KIbb) 
    D34 = -VIbb          
    D43 = (0.5*I2n-KHbb) 
    D44 = (ei/es)*VHbb   
    
    DA1 = 1/(D11*D22-D12*D21)
    DA2 = 1/(D33*D44-D34*D43)
    
    DI11 = D22*DA1
    DI12 = -D12*DA1
    DI21 = -D21*DA1
    DI22 = D11*DA1
    DI33 = D44*DA2
    DI34 = -D34*DA2
    DI43 = -D43*DA2
    DI44 = D33*DA2   
    
    block_mat_precond = bmat([[diags(DI11), diags(DI12), Z1ba, Z2ba], 
                              [diags(DI21), diags(DI22), Z3ba, Z4ba], 
                              [Z3ab, Z4ab, diags(DI33), diags(DI34)], 
                              [Z1ab, Z2ab, diags(DI43), diags(DI44)]]).tocsr()
    
    return aslinearoperator(block_mat_precond)

def BlockDiagonal_4x4_Cavidad(dirichl_space1, neumann_space1, dirichl_space2, neumann_space2, A,es,em,ks):
    #Identity.
    I1d = sparse.identity(dirichl_space1, dirichl_space1, dirichl_space1).weak_form().A.diagonal()  # 1
    I1n = sparse.identity(dirichl_space1, neumann_space1, neumann_space1).weak_form().A.diagonal()  # 1
    I2d = sparse.identity(dirichl_space2, dirichl_space2, dirichl_space2).weak_form().A.diagonal()  # 1
    I2n = sparse.identity(dirichl_space2, neumann_space2, neumann_space2).weak_form().A.diagonal()  # 1
    #Domain of the cavity.
    if ks==0:
        KHaa = laplace.double_layer(dirichl_space1, dirichl_space1, dirichl_space1, assembler="only_diagonal_part").weak_form().A  #K
        VHaa = laplace.single_layer(neumann_space1, dirichl_space1, dirichl_space1, assembler="only_diagonal_part").weak_form().A  #V
    else:    
        KHaa = modified_helmholtz.double_layer(dirichl_space1, dirichl_space1, dirichl_space1, ks, assembler="only_diagonal_part").weak_form().A  #K
        VHaa = modified_helmholtz.single_layer(dirichl_space1, dirichl_space1, dirichl_space1, ks, assembler="only_diagonal_part").weak_form().A  #V        
    Z1ba = ZeroBoundaryOperator(dirichl_space2, dirichl_space1, dirichl_space1).weak_form().A  #0
    Z2ba = ZeroBoundaryOperator(neumann_space2, dirichl_space1, dirichl_space1).weak_form().A  #0
    #Domain of the solute Ωm at the inner interface.
    KLaa = laplace.double_layer(dirichl_space1, neumann_space1, neumann_space1, assembler="only_diagonal_part").weak_form().A  #K
    VLaa = laplace.single_layer(neumann_space1, neumann_space1, neumann_space1, assembler="only_diagonal_part").weak_form().A  #V
    Z3ba = ZeroBoundaryOperator(dirichl_space2, neumann_space1, neumann_space1).weak_form().A  #0
    Z4ba = ZeroBoundaryOperator(neumann_space2, neumann_space1, neumann_space1).weak_form().A  #0
    #Domain of the solute Ωm at the outer interface.
    Z3ab = ZeroBoundaryOperator(dirichl_space1, dirichl_space2, dirichl_space2).weak_form().A  #0
    Z4ab = ZeroBoundaryOperator(neumann_space1, dirichl_space2, dirichl_space2).weak_form().A  #0
    KLbb = laplace.double_layer(dirichl_space2, dirichl_space2, dirichl_space2, assembler="only_diagonal_part").weak_form().A  #K
    VLbb = laplace.single_layer(neumann_space2, dirichl_space2, dirichl_space2, assembler="only_diagonal_part").weak_form().A  #V
    #Domain of the solvent Ωs.
    Z1ab = ZeroBoundaryOperator(dirichl_space1, neumann_space2, neumann_space2).weak_form().A  #0
    Z2ab = ZeroBoundaryOperator(neumann_space1, neumann_space2, neumann_space2).weak_form().A  #0
    if ks==0:
        KHbb = laplace.double_layer(dirichl_space2, neumann_space2, neumann_space2, assembler="only_diagonal_part").weak_form().A  #K
        VHbb = laplace.single_layer(neumann_space2, neumann_space2, neumann_space2, assembler="only_diagonal_part").weak_form().A  #V
    else:
        KHbb = modified_helmholtz.double_layer(dirichl_space2, neumann_space2, neumann_space2, ks, assembler="only_diagonal_part").weak_form().A  #K
        VHbb = modified_helmholtz.single_layer(neumann_space2, neumann_space2, neumann_space2, ks, assembler="only_diagonal_part").weak_form().A  #V
                    
    D11 = (0.5*I1d-KHaa)   
    D12 = VHaa           
    D21 = (0.5*I1n+KLaa) 
    D22 = -(es/em)*VLaa   
    D33 = (0.5*I2d+KLbb) 
    D34 = -VLbb          
    D43 = (0.5*I2n-KHbb) 
    D44 = (em/es)*VHbb   
    
    DA1 = 1/(D11*D22-D12*D21)
    DA2 = 1/(D33*D44-D34*D43)
    
    DI11 = D22*DA1
    DI12 = -D12*DA1
    DI21 = -D21*DA1
    DI22 = D11*DA1
    DI33 = D44*DA2
    DI34 = -D34*DA2
    DI43 = -D43*DA2
    DI44 = D33*DA2   
    
    block_mat_precond = bmat([[diags(DI11), diags(DI12), Z1ba, Z2ba], 
                              [diags(DI21), diags(DI22), Z3ba, Z4ba], 
                              [Z3ab, Z4ab, diags(DI33), diags(DI34)], 
                              [Z1ab, Z2ab, diags(DI43), diags(DI44)]]).tocsr()
    
    return aslinearoperator(block_mat_precond)
