"""
Implementation of different preconditioners. Bempp-cl uses mass-matrix preconditioner by default.
"""

#from .preprocess import PARAMS
import bempp.api
from bempp.api import ZeroBoundaryOperator
from bempp.api.operators.boundary import sparse, laplace, modified_helmholtz
from scipy.sparse import diags, bmat, block_diag
from scipy.sparse.linalg import aslinearoperator

def BlockDiagonal_1x1(neumann_space, A):
    
    VLD = laplace.single_layer(neumann_space, neumann_space, neumann_space, assembler="only_diagonal_part").weak_form().A    
    block_mat_precond = bmat([[diags(1/VLD)]]).tocsr()
    
    return aslinearoperator(block_mat_precond)

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
    K111 = laplace.double_layer(dirichl_space1, dirichl_space1, dirichl_space1, assembler="only_diagonal_part").weak_form().A  #K
    V111 = laplace.single_layer(neumann_space1, dirichl_space1, dirichl_space1, assembler="only_diagonal_part").weak_form().A  #V     
    Y121 = ZeroBoundaryOperator(dirichl_space2, dirichl_space1, dirichl_space1).weak_form().A  #0
    Z121 = ZeroBoundaryOperator(neumann_space2, dirichl_space1, dirichl_space1).weak_form().A  #0   
    #Intermediate domain Ωi at the inner interface.
    if ki==0:
        K211 = laplace.double_layer(dirichl_space1, neumann_space1, neumann_space1, assembler="only_diagonal_part").weak_form().A  #K
        V211 = laplace.single_layer(neumann_space1, neumann_space1, neumann_space1, assembler="only_diagonal_part").weak_form().A  #V
    else:
        K211 = modified_helmholtz.double_layer(dirichl_space1, neumann_space1, neumann_space1, ki, assembler="only_diagonal_part").weak_form().A  #K
        V211 = modified_helmholtz.single_layer(neumann_space1, neumann_space1, neumann_space1, ki, assembler="only_diagonal_part").weak_form().A  #V       
    Y221 = ZeroBoundaryOperator(dirichl_space2, neumann_space1, neumann_space1).weak_form().A  #0
    Z221 = ZeroBoundaryOperator(neumann_space2, neumann_space1, neumann_space1).weak_form().A  #0
    #Intermediate domain Ωi at the outer interface.
    Y212 = ZeroBoundaryOperator(dirichl_space1, dirichl_space2, dirichl_space2).weak_form().A  #0
    Z212 = ZeroBoundaryOperator(neumann_space1, dirichl_space2, dirichl_space2).weak_form().A  #0
    if ki==0:
        K222 = laplace.double_layer(dirichl_space2, dirichl_space2, dirichl_space2, assembler="only_diagonal_part").weak_form().A  #K
        V222 = laplace.single_layer(neumann_space2, dirichl_space2, dirichl_space2, assembler="only_diagonal_part").weak_form().A  #V
    else:
        K222 = modified_helmholtz.double_layer(dirichl_space2, dirichl_space2, dirichl_space2, ki, assembler="only_diagonal_part").weak_form().A  #K
        V222 = modified_helmholtz.single_layer(neumann_space2, dirichl_space2, dirichl_space2, ki, assembler="only_diagonal_part").weak_form().A  #V       
    #Domain of the solvent Ωs.
    Y312 = ZeroBoundaryOperator(dirichl_space1, neumann_space2, neumann_space2).weak_form().A  #0
    Z312 = ZeroBoundaryOperator(neumann_space1, neumann_space2, neumann_space2).weak_form().A  #0
    if ks==0:
        K322 = laplace.double_layer(dirichl_space2, neumann_space2, neumann_space2, assembler="only_diagonal_part").weak_form().A  #K
        V322 = laplace.single_layer(neumann_space2, neumann_space2, neumann_space2, assembler="only_diagonal_part").weak_form().A  #V
    else:
        K322 = modified_helmholtz.double_layer(dirichl_space2, neumann_space2, neumann_space2, ks, assembler="only_diagonal_part").weak_form().A  #K
        V322 = modified_helmholtz.single_layer(neumann_space2, neumann_space2, neumann_space2, ks, assembler="only_diagonal_part").weak_form().A  #V
        
    D11 = (0.5*I1d+K111) # 0.5+K  
    D12 = -V111          # -V
    D21 = (0.5*I1n-K211) # 0.5-K
    D22 = (em/ei)*V211   # (em/ei)V             
    D33 = (0.5*I2d+K222) # 0.5+K
    D34 = -V222          # -V
    D43 = (0.5*I2n-K322) # 0.5-K
    D44 = (ei/es)*V322   # (ei/es)V
    
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
    
    block_mat_precond = bmat([[diags(DI11), diags(DI12), Y121, Z121], 
                              [diags(DI21), diags(DI22), Y221, Z221], 
                              [Y212, Z212, diags(DI33), diags(DI34)], 
                              [Y312, Z312, diags(DI43), diags(DI44)]]).tocsr()
    
    return aslinearoperator(block_mat_precond)

def BlockDiagonal_4x4_Cavidad(dirichl_space1, neumann_space1, dirichl_space2, neumann_space2, A,es,em,ks):
    #Identity.
    I1d = sparse.identity(dirichl_space1, dirichl_space1, dirichl_space1).weak_form().A.diagonal()  # 1
    I1n = sparse.identity(dirichl_space1, neumann_space1, neumann_space1).weak_form().A.diagonal()  # 1
    I2d = sparse.identity(dirichl_space2, dirichl_space2, dirichl_space2).weak_form().A.diagonal()  # 1
    I2n = sparse.identity(dirichl_space2, neumann_space2, neumann_space2).weak_form().A.diagonal()  # 1
    #Domain of the cavity.
    if ks==0:
        K111 = laplace.double_layer(dirichl_space1, dirichl_space1, dirichl_space1, assembler="only_diagonal_part").weak_form().A  #K
        V111 = laplace.single_layer(neumann_space1, dirichl_space1, dirichl_space1, assembler="only_diagonal_part").weak_form().A  #V
    else:    
        K111 = modified_helmholtz.double_layer(dirichl_space1, dirichl_space1, dirichl_space1, ks, assembler="only_diagonal_part").weak_form().A  #K
        V111 = modified_helmholtz.single_layer(dirichl_space1, dirichl_space1, dirichl_space1, ks, assembler="only_diagonal_part").weak_form().A  #V        
    Y121 = ZeroBoundaryOperator(dirichl_space2, dirichl_space1, dirichl_space1).weak_form().A  #0
    Z121 = ZeroBoundaryOperator(neumann_space2, dirichl_space1, dirichl_space1).weak_form().A  #0
    #Domain of the solute Ωm at the inner interface.
    K211 = laplace.double_layer(dirichl_space1, neumann_space1, neumann_space1, assembler="only_diagonal_part").weak_form().A  #K
    V211 = laplace.single_layer(neumann_space1, neumann_space1, neumann_space1, assembler="only_diagonal_part").weak_form().A  #V
    Y221 = ZeroBoundaryOperator(dirichl_space2, neumann_space1, neumann_space1).weak_form().A  #0
    Z221 = ZeroBoundaryOperator(neumann_space2, neumann_space1, neumann_space1).weak_form().A  #0
    #Domain of the solute Ωm at the outer interface.
    Y212 = ZeroBoundaryOperator(dirichl_space1, dirichl_space2, dirichl_space2).weak_form().A  #0
    Z212 = ZeroBoundaryOperator(neumann_space1, dirichl_space2, dirichl_space2).weak_form().A  #0
    K222 = laplace.double_layer(dirichl_space2, dirichl_space2, dirichl_space2, assembler="only_diagonal_part").weak_form().A  #K
    V222 = laplace.single_layer(neumann_space2, dirichl_space2, dirichl_space2, assembler="only_diagonal_part").weak_form().A  #V
    #Domain of the solvent Ωs.
    Y312 = ZeroBoundaryOperator(dirichl_space1, neumann_space2, neumann_space2).weak_form().A  #0
    Z312 = ZeroBoundaryOperator(neumann_space1, neumann_space2, neumann_space2).weak_form().A  #0
    if ks==0:
        K322 = laplace.double_layer(dirichl_space2, neumann_space2, neumann_space2, assembler="only_diagonal_part").weak_form().A  #K
        V322 = laplace.single_layer(neumann_space2, neumann_space2, neumann_space2, assembler="only_diagonal_part").weak_form().A  #V
    else:
        K322 = modified_helmholtz.double_layer(dirichl_space2, neumann_space2, neumann_space2, ks, assembler="only_diagonal_part").weak_form().A  #K
        V322 = modified_helmholtz.single_layer(neumann_space2, neumann_space2, neumann_space2, ks, assembler="only_diagonal_part").weak_form().A  #V
                    
    D11 = (0.5*I1d-K111) # 0.5-K  
    D12 = V111           # V
    D21 = (0.5*I1n+K211) # 0.5+K
    D22 = -(es/em)*V211  # -(es/em)V
    D33 = (0.5*I2d+K222) # 0.5+K
    D34 = -V222          # -V
    D43 = (0.5*I2n-K322) # 0.5-K
    D44 = (em/es)*V322   # (em/es)V
    
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
    
    block_mat_precond = bmat([[diags(DI11), diags(DI12), Y121, Z121], 
                              [diags(DI21), diags(DI22), Y221, Z221], 
                              [Y212, Z212, diags(DI33), diags(DI34)], 
                              [Y312, Z312, diags(DI43), diags(DI44)]]).tocsr()
    
    return aslinearoperator(block_mat_precond)
