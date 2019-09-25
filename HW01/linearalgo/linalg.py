import numpy as np

def inverse(A):
    m,n=A.shape
    assert m==n,'A is noninvertible'
    P,L,D,U=PALU_decomposition(A)
    #get inverse_L
    L_inv=lowTriangularInverse(L)
    #get inverse_D
    D_inv=D.copy()
    for i in range(m):
        D_inv[i,i]=1/D_inv[i,i]
    #get inverse_U
    U_inv=upperTriangularInverse(U)

    return U_inv@D_inv@L_inv@P

def PALU_decomposition(A):
    m,n=A.shape
    U=A.copy()
    D=np.identity(m)
    L=np.identity(m)
    P=np.identity(m)

    for i in range(m):
        #find the max element in column i
        maxEle=abs(U[i,i])
        maxRow=i
        for k in range(i+1,m):
            if abs(U[k,i])>maxEle:
                maxEle=U[k,i]
                maxRow=k
        #swap Rowi&maxRow
        U[[i,maxRow],i:]=U[[maxRow,i],i:]
        P[[i,maxRow], :] = P[[maxRow, i], :]
        L[[i,maxRow],:]=L[[maxRow,i],:]
        L[:,[i,maxRow]]=L[:,[maxRow,i]]
        #eliminate
        for k in range(i+1,m):
            c=-U[k,i]/U[i,i]
            U[k,i:]=U[k,i:]+c*U[i,i:]
            L[k:,i]=L[k:,i]-c*L[k:,k]

    # from U to D*U
    for i in range(m):
        D[i,i]=U[i,i]
        U[i,i:]=U[i,i:]/D[i,i]

    return P,L,D,U

'''lower triangular matrix with diagonal elements =1'''
def lowTriangularInverse(A):
    m,n=A.shape
    A_inv=np.identity(m)
    for i in range(m-1):
        for k in range(i+1,m):
            A_inv[k,:]-=A_inv[i,:]*A[k,i]
    return A_inv

'''upper triangulat matrix with diagonal elements =1'''
def upperTriangularInverse(A):
    m,n=A.shape
    A_inv=np.identity(m)
    for i in range(m-1,0,-1):
        for k in range(i-1,-1,-1):
            A_inv[k,:]-=A_inv[i,:]*A[k,i]
    return A_inv