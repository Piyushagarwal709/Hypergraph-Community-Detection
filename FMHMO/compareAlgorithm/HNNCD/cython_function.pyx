# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 18:56:18 2022

@author: WYW
"""
# Cython —— Q函数
cimport cython


# =============================================================================
#     bound_check_revise: 边界约束检查与修正
#     X: 个体隶属度矩阵
#     c: 社区划分的数目
#     n: 网络节点数目
# =============================================================================
@cython.boundscheck(False) 
@cython.wraparound(False) 
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef bound_check_revise(long[:] X, long n, long low, long up):
    for i in range(n):
        if X[i] < low:
            X[i] = -X[i]
            if X[i] > up:
                X[i] = up
        elif X[i] > up:
            X[i] = 2*up - X[i]
            if X[i] < low:
                X[i] = low
                


# =============================================================================
#     modularityG: G模块度值计算
#     return x 在 arr 中的索引，如果不存在返回 -1
# =============================================================================
@cython.boundscheck(False) 
@cython.wraparound(False) 
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef double modularityG(long[:,:] A, long A_len, long rl, long[:,:] e, long e_len, double[:] w, double w_sum, double[:] d, double d_sum):
    cdef double ec=0.0, dt=0.0, s=0.0
    cdef int i=0,j=0, v=0
    
    for i in range(A_len):
        for j in range(e_len):
            if(LineSearch(A[i,:], rl, e[j,0])>-1 and LineSearch(A[i,:], rl, e[j,1])>-1):
                ec+=w[j]
    ec /= w_sum
    
    for i in range(A_len):
        s = 0.0
        for j in range(rl): 
            v = A[i,j]
            if v==-1:
                break
            s+=d[v]
        s = s**2
        dt+=s
    # dt /= (4*(d_sum/2)**2)
    dt = dt/d_sum**2
    
    return ec - dt


# =============================================================================
#     modularityG: G模块度值计算
#     return x 在 arr 中的索引，如果不存在返回 -1
# =============================================================================
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.profile(False)
def modularityH(long[:,:] H, long[:,:] A, long[:] m, long M, long[:] D, long vol):
    cdef double ec = 0.0
    cdef double dt = 0.0
    cdef double S = 0.0
    cdef int i, j, k

    # Calculate ec
    for k in range(A.shape[0]):
        for i in range(H.shape[0]):
            if isSubset(H[i,:], A[k,:]):
                ec = ec + 1
                break

    ec = ec / M
    print("ec=",ec)
    # Calculate volA
    cdef list volA = [0.0] * A.shape[0]
    for i in range(A.shape[0]):
        for j in A[i,:]:
            if j == -1: break
            volA[i] = volA[i] + D[j]
        volA[i] = volA[i] / vol

    # Calculate S
    for i in range(len(m)):
        if m[i] > 0:
            x = sum([a ** i for a in volA]) * m[i] / M
            S = S + x
            
    print("S=",S)

    return ec - S



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def isSubset(long[:] A, long[:] B):
    cdef bint flag=True
    for i in A:
        if i ==-1:
            break
        elif i not in B:
            flag = False
            return flag
    return flag






# =============================================================================
#     LineSearch: 线性查找数据
#     return x 在 arr 中的索引，如果不存在返回 -1
# =============================================================================
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef int LineSearch(long[:] arr, int r1, int x1):
    cdef int index=0,
    for i in range(r1):
        if arr[i] == x1:
            return i
        elif arr[i] == -1:
            return -1
        index = index + 1
    return -1




# =============================================================================
# 计算i节点对各个邻域社区的隶属程度
# =============================================================================
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.profile(False)
cpdef mem_func(long i, double[:,:] short_path_adj, double[:,:] HEW_adj, dict Xpartition, long[:] j_nodes_setc, long[:] j_nodes_c, list j_nodes):
    cdef dict i_jcnei_wsum, i_jcnei_evg, jm_i_jc, i_jcnei_mem
    cdef int jcno, j_index, node_num, max_cno
    cdef double max_mem, _mem, jm_i_jc_sum

    i_jcnei_wsum = {}
    i_jcnei_evg = {}
    jm_i_jc = {}
    i_jcnei_mem = {}

    for c_nei in j_nodes_setc:
        i_jcnei_wsum[c_nei] = 0.0
        i_jcnei_evg[c_nei] = 0.0

    # Calculate average distance
    for jcno in i_jcnei_evg.keys():
        for j in Xpartition[jcno]:
            i_jcnei_evg[jcno] += short_path_adj[i, j]
        # print("i={}, i_jcnei_evg={}".format(i, i_jcnei_evg))
        node_num = len(Xpartition[jcno]) - 1 if i in Xpartition[jcno] else len(Xpartition[jcno])
        i_jcnei_evg[jcno] /= node_num
        
    # print("i={}, i_jcnei_evg={}".format(i, i_jcnei_evg))

    # Calculate connection strength
    for j_index, j in enumerate(j_nodes):
        i_jcnei_wsum[j_nodes_c[j_index]] += HEW_adj[i, j]

    # Calculate membership strength
    for jcno in j_nodes_setc:
        jm_i_jc[jcno] = i_jcnei_wsum[jcno] / i_jcnei_evg[jcno]

    jm_i_jc_sum = sum(list(jm_i_jc.values()))

    # Calculate normalized membership strengths and find the maximum
    max_mem, max_cno = 0.0, 0
    for jcno in jm_i_jc.keys():
        _mem = jm_i_jc[jcno] / jm_i_jc_sum
        i_jcnei_mem[jcno] = _mem
        if _mem > max_mem:
            max_mem, max_cno = _mem, jcno

    return i_jcnei_mem, max_cno

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.profile(False)
cpdef mem_func_w(long i, double[:,:] short_path_adj, double[:,:] HEW_adj, dict Xpartition, long[:] j_nodes_setc, long[:] j_nodes_c, list j_nodes):
    cdef dict i_jcnei_wsum, i_jcnei_mem
    cdef int jcno, j_index, node_num, max_cno
    cdef double max_mem, _mem, jm_i_jc_sum

    i_jcnei_wsum = {}
    i_jcnei_mem = {}

    for c_nei in j_nodes_setc:
        i_jcnei_wsum[c_nei] = 0.0

    # Calculate connection strength
    for j_index, j in enumerate(j_nodes):
        i_jcnei_wsum[j_nodes_c[j_index]] += HEW_adj[i, j]


    jm_i_jc_sum = sum(list(i_jcnei_wsum.values()))

    # Calculate normalized membership strengths and find the maximum
    max_mem, max_cno = 0.0, 0
    for jcno in j_nodes_setc:
        _mem = i_jcnei_wsum[jcno] / jm_i_jc_sum
        i_jcnei_mem[jcno] = _mem
        if _mem > max_mem:
            max_mem, max_cno = _mem, jcno

    return i_jcnei_mem, max_cno


