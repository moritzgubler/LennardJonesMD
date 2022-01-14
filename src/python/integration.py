import numpy as np
from numba import jit, cuda
from lennard_jones import force_cuda


@jit(nopython=True)
def verlet(ats, v, m, neis, nneis, force, dt, n=1):
    e, f = force(ats, neis, nneis)
    a = f / m
    for i in range(n):
        ats = ats + v * dt + 0.5 * a * dt**2
        e, ff = force(ats, neis, nneis)
        aa = ff / m
        v = v + 0.5 * (a + aa) * dt
        a = aa
        f = ff
    return ats, v


@cuda.jit
def verlet_cuda_step0(ats, v, m, neis, nneis, f, dt):
    threadId = cuda.threadIdx.x
    blockId = cuda.blockIdx.x
    blockDim = cuda.blockDim.x
    i = blockDim * blockId + threadId
    nat = ats.shape[0]
    if i<nat:
        ats[i,0] = ats[i,0] + v[i,0] * dt + 0.5 * f[i,0] / m[i] * dt**2
        ats[i,1] = ats[i,1] + v[i,1] * dt + 0.5 * f[i,1] / m[i] * dt**2

@cuda.jit
def verlet_cuda(ats, v, m, f, lastf, dt, n):
    threadId = cuda.threadIdx.x
    blockId = cuda.blockIdx.x
    blockDim = cuda.blockDim.x
    i = blockDim * blockId + threadId
    nat = ats.shape[0]

    if i < nat:
        v[i,0] = v[i,0] + 0.5 * (f[i,0] + lastf[i,0]) * dt / m[i]
        lastf[i,0] = f[i,0]
        ats[i,0] = ats[i,0] + v[i,0] * dt + 0.5 * f[i,0] * dt**2 / m[i]
        v[i,1] = v[i,1] + 0.5 * (f[i,1] + lastf[i,1]) * dt / m[i]
        lastf[i,1] = f[i,1]
        ats[i,1] = ats[i,1] + v[i,1] * dt + 0.5 * f[i,1] * dt**2 / m[i]

@jit(nopython=True)
def relax(ats, neis, nneis, force, n):
    e, f = force(ats, neis, nneis)
    stepsize = 0.001
    #while np.sum(f**2) > 0.01:
    for i in range(n):
        ats += f * stepsize
        laste = e
        e, f = force(ats, neis, nneis)
        if e < laste:
            stepsize *= 1.05
        else:
            stepsize *= 0.5
        print(e, np.sum(f**2), stepsize)
