import numpy as np
from numba import jit, cuda
#from sklearn.neighbors import NearestNeighbors
import scipy.spatial


#@jit(nopython=True, parallel=True)
def force(ats):
    rcut = 3.0
    nat = ats.shape[0]
    e = 0.0
    f = np.zeros(ats.shape)
    tree = scipy.spatial.cKDTree(ats)

    for i in range(nat):
        neis = tree.query_ball_point(ats[:,i], rcut)
        print(neis)

    for i in range(nat):
        for j in range(i):
            dr = ats[i,:] - ats[j,:]
            dd = np.sum(dr**2)
            dd2 = 1.0 / dd
            dd6 = dd2 * dd2 * dd2
            dd12 = dd6 * dd6
            e += 4.0 * (dd12 - dd6)
            tt = 24.0 * dd2 * (2.0 * dd12 - dd6)
            t = dr * tt
            f[i,:] += t
            f[j,:] -= t
    return e, f

@cuda.jit
def split_cuda(ats, atsx, atsy):
    threadId = cuda.threadIdx.x
    blockId = cuda.blockIdx.x
    blockDim = cuda.blockDim.x
    i = blockDim * blockId + threadId
    if i < ats.shape[0]:
        atsx[i] = ats[i,0]
        atsy[i] = ats[i,1]

min_cuda = cuda.reduce(lambda a, b: min(a, b))
max_cuda = cuda.reduce(lambda a, b: max(a, b))

#@jit(nopython=True)
def buildNeighbourList_cuda(ats, rc, maxNeis, nThreads):
    nat = ats.shape[0]
    atsx = cuda.device_array(nat, ats.dtype)
    atsy = cuda.device_array(nat, ats.dtype)
    nBlocks = int((nat + nThreads - 1) / nThreads)
    split_cuda[nBlocks, nThreads](ats, atsx, atsy)
    minx = min_cuda(atsx, init=ats[0,0]) #np.min(ats[:,0])
    miny = min_cuda(atsy, init=ats[0,1]) #np.min(ats[:,1])
    maxx = max_cuda(atsx, init=ats[0,0]) #np.max(ats[:,0])
    maxy = max_cuda(atsy, init=ats[0,1]) #np.max(ats[:,1])
    nx = int(np.ceil((maxx - minx) / rc))
    ny = int(np.ceil((maxy - miny) / rc))
    grid = cuda.device_array((nx * ny, maxNeis), np.int32)
    gridN = cuda.device_array((nx * ny), np.int32)
    atIdx = cuda.device_array(nat, np.int32)
    nBlocks = int((nx * ny + nThreads - 1) / nThreads)          # here nThreads might be too large sometimes
    zero_cuda[nBlocks, nThreads](gridN)
    nBlocks = int((nat + nThreads - 1) / nThreads)
    addNeighbors_cuda[nBlocks, nThreads](ats, atIdx, rc, maxNeis, nx, ny, minx, maxx, miny, maxy, grid, gridN)

    #neis = np.zeros((nat, maxNeis), dtype=np.int_)
    #nneis = np.zeros((nat,), dtype=np.int_)
    neis = cuda.device_array((nat * maxNeis), np.int32)
    nneis = cuda.device_array((nat,), np.int32)
    nBlocks = int((nat + nThreads - 1) / nThreads)
    zero_cuda[nBlocks, nThreads](nneis)
    getNeighbours_cuda[nBlocks, nThreads](ats, atIdx, grid, gridN, nx, ny, maxNeis, rc, neis, nneis)
    return neis.reshape(nat, maxNeis, order='C'), nneis





@cuda.jit
def zero_cuda(x):
    threadId = cuda.threadIdx.x
    blockId = cuda.blockIdx.x
    blockDim = cuda.blockDim.x
    i = blockDim * blockId + threadId
    if i < x.size:
        x[i] = 0

@cuda.jit
def addNeighbors_cuda(ats, atIdx, rc, maxNeis, nx, ny, minx, maxx, miny, maxy, grid, gridN):
    nat = ats.shape[0]
    threadId = cuda.threadIdx.x
    blockId = cuda.blockIdx.x
    blockDim = cuda.blockDim.x
    i = blockDim * blockId + threadId

    if i < nat:
        x = int((ats[i,0] - minx) / rc)
        y = int((ats[i,1] - miny) / rc)
        idx = y + ny * x
        atIdx[i] = idx
        n = cuda.atomic.add(gridN, idx, 1)
        grid[idx,n] = i
        #grid[x,y,gridN[idx]] = i
        #gridN[x,y] += 1

    #print('JJ', np.max(gridN))
@cuda.jit
def getNeighbours_cuda(ats, atIdx, grid, gridN, nx, ny, maxNeis, rc, neis, nneis):
    nat = ats.shape[0]
    threadId = cuda.threadIdx.x
    blockId = cuda.blockIdx.x
    blockDim = cuda.blockDim.x
    i = blockDim * blockId + threadId
    if i < nat:
        idx = atIdx[i]
        y = idx % ny
        x = idx / ny
        for ix in range(max(0, x-1), min(nx, x+2)):
            for iy in range(max(0, y - 1), min(ny, y + 2)):
                iidx = iy + ny * ix
                for j in grid[iidx, :gridN[iidx]]:
                    if j!=i:
                        #r2 = np.sum((ats[i,:] - ats[j,:])**2)
                        r2 = (ats[i,0] - ats[j,0])**2 + (ats[i,1] - ats[j,1])**2
                        if r2 < rc**2:
                            neis[i*maxNeis + nneis[i]] = j
                            nneis[i] += 1



@jit(nopython=True)
def buildNeighbourList(ats, rc, maxNeis=100):
    nat = ats.shape[0]
    minx = np.min(ats[:,0])
    miny = np.min(ats[:,1])
    maxx = np.max(ats[:,0])
    maxy = np.max(ats[:,1])
    nx = int(np.ceil((maxx - minx) / rc))
    ny = int(np.ceil((maxy - miny) / rc))
    grid = np.zeros((nx, ny, maxNeis), dtype=np.int_)
    gridN = np.zeros((nx, ny), dtype=np.int_)

    for i in range(nat):
        x = int((ats[i,0] - minx) / rc)
        y = int((ats[i,1] - miny) / rc)
        grid[x,y,gridN[x,y]] = i
        gridN[x,y] += 1

    #print('JJ', np.max(gridN))
    neis = np.zeros((nat, maxNeis), dtype=np.int_)
    nneis = np.zeros((nat,), dtype=np.int_)

    for x in range(nx):
        for y in range(ny):
            for i in grid[x,y,:gridN[x,y]]:
                for ix in range(max(0, x-1), min(nx, x+2)):
                    for iy in range(max(0, y - 1), min(ny, y + 2)):
                        for j in grid[ix, iy, :gridN[ix, iy]]:
                            if j!=i:
                                r2 = np.sum((ats[i,:] - ats[j,:])**2)
                                if r2 < rc**2:
                                    neis[i,nneis[i]] = j
                                    nneis[i] += 1


    return neis, nneis


@jit(nopython=False, parallel=False)
def force_co(ats, neis, nneis):

    #lj_rc = 4.0 * (1.0 / rc**12 - 1.0 / rc**6)
    nat = ats.shape[0]
    e = 0.
    f = np.zeros(ats.shape)

    for i in range(nat):
        #for j in range(i):
        #neis = tree.query_ball_point(ats[i,:], rc)
        for j in neis[i,:nneis[i]]:
            if True: #i < j:
                dr = ats[i,:] - ats[j,:]
                dd = np.sum(dr**2)
                dd2 = 1.0 / dd
                dd6 = dd2 * dd2 * dd2
                dd12 = dd6 * dd6
                e += 4.0 * (dd12 - dd6) #+ lj_rc
                tt = 24.0 * dd2 * (2.0 * dd12 - dd6)
                t = dr * tt
                f[i,:] += t
                #f[j,:] -= t
    return e, f

# not computing e for the moment
@cuda.jit
def force_cuda(ats, neis, nneis, f, eat):
    threadId = cuda.threadIdx.x
    blockId = cuda.blockIdx.x
    blockDim = cuda.blockDim.x
    i = blockDim * blockId + threadId
    nat = ats.shape[0]
    if i < nat:
        f[i,0] = 0.
        f[i,1] = 0.
        eat[i] = 0.
        #for j in range(i):
        #neis = tree.query_ball_point(ats[i,:], rc)
        for ij in range(nneis[i]): #neis[i,:nneis[i]]:
            j = neis[i,ij]
            if True: #i < j:
                drx = ats[i,0] - ats[j,0]
                dry = ats[i,1] - ats[j,1]
                dd = drx**2 + dry**2
                dd2 = 1.0 / dd
                dd6 = dd2 * dd2 * dd2
                dd12 = dd6 * dd6
                #e += 4.0 * (dd12 - dd6) #+ lj_rc
                eat[i] += 4.0 * (dd12 - dd6)
                tt = 24.0 * dd2 * (2.0 * dd12 - dd6)
                f[i,0] += tt * drx
                f[i,1] += tt * dry
                #f[j,:] -= t
