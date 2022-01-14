import numpy as np
from numba import cuda
from time import time
from lennard_jones import split_cuda


reduce = cuda.reduce(lambda a, b: min(a, b))

def main():
    n = 22000
    ats = np.random.rand(n*2).astype(np.float32).reshape((n, 2)) + 0.1
    print(ats.dtype)
    ats_d = cuda.to_device(ats)
    atsx_d = cuda.device_array(n, ats_d.dtype)
    atsy_d = cuda.device_array(n, ats_d.dtype)
    nThreads = 64
    nBlocks = int(n / nThreads) + 1
    split_cuda[nBlocks, nThreads](ats_d, atsx_d, atsy_d)
    print(np.min(ats[:,0]))
    a = reduce(atsx_d, init=ats_d[0,0])
    print(a)
    return

    m = minmax_cuda_n
    nThreads = minmax_cuda_nThreads
    nBlocks = int(n / nThreads / minmax_cuda_nThreads) + 1
    minmax = np.zeros(4, dtype=np.float32)
    minmax[0] = ats[0,0]
    minmax[1] = ats[0,0]
    minmax[2] = ats[0,1]
    minmax[3] = ats[0,1]
    minmax_d = cuda.to_device(minmax)
    minmax_cuda[nBlocks, nThreads](ats_d, minmax_d)
    print(minmax_d.copy_to_host())
    print(np.min(ats[:,0]))
    print(np.max(ats[:,0]))
    print(np.min(ats[:,1]))
    print(np.max(ats[:,1]))
    return


    n = 4 * 1024**2
    x = np.random.rand(n).astype(np.float32)
    y = np.random.rand(n).astype(np.float32)
    nThreads = 64
    nBlocks = int(n / nThreads)
    axpy[nBlocks, nThreads](1.4, x, y)

    st = time()
    axpy[nBlocks, nThreads](1.4, x, y)
    axpy[nBlocks, nThreads](1.4, x, y)
    et = time()
    print(et-st)

    st = time()
    axpy[nBlocks, nThreads](1.4, x, y)
    axpy[nBlocks, nThreads](1.4, x, y)
    et = time()
    print(et-st)

    st = time()
    for i in range(200):
        y = 1.4 * x + y
    et = time()
    print(et-st)

    x = np.random.rand(4).reshape((2,2))
    print(x)
    arrTest[1,4](x)
    print(x)

@cuda.jit
def arrTest(x):
    threadId = cuda.threadIdx.x
    blockId = cuda.blockIdx.x
    blockDim = cuda.blockDim.x
    i = blockDim * blockId + threadId
    if i<x.size:
        pass
        #x.reshape(x.size) = x[1,i] + 1


@cuda.jit
def axpy(a, x, y):
    threadId = cuda.threadIdx.x
    blockId = cuda.blockIdx.x
    blockDim = cuda.blockDim.x
    i = blockDim * blockId + threadId

    n = x.size
    if i < n:
        for i in range(200):
            y[i] = a * x[i] + y[i]




if __name__ == '__main__':
    main()
