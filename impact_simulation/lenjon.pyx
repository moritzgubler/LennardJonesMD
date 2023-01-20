import numpy as np
cimport numpy as np
cimport cython




cpdef verlet(np.ndarray ats, np.ndarray v, double[:] m, double dt, int n=1, int rc = 5, int maxNeis = 100):
    cdef np.ndarray[np.float64_t, ndim=2] f = np.zeros((ats.shape[0], ats.shape[1]))
    cdef np.ndarray[np.float64_t, ndim=2] ff = np.zeros((ats.shape[0], ats.shape[1]))
    cdef np.ndarray[np.float64_t, ndim=2] aa = np.zeros((ats.shape[0], ats.shape[1]), dtype = np.float64)
    f = force_co(ats, rc, maxNeis)
    cdef np.ndarray[np.float64_t, ndim=2] a = np.zeros((ats.shape[0], ats.shape[1]), dtype = np.float64)
    cdef int i, k, l
    for i in range(ats.shape[0]):
        a[i][0] = f[i][0] / m[i]
        a[i][1] = f[i][1] / m[i]
    for i in range(n):
        for k in range(ats.shape[0]):
            for l in range(ats.shape[1]):
                ats[k][l] = ats[k][l] + v[k][l] * dt + 0.5 * a[k][l] * dt**2
        ff = force_co(ats, rc, maxNeis)
        for k in range(ats.shape[0]):
            for l in range(ats.shape[1]):
                aa[k][l] = ff[k][l] / m[l]
                v[k][l] = v[k][l] + 0.5 * (a[k][l] + aa[k][l]) * dt
        a = aa
        f = ff
    return ats, v

cdef buildNeighbourList(double[:,:] ats, int rc, int maxNeis=100):
    cdef int nat = ats.shape[0]
    cdef double minx = np.min(ats[:,0])
    cdef double miny = np.min(ats[:,1])
    cdef double maxx = np.max(ats[:,0])
    cdef double maxy = np.max(ats[:,1])
    cdef int nx = int(np.ceil((maxx - minx) / rc))
    cdef int ny = int(np.ceil((maxy - miny) / rc))
    cdef np.ndarray[np.int_t, ndim=3] grid = np.zeros((nx, ny, maxNeis), dtype=np.int_)
    cdef np.ndarray[np.int_t, ndim=2] gridN = np.zeros((nx, ny), dtype=np.int_)
    cdef int i
    cdef int x
    cdef int y
    cdef int ix
    cdef int iy
    cdef int j

    for i in range(nat):
        x = int((ats[i,0] - minx) / rc)
        y = int((ats[i,1] - miny) / rc)
        grid[x,y,gridN[x,y]] = i
        gridN[x,y] += 1

    #print('JJ', np.max(gridN))
    cdef np.ndarray neis = np.zeros((nat, maxNeis), dtype=np.int_)
    cdef np.ndarray nneis = np.zeros((nat), dtype=np.int_)
    cdef double r2

    for x in range(nx):
        for y in range(ny):
            for i in grid[x,y,:gridN[x,y]]:
                for ix in range(max(0, x-1), min(nx, x+2)):
                    for iy in range(max(0, y - 1), min(ny, y + 2)):
                        for j in grid[ix, iy, :gridN[ix, iy]]:
                            if j!=i:
                                r2 = (ats[i,0] - ats[j,0] + ats[i,1] - ats[j,1]) * (ats[i,0] - ats[j,0] + ats[i,1] - ats[j,1])
                                if r2 < rc*rc:
                                    neis[i,nneis[i]] = j
                                    nneis[i] = nneis[i] + 1


    return neis, nneis

cdef np.ndarray force_co(np.ndarray ats, int rc=5, int maxNeis=100):
    cdef int nat = ats.shape[0]
    cdef np.ndarray neis = np.zeros((nat, maxNeis), dtype=np.int_)
    cdef np.ndarray nneis = np.zeros((nat,), dtype=np.int_)
    neis, nneis = buildNeighbourList(ats, rc, maxNeis)
    #lj_rc = 4.0 * (1.0 / rc**12 - 1.0 / rc**6)
    cdef int maxNeisp = np.max(nneis)

    cdef double e=0.0
    cdef np.ndarray f = np.zeros((ats.shape[0], ats.shape[1]))
    cdef np.ndarray t = np.zeros((maxNeisp, 2))
    cdef np.ndarray dr = np.zeros((maxNeisp, 2))
    cdef np.ndarray dd2 = np.zeros(maxNeisp)
    cdef np.ndarray dd6 = np.zeros(maxNeisp)
    cdef np.ndarray dd12 = np.zeros(maxNeisp)
    cdef np.ndarray tt = np.zeros(maxNeisp)
    cdef int i
    cdef int n

    for i in range(nat):
        #for j in range(i):
        #neis = tree.query_ball_point(ats[i,:], rc)
        n = nneis[i]
        dr[:n, :] = ats[i,:] - ats[neis[i,:nneis[i]],:]
        dd2[:n] = 1.0 / np.sum(dr[:n]*dr[:n], axis = 1)
        dd6[:n] = dd2[:n] * dd2[:n] * dd2[:n]
        dd12[:n] = dd6[:n] * dd6[:n]

        #e += 4 * np.sum(dd12[:n] - dd6[:n])

        tt[:n] = 24.0 * dd2[:n] * (2.0 * dd12[:n] - dd6[:n])

        t[:n, 0] = dr[:n, 0] * tt[:n]
        t[:n, 1] = dr[:n, 1] * tt[:n]
        f[i, :] += np.sum(t[:n, :], axis = 0)
    return f
