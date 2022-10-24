import numpy as np
from numba import jit
dotsize = 5
dot = np.zeros((dotsize, dotsize))
for i in range(dotsize):
    for j in range(dotsize):
        r = ((i+0.5-dotsize/2)**2 + (j+0.5-dotsize/2)**2) / (dotsize / 2)**2
        dot[i,j] = np.exp(-r / 0.5)

@jit(nopython=True)
def draw(nx, ny, dx, x0, ats, colors, dotsize):

    nat = ats.shape[0]
    nx = int(nx)
    ny = int(ny)
    im = np.ones((nx, ny, 3))
    for i in range(nat):
        x = int((ats[i,0] - x0[0]) / dx - (dotsize-1) / 2)
        y = int((ats[i,1] - x0[1]) / dx - (dotsize-1) / 2)
        if x >= 0 and y >= 0 and x < nx-dotsize and y < ny-dotsize:
            for ix in range(dotsize):
                for iy in range(dotsize):
                    rx = x0[0] + dx * (x + ix + 0.5)
                    ry = x0[1] + dx * (y + iy + 0.5)
                    d = (rx-ats[i,0])**2 + (ry-ats[i,1])**2
                    d = d/dx**2 / (dotsize / 2)**2 * 2.
                    d = np.exp(-d)
                    #im[x+ix, y+iy, :] = (1. - dot[ix, iy]) * im[x+ix, y+iy, :] + colors[i] * dot[ix, iy]
                    im[x+ix, y+iy, :] = (1. - d) * im[x+ix, y+iy, :] + colors[i] * d
            #im[x:x+dotsize, y:y+dotsize,:] = \
            #    np.outer(dot[:,:], colors[i]).reshape((dotsize, dotsize, 3)) + \
            #    (1. - dot[:, :]).repeat * im[x:x+dotsize, y:y+dotsize,:]
            #for ix in range(dotsize):
            #    for iy in range(dotsize):
            #        if dot[ix, iy] > 0:
            #            im[x+ix, y+iy, :] = colors[i,:]
        #if x >= 0 and y >= 0 and x < nx-dotsize and y < ny-dotsize:
        #    im[x:x+dotsize, y:y+dotsize,:] = np.outer(dot[:,:], colors[i]).reshape((dotsize, dotsize, 3))

    return im


if __name__ == '__main__':
    print(dot)


