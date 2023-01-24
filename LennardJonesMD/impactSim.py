from LennardJonesMD.integration import *
from LennardJonesMD.lennard_jones import force_co, buildNeighbourList, buildNeighbourList_cuda
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import time
import scipy.spatial
from numba.typed import List
from numba import cuda
from LennardJonesMD.drawing import draw
from skimage.io import imsave
from skimage.transform import rescale
import os
import json

class ImpactSim:
    def __init__(self, velocity=10., n_layers=1, thickness=10, dt=0.01, theta=0, layerFlip = [False]):
        self.cuda = False
        self.nCudaThreads=32
        self.dtype = np.float64
        if self.cuda:
            self.dtype = np.float32
        self.rcut = 3. #2.5
        self.maxNeis = 100
        self.vmax = 2

        #___________________input variables_________________
        self.v0 = velocity
        self.n_layers = n_layers
        self.thickness = thickness
        #___________________________________________________
        self.nx = 5
        self.ny = 100 #400 #80
        self.d = 1.1 #1.222
        self.nat = 0
        self.ats = np.zeros((0, 2), dtype=self.dtype)
        self.v = np.zeros((0, 2), dtype=self.dtype)
        self.dt = dt

        d = 0
        for i in range(self.n_layers):
            self.addLayer(d, self.thickness[i], flip=layerFlip[i]) # -42, 50
            d = self.thickness[i] + (i+1) * 20

        # theta = 0 #np.pi/2 * 0.7
        v = [np.cos(theta) * self.v0, -np.sin(theta) * self.v0]
        self.addParticle(np.array([-20., 0.]), v, 6, 6)

        print('Number of particles: ', self.nat)


        #self.v = np.zeros((self.nat, 2))
        if self.cuda:
            self.m = np.ones((self.nat,), dtype=self.dtype)
        else:
            self.m = np.ones((self.nat, 2), dtype=self.dtype)

        self.v += np.random.randn(self.v.size).reshape(self.v.shape) * 0.2

        # self.fig = plt.figure()
        # #self.ax = plt.axes(xlim=(-120, 30), ylim=(-50, 50))
        # self.ax = plt.axes(xlim=(-150, 600), ylim=(-250, 250))
        # self.anim = animation.FuncAnimation(
        #     self.fig,
        #     self.update,
        #     init_func=self.setup,
        #     frames=100,
        #     interval=1. / 60,
        #     blit=True)
        if self.cuda:
            self.ats_d = cuda.to_device(self.ats)
            self.v_d = cuda.to_device(self.v)
            self.m_d = cuda.to_device(self.m)
            self.eat_d = cuda.to_device(np.zeros((self.nat,), dtype=self.dtype))
            #neis, nneis = buildNeighbourList(self.ats, self.rcut, maxNeis=self.maxNeis)
            #neis_d = cuda.to_device(neis)
            #nneis_d = cuda.to_device(nneis)
            neis_d, nneis_d = buildNeighbourList_cuda(self.ats_d, self.rcut, self.maxNeis, self.nCudaThreads)
            self.f_d = cuda.to_device(np.zeros(self.ats.shape, dtype=self.dtype))
            self.lastf_d = cuda.to_device(np.zeros(self.ats.shape, dtype=self.dtype))
            nCudaBlocks = int((self.nat + self.nCudaThreads-1) / self.nCudaThreads)
            force_cuda[nCudaBlocks, self.nCudaThreads](self.ats_d, neis_d, nneis_d, self.f_d, self.eat_d)
            print(self.ats.dtype)
            verlet_cuda_step0[nCudaBlocks, self.nCudaThreads](self.ats_d, self.v_d, self.m_d, neis_d, nneis_d, self.f_d, self.dt)
        self.t_start_draw = time.time()

    def addParticle(self, pos, v, nx, ny, flip = True):
        newats = np.zeros((nx * ny, 2), dtype=self.dtype)
        newv = np.zeros((nx * ny, 2), dtype=self.dtype)
        h = np.sqrt(3) / 2 * self.d
        for i in range(nx):
            for j in range(ny):
                newv[i+j*nx, :] = v
                if flip:
                    newats[i + j * nx, :] = pos + np.array((i * self.d + 0.5 * self.d * (j % 2), (j -ny/2) * h))
                else:
                    newats[i + j * nx, :] = pos + np.array((i * h, (j - ny/2) * self.d + 0.5 * self.d * (i % 2)))
        self.nat += nx * ny
        self.ats = np.concatenate((self.ats, newats), axis=0)
        self.v = np.concatenate((self.v, newv), axis=0)


    def addLayer(self, x, nx, flip=False):
        h = np.sqrt(3) / 2 * self.d
        newats = np.zeros((nx * self.ny, 2), dtype=self.dtype)
        for i in range(nx):
            for j in range(self.ny):
                if flip:
                    newats[i + j * nx, :] = (x + i * self.d + 0.5 * self.d * (j % 2), (j - self.ny/2) * h)
                else:
                    newats[i + j * nx, :] = (x + i * h, (j - self.ny/2) * self.d + 0.5 * self.d * (i % 2))
        self.nat += nx * self.ny
        self.ats = np.concatenate((self.ats, newats), axis=0)
        self.v = np.concatenate((self.v, np.zeros(newats.shape, dtype=self.dtype)), axis=0)

    # def setup(self):
    #     self.scat = self.ax.scatter(self.ats[:,0], self.ats[:,1])
    #     return self.scat,

    def update(self, iteration):
        print('iteration ', iteration)
        self.t_end_draw = time.time()
        print('T(draw)', self.t_end_draw - self.t_start_draw)
        self.updateSim()
        self.scat.set_offsets(self.ats) #(ats[:,0], ats[:,1])
        if self.cuda:
            self.scat.set_sizes(self.m)
        else:
            self.scat.set_sizes(self.m[:,0])
        c = np.sqrt(np.sum(self.v**2, axis=1)) / self.vmax
        #print(np.max(self.eat))
        #emax = np.max(self.eat)
        #emin = np.min(self.eat)
        #c = (self.eat - emin) / (emax-emin)
        c = np.concatenate((c.reshape((self.nat, 1)), np.zeros((self.nat, 2))), axis=1)
        c = np.minimum(c, 1.0)
        #self.scat.set_color([(255, 0, 0)] * self.nat)
        self.scat.set_color(c)
        self.t_start_draw = time.time()



        return self.scat,

    def savePic(self, iteration, path):
        t_start = time.time()
        c = np.sqrt(np.sum(self.v**2, axis=1)) / self.vmax
        #print(np.max(self.eat))
        #emax = np.max(self.eat)
        #emin = np.min(self.eat)
        #c = (self.eat - emin) / (emax-emin)
        c = np.concatenate((c.reshape((self.nat, 1)), np.zeros((self.nat, 2))), axis=1)
        c = np.minimum(c, 1.0)
        #self.scat.set_color([(255, 0, 0)] * self.nat)

        resolution = 1.7 # pixels per unit
        dotsize = 3 # pixels
        im = draw(400 * resolution, 500 * resolution, 1.0 / resolution, np.array([-150, -250]), self.ats, c, dotsize)
        #im = rescale(im, 0.2, anti_aliasing=True, multichannel=True)
        im = np.uint8(im*255)
        filename = path+'/{:05d}.png'.format(iteration)
        imsave(filename, im)
        t_end = time.time()
        # print('T(image) ', t_end - t_start)

    def savePos(self, iteration, shelve_pos):
        c = np.sqrt(np.sum(self.v**2, axis=1)) / self.vmax
        c = np.concatenate((c.reshape((self.nat, 1)), np.zeros((self.nat, 2))), axis=1)
        c = np.minimum(c, 1.0)
        shelve_pos[str(iteration)] = [self.ats, c]

    def updateSim(self):
        t_start_update = time.time()
        # neis, nneis = buildNeighbourList(self.ats, self.rcut, maxNeis=self.maxNeis)
        # self.neilist()
        # t_end = time.time()
        # print('neislit', t_end - t_start)
        for i in range(1):
            #        t_start = time.time()
            if self.cuda:
                neis_d, nneis_d = buildNeighbourList_cuda(self.ats_d, self.rcut, self.maxNeis, self.nCudaThreads)
            else:
                neis, nneis = buildNeighbourList(self.ats, self.rcut, maxNeis=self.maxNeis)
            #    absv = np.sqrt(np.sum(self.v**2, axis=1))
            #    vmax = np.max(absv[nneis > 0]) # what if fast clusters of particles escape but their relative velocity is small?
            #    dt = 20. / vmax * 0.003#0.003
            #    #dt = min(0.01, dt)
            #    dt = min(self.rcut / 2 / np.max(absv), dt)
            #    print('dt', dt, self.rcut / np.max(absv) / 10)
            #    dt = 0.003
            #        t_end = time.time()
            #        print('T(neilist)', t_end - t_start)
            #        t_start = time.time()
            if self.cuda:
                # neis_d = cuda.to_device(neis)
                # nneis_d = cuda.to_device(nneis)
                nCudaBlocks = int((self.nat + self.nCudaThreads - 1) / self.nCudaThreads)
                force_cuda[nCudaBlocks, self.nCudaThreads](self.ats_d, neis_d, nneis_d, self.f_d, self.eat_d)
                verlet_cuda[nCudaBlocks, self.nCudaThreads](self.ats_d, self.v_d, self.m_d, self.f_d, self.lastf_d,
                                                            self.dt, 10)
            else:
                self.ats, self.v = verlet(self.ats, self.v, self.m, neis, nneis, force_co, self.dt, 10)
        #        t_end = time.time()
        #        print('T(verlet)', t_end - t_start)
        # relax(self.ats, neis, nneis, force, 10)

        if self.cuda:
            #       t_start = time.time()
            self.ats = self.ats_d.copy_to_host()
            self.v = self.v_d.copy_to_host()

    #       self.eat = self.eat_d.copy_to_host()
    #       t_end = time.time()
    #       print('T(copy)', t_end - t_start)
        t_end_update = time.time()
        # print('T(update)', t_end_update - t_start_update)

    # use a grid for performance?
    def neilist(self):
        tree = scipy.spatial.cKDTree(self.ats, leafsize=16)
        ns = tree.query_ball_point(self.ats, self.rcut)
        neis = List()
        [neis.append(np.array(n)) for n in ns]
        #for n in ns:
        #    neis.append(np.array(n))
        return neis


def read_in(filename):
    paramDict = {
        'schritte': 500,
        'geschwindigkeit': 5.0,
        'anzahl schichten': 2,
        'schichtdicke': 10,
        'winkel': 0.0,
        'zeitschritt': 0.01,
        'schichtung': False
    }
    with open(filename) as f:
        d = json.load(f)
    d = paramDict | d
    if not type(d["schichtung"]) is list: 
        d["schichtung"] = [d["schichtung"]]
    if not type(d["schichtdicke"]) is list: 
        d["schichtdicke"] = [d["schichtdicke"]]

    if len(d["schichtung"]) is not d['anzahl schichten']:
        temp = d["schichtung"][0]
        d["schichtung"] = []
        for _ in range(d['anzahl schichten']):
            d["schichtung"].append(temp)

    if len(d["schichtdicke"]) is not d['anzahl schichten']:
        temp = d["schichtdicke"][0]
        d["schichtdicke"] = []
        for _ in range(d['anzahl schichten']):
            d["schichtdicke"].append(temp)

    d["winkel"] = d["winkel"] * 0.0174533

    return d["schritte"], d["geschwindigkeit"], d["anzahl schichten"], d['schichtdicke'], d["winkel"]\
        , d["zeitschritt"], d["schichtung"]





# if __name__ == '__main__':
#     filename = 'input.txt'
#     n_steps, velocity, n_layers, thickness = read_in(filename)
#     impactSim = ImpactSim(velocity=velocity,
#                             n_layers = n_layers,
#                             thickness = thickness)
#     #impactSim.anim.save('impact.mp4', writer=animation.FFMpegWriter(fps=30))
#     #plt.show()
#     impactSim.setup()

#     path = 'images'
#     if not os.path.exists(path):
#         os.mkdir(path)

#     k = 0
#     for i in range(n_steps):
#         impactSim.updateSim()
#         if i%10 == 0:
#             impactSim.savePic(k, path)
#             k+=1
