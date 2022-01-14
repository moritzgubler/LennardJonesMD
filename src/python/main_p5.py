from integration import verlet
from lennard_jones import force_co as force
from main import ImpactSim

import numpy as np
import timeit
from p5 import *

offset = np.array([120, 40])
scale = 10
impactSim = ImpactSim()

def setup():
    size(150 * scale, 80 * scale)
    background(255)



def draw():
    background(255)
    fill(0)
    impactSim.update(0)
    for i in range(impactSim.nat):
        x = impactSim.ats[i,:] + offset
        x *= scale
        circle(x[0], x[1], 0.5 * scale)


if __name__ == '__main__':
    run()