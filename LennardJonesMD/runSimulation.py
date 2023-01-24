import LennardJonesMD.impactSim
import os
import shelve
from tqdm import tqdm
import argparse

def main():

    inputFilename = 'input.json'
    outputFilename = 'positions.dat'

    parser = argparse.ArgumentParser(description ='Run a 2d Lennard Jones Simulation')
    parser.add_argument('-i', '--inputfile', dest ='inputFilename',
                    action ='store', help ='input filename. Defauls is '+inputFilename, default=inputFilename)
    parser.add_argument('-o', '--outputfile', dest ='outputFilename',
                    action ='store', help ='output file. Default is'  + outputFilename, default=outputFilename)

    args = parser.parse_args()

    inputFilename = args.inputFilename
    outputFilename = args.outputFilename

    if not os.path.exists(inputFilename):
        print("Input file does not exist. Aborting...")
        quit()

    n_steps, velocity, n_layers, thickness, theta, dt, layerflip = LennardJonesMD.impactSim.read_in(inputFilename)
    impactSim = LennardJonesMD.impactSim.ImpactSim(velocity=velocity,
                            n_layers = n_layers,
                            thickness = thickness,
                            theta = theta,
                            layerFlip=layerflip,
                            dt=dt)
    #impactSim.anim.save('impact.mp4', writer=animation.FFMpegWriter(fps=30))
    #plt.show()
    # impactSim.setup()

    path = 'images'

    k = 0

    if os.path.exists(outputFilename):
        os.remove(outputFilename)
    with shelve.open(outputFilename) as fp:
        for i in tqdm( range(n_steps), desc="Calculating"):
            impactSim.updateSim()
            impactSim.savePos(k, fp)
            k += 1


if __name__ == '__main__':
    import cProfile, pstats, io
    from pstats import SortKey
    pr = cProfile.Profile()
    pr.enable()

    main()

    pr.disable()
    s = io.StringIO()
    sortby = SortKey.CUMULATIVE
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    # print(s.getvalue())