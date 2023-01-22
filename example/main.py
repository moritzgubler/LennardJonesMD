import impact_simulation.impactSim
import os
import shelve
from tqdm import tqdm

def main():
    filename = 'input.txt'
    n_steps, velocity, n_layers, thickness = impact_simulation.impactSim.read_in(filename)
    impactSim = impact_simulation.impactSim.ImpactSim(velocity=velocity,
                            n_layers = n_layers,
                            thickness = thickness)
    #impactSim.anim.save('impact.mp4', writer=animation.FFMpegWriter(fps=30))
    #plt.show()
    # impactSim.setup()

    path = 'images'

    k = 0

    if os.path.exists('fp.npz'):
        os.remove('fp.npz')
    if os.path.exists('fc.npz'):
        os.remove('fc.npz')
    with shelve.open('fp.npz') as fp, shelve.open('fc.npz') as fc:
        for i in tqdm( range(n_steps), desc="Calculating"):
            impactSim.updateSim()
            if i%5 == 0:
                # impactSim.savePic(k, path)
                impactSim.savePos(k, fp, fc)
                k+=1


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