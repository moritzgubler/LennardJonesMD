import LennardJonesMD.impactSim
import os
import shelve
from tqdm import tqdm

def main():
    filename = 'input.json'
    n_steps, velocity, n_layers, thickness, theta, dt, layerflip = LennardJonesMD.impactSim.read_in(filename)
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

    if os.path.exists('fp.npz'):
        os.remove('fp.npz')
    with shelve.open('fp.npz') as fp:
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