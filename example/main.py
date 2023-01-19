import impact_simulation.impactSim
import os

def main():
    filename = 'input.txt'
    n_steps, velocity, n_layers, thickness = impact_simulation.impactSim.read_in(filename)
    impactSim = impact_simulation.impactSim.ImpactSim(velocity=velocity,
                            n_layers = n_layers,
                            thickness = thickness)
    #impactSim.anim.save('impact.mp4', writer=animation.FFMpegWriter(fps=30))
    #plt.show()
    impactSim.setup()

    path = 'images'
    if not os.path.exists(path):
        os.mkdir(path)

    k = 0
    for i in range(n_steps):
        impactSim.updateSim()
        if i%10 == 0:
            impactSim.savePic(k, path)
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
    print(s.getvalue())