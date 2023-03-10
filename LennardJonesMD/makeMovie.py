import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import shelve
import progressbar
import argparse

def main():

    inputFilename = 'positions.dat'
    outputFilename = 'movie.mkv'

    parser = argparse.ArgumentParser(description ='Make movie of a 2d Lennard Jones Simulation')
    parser.add_argument('-i', '--inputfile', dest ='inputFilename',
                    action ='store', help ='input filename. Defauls is '+inputFilename, default=inputFilename)
    parser.add_argument('-o', '--outputfile', dest ='outputFilename',
                    action ='store', help ='output file. Default is'  + outputFilename, default=outputFilename)

    args = parser.parse_args()

    inputFilename = args.inputFilename
    outputFilename = args.outputFilename

    pos_shelve = shelve.open(inputFilename)

    n_steps = len(pos_shelve)
    [p0, c0] = pos_shelve[str(0)]

    os = 5
    fig, ax = plt.subplots()
    plt.axis("off")
    fig.set_size_inches((16,9)) 
    s = ax.scatter([], [])

    movie=True
    widgets = [' [',
         progressbar.Timer(format= 'elapsed time: %(elapsed)s'),
         '] ',
           progressbar.Bar('*'),' (',
           progressbar.ETA(), ') ',
          ]
    print('Creating movie')
    bar = progressbar.ProgressBar(maxval=n_steps,
                              widgets=widgets).start()

    def animate(i):
        [p, c] = pos_shelve[str(i)]
        # ax.clear()
        # ax.axis([np.min(p0[:, 0]) - os,np.max(p0[:, 0]) + os,np.min(p0[:, 1]) - os,np.max(p0[:, 1]) + os])

        # ax.scatter(p[:, 0], p[:, 1], s = n_steps, c=c)
        # print(i)
        s.set_offsets(p)
        s.set_facecolor(c)
        if movie:
            bar.update(i+1)

        # mat.set_data(p[:, 0], p[:, 1])
        # # ax.axis([np.min(p[:, 0]) - os,np.max(p[:, 0]) + os,np.min(p[:, 1]) - os,np.max(p[:, 1]) + os])
        # return mat)

    ax.axis([np.min(p0[:, 0]) - os,np.max(p0[:, 0]) + os,np.min(p0[:, 1]) - os,np.max(p0[:, 1]) + os])
    # mat, = plt.scatter(p0[:, 0], p0[:, 1], s = n_steps, c=c0)


    ani = animation.FuncAnimation(fig, animate, interval=30, frames=n_steps)
    ani.save(outputFilename)
    movie = False
    plt.show()
    # ani.resume()

if __name__ == '__main__':
    main()