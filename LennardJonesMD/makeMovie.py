import itertools
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import shelve

def main():
    pos_shelve = shelve.open('fp.npz')

    n_steps = len(pos_shelve)
    [p0, c0] = pos_shelve[str(0)]

    os = 5
    fig, ax = plt.subplots()
    plt.axis("off")
    s = ax.scatter([], [])

    def animate(i):
        [p, c] = pos_shelve[str(i)]
        # ax.clear()
        # ax.axis([np.min(p0[:, 0]) - os,np.max(p0[:, 0]) + os,np.min(p0[:, 1]) - os,np.max(p0[:, 1]) + os])

        # ax.scatter(p[:, 0], p[:, 1], s = n_steps, c=c)
        # print(i)
        s.set_offsets(p)
        s.set_facecolor(c)

        # mat.set_data(p[:, 0], p[:, 1])
        # # ax.axis([np.min(p[:, 0]) - os,np.max(p[:, 0]) + os,np.min(p[:, 1]) - os,np.max(p[:, 1]) + os])
        # return mat)

    ax.axis([np.min(p0[:, 0]) - os,np.max(p0[:, 0]) + os,np.min(p0[:, 1]) - os,np.max(p0[:, 1]) + os])
    # mat, = plt.scatter(p0[:, 0], p0[:, 1], s = n_steps, c=c0)


    ani = animation.FuncAnimation(fig, animate, interval=20, frames=n_steps)
    ani.save('movie.mkv')
    plt.show()
    # ani.resume()

if __name__ == '__main__':
    main()