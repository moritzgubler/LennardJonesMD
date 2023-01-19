import itertools
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import shelve

def main():
    pos_shelve = shelve.open('fp.npz')
    col_shelve = shelve.open('fc.npz')

    n_steps = len(pos_shelve)
    p0 = pos_shelve[str(0)]
    c0 = col_shelve[str(0)]

    os = 5

    fig, ax = plt.subplots()

    def animate(i):
        p = pos_shelve[str(i)]
        c = col_shelve[str(i)]
        ax.clear()
        ax.axis([np.min(p0[:, 0]) - os,np.max(p0[:, 0]) + os,np.min(p0[:, 1]) - os,np.max(p0[:, 1]) + os])

        ax.scatter(p[:, 0], p[:, 1], s = n_steps, c=c)
        print(i)


        # mat.set_data(p[:, 0], p[:, 1])
        # # ax.axis([np.min(p[:, 0]) - os,np.max(p[:, 0]) + os,np.min(p[:, 1]) - os,np.max(p[:, 1]) + os])
        # return mat)

    print(c0.shape)

    ax.axis([np.min(p0[:, 0]) - os,np.max(p0[:, 0]) + os,np.min(p0[:, 1]) - os,np.max(p0[:, 1]) + os])
    # mat, = plt.scatter(p0[:, 0], p0[:, 1], s = n_steps, c=c0)


    ani = animation.FuncAnimation(fig, animate, interval=10, frames=n_steps)
    plt.show()
    # ani.resume()

if __name__ == '__main__':
    main()
    quit()





def neighbors(point):
    x, y = point
    for i, j in itertools.product(range(-1, 2), repeat=2):
        if any((i, j)):
            yield (x + i, y + j)

def advance(board):
    newstate = set()
    recalc = board | set(itertools.chain(*map(neighbors, board)))

    for point in recalc:
        count = sum((neigh in board)
                for neigh in neighbors(point))
        if count == 3 or (count == 2 and point in board):
            newstate.add(point)

    return newstate

glider = set([(0, 0), (1, 0), (2, 0), (0, 1), (1, 2)])

fig, ax = plt.subplots()

x, y = zip(*glider)
mat, = ax.plot(x, y, 'o')

def animate(i):
    global glider
    glider = advance(glider)
    x, y = zip(*glider)
    mat.set_data(x, y)
    return mat,

ax.axis([-15,5,-15,5])
ani = animation.FuncAnimation(fig, animate, interval=50)
plt.show()