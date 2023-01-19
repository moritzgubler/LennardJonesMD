


filename = 'input.txt'
n_steps, velocity, n_layers, thickness = read_in(filename)
impactSim = ImpactSim(velocity=velocity,
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