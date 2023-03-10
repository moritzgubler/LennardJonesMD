import sys
import time
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
import numpy as np
import pyqtgraph as pg
import shelve
import time
import argparse

class App(QtWidgets.QMainWindow):
    def __init__(self, parent=None, shelveFilename='positions.dat'):
        super(App, self).__init__(parent)


        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')

        self.pos_shelve = shelve.open(shelveFilename)

        self.scale = 8.0

        self.targetfps = 30
        self.targetTime = 1 / self.targetfps
        self.ncolors = 32

        #### Create Gui Elements ###########
        self.mainbox = QtWidgets.QWidget()
        self.setCentralWidget(self.mainbox)
        self.mainbox.setLayout(QtWidgets.QVBoxLayout())

        self.canvas = pg.GraphicsLayoutWidget()
        self.mainbox.layout().addWidget(self.canvas)

        self.label = QtWidgets.QLabel()
        self.mainbox.layout().addWidget(self.label)

        self.view = self.canvas.addViewBox()
        self.view.setAspectLocked(True)
        self.p0, self.c0 = self.pos_shelve[str(0)]
        self.p0 = self.p0 * self.scale

        self.offset=2.0
        self.minx = np.min(self.p0[:,0]) - self.offset
        self.maxx = np.max(self.p0[:,0]) + self.offset
        self.miny = np.min(self.p0[:,1]) - self.offset
        self.maxy = np.max(self.p0[:,1]) + self.offset
        self.dx = self.maxx - self.minx
        self.dy = self.maxy - self.miny

        self.brushList = []
        for i in range(self.ncolors):
            self.brushList.append(QtGui.QBrush( QtGui.QColor(int(i*256/self.ncolors), 0, 0, 255)))

        self.brushes = []
        for i, c in enumerate(self.c0):
            self.brushes.append(self.brushList[int(c[0]*(self.ncolors-1))])

        


        self.view.setRange(QtCore.QRectF(self.minx,self.miny, self.dx, self.dy))

        self.sc = pg.ScatterPlotItem()
        self.view.addItem(self.sc)

        self.canvas.nextRow()


        #### Set Data  #####################

        self.counter = 0
        self.fps = 0.
        self.lastupdate = time.time()


        self.n_steps = len(self.pos_shelve)
        # self.p = self.pos_shelve[str(self.counter)]
        self._update()

    def _update(self):

        self.p, self.col = self.pos_shelve[str(self.counter)]
        self.p = self.p * self.scale

        for i, c in enumerate(self.col):
            self.brushes[i] = self.brushList[int(c[0]*(self.ncolors-1))]

        # t1 = time.time()
        self.sc.setData(self.p[:, 0], self.p[:, 1], pen=None, brush=self.brushes, antialias=False)
        # t2 = time.time()
        # print((t2 - t1))

        now = time.time()
        dt = (now-self.lastupdate)
        if dt <= 0:
            dt = 0.000000000001
        if dt <= self.targetTime:
            # print("delaying")
            time.sleep(self.targetTime - dt)
        now = time.time()
        dt = (now-self.lastupdate)
        fps2 = 1.0 / dt
        self.lastupdate = now
        self.fps = self.fps * 0.9 + fps2 * 0.1
        tx = '{fps:.3f} FPS'.format(fps=self.fps )
        self.label.setText(tx)
        QtCore.QTimer.singleShot(1, self._update)
        self.counter += 1
        if self.counter >= self.n_steps:
            self.counter = 0

def main():
    inputFilename = 'positions.dat'

    parser = argparse.ArgumentParser(description ='Plot a 2d Lennard Jones Simulation')
    parser.add_argument('-i', '--inputfile', dest ='inputFilename',
                    action ='store', help ='name of input file. Default is '+inputFilename, default=inputFilename)

    args = parser.parse_args()

    inputFilename = args.inputFilename

    pg.setConfigOptions(useOpenGL=False)
    # QtWidgets.QApplication.setHighDpiScaleFactorRoundingPolicy(QtCore.Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    # QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)

    app = QtWidgets.QApplication(sys.argv)
    # app.setAttribute(QtCore.Qt.AA_Use96Dpi)
    thisapp = App(shelveFilename=inputFilename)
    thisapp.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()

