import sys
import time
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
import numpy as np
import pyqtgraph as pg
import shelve
import time

class App(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super(App, self).__init__(parent)


        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')

        self.pos_shelve = shelve.open('fp.npz')

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

        for i, c in enumerate(self.col):
            self.brushes[i] = self.brushList[int(c[0]*(self.ncolors-1))]

        # t1 = time.time()
        self.sc.setData(self.p[:, 0], self.p[:, 1], pen=None, brush=self.brushes)
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
        tx = 'Mean Frame Rate:  {fps:.3f} FPS'.format(fps=self.fps )
        self.label.setText(tx)
        QtCore.QTimer.singleShot(1, self._update)
        self.counter += 1
        if self.counter >= self.n_steps:
            self.counter = 0

def pprint():
    print("hi")


if __name__ == '__main__':

    app = QtWidgets.QApplication(sys.argv)
    thisapp = App()
    thisapp.show()
    sys.exit(app.exec_())