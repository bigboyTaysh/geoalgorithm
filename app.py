from PyQt5 import uic, QtWidgets
from lib.modules import evolution
from time import time
import numba
import numpy

Form, Window = uic.loadUiType("geo.ui")
app = QtWidgets.QApplication([])
window = Window()
#window.setWindowFlags(QtCore.Qt.FramelessWindowHint)
form = Form()
form.setupUi(window)
window.show()

def run_evolution():
    range_a = float(str(form.input_a.text()))
    range_b = float(str(form.input_b.text()))
    precision = int(str(form.input_d.text()))
    generations_number = int(str(form.input_t.text()))
    tau = float(str(form.input_tau.text()))
    
    start = time()
    best = evolution(range_a, range_b, precision, tau, generations_number, save_file=True)
    print(best.real)
    form.best_table.item(1,0).setText(str(best.real))
    form.best_table.item(1,1).setText(''.join(map(str, best.binary)))
    form.best_table.item(1,2).setText(str(best.fx))
    stop = time()-start
    print(stop)

form.button_start.clicked.connect(run_evolution)
app.exec()