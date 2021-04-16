from PyQt5 import uic, QtWidgets
import modules, modules2
from time import time
import numba
import numpy

Form, Window = uic.loadUiType("geo.ui")
app = QtWidgets.QApplication([])
window = Window()
form = Form()
form.setupUi(window)
window.show()

start = time()
range_a = -4.0
range_b = 12.0 
precision = 3
generations_number = 200000
tau = 1.5

def run_evolution():
    start = time()
    #modules.evolution(range_a, range_b, precision, population_size, generations_number, crossover_probability, mutation_probability, elite_number, False)
    #modules.evolution(range_a, range_b, precision, tau, generations_number, save_file=True)
    stop = time()-start
    print(stop)

    start = time()
    modules2.evolution(range_a, range_b, precision, tau, generations_number, save_file=True)
    stop = time()-start
    print(stop)
    #form.label_time.setText(str(modules.real(798 ,range_a, range_b, precision)))
    
    #print([int(n) for n in bin(798)[2:].zfill(14)])
    #form.label_time.setText(str(stop))

form.button_start.clicked.connect(run_evolution)
app.exec()