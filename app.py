from PyQt5 import uic, QtWidgets, QtCore, QtGui
from lib.modules import evolution, test_tau
from time import time
import numpy
from PyQt5.QtChart import QChart, QLineSeries, QScatterSeries
from lib.models import Test

Form, Window = uic.loadUiType("geo.ui")
app = QtWidgets.QApplication([])
window = Window()
form = Form()
form.setupUi(window)
chart = QChart()
chart.setBackgroundBrush(QtGui.QColor(41, 43, 47))
form.widget.setChart(chart)
form.widget_test.setChart(chart)
form.tabWidget.setTabText(0, "Algorytm")
form.tabWidget.setTabText(1, "Testy")
window.show()

def run_evolution():
    range_a = float(str(form.input_a.text()))
    range_b = float(str(form.input_b.text()))
    precision = int(str(form.input_d.text()))
    tau = float(str(form.input_tau.text()))
    generations_number = int(str(form.input_t.text()))

    start = time()
    app.setOverrideCursor(QtCore.Qt.WaitCursor)

    best, fxs, best_fx = evolution(range_a, range_b, precision, tau, generations_number, save_file=True)
    
    form.best_table.item(1,0).setText(str(best.real))
    form.best_table.item(1,1).setText(''.join(map(str, best.binary)))
    form.best_table.item(1,2).setText(str(best.fx))
    
    chart = QChart()
    series = QLineSeries()
    bests = QLineSeries()
    points = QScatterSeries()

    pen = series.pen()
    pen.setWidth(1)
    pen.setBrush(QtGui.QColor(114, 137, 218))
    series.setPen(pen)

    pen_best = bests.pen()
    pen_best.setWidth(1)
    pen_best.setBrush(QtGui.QColor("red"))
    bests.setPen(pen_best)

    pen_points = points.pen()
    pen_points.setBrush(QtGui.QColor("white"))
    points.setMarkerSize(8)
    points.setColor(QtGui.QColor("red"))
    points.setPen(pen_points)

    for i in range(1, generations_number+1):
        if fxs[i-1] == best.fx:
            points.append(i, fxs[i-1])
        series.append(i, fxs[i-1])
        bests.append(i, best_fx[i-1])
    
    chart.addSeries(series)
    chart.addSeries(bests)
    chart.addSeries(points)
    
    chart.setBackgroundBrush(QtGui.QColor(41, 43, 47))
    chart.createDefaultAxes()
    chart.legend().hide()
    chart.setContentsMargins(-10, -10, -10, -10)
    chart.layout().setContentsMargins(0, 0, 0, 0)
    chart.axisY().setRange(-2,2)
    chart.axisX().setLabelsColor(QtGui.QColor("white"))
    chart.axisX().setLabelFormat("%i")
    chart.axisY().setLabelsColor(QtGui.QColor("white"))
    form.widget.setChart(chart)

    app.restoreOverrideCursor()
    stop = time()-start
    print(stop)

def test_generations():
    tau = float(str(form.input_tau.text()))
    print(str(tau))

def test_taus():
    range_a = float(str(form.input_a_test.text()))
    range_b = float(str(form.input_b_test.text()))
    precision = int(str(form.input_d_test.text()))
    generations_number = int(str(form.input_t_test.text()))

    start = time()
    tests = test_tau(range_a, range_b, precision, generations_number)
    print(time()-start)

    chart = QChart()
    series_bests = QLineSeries()

    form.test_table.setRowCount(0)

    for i in range(0, 50):
        series_bests.append(i, tests[i].fx)
        form.test_table.insertRow(i)
        form.test_table.setItem(i, 0, QtWidgets.QTableWidgetItem(str(tests[i].tau)))
        form.test_table.setItem(i, 1, QtWidgets.QTableWidgetItem(str(tests[i].generations_number)))
        form.test_table.setItem(i, 2, QtWidgets.QTableWidgetItem(str(tests[i].fx)))



    chart.addSeries(series_bests)

    chart.setBackgroundBrush(QtGui.QColor(41, 43, 47))
    chart.createDefaultAxes()
    chart.legend().hide()
    chart.setContentsMargins(-10, -10, -10, -10)
    chart.layout().setContentsMargins(0, 0, 0, 0)
    chart.axisX().setRange(0.1, 5.0)
    chart.axisY().setRange(-2, 2)
    chart.axisX().setLabelsColor(QtGui.QColor("white"))
    chart.axisY().setLabelsColor(QtGui.QColor("white"))
    form.widget_test.setChart(chart)

form.button_start.clicked.connect(run_evolution)
form.button_test_generations.clicked.connect(test_generations)
form.button_test_tau.clicked.connect(test_taus)
app.exec()