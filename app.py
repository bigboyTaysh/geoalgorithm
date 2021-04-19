from PyQt5 import uic, QtWidgets, QtCore, QtGui
from lib.modules import evolution
from time import time
import numpy
from PyQt5.QtChart import QChart, QLineSeries, QScatterSeries

Form, Window = uic.loadUiType("geo.ui")
app = QtWidgets.QApplication([])
window = Window()
form = Form()
form.setupUi(window)
chart = QChart()
chart.setBackgroundBrush(QtGui.QColor(41, 43, 47))
form.widget.setChart(chart)
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

    best, fxs = evolution(range_a, range_b, precision, tau, generations_number, save_file=True)
    
    form.best_table.item(1,0).setText(str(best.real))
    form.best_table.item(1,1).setText(''.join(map(str, best.binary)))
    form.best_table.item(1,2).setText(str(best.fx))
    
    chart = QChart()
    series = QLineSeries()
    points = QScatterSeries()

    pen = series.pen()
    pen.setWidth(1)
    pen.setBrush(QtGui.QColor(114, 137, 218))
    series.setPen(pen)

    pen_points = points.pen()
    pen_points.setBrush(QtGui.QColor("white"))
    points.setMarkerSize(8)
    points.setColor(QtGui.QColor("red"))
    points.setPen(pen_points)

    for i in range(1, generations_number+1):
        if fxs[i-1] == best.fx:
            points.append(i, fxs[i-1])
        series.append(i, fxs[i-1])
    
    chart.addSeries(series)
    chart.addSeries(points)
    
    chart.setBackgroundBrush(QtGui.QColor(41, 43, 47))
    chart.createDefaultAxes()
    chart.legend().hide()
    chart.setContentsMargins(-10, -10, -10, -10)
    chart.layout().setContentsMargins(0, 0, 0, 0)
    chart.axisY().setRange(-2.1,2)
    chart.axisX().setLabelsColor(QtGui.QColor("white"))
    chart.axisX().setLabelFormat("%i")
    chart.axisY().setLabelsColor(QtGui.QColor("white"))
    form.widget.setChart(chart)

    app.restoreOverrideCursor()
    stop = time()-start
    print(stop)

form.button_start.clicked.connect(run_evolution)
app.exec()