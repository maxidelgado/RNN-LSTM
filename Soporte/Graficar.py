# coding=utf-8
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.finance import candlestick_ohlc, index_bar
import matplotlib.dates as md
import matplotlib.ticker as mticker
from matplotlib.finance import date2num

class Graficar:
    __path = ''

    def __init__(self, path = ''):
        self.__path = path

    def csv_ohcl(self,nombre_moneda):
        try:
            # Generar numpy.array con los 4 datos, convirtiendo la fecha en flotante para graficar
            convertir = lambda x: date2num(datetime.strptime(x.decode("utf-8"), '"%Y-%m-%d %H:%M:%S"'))
            fecha, apertura, alto, bajo, cierre = np.genfromtxt(
                self.__path, delimiter=',', unpack=True,
                skip_header=1, skip_footer=3,
                converters={0: convertir}
            )

            # Agrego todos los valores a mi Array de ohcl
            ohlc = []
            for i in range(len(fecha)):
                agregar = fecha[i], apertura[i], alto[i], bajo[i], cierre[i]
                ohlc.append(agregar)

            # Generar el gráfico ohlc y configuraciones básicas del eje x
            figura = plt.figure()
            ax = plt.subplot2grid((1, 1), (0, 0))
            candlestick_ohlc(ax, ohlc, width=0.00035)
            xfmt = md.DateFormatter('%H:%M')
            ax.xaxis.set_major_formatter(xfmt)

            for label in ax.xaxis.get_ticklabels():
                label.set_rotation(90)

            ax.xaxis.set_major_locator(mticker.MaxNLocator(50))
            plt.xlabel('Hora')
            plt.ylabel('Precio')
            plt.title(nombre_moneda)
            plt.grid(False)
            plt.show()

        except IOError:
            print 'No se pudo leer el archivo o está vacío'

    def cvs_closing_price(self, nombre_moneda):
        try:
            # Generar numpy.array con los 4 datos, convirtiendo la fecha en flotante para graficar
            convertir = lambda x: date2num(datetime.strptime(x.decode("utf-8"), '"%Y-%m-%d %H:%M:%S"'))
            fecha, cierre = np.genfromtxt(
                self.__path, delimiter=',', unpack=True,
                skip_header=1, skip_footer=3,
                converters={0: convertir}
            )

            xy = []

            for i in range(len(fecha)):
                agregar = fecha[i], cierre[i]
                xy.append(agregar)

            plt.gca().xaxis.set_major_formatter(md.DateFormatter('%H:%M'))
            plt.gca().xaxis.set_major_locator(md.HourLocator())
            plt.gca().xaxis.set_major_locator(mticker.MaxNLocator(50))

            plt.plot(fecha,cierre)
            plt.gcf().autofmt_xdate()
            plt.xlabel('Hora')
            plt.ylabel('Precio')
            plt.title(nombre_moneda)
            plt.grid(False)
            plt.show()

        except IOError:
            print 'No se pudo leer el archivo o está vacío'