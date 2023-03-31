#!/usr/bin/env python
# coding: utf-8

# LIBRERÍAS

import pandas as pd
import numpy as np

from bokeh.io import output_notebook, show, curdoc
from bokeh.plotting import figure
from bokeh.layouts import row, column
from bokeh.models import ColumnDataSource, Slider, Select, PreText, LinearAxis, Range1d, Div

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Presa seleccionada: Quebrada de Ullúm (Inventario de Presas - Tomo III)

# Topografía tomada del inventario de presas

COTA_01 = [726.19, 730.07, 735.03, 739.99, 745.06, 750.13, 755.10, 760.18, 765.05, 770.03, 772.30] # msnm
AREA_01 = [0, 1.06, 2.56, 4.45, 6.88, 10.03, 14.05, 19.19, 25.47, 33.86, 37.68] # km^2


DF_COTA_AREA = pd.DataFrame({"COTA_01" : COTA_01,
                             "AREA_01" : AREA_01})

# Cambiar el cero de referencia al nivel mínimo del embalse
REF = min(COTA_01)

COTA_02 = [i - REF for i in COTA_01]

DF_COTA_AREA["COTA_02"] = DF_COTA_AREA["COTA_01"] - REF

# Encontrar una función cuadrática que vincule A para todo H

poly = PolynomialFeatures(degree=2, include_bias=False) # Include_bias = True forces y-intercept to equal zero
poly_features = poly.fit_transform(np.array(COTA_02).reshape(-1, 1))


poly_reg_model = LinearRegression(fit_intercept=False)
poly_reg_model.fit(poly_features, [i*1000**2 for i in AREA_01]) # Tal que devuelva áreas en m^2

AREA_COEF_A = poly_reg_model.coef_[1] # Multiplica el termino al cuadrado
AREA_COEF_B = poly_reg_model.coef_[0] # Multiplica el termino lineal

# Aplicar A(h) para un rango de alturas
DF_COTA_AREA_2 = pd.DataFrame()

DF_COTA_AREA_2["COTA_01"] = np.arange(min(COTA_01), max(COTA_01)+1, 1)
DF_COTA_AREA_2["COTA_02"] = DF_COTA_AREA_2["COTA_01"] - REF
DF_COTA_AREA_2["AREA_02"] = [((AREA_COEF_A*i**2) + (AREA_COEF_B*i)) / 1000**2 for i in DF_COTA_AREA_2["COTA_02"]]

# Encontrar los coeficientes para vincular volumen con cota a través de la integración de la expresión cuadrática A=f(h)

VOL_COEF_C = AREA_COEF_A / 3 # Multiplica el termino al cubo
VOL_COEF_D = AREA_COEF_B / 2 # Multiplica el termino cuadrático

# Estos coeficientes devuelven volúmenes en m^3

# Aplicar Vol(h) para un rango de alturas
DF_COTA_VOL = pd.DataFrame()

DF_COTA_VOL["COTA_01"] = np.arange(min(COTA_01), max(COTA_01)+1, 1)
DF_COTA_VOL["COTA_02"] = DF_COTA_VOL["COTA_01"] - REF
DF_COTA_VOL["VOL_02"] = [((VOL_COEF_C*i**3) + (VOL_COEF_D*i**2)) / 100**3 for i in DF_COTA_VOL["COTA_02"]]


# Definir funciones que devuelven A en m2 y Vol en m3 en función de la altura

def AREA(H):
    """
    Devuelve A en m^2
    """
    
    A = AREA_COEF_A * (H**2) + AREA_COEF_B * (H)
    
    return A


def VOL(H):
    """
    Devuelve VOL en m^3
    """
    
    VOL = VOL_COEF_C * (H**3) + VOL_COEF_D * (H**2)
    
    return VOL


# Definición del hidrograma de entrada variable - Kozeny

Q_base = 60.0 # m^3/s (propuesto)

def kozeny(Q_tr, tp, m, t):
    
    """
    Devuelve ordenada del hidrograma de entrada I(t) en m3/s para el caudal al pico,
    tiempo al pico y factor de forma m dados
    """
    
    q = ((t / tp) * np.exp(1 - t/tp))**m
    
    I_t = (Q_tr - Q_base)*q + Q_base
    
    return I_t


# Caudal de diseño del vertedero tomado del Inventario de Presas
Q_DIS = 6500 # m3/s . 
T_PICO = 12 # hs
FACTOR_FORMA = 4


DF_HID_I = pd.DataFrame()

DF_HID_I["T"] = [(1/6)*i for i in range(100*6 + 1)] # En hs

DF_HID_I["I"] = DF_HID_I["T"].apply(lambda x: kozeny(Q_DIS, T_PICO, FACTOR_FORMA, x))

# Definición de las características del vertedero

# Ley de VERTEDERO
c = 2.2 # m^0.5/s, coeficiente de descarga
B_ORIGINAL = 70 # m, Ancho original del vertedero
H_CRESTA = 40 #768.0 - REF #m , NMN original de la presa

# Ley de VERTEDERO

def LEY_VERTEDERO(NIVEL, B, CRESTA):
    
    """
    Devuelve el caudal erogado por vertedero libre para el nivel de embalse, altura de cresta y ancho de vertedero dados
    """
   
    if NIVEL >= CRESTA:   
        
        O = c * B * ((NIVEL - CRESTA)**(3/2))
        
        return O
    
    if NIVEL < CRESTA:
        
        return 0.0
   
    else:
        
        return np.nan

# Definición del método numérico de resolución (Runge Kutta - Orden 3) - Al igual que en el Ven Te Chow.

k = 600 #s (se trata de un hidrograma de entrada equiespaciado en el tiempo cada 600 segundos)


def RK_3(NIVEL, B, CRESTA, I_t1, I_t2, I_t3):
    
    """
    Se entra con un nivel inicial, devuelve el nivel final luego de un tiempo k.
    """
    
    # h1 se evalua en h0 y t0
    h1 = k * (((I_t1 - LEY_VERTEDERO(NIVEL, B, CRESTA))) / AREA(NIVEL)) # m

    # h2 se evalua en h0+(1/3)h1 y 1/3(t1-t0)
    h2 = k * (((I_t2 - LEY_VERTEDERO(NIVEL + (h1/3), B, CRESTA))) / AREA(NIVEL + (h1/3))) # m
    
    # h3 se evalua en h0+(2/3)h2 y 2/3(t1-t0)
    h3 = k * (((I_t2 - LEY_VERTEDERO(NIVEL + (2*h1/3), B, CRESTA))) / AREA(NIVEL + (2*h1/3))) # m

    h_RK3 = NIVEL + ((1/4)*h1) + ((3/4)*(h3))

    return h_RK3


# Resolver laminación (para nivel inicial coincidente con NCV)


DF_TAC_01 =  DF_HID_I.copy()

# Proponer un nivel inicial del embalse
DF_TAC_01["NIVEL"] = np.nan
DF_TAC_01["NIVEL"].iloc[0] = H_CRESTA - 1

# Aplicar RK3

for i in range(1, len(DF_TAC_01)):
    
    N_0 = DF_TAC_01.loc[i-1, "NIVEL"]
    
    I_1 = DF_TAC_01.loc[i-1, "I"]
    
    I_2 = (2/3)*I_1 + (1/3)*DF_TAC_01.loc[i, "I"]
    
    I_3 = (1/3)*I_1 + (2/3)*DF_TAC_01.loc[i, "I"]
    
    DF_TAC_01.loc[i, "NIVEL"] = RK_3(N_0, B_ORIGINAL, H_CRESTA, I_1, I_2, I_3)

# Se calcula el caudal de salida para cada paso de tiempo
DF_TAC_01["O"] = DF_TAC_01["NIVEL"].apply(lambda x: LEY_VERTEDERO(x, B_ORIGINAL, H_CRESTA))

# Se calcula el área inundada para cada paso de tiempo en km2
DF_TAC_01["A"] = DF_TAC_01["NIVEL"].apply(lambda x: AREA(x)/1000**2)

# Se calcula el volumen almacenado para cada paso de tiempo en hm3
DF_TAC_01["VOL"] = DF_TAC_01["NIVEL"].apply(lambda x: VOL(x)/100**3)

# H_CRESTA constante en todo t
DF_TAC_01["NMN"] = H_CRESTA

TEXT_STATS = PreText(text='', styles={"font-size" : "100%", "font-family" : "Arial"})
def RETURS_STATS(I_MAX, O_MAX, T_I_MAX, T_O_MAX, H_INI, H_MAX, A_MAX, VOL_MAX):
    
    ATEN = (I_MAX - O_MAX) / I_MAX

    DESF = T_O_MAX - T_I_MAX

    DELTA_H = H_MAX - H_INI

    A = f"El caudal máximo de entrada y salida es igual a {I_MAX:.1f} m³/s y {O_MAX:.1f} m³/s respectivamente. La atenuación es igual al {(100*ATEN):.0f} %."
    
    B = f"El caudal máximo de entrada ocurre a las {T_I_MAX:.1f} horas y el de salida a las {T_O_MAX:.1f} hs y por lo tanto el desfasaje es de {DESF:.1f} horas."

    C = f"El nivel inicial del embalse es igual a {H_INI:.1f} m y el máximo alcanzado igual a {H_MAX:.1f} m y por lo tanto el nivel aumento en {DELTA_H:.1f} m."

    D = f"La máxima superficie inundada es de {A_MAX:.1f} km² y el máximo volumen almacenado igual a {VOL_MAX:.1f} hm³."
    
    TEXT_STATS.text =  A + "\n" + B + "\n" + C + "\n" + D

# Create CDSs

CDS_RK3 = ColumnDataSource(data={"T" : DF_TAC_01["T"],
                                 "I" : DF_TAC_01["I"],
                                 "NIVEL" : DF_TAC_01["NIVEL"],
                                 "O" : DF_TAC_01["O"],
                                 "A" : DF_TAC_01["A"],
                                 "VOL" : DF_TAC_01["VOL"],
                                 "NMN" : DF_TAC_01["NMN"],
                                })


# WIDGETS

# Sliders ancho del vertedero y nivel incial del embalse

slider_1 = Slider(start=30, end=50, value=H_CRESTA, step=1, title="NMN de la presa [m]") # Slider altura de la presa
slider_2 = Slider(start=H_CRESTA-10, end=H_CRESTA, value=H_CRESTA-1, step=0.5, title="Nivel inicial del embalse [m]") # Slider nivel inicial de embalse
slider_3 = Slider(start=20, end=100, value=B_ORIGINAL, step=1, title="Largo del vertedero [m]") # Slider ancho de vertedero

# Sliders definición del hidrograma de Entrada
slider_Q_DIS = Slider(start=1000, end=10000, value=Q_DIS, step=50, title="Caudal pico de entrada [m³/s]") # Slider caudal pico
slider_T_PICO = Slider(start=1, end=50, value=T_PICO, step=1, title="Tiempo al pico de entrada [hs]") # S lider Tiempo al pico
slider_FACTOR_FORMA = Slider(start=1, end=20, value=FACTOR_FORMA, step=1, title="Factor de forma") # Slider ancho de vertedero


def callback(attr, old, new):
    
    # Slider values
    
    # Nivel inicial y largo del vertedero
    H_CRESTA = slider_1.value

    NIVEL_INICIAL = slider_2.value
    slider_2.update(start=30, end=slider_1.value, value=NIVEL_INICIAL if NIVEL_INICIAL < H_CRESTA else H_CRESTA, step=0.5, title="Nivel inicial del embalse [m]") # Slider nivel inicial de embalse    
    NIVEL_INICIAL = slider_2.value
    
    B = slider_3.value 
    
    # Hidrograma de entrada
    Q_DIS = slider_Q_DIS.value
    T_PICO = slider_T_PICO.value
    FACTOR_FORMA = slider_FACTOR_FORMA.value
    
    # Update laminación
    
    DF_TAC_01["I"] = DF_TAC_01["T"].apply(lambda x: kozeny(Q_DIS, T_PICO, FACTOR_FORMA, x))


    DF_TAC_01.loc[0, "NIVEL"] = NIVEL_INICIAL

    # Aplicar RK3

    for i in range(1, len(DF_TAC_01)):
    
        N_0 = DF_TAC_01.loc[i-1, "NIVEL"]

        I_1 = DF_TAC_01.loc[i-1, "I"]

        I_2 = (2/3)*I_1 + (1/3)*DF_TAC_01.loc[i, "I"]

        I_3 = (1/3)*I_1 + (2/3)*DF_TAC_01.loc[i, "I"]

        DF_TAC_01.loc[i, "NIVEL"] = RK_3(N_0, B, H_CRESTA, I_1, I_2, I_3)
        
    # Calcular hidrograma de salida a partir de los niveles
    DF_TAC_01["O"] = DF_TAC_01["NIVEL"].apply(lambda x: LEY_VERTEDERO(x, B, H_CRESTA))
    
    # Se calcula el área inundada para cada paso de tiempo en km2
    DF_TAC_01["A"] = DF_TAC_01["NIVEL"].apply(lambda x: AREA(x)/1000**2)

    # Se calcula el volumen almacenado para cada paso de tiempo en hm3
    DF_TAC_01["VOL"] = DF_TAC_01["NIVEL"].apply(lambda x: VOL(x)/100**3)

    DF_TAC_01["NMN"] = H_CRESTA

    # Calcular el caudal máximo de entrada y salida
    TEXT_STATS = RETURS_STATS(DF_TAC_01["I"].max(), DF_TAC_01["O"].max(), float(DF_TAC_01["T"][DF_TAC_01["I"] == DF_TAC_01["I"].max()]), float(DF_TAC_01["T"][DF_TAC_01["O"] == DF_TAC_01["O"].max()]), DF_TAC_01["NIVEL"].iloc[0], DF_TAC_01["NIVEL"].max(), DF_TAC_01["A"].max(), DF_TAC_01["VOL"].max())
    
    UPDATE_CDS = {
                        "T" : DF_TAC_01["T"],
                         "I" : DF_TAC_01["I"],
                         "NIVEL" : DF_TAC_01["NIVEL"],
                         "O" : DF_TAC_01["O"],
                         "A" : DF_TAC_01["A"],
                         "VOL" : DF_TAC_01["VOL"],
                         "NMN" : DF_TAC_01["NMN"]
                     }

        
    CDS_RK3.data = UPDATE_CDS


slider_1.on_change('value', callback)
slider_2.on_change('value', callback)
slider_3.on_change('value', callback)

slider_Q_DIS.on_change('value', callback)
slider_T_PICO.on_change('value', callback)
slider_FACTOR_FORMA.on_change('value', callback)

# Hidrogramas de entrada y salida

p_1 = figure(width=900, height=250, x_axis_label="Tiempo [hs]", y_axis_label="Q [m³/s]", title="Hidrogramas de entrada y salida")

p_1.line(x="T", y="I", source=CDS_RK3, color="blue", legend_label="Hidrograma de entrada")
p_1.line(x="T", y="O", source=CDS_RK3, color="red", legend_label="Hidrograma de salida")

p_1.varea(x="T", y1=0, y2="I", source=CDS_RK3, alpha=0.15, fill_color="blue")
p_1.varea(x="T", y1=0, y2="O", source=CDS_RK3, alpha=0.15, fill_color="red")

# Nivel

p_2 = figure(width=900, height=250, x_axis_label="Tiempo [hs]", y_axis_label="Nivel [m]", title="Nivel del embalse")

p_2.line(x="T", y="NIVEL", source=CDS_RK3, color="green", legend_label="Nivel del embalse")

p_2.line(x="T", y="NMN", source=CDS_RK3, color="black", legend_label="NMN de la presa")

# Área vs H
p_3 = figure(width=500, height=250, x_axis_label='Nivel [m]', y_axis_label='Área [km²]', title='Área en función del nivel del embalse')
p_3.circle(DF_COTA_AREA['COTA_02'], DF_COTA_AREA['AREA_01'], color='blue', legend_label="Datos")
p_3.line(DF_COTA_AREA_2['COTA_02'], DF_COTA_AREA_2['AREA_02'], color='black', legend_label="A(h²)")

p_3.legend.location = "top_left"

# Volumne vs H
p_4 = figure(width=500, height=250, x_axis_label='Nivel [m]', y_axis_label='Volumen [hm³]', title='Volumen en función del nivel del embalse')
p_4.line(DF_COTA_VOL['COTA_02'], DF_COTA_VOL['VOL_02'], color='black', legend_label="V(h³)")

p_4.legend.location = "top_left"

for p in [p_1, p_2, p_3, p_4]:
    
    p.background_fill_color = "lightblue"
    p.background_fill_alpha = 0.20

widgets_1 = column(Div(text="<b>Características del hidrogama de entrada</b>"), slider_Q_DIS, slider_T_PICO, slider_FACTOR_FORMA)
widgets_2 = column(Div(text="<b>Características de la presa y el embalse</b>"), slider_1, slider_2, slider_3)
widgets_3 = column(Div(text="<b>Resultados generales de la laminación</b>"), TEXT_STATS)


text_title = "<b><i>Tránsito agregado de crecidas en un embalse a través de una apliación web interactiva</b></i>"

text_intro_1 = "<i>- Para el hidrograma de entrada, es posible variar el caudal pico, el tiempo al pico y la forma del hidrogama a través de un factor de forma.</i>"

text_intro_2 = "<i>- En cuanto a las características de la presa y el embalse, es posible variar la cota a la cual se encuentra en nivel máximo normal del embalse (NMN), el ancho del vertedero y el nivel inicial del embalse.</i>"

text_intro_3 = "<i>- Por consultas o comentarios enviar un correo a <a href='mailto: jmarcenaro@fi.uba.ar'>jmarcenaro@fi.uba.ar</a>, <a href='mailto: mdevoto@fi.uba.ar'>mdevoto@fi.uba.ar</a> o <a href='mailto: msuriano@fi.uba.ar'>msuriano@fi.uba.ar</a>."

layout = column(Div(text=text_title, styles={"font-size" : "150%", "margin": "auto"}),
                Div(text=text_intro_1),
                Div(text=text_intro_2),
                Div(text=text_intro_3),
                row(widgets_1, widgets_2, widgets_3), 
                row(column(p_1, p_2), column(p_3,p_4))
                )
                
curdoc().add_root(layout)

curdoc().title = "TAC Embalses"