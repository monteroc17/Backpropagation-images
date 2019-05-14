import numpy
from math import exp
from decimal import *
from random import seed
from random import random

c_oculta = 1
n_capa_entrada = 2
n_capa_oculta = 2
n_capa_salida = 1
ALFA = 0.25

weights_capa_oculta = [
    [0.1, -0.7],
    [0.5, 0.3]
]

weights_capa_salida = [[0.2], [0.4]]

def sigmoid(gamma):
    return 1.0 / (1.0 + exp(-gamma))

def calcular_capa_salida(resultados_for_prop, n_c_salida):
    result = 0
    for i in range(len(weights_capa_salida)):
        for k in range(n_c_salida):
            result += weights_capa_salida[i][k] * resultados_for_prop[i]
    return sigmoid(result)

def calc_error_c_oculta(result_salida, error_neuronal,mult_wei_nuevos_pesos):
    return result_salida * error_neuronal * mult_wei_nuevos_pesos

def ajustar_pesos_oculta(result_salida,result_individuales, entrada, resultado_esperado, n_c_oculta):
    error_real_obtenido = 0
    nuevos_pesos = 0

    for i in range(len(weights_capa_oculta)):
        nuevos_pesos = result_salida*(1-result_salida)*(1-result_salida)
        error_real_obtenido = calc_error_c_oculta(result_individuales[i], (1-result_individuales[i]),
                                                  (weights_capa_salida[i][0] * nuevos_pesos))
        for j in range(len(weights_capa_oculta)):
            weights_capa_oculta[i][j] = weights_capa_oculta[i][j] + ALFA * entrada[j] * error_real_obtenido


def ajustar_pesos_salida(result_salida, result_individuales, resultado_esperado, n_c_salida):
    error_real_obtenido = 0
    nuevos_pesos = 0

    #valor esperado menos posicion x result salida = ERROR REAL OBTENIDO
    error_real_obtenido = resultado_esperado - result_salida

    #nuevos pesos para la neurona
    #cambiar cada posicion del weights capa salida usando la siguiente formula:
    #posicion capa salida * ( result esperado menos posicion capa salida ) * ERROR REAL OBTENIDO
    nuevos_pesos = result_salida * error_real_obtenido * error_real_obtenido

    for j in range(len(weights_capa_salida)):
        for k in range(n_c_salida):
            weights_capa_salida[j][k] = weights_capa_salida[j][k] + ALFA * result_individuales[j] * nuevos_pesos

def forward_propagation(n_c_oculta, n_c_salida, entradas, result_esperados, interacciones):
    result_temp = []
    result_individual = 0
    for i in range(interacciones):
        #multiplicacion de las neuronas de capa oculta
        for e in range(len(entradas)):
            for n in range(n_c_oculta):
                for k in range(n_c_oculta):
                    result_individual += weights_capa_oculta[n][k] * entradas[e][k]
                result_temp.append(sigmoid(result_individual))
                result_individual = 0

            # ajustar pesos
            ajustar_pesos_oculta(calcular_capa_salida(result_temp, n_c_salida), result_temp, entradas[e], result_esperados[e],n_c_oculta) #capa oculta
            ajustar_pesos_salida(calcular_capa_salida(result_temp, n_c_salida), result_temp, result_esperados[e], n_c_salida)

            result_temp = [] #se reinicia el proceso

            print("Epoch #" + str(i))
            print(weights_capa_oculta)
            print(weights_capa_salida)


forward_propagation(n_capa_oculta, n_capa_salida, [
    [0, 1],
    [1, 0],
    [1, 1],
    [0, 0]], [1, 1, 0, 0], 100)

