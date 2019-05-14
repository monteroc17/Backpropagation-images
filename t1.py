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

# weights_capa_oculta = [
#     [0.1, -0.7],
#     [0.5, 0.3]
# ]

# weights_capa_salida = [[0.2], [0.4]]

weights_capa_oculta = [[random() for i in range(n_capa_entrada)] for i in range(n_capa_oculta)]
weights_capa_salida = [[random()] for i in range(n_capa_oculta) for i in range(n_capa_salida)]

def sigmoid(gamma):
    return 1.0 / (1.0 + exp(-gamma))

def calcular_salida_capa(resultados_for_prop,num_neuronas, pesos_capa):
    result = 0
    for i in range(num_neuronas):
        for k in range(num_neuronas):
            result += pesos_capa[i][k] * resultados_for_prop[i]
    return sigmoid(result)

def calc_error_c_oculta(result_salida, error_neuronal,mult_wei_nuevos_pesos):
    return result_salida * error_neuronal * mult_wei_nuevos_pesos

def ajustar_pesos_oculta(result_salida,result_individuales, entrada, resultado_esperado):
    error_estimado = 0
    error_real_obtenido_ = 0

    print("capa oculta:" + str(result_individuales))

    for i in range(len(weights_capa_oculta)):
        error_real_obtenido_ = result_salida*(1-result_salida)*(1-result_salida)
        error_estimado = calc_error_c_oculta(result_individuales[i], (resultado_esperado-result_individuales[i]),
                                                  (weights_capa_salida[i][0] * error_real_obtenido_))
        for j in range(len(weights_capa_oculta)):
            weights_capa_oculta[i][j] = weights_capa_oculta[i][j] + ALFA * entrada[j] * error_estimado

def ajustar_pesos_salida(result_salida, result_individuales, resultado_esperado, n_c_salida):
    error_real_obtenido = 0
    nuevos_pesos = 0

    print("capa salida: "+ str(result_salida))

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
    result_capa_oculta = []
    result_capa_salida = 0
    result_individual = 0
    for i in range(interacciones):
        #multiplicacion de las neuronas de capa oculta
        for e in range(len(entradas)):
            for n in range(n_c_oculta):
                for k in range(n_c_oculta):
                    result_individual += weights_capa_oculta[n][k] * entradas[e][k]
                result_capa_oculta.append(sigmoid(result_individual))
                result_individual = 0

            # ajustar pesos
            ajustar_pesos_oculta(calcular_salida_capa(result_capa_oculta, n_c_oculta, weights_capa_oculta), result_capa_oculta, entradas[e], result_esperados[e]) #capa oculta

            for l in range(n_c_oculta):
                result_individual += weights_capa_salida[l][0]*result_capa_oculta[l]
            result_capa_salida = sigmoid(result_individual)
            ajustar_pesos_salida(result_capa_salida, result_capa_oculta, result_esperados[e], n_c_salida)

            result_capa_oculta = [] #se reinicia el proceso
            result_capa_salida = 0

            print("Epoch #" + str(i+1))
            print(weights_capa_oculta)
            print(weights_capa_salida)
            print("-------------------------------------------------------------------------")

print(weights_capa_oculta)
print(weights_capa_salida)
forward_propagation(n_capa_oculta, n_capa_salida, [
    [0, 1],
    [1, 0],
    [1, 1],
    [0, 0]], [0, 0, 1, 0], 10)
