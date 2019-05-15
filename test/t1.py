import numpy
import math
from decimal import *
from random import seed
from random import random
import pickle

c_oculta = 1
n_capa_entrada = 2
n_capa_oculta = 2
n_capa_salida = 1
ALFA = 0.25

#weights_capa_oculta = [
#    [0.1, -0.7],
#    [0.5, 0.3]
#]

#weights_capa_salida = [[0.2], [0.4]]

weights_capa_oculta = [[random() for i in range(n_capa_entrada)]for i in range(n_capa_oculta)]
weights_capa_salida = [[random()] for i in range(n_capa_oculta)for i in range(n_capa_salida)]

def sigmoid(gamma):
    if gamma < 0:
        return math.exp(gamma) / (1 + math.exp(gamma))
    return 1 / (1 + math.exp(-gamma))


def calcular_salida_capa(resultados_for_prop, num_neuronas, pesos_capa):
    result = 0
    for i in range(num_neuronas):
        for k in range(num_neuronas):
            result += pesos_capa[i][k] * resultados_for_prop[i]
    return sigmoid(result)


def calc_error_c_oculta(result_salida, error_neuronal, mult_wei_nuevos_pesos):
    return result_salida * error_neuronal * mult_wei_nuevos_pesos

def calc_error_real_obtenido(result_esperado,result_c_salida):
    return result_c_salida * (1-result_c_salida) * (result_esperado-result_c_salida)

# ajuste pesos para la capa oculta
def ajustar_pesos_oculta(salidas_c_oculta,p_c_salida, result_esperado, p_c_oculta, entrada,result_c_salida):
    for n in range(len(weights_capa_oculta)): #recorrer cada neurona
        #calculo del delta para la neurona h
        delta = salidas_c_oculta[n]*(result_esperado-salidas_c_oculta[n])*(p_c_salida[n][0]*calc_error_real_obtenido(result_esperado,result_c_salida))
        for j in range(len(weights_capa_oculta)): #recorrer cada elemento peso de la capa oculta
            weights_capa_oculta[n][j] = weights_capa_oculta[n][j]+ALFA*entrada[j]*delta

# ajuste pesos para la capa oculta
def ajustar_pesos_salida(result_esperado, result_c_salida, pesos_c_salida, r_c_oculta):
    delta = calc_error_real_obtenido(result_esperado,result_c_salida)
    for p in range(len(pesos_c_salida)):
        weights_capa_salida[p][0] = weights_capa_salida[p][0] + ALFA * r_c_oculta[p]*delta

# para la parte de prueba
def forward_propagation(n_c_oculta, entradas):
    result_capa_oculta = []
    result_neu_capa_oculta = 0
    for n in range(n_c_oculta):
        for k in range(n_c_oculta):
            result_neu_capa_oculta += weights_capa_oculta[n][k] * entradas[k]
        result_capa_oculta.append(sigmoid(result_neu_capa_oculta))
        result_neu_capa_oculta = 0
        
    #se saca la neurona de la capa de salida
    for elem_c_salida in weights_capa_salida:
        result_neu_capa_oculta += elem_c_salida[0] *  result_capa_oculta[weights_capa_salida.index(elem_c_salida)]    
    result_capa_salida = sigmoid(result_neu_capa_oculta)
    print(result_capa_salida)

def calc_total_error(target, output):
    return 0.5 * math.pow((target - output),2)

def train(n_c_oculta, n_c_salida, entradas, result_esperados, interacciones):
    result_capa_oculta = [] #contiene las salidas de las neuronas de la capa oculta
    result_capa_salida = 0 #contiene la salida de la neurona de la capa oculta
    result_individual = 0 #contiene el valor temporal de la capa oculta
    total_error = 0 #error obtenido al sumar y operar todas las salidas de las capas de salida
    for i in range(interacciones):
        # multiplicacion de las neuronas de capa oculta
        for e in range(len(entradas)):
            #se hace el forward propagation
            #result_capa_oculta = forward_propagation(n_c_oculta, entradas[e])
            for n in range(n_c_oculta):
                for k in range(n_c_oculta):
                    result_individual += weights_capa_oculta[n][k] * entradas[e][k]
                result_capa_oculta.append(sigmoid(result_individual))
                result_individual = 0

            #se saca la neurona de la capa de salida
            for elem_c_salida in weights_capa_salida:
                result_individual += elem_c_salida[0] *  result_capa_oculta[weights_capa_salida.index(elem_c_salida)]
            
            result_capa_salida = sigmoid(result_individual)
            
            total_error += calc_total_error(result_esperados[e],result_capa_salida) # sumatoria total error
            
            # ajustar pesos
            #delta ocupa = result individuales, pesos capa salida, result esperado en cuestion, result error real obt
            #nuevos pesos = pesos capa oculta, alpha, entrada en cuestión y delta
            ajustar_pesos_oculta(
                result_capa_oculta, weights_capa_salida, result_esperados[e],
                weights_capa_oculta, entradas[e], result_capa_salida
            )
            ajustar_pesos_salida(result_esperados[e],result_capa_salida,weights_capa_salida, result_capa_oculta)
            
            print("Epoch #" + str(i+1))
            print("Error: "+str(total_error))
            print("RESULTADO CAPA SALIDA: "+str(result_capa_salida))

            result_capa_oculta = []  # se reinicia el proceso
            result_capa_salida = 0
            
            print("-------------------------------------------------------------------------")
        total_error = 0

    with open('pesos.txt', 'wb') as file:
        pickle.dump(weights_capa_oculta, file)
    

def predict(weights, row):
    result = []
    result = forward_propagation(len(weights), row)

def testing():
    train(n_capa_oculta, n_capa_salida, [
    [0, 1],
    [1, 0],
    [1, 1],
    [0, 0]], [1, 1, 1, 0], 5000)

    dataset = [
        [0, 1],
        [1, 0],
        [1, 1],
        [0, 0]]
    expected = [1, 1, 1, 0]

    print("-------------RESULTS--------------------")
    for row in range(len(dataset)):
        prediction = predict(weights_capa_oculta, dataset[row])
        #rint(prediction)
        #print('Expected=%d, Got=%d' % (expected[row], prediction))

testing()





