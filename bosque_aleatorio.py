import random
from collections import Counter
import arboles_numericos as an

def entrena_bosque(datos, target, clase_default,M=10,
                   tam_muestra=None, max_profundidad=None,
                   acc_nodo=1.0, min_ejemplos=0,
                   variables_por_nodo=None, seed=None):
"""
    Entrena un bosque aleatorio compuesto por M árboles.

    datos: list(dict) 
        Conjunto de datos de entrenamiento.
    target: str 
        Nombre del atributo a predecir.
    M: int 
        El número de árboles en el bosque.
    tam_muestra: int 
        Tamaño de cada subconjunto
    max_profundidad: int 
        Profundidad máxima permitida para cada árbol.
    acc_nodo: float 
        Precisión mínima para convertir un nodo en hoja.
    min_ejemplos: int 
        Número de ejmplo para continuar dividiendo.
    variables_por_nodo: int 
        Número de variables seleccionadas aleatoriamente en cada nodo del árbol.

    Regresa: Una lista de nodos raíz que foman el bosque.
    """
    if seed is not None:
        random.seed(seed)

    bosque = []
    for _ in range(M):
        tam = len(datos) if tam_muestra is None else tam_muestra
        subconjunto = random.choices(datos, k=tam)

        arbol = an.entrena_arbol(
            subconjunto,
            target,
            clase_default,
            max_profundidad=max_profundidad,
            acc_nodo=acc_nodo,
            min_ejemplos=min_ejemplos,
            variables_seleccionadas=variables_por_nodo
        )
        bosque.append(arbol)

    return bosque


def predice_bosque(bosque, instancia):
   """
    Predice la clase de una sola instancia utlizando votación mayoritaria entre los árboles.

    bosque: list
        lista de árboles entrenados
    instancia: dict
        Es la instancia a clasificar.
    
    Regresa: any
        clase predicha por el bosque
    """
    votos = [arbol.predice(instancia) for arbol in bosque]
    return Counter(votos).most_common(1)[0][0]

def predice_bosque_lista(bosque, datos):
  """
    Predice una lista de instancias.

    bosque: list
        lista de árboles entrenados
    datos: list
        Conjunto de instancias

    Regresa: list
        Lista de predicciones
    """
    return [predice_bosque(bosque, d) for d in datos]


def evalua_bosque(bosque, datos, target):
  """
    Calcula la exactitud del bosque sobre un conjunto de datos.

    bosque: list
        lista de árboles entrenados
    datos: list(dict)
        Conjunto de evaluación
    target: str
        Nombre del atributo real a comparar.

    Regresa: float
        Proporción de aciertos.
    """
    predicciones = predice_bosque_lista(bosque, datos)
    aciertos = 0
    for p, d in zip(predicciones, datos):
        if p == d[target]:
            aciertos += 1

    return aciertos / len(datos)

