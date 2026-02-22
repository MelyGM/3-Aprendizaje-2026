import random
from collections import Counter
import arboles_numericos as an

def entrena_bosque(datos, target, clase_default,M=10,
                   tam_muestra=None, max_profundidad=None,
                   acc_nodo=1.0, min_ejemplos=0,
                   variables_por_nodo=None, seed=None):
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
    votos = [arbol.predice(instancia) for arbol in bosque]
    return Counter(votos).most_common(1)[0][0]

def predice_bosque_lista(bosque, datos):
    return [predice_bosque(bosque, d) for d in datos]


def evalua_bosque(bosque, datos, target):
    predicciones = predice_bosque_lista(bosque, datos)
    aciertos = 0
    for p, d in zip(predicciones, datos):
        if p == d[target]:
            aciertos += 1

    return aciertos / len(datos)
