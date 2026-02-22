import os
import random
import utileria as ut
import bosque_aleatorio as ba

__author__ = "Melina González Méndez"
__date__ = "Febrero 2026"

url = "http://archive.ics.uci.edu/static/public/109/wine.zip"
archivo_zip = "datos/wine.zip"
archivo_datos = "datos/wine.data"

# Wine: clase + 13 features 
atributos = [
    "class",
    "alcohol", "malic_acid", "ash", "alcalinity_of_ash", "magnesium",
    "total_phenols", "flavanoids", "nonflavanoid_phenols", "proanthocyanins",
    "color_intensity", "hue", "od280_od315", "proline"
]
target = "class"

if not os.path.exists("datos"):
    os.makedirs("datos")

if not os.path.exists(archivo_zip):
    ut.descarga_datos(url, archivo_zip)
    ut.descomprime_zip(archivo_zip)

# Leer archivo SIN perder la primera fila 
datos = []
with open(archivo_datos, "r") as f:
      for linea in f:
        linea = linea.strip()
        if not linea:
            continue
        partes = linea.split(",")
        if len(partes) != len(atributos):
            continue  
        d = {a: v for a, v in zip(atributos, partes)}
        datos.append(d)

for d in datos:
    d["class"] = int(d["class"])
    for a in atributos:
        if a != "class":
            d[a] = float(d[a])

# Split / validación
random.seed(42)
random.shuffle(datos)

N = int(0.8 * len(datos))
datos_ent = datos[:N]
datos_val = datos[N:]

conteo = {}
for d in datos_ent:
    c = d[target]
    conteo[c] = conteo.get(c, 0) + 1
clase_default = max(conteo, key=conteo.get)

# Imprimir tablas
def imprime_tabla(titulo, filas, encabezados):
    print("\n" + titulo)
    print(" | ".join(h.center(12) for h in encabezados))
    print("-" * (len(encabezados) * 15))
    for fila in filas:
        print(" | ".join(str(x).center(12) for x in fila))
    print("-" * (len(encabezados) * 15))

# >>> Pruebas <<<
# 1) Número de árboles (M)
filas_M = []
for M in [1, 5, 10, 25, 50, 100]:
    bosque = ba.entrena_bosque(
        datos_ent, target, clase_default,
        M=M,
        max_profundidad=10,
        variables_por_nodo=4,  
        seed=42
    )
    acc_ent = ba.evalua_bosque(bosque, datos_ent, target)
    acc_val = ba.evalua_bosque(bosque, datos_val, target)
    filas_M.append((M, f"{acc_ent:.3f}", f"{acc_val:.3f}"))

imprime_tabla(
    "1) Aumentar M (número de árboles)  Wine",
    filas_M,
    ["M", "Acc_train", "Acc_val"]
)

# 2) Profundidad máxima
filas_d = []
for prof in [1, 3, 5, 10, 15, 20, None]:
    bosque = ba.entrena_bosque(
        datos_ent, target, clase_default,
        M=50,
        max_profundidad=prof,
        variables_por_nodo=4,
        seed=42
    )
    acc_ent = ba.evalua_bosque(bosque, datos_ent, target)
    acc_val = ba.evalua_bosque(bosque, datos_val, target)
    filas_d.append((prof, f"{acc_ent:.3f}", f"{acc_val:.3f}"))

imprime_tabla(
    "B) Max_profundidad - Wine",
    filas_d,
    ["Prof", "Acc_train", "Acc_val"]
)

# 3) Variables_por_nodo (k)
filas_k = []
for k in [1, 2, 3, 4, 6, 8, 13]:
    bosque = ba.entrena_bosque(
        datos_ent, target, clase_default,
        M=50,
        max_profundidad=10,
        variables_por_nodo=k,
        seed=42
    )
    acc_ent = ba.evalua_bosque(bosque, datos_ent, target)
    acc_val = ba.evalua_bosque(bosque, datos_val, target)
    filas_k.append((k, f"{acc_ent:.3f}", f"{acc_val:.3f}"))

imprime_tabla(
    "3) Variables_por_nodo (k) - Wine",
    filas_k,
    ["k", "Acc_train", "Acc_val"]
)


#Conclusión:
#Al aumentar el número de árboles, la precisión en validación mejora y luego se mantiene estable porque la votación ayuda a 
# reducir errores. Con poca profundidad el modelo aprende poco, pero desde profundidad 3 ya funciona muy bien. 
# Si k es muy grande, los árboles se vuelven muy parecidos y el bosque pierde diversidad, 
# por eso un k intermedio suele dar mejores resultados.
