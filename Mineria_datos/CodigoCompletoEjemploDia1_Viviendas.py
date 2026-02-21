# Cargo las librerias 
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

# Establecemos nuestro escritorio de trabajo
os.chdir('C:\DocumentaciónMineria_Python\Datos')

# Cargo las funciones que voy a utilizar
from FuncionesMineria import (analizar_variables_categoricas, cuentaDistintos, frec_variables_num, 
                           atipicosAmissing, patron_perdidos, ImputacionCuant, ImputacionCuali)

# Cargo los datos
datos = pd.read_excel('VentaViviendas.xlsx')

# Comprobamos el tipo de formato de las variables variable que se ha asignado en la lectura.
# No todas las categoricas estan como queremos
datos.dtypes

# Genera una lista con los nombres de las variables.
variables = list(datos.columns) 

# Indico las categóricas que aparecen como numéricas
numericasAcategoricas = ['Luxury', 'waterfront', 'view', 'basement']

# Las transformo en categóricas
for var in numericasAcategoricas:
    datos[var] = datos[var].astype(str)

# Seleccionar las columnas numéricas del DataFrame
numericas = datos.select_dtypes(include=['int', 'int32', 'int64','float', 'float32', 'float64']).columns

# Seleccionar las columnas categóricas del DataFrame
categoricas = [variable for variable in variables if variable not in numericas]
 
# Comprobamos que todas las variables tienen el formato que queremos  
datos.dtypes

# Frecuencias de los valores en las variables categóricas
analizar_variables_categoricas(datos)

# Cuenta el número de valores distintos de cada una de las variables numéricas de un DataFrame
cuentaDistintos(datos)

# Podríamos considerar 'bedrooms', 'bathrooms', 'floors' como categóricas
# ya que tienen pocos valores distintos. Además Bedrooms presenta una distribución extraña
# con valores muy elevados (inusual casas con 70 habitaciones..) y poco representados
# Se calcula la Tabla frecuencias para numericas para comprobar lo indicado

categoricas2 = ['bedrooms', 'bathrooms', 'floors']
frec_variables_num(datos, categoricas2)

# Descriptivos variables numéricas mediante función describe() de Python
descriptivos_num = datos.describe().T

# Añadimos más descriptivos a los anteriores
for num in numericas:
    descriptivos_num.loc[num, "Asimetria"] = datos[num].skew()
    descriptivos_num.loc[num, "Kurtosis"] = datos[num].kurtosis()    
    descriptivos_num.loc[num, "Rango"] = np.ptp(datos[num].dropna().values)


# Muestra valores perdidos
datos[variables].isna().sum()

# No hay ningún valor perdido en el conjunto de datos original

# Comenzamos la depuración de los datos

# A veces los 'nan' vienen como como una cadena de caracteres, los modificamos a perdidos.
for x in categoricas:
    datos[x] = datos[x].replace('nan', np.nan) 

# Missings no declarados variables cualitativas (NSNC, ?)
datos['condition'] = datos['condition'].replace('?', np.nan)

# Missings no declarados variables cuantitativas (-1, 99999)
datos['waterfront'] = datos['waterfront'].replace('-1', np.nan)

# Missings no declarados variables cuantitativas (-1, 99999)
datos['sqft_lot'] = datos['sqft_lot'].replace(-1, np.nan)

# Por si quisieramos hacerlo para cualquier valor menor igual que cero
datos['sqft_lot'] = datos['sqft_lot'].apply(lambda x: np.nan if x <= 0 else x)

#Variables cualitativas(consideradas numéricas) con categorías poco representadas.
datos['floors'] = datos['floors'].astype(str)
datos['floors'] = datos['floors'].replace({'2.5': '2+', '3.0': '2+', '3.5': '2+'})


datos['bedrooms'] = datos['bedrooms'].map(lambda x: 5 if x > 5 else x)
datos['bedrooms'] = datos['bedrooms'].astype(str)
datos['bedrooms'] = datos['bedrooms'].replace({'0': '0-1', '1': '0-1', '5': '5+'})

datos['bathrooms'] = datos['bathrooms'].map(lambda x: 3 if x > 3 else x)
datos['bathrooms'] = datos['bathrooms'].astype(str)
datos['bathrooms'] = datos['bathrooms'].replace({'3.0': '3+'})

#No hay errores de escritura en variables cualitativas (no diferencias entre mayúsculas y minúsculas..). 

#Variables cualitativas con categorías poco representadas
datos['condition'] = datos['condition'].replace({'D': 'C'})

# Modificación de la variable Yr_renovated, se crea una nueva (llamada prueba) 
datos['prueba'] = np.where(datos['yr_renovated'] != 0, 1, datos['yr_renovated'])
# Cambiar la variable prueba a Object (ahora la variables es numérica)
datos['prueba'] = datos['prueba'].astype(str)
# Se elimina variable yr_renovated
datos = datos.drop(columns = 'yr_renovated')

# Indico la variableObj, el ID y las Input (los atipicos y los missings se gestionan
# solo de las variables input)
varObjCont = datos['price']
varObjBin = datos['Luxury']

# Elimino las variables objetivo del conjunto de datos input, también elinimo las variables year y month por no aportar 
# directamente en este estudio información
datos_input = datos.drop(['year','month','price', 'Luxury'], axis = 1)

# Genera una lista con los nombres de las variables del cojunto de datos input.
variables_input = list(datos_input.columns)  

# Selecionamos las variables numéricas
numericas_input = datos_input.select_dtypes(include = ['int', 'int32', 'int64','float', 'float32', 'float64']).columns

# Selecionamos las variables categóricas
categoricas_input = [variable for variable in variables_input if variable not in numericas_input]

## ATIPICOS

# Cuento el porcentaje de atipicos de cada variable. 

# Seleccionar las columnas numéricas en el DataFrame
# Calcular la proporción de valores atípicos para cada columna numérica
# utilizando una función llamada 'atipicosAmissing'
# 'x' representa el nombre de cada columna numérica mientras se itera a través de 'numericas'
# 'atipicosAmissing(datos_input[x])' es una llamada a una función que devuelve una dupla
# donde el segundo elemento ([1]) es el número de valores atípicos
# 'len(datos_input)' es el número total de filas en el DataFrame de entrada
# La proporción de valores atípicos se calcula dividiendo la cantidad de valores atípicos por el número total de filas
resultados = {x: atipicosAmissing(datos_input[x])[1] / len(datos_input) for x in numericas_input}

# Modifico los atipicos como missings
for x in numericas_input:
    datos_input[x] = atipicosAmissing(datos_input[x])[0]



# MISSINGS
# Visualiza un mapa de calor que muestra la matriz de correlación de valores ausentes en el conjunto de datos.

patron_perdidos(datos_input)

# Muestra total de valores perdidos por cada variable
datos_input[variables_input].isna().sum()

# Muestra proporción de valores perdidos por cada variable (guardo la información)
prop_missingsVars = datos_input.isna().sum()/len(datos_input)

# Creamos la variable prop_missings que recoge el número de valores perdidos por cada observación
datos_input['prop_missings'] = datos_input.isna().mean(axis = 1)

# Realizamos un estudio descriptivo básico a la nueva variable
datos_input['prop_missings'].describe()

# Calculamos el número de valores distintos que tiene la nueva variable
len(datos_input['prop_missings'].unique())

# Elimino las observaciones con mas de la mitad de datos missings (no hay ninguna)
eliminar = datos_input['prop_missings'] > 0.5
datos_input = datos_input[~eliminar]
varObjBin = varObjBin[~eliminar]
varObjCont = varObjCont[~eliminar]


# Transformo la nueva variable en categórica (ya que tiene pocos valores diferentes)
datos_input["prop_missings"] = datos_input["prop_missings"].astype(str)

# Agrego 'prop_missings' a la lista de nombres de variables input
variables_input.append('prop_missings')
categoricas_input.append('prop_missings')


# Elimino las variables con mas de la mitad de datos missings (no hay ninguna)
eliminar = [prop_missingsVars.index[x] for x in range(len(prop_missingsVars)) if prop_missingsVars[x] > 0.5]
datos_input = datos_input.drop(eliminar, axis = 1)

# Recategorizo categoricas con "suficientes" observaciones missings
# Las que más tienen son 'waterfront' y 'condition'
# Se considera una categoria mas los missing.
datos_input['waterfront'] = datos_input['waterfront'].fillna('Desconocido')
datos_input['condition'] = datos_input['condition'].fillna('Desconocido')

## IMPUTACIONES
# Imputo todas las cuantitativas, seleccionar el tipo de imputacion: media, mediana o aleatorio
for x in numericas_input:
    datos_input[x] = ImputacionCuant(datos_input[x], 'mediana')

# Imputo todas las cualitativas, seleccionar el tipo de imputacion: moda o aleatorio
for x in categoricas_input:
    datos_input[x] = ImputacionCuali(datos_input[x], 'aleatorio')

# Reviso que no queden datos missings
datos_input.isna().sum()

# Una vez finalizado este proceso, se puede considerar que los datos estan depurados. Los guardamos
datos_input = pd.concat([varObjBin, varObjCont, datos_input], axis = 1)
with open('datosViviendaDep.pickle', 'wb') as archivo:
    pickle.dump(datos_input, archivo)


    
    
    