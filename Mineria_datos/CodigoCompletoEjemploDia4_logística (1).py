# Cargo las librerias 
import os
import pickle
from sklearn.model_selection import train_test_split
import itertools
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np

os.chdir('C:\DocumentaciónMineria_Python\Datos')

# Cargo las funciones que voy a utilizar
from FuncionesMineria import (pseudoR2, glm, glm_forward, glm_backward, glm_stepwise, validacion_cruzada_glm,
                           crear_data_modelo, summary_glm, graficoVcramer, impVariablesLog, pseudoR2, 
                           sensEspCorte, curva_roc)


# Cargo los datos depurados
with open('todo_bin.pickle', 'rb') as f:
    todo = pickle.load(f)

# Identifico la variable objetivo y la elimino del conjunto de datos
varObjBin = todo['Compra']
todo = todo.drop('Compra', axis = 1)

# Identifico las variables continuas
var_cont = ['Acidez', 'AcidoCitrico', 'Azucar', 'CloruroSodico', 'Densidad', 'pH', 
            'Sulfatos', 'Alcohol',  'PrecioBotella', 'xAcidez', 
            'xAcidoCitrico', 'xAzucar', 'xCloruroSodico', 'xDensidad', 
            'xpH', 'xSulfatos', 'xAlcohol', 'xPrecioBotella']

# Identifico las variables continuas sin transformar
var_cont_sin_transf = ['Acidez', 'AcidoCitrico', 'Azucar', 'CloruroSodico', 'Densidad', 'pH', 
                       'Sulfatos', 'Alcohol',  'PrecioBotella']

# Identifico las variables categóricas
var_categ = ['Etiqueta', 'CalifProductor', 'Clasificacion', 'Region', 'prop_missings']

# Hago la particion
x_train, x_test, y_train, y_test = train_test_split(todo, varObjBin, test_size = 0.2, random_state = 1234567)
# Indico que la variable respuesta es numérica (hay que introducirla en el algoritmo de phython tal y como la va a tratar)
y_train, y_test = y_train.astype(int), y_test.astype(int)


# Construyo el modelo ganador del dia 3
modeloManual = glm(y_train, x_train, [], ['Clasificacion', 'CalifProductor', 'Etiqueta'])
# Resumen del modelo
summary_glm(modeloManual['Modelo'], y_train, modeloManual['X'])

# Calculamos la medida de ajuste R^2 para train
pseudoR2(modeloManual['Modelo'], modeloManual['X'], y_train)

# Preparo datos test
x_test_modeloManual = crear_data_modelo(x_test, [], ['Clasificacion', 'CalifProductor', 'Etiqueta'])
# R-squared del modelo para test
pseudoR2(modeloManual['Modelo'], x_test_modeloManual, y_test)


# Seleccion de variables Stepwise, variables originales sin transformar y sin interacciones, métrica AIC
modeloStepAIC = glm_stepwise(y_train, x_train, var_cont_sin_transf, var_categ, [], 'AIC')
# Resumen del modelo
summary_glm(modeloStepAIC['Modelo'], y_train, modeloStepAIC['X'])
# Calculamos la medida de ajuste R^2 para train
pseudoR2(modeloStepAIC['Modelo'], modeloStepAIC['X'], y_train)
# Preparo datos test
x_test_modeloStepAIC = crear_data_modelo(x_test, modeloStepAIC['Variables']['cont'], 
                                                modeloStepAIC['Variables']['categ'], 
                                                modeloStepAIC['Variables']['inter'])
# R-squared del modelo para test
pseudoR2(modeloStepAIC['Modelo'], x_test_modeloStepAIC, y_test)


# Seleccion de variables Backward, variables originales sin transformar y sin interacciones, métrica AIC
modeloBackAIC = glm_backward(y_train, x_train, var_cont_sin_transf, var_categ, [], 'AIC')
# Resumen del modelo
summary_glm(modeloBackAIC['Modelo'], y_train, modeloBackAIC['X'])
# Calculamos la medida de ajuste R^2 para train
pseudoR2(modeloBackAIC['Modelo'], modeloBackAIC['X'], y_train)
# Preparo datos test
x_test_modeloBackAIC = crear_data_modelo(x_test, modeloBackAIC['Variables']['cont'], 
                                                modeloBackAIC['Variables']['categ'], 
                                                modeloBackAIC['Variables']['inter'])
# R-squared del modelo para test
pseudoR2(modeloBackAIC['Modelo'], x_test_modeloBackAIC, y_test)

# Comparo número de parámetros 
len(modeloStepAIC['Modelo'].coef_[0])
len(modeloBackAIC['Modelo'].coef_[0])


# Mismas variables seleccionadas, mismos parámetros, mismo modelo.
modeloStepAIC['Modelo'].coef_
modeloBackAIC['Modelo'].coef_


# Seleccion de variables Stepwise, métrica BIC
modeloStepBIC = glm_stepwise(y_train, x_train, var_cont_sin_transf, var_categ, [], 'BIC')
# Resumen del modelo
summary_glm(modeloStepBIC['Modelo'], y_train, modeloStepBIC['X'])
# Calculamos la medida de ajuste R^2 para train
pseudoR2(modeloStepBIC['Modelo'], modeloStepBIC['X'], y_train)
# Preparo datos test
x_test_modeloStepBIC = crear_data_modelo(x_test, modeloStepBIC['Variables']['cont'], 
                                                modeloStepBIC['Variables']['categ'], 
                                                modeloStepBIC['Variables']['inter'])
# R-squared del modelo para test
pseudoR2(modeloStepBIC['Modelo'], x_test_modeloStepBIC, y_test)

# Seleccion de variables Backward, métrica BIC
modeloBackBIC = glm_backward(y_train, x_train, var_cont_sin_transf, var_categ, [], 'BIC')
# Resumen del modelo
summary_glm(modeloBackBIC['Modelo'], y_train, modeloBackBIC['X'])
# Calculamos la medida de ajuste R^2 para train
pseudoR2(modeloBackBIC['Modelo'], modeloBackBIC['X'], y_train)
# Preparo datos test
x_test_modeloBackBIC = crear_data_modelo(x_test, modeloBackBIC['Variables']['cont'], 
                                                modeloBackBIC['Variables']['categ'], 
                                                modeloBackBIC['Variables']['inter'])
# R-squared del modelo para test
pseudoR2(modeloBackBIC['Modelo'], x_test_modeloBackBIC, y_test)


# Comparo número de parámetros 
len(modeloStepAIC['Modelo'].coef_[0])
len(modeloBackAIC['Modelo'].coef_[0])


# Mismas variables seleccionadas, mismos parámetros (en otro orden), mismo modelo.
modeloStepAIC['Modelo'].coef_
modeloBackAIC['Modelo'].coef_

# Los métodos Stepwise y Backward han resultado ser iguales tanto para AIC como para BIC


# Interacciones 2 a 2 de todas las variables continuas (sin transformar) con categoricas
interacciones = list(itertools.product(var_cont_sin_transf, var_categ))    
interacciones_unicas = []
for x in interacciones:
    if (sorted(x) not in [sorted(t) for t in interacciones_unicas]) and (x[0] != x[1]):
        interacciones_unicas.append(x)
  
# Seleccion de variables Stepwise, métrica AIC, con interacciones
modeloStepAIC_int = glm_stepwise(y_train, x_train, var_cont_sin_transf, var_categ, 
                                interacciones_unicas, 'AIC')
# Resumen del modelo
summary_glm(modeloStepAIC_int['Modelo'], y_train, modeloStepAIC_int['X'])
# Calculamos la medida de ajuste R^2 para train
pseudoR2(modeloStepAIC_int['Modelo'], modeloStepAIC_int['X'], y_train)
# Preparo datos test
x_test_modeloStepAIC_int = crear_data_modelo(x_test, modeloStepAIC_int['Variables']['cont'], 
                                                modeloStepAIC_int['Variables']['categ'], 
                                                modeloStepAIC_int['Variables']['inter'])
# R-squared del modelo para test
pseudoR2(modeloStepAIC_int['Modelo'], x_test_modeloStepAIC_int, y_test)


# Número de parámetros 
len(modeloStepAIC_int['Modelo'].coef_[0])

# Seleccion de variables Stepwise, métrica BIC, con interacciones
modeloStepBIC_int = glm_stepwise(y_train, x_train, var_cont_sin_transf, var_categ, 
                                interacciones_unicas, 'AIC')
# Resumen del modelo
summary_glm(modeloStepBIC_int['Modelo'], y_train, modeloStepBIC_int['X'])
# Calculamos la medida de ajuste R^2 para train
pseudoR2(modeloStepBIC_int['Modelo'], modeloStepBIC_int['X'], y_train)
# Preparo datos test
x_test_modeloStepBIC_int = crear_data_modelo(x_test, modeloStepBIC_int['Variables']['cont'], 
                                                modeloStepBIC_int['Variables']['categ'], 
                                                modeloStepBIC_int['Variables']['inter'])
# R-squared del modelo para test
pseudoR2(modeloStepBIC_int['Modelo'], x_test_modeloStepBIC_int, y_test)


# Número de parámetros 
len(modeloStepBIC_int['Modelo'].coef_[0])

# Mismo modelo modeloStepAIC_int = modeloStepBIC_int
modeloStepAIC_int['Modelo'].coef_
modeloStepBIC_int['Modelo'].coef_

# Seleccion de variables Stepwise, métrica AIC, variables originales y transformaciones, sin interacciones
modeloStepAIC_trans = glm_stepwise(y_train, x_train, var_cont, var_categ, [],'AIC')
# Resumen del modelo
summary_glm(modeloStepAIC_trans['Modelo'], y_train, modeloStepAIC_trans['X'])
# Calculamos la medida de ajuste R^2 para train
pseudoR2(modeloStepAIC_trans['Modelo'], modeloStepAIC_trans['X'], y_train)
# Preparo datos test
x_test_modeloStepAIC_trans = crear_data_modelo(x_test, modeloStepAIC_trans['Variables']['cont'], 
                                                modeloStepAIC_trans['Variables']['categ'], 
                                                modeloStepAIC_trans['Variables']['inter'])
# R-squared del modelo para test
pseudoR2(modeloStepAIC_trans['Modelo'], x_test_modeloStepAIC_trans, y_test)


# Número de parámetros 
len(modeloStepAIC_trans['Modelo'].coef_[0])


# Seleccion de variables Stepwise, métrica BIC, variables originales y transformaciones, sin interaciones
modeloStepBIC_trans = glm_stepwise(y_train, x_train, var_cont, var_categ, [],'BIC')
# Resumen del modelo
summary_glm(modeloStepBIC_trans['Modelo'], y_train, modeloStepBIC_trans['X'])
# Calculamos la medida de ajuste R^2 para train
pseudoR2(modeloStepBIC_trans['Modelo'], modeloStepBIC_trans['X'], y_train)
# Preparo datos test
x_test_modeloStepBIC_trans = crear_data_modelo(x_test, modeloStepBIC_trans['Variables']['cont'], 
                                                modeloStepBIC_trans['Variables']['categ'], 
                                                modeloStepBIC_trans['Variables']['inter'])
# R-squared del modelo para test
pseudoR2(modeloStepBIC_trans['Modelo'], x_test_modeloStepBIC_trans, y_test)


# Número de parámetros 
len(modeloStepBIC_trans['Modelo'].coef_[0])

# Mismo modelo modeloStepAIC_trans = modeloStepBIC_trans
modeloStepAIC_trans['Modelo'].coef_
modeloStepBIC_trans['Modelo'].coef_

# Seleccion de variables Stepwise, métrica AIC, variables originales, transformaciones e interacciones
modeloStepAIC_transInt = glm_stepwise(y_train, x_train, var_cont, var_categ, interacciones_unicas,'AIC')
# Resumen del modelo
summary_glm(modeloStepAIC_transInt['Modelo'], y_train, modeloStepAIC_transInt['X'])
# Calculamos la medida de ajuste R^2 para train
pseudoR2(modeloStepAIC_transInt['Modelo'], modeloStepAIC_transInt['X'], y_train)
# Preparo datos test
x_test_modeloStepAIC_transInt = crear_data_modelo(x_test, modeloStepAIC_transInt['Variables']['cont'], 
                                                modeloStepAIC_transInt['Variables']['categ'], 
                                                modeloStepAIC_transInt['Variables']['inter'])
# R-squared del modelo para test
pseudoR2(modeloStepAIC_transInt['Modelo'], x_test_modeloStepAIC_transInt, y_test)


# Número de parámetros 
len(modeloStepAIC_transInt['Modelo'].coef_[0])


# Seleccion de variables Stepwise, métrica BIC, variables originales y transformaciones
modeloStepBIC_transInt = glm_stepwise(y_train, x_train, var_cont, var_categ, interacciones_unicas,'BIC')
# Resumen del modelo
summary_glm(modeloStepBIC_transInt['Modelo'], y_train, modeloStepBIC_transInt['X'])
# Calculamos la medida de ajuste R^2 para train
pseudoR2(modeloStepBIC_transInt['Modelo'], modeloStepBIC_transInt['X'], y_train)
# Preparo datos test
x_test_modeloStepBIC_transInt = crear_data_modelo(x_test, modeloStepBIC_transInt['Variables']['cont'], 
                                                modeloStepBIC_transInt['Variables']['categ'], 
                                                modeloStepBIC_transInt['Variables']['inter'])
# R-squared del modelo para test
pseudoR2(modeloStepBIC_transInt['Modelo'], x_test_modeloStepBIC_transInt, y_test)


# Número de parámetros 
len(modeloStepBIC_transInt['Modelo'].coef_[0])

# Mismo modelo modeloStepAIC_transInt = modeloStepBIC_transInt
modeloStepAIC_transInt['Modelo'].coef_
modeloStepBIC_transInt['Modelo'].coef_


# Hago validacion cruzada repetida para ver que modelo es mejor
# Crea un DataFrame vacío para almacenar resultados
results = pd.DataFrame({
    'AUC': []
    , 'Resample': []
    , 'Modelo': []
})

# Realiza el siguiente proceso 20 veces (representado por el bucle `for rep in range(20)`)
for rep in range(20):
    # Realiza validación cruzada en cuatro modelos diferentes y almacena sus R-squared en listas separadas
    
    modelo_Manual = validacion_cruzada_glm(5,
                                          x_train,
                                          y_train,
                                          modeloManual['Variables']['cont'],
                                          modeloManual['Variables']['categ']
                                          )
    modelo_StepAIC = validacion_cruzada_glm(5,
                                          x_train,
                                          y_train,
                                          modeloStepAIC['Variables']['cont'],
                                          modeloStepAIC['Variables']['categ']
                                          )
    modelo_BackAIC = validacion_cruzada_glm(5,
                                           x_train,
                                           y_train,
                                           modeloBackAIC['Variables']['cont'],
                                           modeloBackAIC['Variables']['categ']
                                           )
    modelo_StepAIC_int = validacion_cruzada_glm(5,
                                           x_train,
                                           y_train,
                                           modeloStepAIC_int['Variables']['cont'],
                                           modeloStepAIC_int['Variables']['categ'],
                                           modeloStepAIC_int['Variables']['inter']
                                           )
    modelo_StepAIC_trans= validacion_cruzada_glm(5,
                                           x_train,
                                           y_train,
                                           modeloStepAIC_trans['Variables']['cont'],
                                           modeloStepAIC_trans['Variables']['categ'],
                                           modeloStepAIC_trans['Variables']['inter']
                                           )
    modelo_StepAIC_transInt = validacion_cruzada_glm(5,
                                           x_train,
                                           y_train,
                                           modeloStepAIC_transInt['Variables']['cont'],
                                           modeloStepAIC_transInt['Variables']['categ'],
                                           modeloStepAIC_transInt['Variables']['inter']
                                           )
    
    # Crea un DataFrame con los resultados de validación cruzada para esta repetición
    results_rep = pd.DataFrame({
        'AUC': modelo_Manual + modelo_StepAIC + modelo_BackAIC + modelo_StepAIC_int + modelo_StepAIC_trans + modelo_StepAIC_transInt
        ,'Resample': ['Rep' + str((rep + 1))]*5*6  # Etiqueta de repetición (5 repeticiones 6 modelos)
        , 'Modelo': [1]*5 + [2]*5 + [3]*5 + [4]*5 + [5]*5 + [6]*5 # Etiqueta de modelo (6 modelos 5 repeticiones)
    })
    results = pd.concat([results, results_rep], axis = 0)


    

# Boxplot de la validacion cruzada 
plt.figure(figsize=(10, 6))  # Crea una figura de tamaño 10x6
plt.grid(True)  # Activa la cuadrícula en el gráficoç
# Agrupa los valores de AUC por modelo
grupo_metrica = results.groupby('Modelo')['AUC']
# Organiza los valores de R-squared por grupo en una lista
boxplot_data = [grupo_metrica.get_group(grupo).tolist() for grupo in grupo_metrica.groups]
# Crea un boxplot con los datos organizados
plt.boxplot(boxplot_data, labels=grupo_metrica.groups.keys())  # Etiqueta los grupos en el boxplot
# Etiqueta los ejes del gráfico
plt.xlabel('Modelo')  # Etiqueta del eje x
plt.ylabel('AUC')  # Etiqueta del eje y
plt.show()  # Muestra el gráfico  

 
# Calcular la media del AUC por modelo
media_AUC = results.groupby('Modelo')['AUC'].mean()
# Calcular la desviación estándar del AUC por modelo
std_AUC = results.groupby('Modelo')['AUC'].std()    
# Contar el número de parámetros en cada modelo
num_params = [len(modeloManual['Modelo'].coef_[0]), len(modeloStepAIC['Modelo'].coef_[0]), len(modeloBackAIC['Modelo'].coef_[0]), len(modeloStepAIC_int['Modelo'].coef_[0]), 
 len(modeloStepAIC_trans['Modelo'].coef_[0]), len(modeloStepAIC_transInt['Modelo'].coef_[0])]



# Todos los modelos son parecidos en cuanto a AUC y su desviación estandar
# descartamos el modeloStepAIC_int y modeloStepAIC_transInt por su elevado número de parámetros
# modeloManual y modeloBackAIC son el mismo y son el que elijo por su reducido número de parámetros
modeloManual['Modelo'].coef_[0]
modeloBackAIC['Modelo'].coef_[0]  

## Seleccion aleatoria (se coge la submuestra de los datos de entrenamiento)

# Inicializar un diccionario para almacenar las fórmulas y variables seleccionadas.
variables_seleccionadas = {
    'Formula': [],
    'Variables': [],
    'Interaccion': []
}

# Realizar 20 iteraciones de selección aleatoria. (en clase no se puede correr con 20)
for x in range(20):
    print('---------------------------- iter: ' + str(x))
    
    # Dividir los datos de entrenamiento en conjuntos de entrenamiento y prueba.
    x_train2, x_test2, y_train2, y_test2 = train_test_split(x_train, y_train, 
                                                            test_size = 0.3, random_state = 1234567 + x)
    
    # Realizar la selección stepwise utilizando el criterio BIC en la submuestra.
    modelo = glm_stepwise(y_train2.astype(int), x_train2, var_cont, var_categ, interacciones_unicas, 'BIC')
    
    # Almacenar las variables seleccionadas y la fórmula correspondiente.
    variables_seleccionadas['Variables'].append(modelo['Variables'])
    variables_seleccionadas['Formula'].append(sorted(modelo['X'].columns))
# Unir las variables en las fórmulas seleccionadas en una sola cadena.
variables_seleccionadas['Formula'] = list(map(lambda x: '+'.join(x), variables_seleccionadas['Formula']))
    
# Calcular la frecuencia de cada fórmula y ordenarlas por frecuencia.
frecuencias = Counter(variables_seleccionadas['Formula'])
frec_ordenada = pd.DataFrame(list(frecuencias.items()), columns = ['Formula', 'Frecuencia'])
frec_ordenada = frec_ordenada.sort_values('Frecuencia', ascending = False).reset_index()

# Identificar las tres fórmulas más frecuentes y las variables correspondientes.
var_1 = variables_seleccionadas['Variables'][variables_seleccionadas['Formula'].index(
    frec_ordenada['Formula'][0])]
var_2 = variables_seleccionadas['Variables'][variables_seleccionadas['Formula'].index(
    frec_ordenada['Formula'][1])]
var_3 = variables_seleccionadas['Variables'][variables_seleccionadas['Formula'].index(
    frec_ordenada['Formula'][2])]



#######################################



# De las 20 repeticiones, las 3 que mas se repiten son (no tienen porqué coincidir con los aquí mostrados):
#   1)  'Etiqueta', 'xpH', 'xAzucar', 'prop_missings', ('pH', 'CalifProductor'), ('PrecioBotella', 'Clasificacion'),
#        ('AcidoCitrico', 'prop_missings'), ('Acidez', 'Etiqueta'), ('pH', 'prop_missings'), ('Sulfatos', 'Etiqueta'),
#        ('Azucar', 'prop_missings'), ('Azucar', 'Region'), ('Azucar', 'Clasificacion')
#
#   2)  'Clasificacion', 'Etiqueta', 'xpH', ('pH', 'CalifProductor'), ('PrecioBotella', 'Clasificacion'), 
#       ('pH', 'prop_missings'), ('pH', 'Region')
#
#   3)  'Etiqueta', 'xAzucar', ('Alcohol', 'Clasificacion'), ('Azucar', 'CalifProductor'), ('AcidoCitrico', 'Etiqueta'),
#        ('Azucar', 'prop_missings'), ('PrecioBotella', 'prop_missings'), ('Acidez', 'Region'),
#        ('Alcohol', 'prop_missings'), ('Azucar', 'Clasificacion')


## Comparacion final, tomo el ganador de antes y los nuevos candidatos
results = pd.DataFrame({
    'Rsquared': []
    , 'Resample': []
    , 'Modelo': []
})
for rep in range(20):
    modelo1 = validacion_cruzada_glm(
        5
        , x_train
        , y_train
        , modeloManual['Variables']['cont']
        , modeloManual['Variables']['categ']
        , modeloManual['Variables']['inter']
    )
    modelo2 = validacion_cruzada_glm(
        5
        , x_train
        , y_train
        , var_1['cont']
        , var_1['categ']
        , var_1['inter']
    )
    modelo3 = validacion_cruzada_glm(
        5
        , x_train
        , y_train
        , var_2['cont']
        , var_2['categ']
        , var_2['inter']
    )
    modelo4 = validacion_cruzada_glm(
        5
        , x_train
        , y_train
        , var_3['cont']
        , var_3['categ']
        , var_3['inter']
    )
    results_rep = pd.DataFrame({
        'AUC': modelo1 + modelo2 + modelo3 + modelo4
        , 'Resample': ['Rep' + str((rep + 1))]*5*4
        , 'Modelo': [1]*5 + [2]*5 + [3]*5 + [4]*5
    })
    results = pd.concat([results, results_rep], axis = 0)
     

# Boxplot de la validacion cruzada 
plt.figure(figsize=(10, 6))  # Crea una figura de tamaño 10x6
plt.grid(True)  # Activa la cuadrícula en el gráficoç
# Agrupa los valores de Rsquared por modelo
grupo_metrica = results.groupby('Modelo')['AUC']
# Organiza los valores de R-squared por grupo en una lista
boxplot_data = [grupo_metrica.get_group(grupo).tolist() for grupo in grupo_metrica.groups]
# Crea un boxplot con los datos organizados
plt.boxplot(boxplot_data, labels=grupo_metrica.groups.keys())  # Etiqueta los grupos en el boxplot
# Etiqueta los ejes del gráfico
plt.xlabel('Modelo')  # Etiqueta del eje x
plt.ylabel('Rsquared')  # Etiqueta del eje y
plt.show()  # Muestra el gráfico  

# Calcular la media de las métricas R-squared por modelo
media_AUC2= results.groupby('Modelo')['AUC'].mean()
# Calcular la desviación estándar de las métricas R-squared por modelo
std_AUC2 = results.groupby('Modelo')['AUC'].std()
# Contar el número de parámetros en cada modelo
num_params2 = [len(modeloManual['Modelo'].coef_[0]), 
                 len(frec_ordenada['Formula'][0].split('+')),
                 len(frec_ordenada['Formula'][1].split('+')), 
                 len(frec_ordenada['Formula'][2].split('+'))]


  


# Una vez decidido el mejor modelo, buscamos el mejor punto de corte 
ModeloGanador = modeloManual

## Buscamos el mejor punto de corte
# Probamos dos
sensEspCorte(ModeloGanador['Modelo'], x_test, y_test, 0.4, [], ['Clasificacion', 'CalifProductor', 'Etiqueta'])
sensEspCorte(ModeloGanador['Modelo'], x_test, y_test, 0.6, [], ['Clasificacion', 'CalifProductor', 'Etiqueta'])

# Generamos una rejilla de puntos de corte
posiblesCortes = np.arange(0, 1.01, 0.01).tolist()  # Generamos puntos de corte de 0 a 1 con intervalo de 0.01
rejilla = pd.DataFrame({
    'PtoCorte': [],
    'Accuracy': [],
    'Sensitivity': [],
    'Specificity': [],
    'PosPredValue': [],
    'NegPredValue': []
})  # Creamos un DataFrame para almacenar las métricas para cada punto de corte

for pto_corte in posiblesCortes:  # Iteramos sobre los puntos de corte
    rejilla = pd.concat(
        [rejilla, sensEspCorte(ModeloGanador['Modelo'], x_test, y_test, pto_corte, [], ['Clasificacion', 'CalifProductor', 'Etiqueta'])],
        axis=0
    )  # Calculamos las métricas para el punto de corte actual y lo agregamos al DataFrame

rejilla['Youden'] = rejilla['Sensitivity'] + rejilla['Specificity'] - 1  # Calculamos el índice de Youden
rejilla.index = list(range(len(rejilla)))  # Reindexamos el DataFrame para que los índices sean consecutivos



plt.plot(rejilla['PtoCorte'], rejilla['Youden'])
plt.xlabel('Posibles Cortes')
plt.ylabel('Youden')
plt.title('Youden')
plt.show()

plt.plot(rejilla['PtoCorte'], rejilla['Accuracy'])
plt.xlabel('Posibles Cortes')
plt.ylabel('Accuracy')
plt.title('Accuracy')
plt.show()

rejilla['PtoCorte'][rejilla['Youden'].idxmax()]
rejilla['PtoCorte'][rejilla['Accuracy'].idxmax()]

# El resultado es 0.75 para youden y 0.5 para Accuracy
# Los comparamos
sensEspCorte(ModeloGanador['Modelo'], x_test, y_test, 0.75, [], ['Clasificacion', 'CalifProductor', 'Etiqueta'])
sensEspCorte(ModeloGanador['Modelo'], x_test, y_test, 0.5, [], ['Clasificacion', 'CalifProductor', 'Etiqueta'])


# Vemos los coeficientes del modelo ganador
summary_glm(ModeloGanador['Modelo'], y_train, ModeloGanador['X'])

# Evaluamos la estabilidad del modelo a partir de las diferencias en train y test
pseudoR2(ModeloGanador['Modelo'], ModeloGanador['X'], y_train)


x_test_ModeloGanador = crear_data_modelo(x_test, ModeloGanador['Variables']['cont'],
                                         ModeloGanador['Variables']['categ'], ModeloGanador['Variables']['inter'])

pseudoR2(ModeloGanador['Modelo'], x_test_ModeloGanador, y_test)
    


    
