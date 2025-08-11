import pandas as pd
import numpy as np
from minisom import MiniSom
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import os


pd.set_option('display.max_columns', None)  # Muestra todas las columnas
pd.set_option('display.width', None)        # Evita el truncamiento del ancho
pd.set_option('display.max_rows', None)     # Muestra todas las filas

#leer el archivo excel

#df = pd.read_excel(r'C:\Users\manul\OneDrive\Escritorio\tsi\datatecito (2).xlsx')
import pandas as pd
import os

# Obtener la ruta absoluta donde se encuentra el script
ruta_base = os.path.dirname(os.path.abspath(__file__))

# Construir la ruta relativa hacia el archivo Excel
ruta_excel = os.path.join(ruta_base, 'datos', 'datatecito (2).xlsx')

# Cargar el archivo Excel
df = pd.read_excel(ruta_excel, engine='openpyxl')

print("Datos cargados correctamente:")

#df.shape
#eliminar la columna DOI
df = df.drop('DOI', axis=1)
#convertir las columnas a tipo category
df['TIPO SINTES.'] = df['TIPO SINTES.'].astype('category')
df['LIGANDO'] = df['LIGANDO'].astype('category')
df['SOLVENTE - MEDIO'] = df['SOLVENTE - MEDIO'].astype('category')
df['TIPO'] = df['TIPO'].astype('category')
df['Ln'] = df['Ln'].astype('category')
df['LPOM'] = df['LPOM'].astype('category')
df['SOLVENTE - MEDIO'] = df['SOLVENTE - MEDIO'].astype('category')
df['PRECURSOR'] = df['PRECURSOR'].astype('category')
df['SAL  Ln'] = df['SAL  Ln'].astype('category')

def acortar_nombre(nombre_completo):
    partes_nombre = nombre_completo.split()  # Dividir el nombre en palabras
    iniciales = [parte[0] for parte in partes_nombre]  # Tomar la primera letra de cada palabra
    return ''.join(iniciales)  # Unir las iniciales en una cadena
# Aplicar la función a la columna 'Nombre'
df['TIPO'] = df['TIPO'].apply(acortar_nombre)

df_filtrado = df[(df['TIPO'] != 'PI')]
#print(df['TIPO'])
# Eliminar filas que contienen valores faltantes en las demás columnas
df = df.dropna()

# Define una función para verificar si una celda contiene 'x', 'X' o 0
def contiene_x_o_0(celda):
    if isinstance(celda, str):
        return 'x' in celda or 'X' in celda
    else:
        return celda == 0

# Aplica la función a cada celda del DataFrame para marcar las filas que contienen 'x', 'X' o 0
filas_con_x_o_0 = df.applymap(contiene_x_o_0).any(axis=1)

# Filtra el DataFrame para eliminar las filas que contienen 'x', 'X' o 0
df = df[~filas_con_x_o_0]

# Lista de columnas con valores faltantes
columnas_con_faltantes = ['TIEMPO', 'pH', 't cristalización días', 'FTIR (W=O) term']

# Eliminar filas con valores faltantes solo en las columnas específicas
df= df.dropna(subset=columnas_con_faltantes)

df['CÓDIGO'] = pd.factorize(df['CÓDIGO'])[0]
# Asignar códigos numéricos a las categorías
df['TIPO'] = df['TIPO'].cat.codes + 1  # Sumar 1 para comenzar desde 1 en lugar de 0
# Suponiendo que df es tu DataFrame original
df = df.sort_values(by='CÓDIGO')
df['FÓRMULA_ID'] = pd.factorize(df['FÓRMULA'])[0] + 1

# Suponiendo que df es tu DataFrame original
df = df.sort_values(by='CÓDIGO')

# Filtrar las filas en la columna 'Ln' que contienen 'Ce3+', 'Gd3+', y 'La3+'
filtro_ln = df['Ln'].isin(['Ce3+', 'Gd3+', 'La3+'])
df_filtrado = df[filtro_ln]

# Suponiendo que df es tu DataFrame original
df = df.sort_values(by='CÓDIGO')
df['Ln_ID'] = pd.factorize(df['Ln'])[0] + 1

# Suponiendo que df es tu DataFrame original
df = df.sort_values(by='CÓDIGO')
df['Lpom_ID'] = pd.factorize(df['LPOM'])[0] + 1

# Suponiendo que df es tu DataFrame original
df = df.sort_values(by='CÓDIGO')
df['Lig_ID'] = pd.factorize(df['LIGANDO'])[0] + 1

# Suponiendo que df es tu DataFrame original
df = df.sort_values(by='CÓDIGO')
df['TS_ID'] = pd.factorize(df['TIPO SINTES.'])[0] + 1

# Suponiendo que df es tu DataFrame original
df = df.sort_values(by='CÓDIGO')
df['SM_ID'] = pd.factorize(df['SOLVENTE - MEDIO'])[0] + 1

# Cambio de variables categorica a numerica
df = df.sort_values(by='CÓDIGO')
df['PRE_ID'] = pd.factorize(df['PRECURSOR'])[0] + 1

# Suponiendo que df es tu DataFrame original
df = df.sort_values(by='CÓDIGO')
df['SAL_ID'] = pd.factorize(df['SAL  Ln'])[0] + 1

df = df.sort_values(by='CÓDIGO')
df1 = df[['CÓDIGO','FÓRMULA','FÓRMULA_ID','Ln','Ln_ID', 'LPOM','Lpom_ID','LIGANDO', 'Lig_ID', 'TIPO SINTES.', 'TS_ID','SOLVENTE - MEDIO', 'SM_ID', 'PRECURSOR','PRE_ID','SAL  Ln', 'SAL_ID']]

df['FÓRMULA'] = df['FÓRMULA_ID']
df['Ln'] = df['Ln_ID']
df['LPOM'] = df['Lpom_ID']
df['LIGANDO'] = df['Lig_ID']
df['TIPO SINTES.'] = df['TS_ID']
df['SOLVENTE - MEDIO'] = df['SM_ID']
df['PRECURSOR'] = df['PRE_ID']
df['SAL  Ln'] = df['SAL_ID']

# Eliminacion de columnas id
columnas_a_eliminar = ['FÓRMULA_ID','Ln_ID', 'Lpom_ID', 'Lig_ID', 'TS_ID', 'SM_ID', 'PRE_ID', 'SAL_ID']
df = df.drop(columnas_a_eliminar, axis=1)

df = df.drop(['CÓDIGO'], axis=1)

#Comienza XGBoost

# Convert the 'TIPO' column back to a categorical type before using .cat.codes
df['TIPO'] = df['TIPO'].astype('category')
df['TIPO'] = df['TIPO'].cat.codes

X = df.drop(columns=['TIPO'], axis=1)
y = df['TIPO']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

parametros = {
    #'objective':'multi:softmax',
    'eval_metric': 'logloss',        # Métrica para evaluar el rendimiento (log-loss para clasificación)
    #'max_depth': 4,                  # Árboles poco profundos para evitar sobreajuste
    #'learning_rate': 0.1,            # Tasa de aprendizaje moderada
    #'n_estimators': 50,              # Número de árboles más bajo, ideal para datasets pequeños
    #'subsample': 0.8,                # Utiliza el 80% de los datos en cada iteración
    #'colsample_bytree': 0.8,         # Utiliza el 80% de las características en cada árbol
    #'gamma': 1,                      # Regularización para evitar divisiones insignificantes
    #'lambda': 1,                     # Regularización L2 (predeterminada)
    #'alpha': 0,                      # Regularización L1 (sin sparsidad por defecto)
    #'seed': 42                       # Semilla para reproducibilidad
}

model = xgb.XGBClassifier(**parametros)
model.fit(X_train, y_train)

def esc_data():
    try:
        Formula= float(input("Ingresa valor Formula:"))
        Ln= float(input("Ingresa valor Ln:"))
        LPOM= float(input("Ingresa valor LPOM:"))
        LIGANDO= float(input("Ingresa valor LIGANDO:"))
        TIPOSINTES= float(input("Ingresa valor TIPO Sintensis:"))
        TCelcius= float(input("Ingresa valor TCelcius:"))
        TIEMPO= float(input("Ingresa valor Tiempo:"))
        SOLVENTE_MEDIO= float(input("Ingresa valor Solvente medio:"))
        pH= float(input("Ingresa valor Ph:"))
        PRECURSOR= float(input("Ingresa valor Precursor:"))
        SALLn= float(input("Ingresa valor Sal ln:"))
        tcristalizacióndias= float(input("Ingresa valor tcristalizacion:"))
        FTIRterm= float(input("Ingresa valor FTIR:"))
        parametroD= float(input("Ingresa valor parametro D:"))
        AnguloLn_O_W= float(input("Ingresa valor AnguloLNOW:")) # Changed AnguloLn-O-W to AnguloLn_O_W as - is not a valid character in variable names
        #Corrected the indentation of the return statement to align with the try block
        return np.array([[Formula,Ln,LPOM,LIGANDO,TIPOSINTES,TCelcius,TIEMPO,SOLVENTE_MEDIO,pH,PRECURSOR,SALLn,tcristalizacióndias,FTIRterm,parametroD,AnguloLn_O_W]])  # Devuelve un array con los datos del usuario
    except ValueError:
        print("Por favor, introduce un valor numérico válido.")
        return esc_data()  # Vuelve a pedir los datos si hay error

# 7. Hacer predicciones
#prec = esc_data()
#y_pred = model.predict(prec)

#predecir valores para cada clase

def predecir2():
  import numpy as np
  import pandas as pd
  import xgboost as xgb

  # Generamos nuevas combinaciones aleatorias de variables independientes
  n_random = 10000  # Número de combinaciones a probar

  rangos = [(1,62),(1,14),(1,4),(1,9),(1,1),(50,90),(0.5,4),(1,4),(2,6),(1,8),(1,21),(3,40),(900,960),(0,2),(100,160)]

  # Use the correct feature names from your training data
  feature_names = ['FÓRMULA', 'Ln', 'LPOM', 'LIGANDO', 'TIPO SINTES.', 'T° Celcius', 'TIEMPO',
                  'SOLVENTE - MEDIO', 'pH', 'PRECURSOR', 'SAL  Ln', 't cristalización días',
                  'FTIR (W=O) term', ' parametro D', 'Angulo Ln-O-W']

  X_random = pd.DataFrame({
    feature_names[i]: np.random.randint(low=r[0], high=r[1]+1, size=n_random)
    for i, r in enumerate(rangos)
})
  #np.random.randint(low=r[0], high=r[1]+1
  #np.random.uniform(low=r[0], high=r[1]


  # **Removed DMatrix creation here**
  #X_random_dmatrix = xgb.DMatrix(X_random)

  # Realizar predicciones con el modelo entrenado
  predictions = model.predict(X_random) # **Pass the raw data (X_random) to predict**

  # 5. Filtrar combinaciones donde la predicción sea igual a la clase deseada
  escribetop = input("Escriba el tipo que desea predecir: ")  # Valor específico de la clase (por ejemplo, clase 2)
  desired_class = escribetop.lower()

  if desired_class == "dinuclear hibrido":
        vtipo = 0
  elif desired_class == "dinuclear inorganico":
        vtipo = 1
  elif desired_class == "mononuclear hibrido":
        vtipo = 2
  elif desired_class == "mononuclear inorganico":
        vtipo = 3

  X_desired = X_random[predictions == vtipo]
  df_desired = X_desired.head(10)
  # Mostrar los resultados
  #print(f"\nCombinaciones de variables independientes que predicen la clase '{desired_class}' '{vtipo}':")
  #return print(df_desired.head(10))
  df_desired.to_excel('Datos Predecidos.xlsx', index=False, engine='openpyxl')
  print("\nCombinaciones de variables independientes que predicen la clase '{}' '{}':"
        .format(desired_class, vtipo))
  return df_desired

def cmid2():
    """
    This function calls the `esc_data()` function (which is assumed to be defined elsewhere)
    to get data to make predictions.
    The data is converted into an XGBoost DMatrix before passing it to the model.predict() method.

    The function is edited to ensure the prediction data has the same column names
    as the training data.

    **Fix:** Converts the output of `esc_data()` to a Pandas DataFrame
             if it's a NumPy array, allowing the use of `.columns`.

    **Further fix:** Removes the extra DMatrix creation inside cmid2()
                    to avoid the error.
    """
    prec = esc_data() #Get the data

    # 1. Ensure 'prec' has the correct columns
    feature_names = ['FÓRMULA', 'Ln', 'LPOM', 'LIGANDO', 'TIPO SINTES.', 'T° Celcius', 'TIEMPO',
                'SOLVENTE - MEDIO', 'pH', 'PRECURSOR', 'SAL  Ln', 't cristalización días',
                'FTIR (W=O) term', ' parametro D', 'Angulo Ln-O-W']

    # **Convert prec to a DataFrame if it's a NumPy array**
    if isinstance(prec, np.ndarray):
        prec = pd.DataFrame(prec, columns=feature_names) # Assume columns order matches esc_data() output

    #Check if prec has the correct columns, if not, create them and fill with 0 if needed
    missing_cols = [col for col in feature_names if col not in prec.columns]
    for col in missing_cols:
        prec[col] = 0  # or any other default value

    prec = prec[feature_names] #Order the columns to match training data


    # **Removed DMatrix creation here - dprec = xgb.DMatrix(prec)**
    y_pred = model.predict(prec) # **Pass the raw data (prec) to predict**

    if y_pred == 0:
      print("El tipo elegido es Dinuclear_Hibrido 0")
    elif y_pred ==1:
      print("El tipo elegido es Dinuclear_Inorganico 1")
    elif y_pred == 2:
      print("El tipo elegido es Mono_Hibrido 2")
    elif y_pred == 3:
      print("El tipo elegido es Mono_Ingorganico 3")

def ident():

  # Opciones disponibles
  opciones = ['FÓRMULA_ID','Ln_ID', 'Lpom_ID', 'Lig_ID', 'TS_ID', 'SM_ID', 'PRE_ID', 'SAL_ID']

  # Mostrar opciones
  print("Selecciona una opción:")
  for i, opcion in enumerate(opciones, start=1):
      print(f"{i}. {opcion}")

  # Capturar la selección del usuario
  seleccion = input("Ingresa los números de tus opciones: ")

  # Convertir la selección en una lista de índices
  seleccion_indices = [int(x.strip()) - 1 for x in seleccion.split(',') if x.strip().isdigit()]
  idselc = int(input("Ingresa el id: "))
  columna = None
  resultado2 = None

  if seleccion_indices ==[0]:
    resultado = df1.loc[df1['FÓRMULA_ID'] == idselc, ['CÓDIGO']]
    codigo_value = resultado['CÓDIGO'].values[0]
    resultado2 = str(df1.loc[df1['CÓDIGO'] == codigo_value, ['FÓRMULA']])

  elif seleccion_indices ==[1]:
    resultado = df1.loc[df1['Ln_ID'] == idselc, ['CÓDIGO']]
    codigo_value = resultado['CÓDIGO'].values[0]
    resultado2 = str(df1.loc[df1['CÓDIGO'] == codigo_value, ['Ln']])

  elif seleccion_indices ==[2]:
    resultado = df1.loc[df1['Lpom_ID'] == idselc, ['CÓDIGO']]
    codigo_value = resultado['CÓDIGO'].values[0]
    resultado2 = str(df1.loc[df1['CÓDIGO'] == codigo_value, ['LPOM']])

  elif seleccion_indices ==[3]:
    resultado = df1.loc[df1['Lig_ID'] == idselc, ['CÓDIGO']]
    codigo_value = resultado['CÓDIGO'].values[0]
    resultado2 = str(df1.loc[df1['CÓDIGO'] == codigo_value, ['LIGANDO']])

  elif seleccion_indices ==[4]:
    resultado = df1.loc[df1['TS_ID'] == idselc, ['CÓDIGO']]
    codigo_value = resultado['CÓDIGO'].values[0]
    resultado2 = str(df1.loc[df1['CÓDIGO'] == codigo_value, ['TIPO SINTES.']])

  elif seleccion_indices ==[5]:
    resultado = df1.loc[df1['SM_ID'] == idselc, ['CÓDIGO']]
    codigo_value = resultado['CÓDIGO'].values[0]
    resultado2 = str(df1.loc[df1['CÓDIGO'] == codigo_value, ['SOLVENTE - MEDIO']])

  elif seleccion_indices ==[6]:
    resultado = df1.loc[df1['PRE_ID'] == idselc, ['CÓDIGO']]
    codigo_value = resultado['CÓDIGO'].values[0]
    resultado2 = str(df1.loc[df1['CÓDIGO'] == codigo_value, ['PRECURSOR']])

  elif seleccion_indices ==[7]:
    resultado = df1.loc[df1['SAL_ID'] == idselc, ['CÓDIGO']]
    codigo_value = resultado['CÓDIGO'].values[0]
    resultado2 = str(df1.loc[df1['CÓDIGO'] == codigo_value, ['SAL  Ln']])

  # Obtener las opciones seleccionadas
  opciones_seleccionadas = [opciones[i] for i in seleccion_indices if 0 <= i < len(opciones)]
  datoreal = resultado2
  #print("El verdadero dato es: ", resultado2)
  print("El verdadero dato es de ", opciones_seleccionadas, "es: ",datoreal)


import tkinter as tk
from tkinter import messagebox

# Crear la ventana principal
ventana = tk.Tk()
ventana.title("Selecciona la función que deseas ejecutar")
ventana.geometry("400x200")
# Crear etiquetas para guiar al usuario
label = tk.Label(ventana, text="Selecciona una opción para ejecutar la función:")
label.pack(pady=10)

# Botón para Método 1
boton1 = tk.Button(ventana, text="Predecir", command=predecir2, width=20, height=2)
boton1.pack(pady=5)

# Botón para Método 2
boton2 = tk.Button(ventana, text="Evaluar", command=cmid2, width=20, height=2)
boton2.pack(pady=5)

# Botón para Método 3
boton3 = tk.Button(ventana, text="Identificar", command=ident, width=20, height=2)
boton3.pack(pady=5)

# Iniciar el bucle principal de la ventana
ventana.mainloop()