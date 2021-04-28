"""
PRIMERA ENTREGA HACK MX
RETO NDS 

EQUIPO: CYBERBOTS
ESCUELA: ITESM CEM
REALIZADO POR:
    ANA PATRICIA ISLAS MAINOU
    PAULO OGANDO GULIAS
    CESAR EMILIANO PALOME LUNA
    JOSE LUIS MADRIGAL SANCHEZ
"""

# Importar Liberiras
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

# Importar la libreria que nos permite crear la red y poner capas de neurones
import tensorflow

# Leer la base de datos de transacciones de tarjetas de credito
datos = pd.read_csv("creditcard.csv")

# Standardizzacion de los datos del monto
scaler = preprocessing.StandardScaler()
# Normalizacion de los montos de -1 a 1
datos["NORMALIZADO"] = scaler.fit_transform(datos["Amount"\
                                                  ].values.reshape(-1, 1))

# Borar las columnas de tiempo y monto no normalizado
datos = datos.drop(["Amount", "Time"], axis = 1)

# La clase de y contiene si es fraude (1) o no (0)
y = datos["Class"] 
# La clase de x tiene las transaccioens y el monto normalizado
X = datos.drop(["Class"], axis = 1)

# Crear conjuntos de entrenamiento y de pruebas
# El 70 % de los datos es para entrenar la red neuronal
# El 30 % de los datos es para probar la red neuronal
X_train, X_test, y_train, y_test =train_test_split(X, y, test_size = 0.3, \
                                                   random_state = 0)

#CREAR LA RED
# Creamos una lugar donde vamos a construir la red
modelo = tensorflow.keras.Sequential()

# Agreamos la primera capa de neuronas, tiene 30 neuronas y de ahí salen 15 posibles resultados
modelo.add(tensorflow.keras.layers.Dense(input_dim = 30, units = 15, activation = "relu"))

# Agreamos la segunda capa de neuronas de ahí salen 24 posibles resultados y tiene 30 neuronas
modelo.add(tensorflow.keras.layers.Dense(units = 24, activation = "relu"))

# Agregamos una capa de ayuda en el entrenamiento, esta capa ayuda a evitar que la red neuronal se sature
modelo.add(tensorflow.keras.layers.Dropout(0.5))

# Agreamos la tecera capa de neuronas y de ahí salen 20 posibles resultados
modelo.add(tensorflow.keras.layers.Dense(units = 20, activation = "relu"))

# Agreamos la cuarta capa de neuronas y de ahí salen 24 posibles resultados
modelo.add(tensorflow.keras.layers.Dense (units = 24, activation = "relu"))

# Agreamos la quinta capa de neuronas y de ahí salen 1 posible resultado, que es un booleano 0 o 1.
modelo.add(tensorflow.keras.layers.Dense(units =1, activation = "sigmoid"))

# Confirmamos que la salida que nos da el modelo es de 1 valor
modelo.output_shape
