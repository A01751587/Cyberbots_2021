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

#Cosntruir red :D
