import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
from torch.utils.tensorboard import SummaryWriter
#========================================================
# 1.Configure visual style and seeds for reproducibility
#========================================================

sns.set_theme(style="whitegrid")
torch.manual_seed(42)
np.random.seed(42)

print("Environment ready.")

#========================================================
# 2.Load and preprocess the data
#========================================================

print("\nLoading and preprocessing data...")

df = pd.read_csv('/home/ikermorrondo/deepLearning/1assign/insurance.csv')

# Transformamos categorías en columnas de 0 y 1 ya que nuestro modelo no puede trabajar con texto porque las redes neuronales trabajan con números, no con texto.
df = pd.get_dummies(df, columns=['sex', 'smoker', 'region'], drop_first=True)   #el drop_first=True es para evitar redundacia, es decir que si no es hombre, entonces es mujer, si no es fumador, entonces es no fumador, etc.

# Separamos las características (X) de la variable objetivo (y) que es 'charges'. 
X = df.drop('charges', axis=1).values
y = df['charges'].values.reshape(-1, 1)

# Separamos 80% para entrenar y 20% para el resto
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)

# Ese 20% lo dividimos en dos: 10% para Validar y 10% para el Test final
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

#========================================================
# 3.Exploratory data analysis
#========================================================

print("--- Dataset Head ---")
print(df.head())

print("\n--- General Information and Null Values ---")
df.info()
print("\nNull values per column:\n", df.isnull().sum())

print("\n--- Descriptive Statistics ---")
print(df.describe())

print("\n--- Some plots ---")

#Barchart 
plt.figure(figsize=(10, 6))
sns.histplot(df['charges'], kde=True, color='blue', bins=50)
plt.title('Distribución de los Costos de Seguro (Charges)')
plt.xlabel('Cargos ($)')
plt.ylabel('Frecuencia')
plt.show()

#Correlation matrix
plt.figure(figsize=(10, 8))
# Calculamos la correlación
correlacion = df.corr()
# Dibujamos un mapa de calor (heatmap)
sns.heatmap(correlacion, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Matriz de Correlación de las Variables')
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Boxplot: Fumador vs Cargos
sns.boxplot(ax=axes[0], x='smoker_yes', y='charges', data=df, palette='Set2')
axes[0].set_title('Impacto de Fumar en los Costos Médicos')
axes[0].set_xlabel('Fumador')
axes[0].set_ylabel('Cargos ($)')
plt.tight_layout()
plt.show()

#========================================================
# 4.Data preprocessing for the neural network
#========================================================

#Estandarizamos los datos porque las redes neuronales funcionan mejor con datos centrados y escalados
#Si no lo hacemos, algunas características podrían dominar a otras debido a sus diferentes escalas, lo que podría dificultar el aprendizaje del modelo.
scaler = StandardScaler()

# Ajustamos el scaler solo con los datos de entrenamiento para evitar "fugas de información" 
X_train = scaler.fit_transform(X_train)

# Luego transformamos los datos de validación y test usando el mismo scaler para mantener la consistencia
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)



# La red neuronal no entiende tablas de Excel o listas normales, solo entiende "Tensores".
# Convertimos los números a Floats (decimales) para que el motor de PyTorch pueda hacer cálculos rápidos.
X_train = torch.FloatTensor(X_train)
y_train = torch.FloatTensor(y_train)
X_val = torch.FloatTensor(X_val)
y_val = torch.FloatTensor(y_val)

X_test, y_test = torch.FloatTensor(X_test), torch.FloatTensor(y_test)