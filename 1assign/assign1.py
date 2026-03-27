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
from sklearn.metrics import mean_squared_error


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

df = pd.read_csv('insurance.csv')

# Transform categorical columns into 0 and 1 columns since our model cannot work with text because neural networks work with numbers, not text.
df = pd.get_dummies(df, columns=['sex', 'smoker', 'region'], drop_first=True)   # drop_first=True avoids redundancy, meaning if it's not male, then it's female, if not a smoker, then it's a non-smoker, etc.

# Separate the features (X) from the target variable (y) which is 'charges'.
X = df.drop('charges', axis=1).values
y = df['charges'].values.reshape(-1, 1)

# Split the data into training (80%) and temporary (20%) sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)

# Split that 20% into two: 10% for Validation and 10% for the final Test
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
plt.title('Distribution of Insurance Costs (Charges)')
plt.xlabel('Charges ($)')
plt.ylabel('Frequency')
plt.show()

#Correlation matrix
plt.figure(figsize=(10, 8))
# We calculate the correlation matrix to see how the features relate to each other and to the target variable 'charges'
correlacion = df.corr()
# We draw a heatmap to visualize the correlation matrix.
sns.heatmap(correlacion, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Matrix of the Variables')
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Boxplot: Smoker vs Charges
sns.boxplot(ax=axes[0], x='smoker_yes', y='charges', data=df, palette='Set2')
axes[0].set_title('Impact of Smoking on Medical Costs')
axes[0].set_xlabel('Smoker')
axes[0].set_ylabel('Charges ($)')
plt.tight_layout()
plt.show()

#========================================================
# 4.Data preprocessing for the neural network
#========================================================

# Standardization: Scaling features to have mean=0 and variance=1
# This prevents features with larger scales from dominating the learning process
scaler = StandardScaler()

# Fit the scaler only on the training data to avoid "data leakage"
X_train = scaler.fit_transform(X_train)

# Transform validation and test sets using the same parameters
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)



# We convert the number into floats for Pytorch motor 
X_train = torch.FloatTensor(X_train)
y_train = torch.FloatTensor(y_train)
X_val = torch.FloatTensor(X_val)
y_val = torch.FloatTensor(y_val)

X_test, y_test = torch.FloatTensor(X_test), torch.FloatTensor(y_test)



#========================================================
# 5.Base Model: Linear Regression
#========================================================



# Train a traditional Linear Regression model as a benchmark
base_model = LinearRegression()
base_model.fit(X_train.numpy(), y_train.numpy())

# Predict using the test set
base_predictions = base_model.predict(X_test.numpy())

# Calculate evaluation metrics
rmse_base = np.sqrt(mean_squared_error(y_test.numpy(), base_predictions))
mae_base = mean_absolute_error(y_test.numpy(), base_predictions)

print("--- BASELINE MODEL RESULTS (LINEAR REGRESSION) ---")
print(f"Base RMSE: ${rmse_base:.2f}")
print(f"Base MAE: ${mae_base:.2f}")
print("-" * 50)


#========================================================
# 6.Neuronal Network and Training 
#========================================================


class RedNeuronalSeguros(nn.Module):
    # Here we add  n1 y n2 for Optuna to modify the number of neurons in each layer
    def __init__(self, input_size, n1=64, n2=32): 
        super(RedNeuronalSeguros, self).__init__()
        self.capa1 = nn.Linear(input_size, n1)
        self.capa2 = nn.Linear(n1, n2)
        self.capa_salida = nn.Linear(n2, 1)
        self.relu = nn.ReLU()
        
        # Using Kaiming Uniform initialization for layers with ReLU activation
        init.kaiming_uniform_(self.capa1.weight, nonlinearity='relu')
        init.kaiming_uniform_(self.capa2.weight, nonlinearity='relu')
        
    def forward(self, x):
        x = self.relu(self.capa1(x))
        x = self.relu(self.capa2(x))
        x = self.capa_salida(x)
        return x




def model_training(modelo, criterion, optimizer, X_train, y_train, X_test, y_test, epochs=500):
    historial_train = []
    historial_val = []
    
    writer = SummaryWriter('runs/experimento_base')
    
    for epoch in range(epochs):
        # training 
        modelo.train()
        optimizer.zero_grad()
        predicciones = modelo(X_train)
        loss = criterion(predicciones, y_train)
        loss.backward()
        optimizer.step()
        historial_train.append(loss.item())
        
        # validation 
        modelo.eval()
        with torch.no_grad():
            pred_val = modelo(X_test)
            loss_val = criterion(pred_val, y_test)
            historial_val.append(loss_val.item())
            
        # TensorBoard
        writer.add_scalar('Loss/Train', loss.item(), epoch)
        writer.add_scalar('Loss/Validation', loss_val.item(), epoch)
        
        if (epoch + 1) % 50 == 0:
            print(f'Época {epoch+1}/{epochs} | MSE Train: {loss.item():.2f} | MSE Val: {loss_val.item():.2f}')
            
    writer.close()
    return historial_train, historial_val



# Initial model test 
input_size = X_train.shape[1] 
modelo_dl = RedNeuronalSeguros(input_size)

# define the loss and optimizer 
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(modelo_dl.parameters(), lr=0.01)

print("\nStarting initial training...")
historial_train, historial_val = model_training(
    modelo_dl, criterion, optimizer, 
    X_train, y_train, X_test, y_test, 
    epochs=500
)

# Final predictions for graph 2 
modelo_dl.eval()
with torch.no_grad():
    predicciones_finales = modelo_dl(X_test)
    
print("Initial training completed!")



#========================================================
# 7.Optuna 
#========================================================



def objective(trial):
    # hiperparameter suggestions
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    
    # testing round neuron counts, as multiples of 16 for better performance 
    n1 = trial.suggest_int("n1", 32, 256, step=16) 
    n2 = trial.suggest_int("n2", 16, 128, step=16)
    
    #reusing modular class
    model_opt = RedNeuronalSeguros(input_size=X_train.shape[1], n1=n1, n2=n2)
    optimizer_opt = torch.optim.Adam(model_opt.parameters(), lr=lr)
    criterion_opt = nn.MSELoss()

    # fast training 
    for epoch in range(200):
        model_opt.train()
        optimizer_opt.zero_grad()
        loss = criterion_opt(model_opt(X_train), y_train)
        loss.backward()
        optimizer_opt.step()
    

    # evaluation using validation set
    model_opt.eval()
    with torch.no_grad():
        v_loss = criterion_opt(model_opt(X_test), y_test)
    
    return np.sqrt(v_loss.item()) 



# Launching the Optuna study , it will do 20 tries automatically 
# Optuna will try the 20 different ones and will take better one 
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=20)

print("\n--- HYPERPARAMETER SEARCH COMPLETED ---")
print(f"Best RMSE found: ${study.best_value:.2f}")
print(f"Best Parameters: {study.best_params}")



#========================================================
# 8. Final Model Training & Performance Evaluation
#========================================================


# we initialize the final model with the best hyperparameters found by Optuna
bests_params = study.best_params
print(f"Construyendo modelo definitivo con: {bests_params}")


modelo_final = RedNeuronalSeguros(
    input_size=X_train.shape[1], 
    n1=bests_params['n1'], 
    n2=bests_params['n2']
)


# setup the final optimizer and criterion 
criterion_final = nn.MSELoss()
optimizer_final = torch.optim.Adam(modelo_final.parameters(), lr=bests_params['lr'])

 
#final training session
print("\nStarting final training...")
historial_train, historial_val = model_training(
    modelo_final, criterion_final, optimizer_final, 
    X_train, y_train, X_test, y_test, 
    epochs=600  # we put more epochs for letting him learn at maximum 
)

# final test set predictions
modelo_final.eval()
with torch.no_grad():
    predicciones_finales = modelo_final(X_test)

# calculate final performance metrics in Dollars
rmse_final = mean_squared_error(y_test, predicciones_finales.numpy()) ** 0.5

print(f"\n--- FINAL PERFORMANCE COMPARISON ---")
print(f"Baseline Linear Regression RMSE: ${rmse_base:.2f}")
print(f"Optimized Neural Network RMSE:   ${rmse_final:.2f}")



#========================================================
# 9. Final Visualizations
#========================================================


# Learning Curve: Checking for Overfitting
plt.figure(figsize=(10, 6))
plt.plot(historial_train, label='Training Loss')
plt.plot(historial_val, label='Validation Loss')
plt.title('Learning Curves: Generalization Assessment')
plt.xlabel('Epochs')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.show()


# Predicted vs Actual Values Scatter Plot
# We are gonna see visualizing how the points deviate from reality

plt.figure(figsize=(8, 8))
plt.scatter(y_test, predicciones_finales, alpha=0.5, color='green')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.title('Predicted vs Actual Values (Set de Test)')
plt.xlabel('Actual Value($)')
plt.ylabel('AI Prediction($)')
plt.show()

