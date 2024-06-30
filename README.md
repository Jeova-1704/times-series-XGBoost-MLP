# Previsão da Série Temporal de Produção de Energia Nuclear

Este projeto tem como objetivo prever a produção de energia nuclear utilizando dois modelos de previsão: MLP (Multi-Layer Perceptron) e XGBoost.

## Requisitos

Certifique-se de ter as seguintes bibliotecas instaladas:

```python
pip install pandas numpy statsmodels matplotlib scikit-learn xgboost
```

## Importação das Bibliotecas

```python
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
import warnings
warnings.filterwarnings('ignore')

from xgboost import XGBRegressor
```

## Carregamento e Preparação dos Dados

### Carregamento dos Dados

```python
path = '/content/electricity_Consumption_Productioction.csv'
dados = pd.read_csv(path)
dados['Date'] = pd.to_datetime(dados['Date'])
dados = dados.set_index('Date')
dados = pd.DataFrame(index=dados.index, data=dados['Nuclear'])
```

### Visualização da Série Temporal

```python
rcParams['figure.figsize'] = 15, 6
dados.plot()
plt.show()
```

### Divisão dos Dados em Treino, Validação e Teste

```python
train = dados.loc['2019': '2021']
val = dados.loc['2022']
test = dados.loc['2023':]

plt.plot(train, label='train')
plt.plot(val, label='val')
plt.plot(test, label='test')
plt.legend(loc='best')
plt.show()
```

## Análise dos Dados

### Normalização dos Dados

```python
scaler = MinMaxScaler()
scaler.fit(train)

X_train_scaler = scaler.transform(train)
X_val_scaler = scaler.transform(val)
X_test_scaler = scaler.transform(test)
```

### Função para Plotar ACF e PACF

```python
def acf_pacf(x, qtd_lag):
    fig = plt.figure(figsize=(16,10))
    ax1 = fig.add_subplot(221)
    fig = sm.graphics.tsa.plot_acf(x, lags=qtd_lag, ax=ax1)
    ax2 = fig.add_subplot(222)
    fig = sm.graphics.tsa.plot_pacf(x, lags=qtd_lag, ax=ax2)
    plt.show()

acf_pacf(train.values, 30)
```

### Criação de Janelas Deslizantes

```python
def create_sliding_windows(series, window_size):
    list_of_sliding_windows = []
    list_size_to_iterate = len(series) - window_size
    for i in range(0, list_size_to_iterate):
        window = series[i: i + window_size + 1]
        list_of_sliding_windows.append(window)
    return np.array(list_of_sliding_windows).reshape(len(list_of_sliding_windows), window_size+1)

train_windows = create_sliding_windows(X_train_scaler, 24)
val_windows = create_sliding_windows(X_val_scaler, 24)
test_windows = create_sliding_windows(X_test_scaler, 24)

X_train = train_windows[:, 0:-1]
y_train = train_windows[:, -1]

X_val = val_windows[:, 0: -1]
y_val = val_windows[:, -1]

X_test = test_windows[:, 0: -1]
y_test = test_windows[:, -1]

X_train_full = np.vstack([X_train, X_val])
y_train_full = np.hstack([y_train, y_val])
```

## Modelos de Previsão

### MLP - Multi-Layer Perceptron

#### Treinamento e Validação

```python
param_grid = {
    'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50), (50, 50, 50)],
    'activation': ['identity', 'logistic', 'tanh', 'relu'],
    'solver': ['adam', 'lbfgs'],
    'alpha': [0.0001, 0.001, 0.01],
    'learning_rate_init': [0.001, 0.01, 0.1]
}

results = []
best_mse = np.inf

for hidden_layer_sizes in param_grid['hidden_layer_sizes']:
    for activation in param_grid['activation']:
        for solver in param_grid['solver']:
            for alpha in param_grid['alpha']:
                for learning_rate_init in param_grid['learning_rate_init']:
                    model = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes,
                                         activation=activation,
                                         solver=solver,
                                         alpha=alpha,
                                         learning_rate_init=learning_rate_init,
                                         random_state=42)

                    model.fit(X_train, y_train)

                    prev_train = model.predict(X_train)
                    prev_val = model.predict(X_val)

                    train_error = mean_absolute_percentage_error(y_train, prev_train)
                    val_error = mean_absolute_percentage_error(y_val, prev_val)

                    if best_mse > val_error:
                        results.append({
                            'hidden_layer_sizes': hidden_layer_sizes,
                            'activation': activation,
                            'solver': solver,
                            'alpha': alpha,
                            'learning_rate_init': learning_rate_init,
                            'train_error': train_error,
                            'val_error': val_error
                        })
                        best_mse = val_error

best_params = min(results, key=lambda x: x['val_error'])
print("Melhores parâmetros encontrados:", best_params)
```

#### Treinamento do Melhor Modelo

```python
melhor_modelo_mlp = MLPRegressor(hidden_layer_sizes=(50, 50),
                                 activation='logistic',
                                 solver='lbfgs',
                                 alpha=best_params['alpha'],
                                 learning_rate_init=best_params['learning_rate_init'],
                                 learning_rate='adaptive',
                                 random_state=42)

melhor_modelo_mlp.fit(X_train_full, y_train_full)
```

#### Avaliação do Modelo MLP

```python
prev_train = melhor_modelo_mlp.predict(X_train_full)
plt.plot(prev_train, label='Previsão')
plt.plot(y_train_full, label='treino')
plt.legend(loc='best')
plt.title('Previsões vs. treino (Conjunto de Treinamento)')
plt.show()

rmse_mlp = np.sqrt(mean_squared_error(y_train_full, prev_train))
mse_mlp = mean_squared_error(y_train_full, prev_train)
mape = mean_absolute_percentage_error(prev_train, y_train_full)

print("RMSE:", rmse_mlp)
print("MSE:", mse_mlp)
print("MAPE:", mape)

pred_test = melhor_modelo_mlp.predict(X_test)
plt.plot(pred_test, label='Previsão')
plt.plot(y_test, label='Real')
plt.legend(loc='best')
plt.title('Previsões vs. Real (Conjunto de Teste)')
plt.show()

rmse_test_mlp = np.sqrt(mean_squared_error(y_test, pred_test))
mse_test_mlp = mean_squared_error(y_test, pred_test)
mape_test_mlp = mean_absolute_percentage_error(y_test, pred_test)

print("RMSE Test:", rmse_test_mlp)
print("MSE Test:", mse_test_mlp)
print("MAPE Test:", mape_test_mlp)
```

### XGBoost

#### Treinamento e Validação

```python
n_estimators_values = [100, 200, 300]
max_depth_values = [3, 5, 7]
learning_rate_values = [0.01, 0.05, 0.1]

best_score = float('-inf')
best_params = {}

for n_estimators in n_estimators_values:
    for max_depth in max_depth_values:
        for learning_rate in learning_rate_values:
            model = XGBRegressor(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate)
            scores = cross_val_score(model, X_train, y_train, cv=3, scoring='neg_mean_squared_error')
            mean_score = scores.mean()

            if mean_score > best_score:
                best_score = mean_score
                best_params = {'n_estimators': n_estimators, 'max_depth': max_depth, 'learning_rate': learning_rate}

print("Melhores parâmetros:", best_params)
```

#### Treinamento do Melhor Modelo

```python
best_model = XGBRegressor(n_estimators=best_params['n_estimators'],
                          max_depth=best_params['max_depth'],
                          learning_rate=best_params['learning_rate'])

best_model.fit(X_train, y_train)
best_model.fit(X_train_full, y_train_full)
```

#### Avaliação do Modelo XGBoost

```python
prev_train = best_model.predict(X_train_full)
plt.title("Previsão em treino")
plt.plot(prev_train, label='predict')
plt.plot(y_train_full, label='target')
plt.legend(loc='best')
plt.show()

resultados = {
    "Métricas": ["MAPE", "MSE", "RMSE"],
    "Resultado": [mean_absolute_percentage_error(prev_train, y_train_full),
                  mean_squared_error(prev_train, y_train_full),
                  np.sqrt(mean_squared_error(prev_train, y_train_full))]
}

df_resultados_XGB_train = pd.DataFrame(resultados)


print(df_resultados_XGB_train)

prev_test = best_model.predict(X_test)
plt.title("Previsão em teste")
plt.plot(prev_test, label='predict')
plt.plot(y_test, label='target')
plt.legend(loc='best')
plt.show()

resultados = {
    "Métricas": ["MAPE", "MSE", "RMSE"],
    "Resultado": [mean_absolute_percentage_error(prev_test, y_test),
                  mean_squared_error(prev_test, y_test),
                  np.sqrt(mean_squared_error(prev_test, y_test))]
}

df_resultados_XGB_test = pd.DataFrame(resultados)
print(df_resultados_XGB_test)
```

## Comparação dos Modelos

```python
metricas = {
    'Métrica': ['MAPE', 'MSE', 'RMSE'],
    'Valor': [mape_test_mlp, mse_test_mlp, rmse_test_mlp]
}

df_resultados_mlp_test = pd.DataFrame(metricas)
print(df_resultados_mlp_test)
```
