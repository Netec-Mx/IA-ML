# Práctica 3. Aplicación real de una Red Neuronal

## Objetivo de la práctica:

Al finalizar la práctica, serás capaz de:

- Aplicar los conocimientos aprendidos para comprender cómo se crea una red neuronal para detectar si una transacción bancaria es legítima y obtener un resultado mayor al 80% de efectividad.

## Objetivo visual:

![Objetivo visual](../images/cap3_obj_vis.png) 

## Duración aproximada:

- 60 minutos.

**[⬅️ Atrás](https://netec-mx.github.io/IA-ML/Cap%C3%ADtulo2/Lab2.html)** | **[Lista General](https://netec-mx.github.io/IA-ML/)** | **[Siguiente ➡️](https://netec-mx.github.io/IA-ML/Cap%C3%ADtulo4/Lab4.html)**

## Instrucciones:

### Configuración del entorno de trabajo:

1. Dentro de Google Drive, selecciona `Nuevo`.

![imagen resultado](../images/conf_1.png)

2. Da clic en `Más` y después en `Conectar con más aplicaciones`.

![imagen resultado](../images/conf_2.png)

3. En el buscador, escribe `Colab` y selecciona el que dice `Colaboratory`.

![imagen resultado](../images/conf_3.png)

4. Da clic en `Instalar`.

![imagen resultado](../images/conf_4.png)

5. En el botón de`Nuevo`, selecciona `Colaboratory`.

![imagen resultado](../images/conf_5.png)

6. Cuando se abra un nuevo archivo, selecciona `Entorno de ejecución`.

![imagen resultado](../images/conf_6.png)

7. Selecciona `Cambiar tipo de entorno de ejecución`.

![imagen resultado](../images/conf_7.png)

8. Selecciona `T4 GPU` y da clic en `Guardar`.

![imagen resultado](../images/conf_8.png)

9. Finalmente, conéctate a los recursos seleccionados.

![imagen resultado](../images/conf_9.png)

### Tarea 1. Cargar el dataset.

Paso 1. Descarga el siguiente [archivo](https://drive.google.com/file/d/1afk1TL6_91oYtp-FH5KbQnpdKyCNueZB/view?usp=sharing) csv y súbelo al entorno de trabajo de Colab.

![imagen resultado](../images/cap3_1.png)
![imagen resultado](../images/cap3_2.png)

Paso 2. Importa las librerías necesarias para el procesamiento de datos, gráficas, etc.

```python
%matplotlib inline
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib.gridspec as gridspec
from collections import Counter
from sklearn import metrics
import numpy as np
from sklearn.model_selection import train_test_split
from matplotlib.colors import LogNorm
from sklearn.metrics import f1_score
```

Paso 3. Se dividirán los datos en el siguiente orden:
  - Primero, el conjunto de entrenamiento (train_set) recibe el 60% de los datos.
  - Luego, el 40% restante se divide nuevamente en dos partes iguales:
    - 20% para validación (val_set).
    - 20% para pruebas (test_set).

```python
def train_val_test_split(df, rstate=42, shuffle=True, stratify=None):
    strat = df[stratify] if stratify else None
    train_set, test_set = train_test_split(
        df, test_size=0.4, random_state=rstate, shuffle=shuffle, stratify=strat)
    strat = test_set[stratify] if stratify else None
    val_set, test_set = train_test_split(
        test_set, test_size=0.5, random_state=rstate, shuffle=shuffle, stratify=strat)
    return (train_set, val_set, test_set)

def remove_labels(df, label_name):
    X = df.drop(label_name, axis=1)
    y = df[label_name].copy()
    return (X, y)
```

Paso 4. Importa los datos y visualiza los 10 primeros.

```python
df = pd.read_csv('creditcard.csv')
df.head(10)
```

Paso 5. Asímismo, es posible visualizar características particulares del dataset.

```python
# Información general del DataFrame
print("="*40)
print("INFORMACIÓN GENERAL DEL DATAFRAME")
print("="*40)
df.info()

# Número total de características (columnas)
print("\nNúmero de características:", len(df.columns))

# Longitud del conjunto de datos (número de filas)
print("Longitud del conjunto de datos:", len(df))

# Verificación de valores nulos en cada columna
print("\n" + "="*40)
print("VERIFICACIÓN DE VALORES NULOS")
print("="*40)
print(df.isna().sum())

# Distribución de la variable objetivo (si 'Class' es relevante)
print("\n" + "="*40)
print("DISTRIBUCIÓN DE LA COLUMNA 'Class'")
print("="*40)
print(df["Class"].value_counts())

# Estadísticas descriptivas del DataFrame
print("\n" + "="*40)
print("RESUMEN ESTADÍSTICO")
print("="*40)
print(df.describe())

```
Paso 6. Para agilizar el procesamiento, se usarán unicamente dos variables (V10 y V14), las cuales se pueden visualizar en una gráfica.

```python
plt.figure(figsize=(14, 6))
plt.scatter(df["V10"][df['Class'] == 0], df["V14"][df['Class'] == 0], c="g", marker=".")
plt.scatter(df["V10"][df['Class'] == 1], df["V14"][df['Class'] == 1], c="r", marker=".")
plt.xlabel("V10", fontsize=14)
plt.ylabel("V14", fontsize=14)
plt.show()
```

Paso 7. Dada esta nueva reduccion en las clases a usar, se hara la misma division de los conjuntos de datos pero unicamente con las 2 variables.

```python

df = df.drop(["Time", "Amount"], axis=1)

train_set, val_set, test_set = train_val_test_split(df)

X_train, y_train = remove_labels(train_set, 'Class')
X_val, y_val = remove_labels(val_set, 'Class')
X_test, y_test = remove_labels(test_set, 'Class')

X_train_reduced = X_train[["V10", "V14"]].copy()
X_val_reduced = X_val[["V10", "V14"]].copy()
X_test_reduced = X_test[["V10", "V14"]].copy()

X_train_reduced
```

### Tarea 2. Entrenar el modelo.

Paso 8. Instala el módulo de métricas de Keras.

```python
!pip install keras-metrics
```

Paso 9. Define la arquitectura del modelo.

```python
from keras import models
from keras import layers
import keras_metrics as km
from sklearn.metrics import f1_score

model = models.Sequential()
model.add(layers.Dense(128, activation='relu', input_shape=(X_train_reduced.shape[1],)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='adam',
             loss='binary_crossentropy',
             metrics=[f1_score])


print("Using TensorFlow backend.")
```

Paso 10. Visualiza la arquitectura construida.

```python
model.summary()
```

### Tarea 4. Entrena el modelo.

Paso 11. Compila el modelo usando **Adam** como optimizador, una función de pérdida `binary_crossentropy` y obetén las métricas de calidad del modelo durante el entrenamiento:

```python
from sklearn.metrics import f1_score

model.compile(optimizer='rmsprop',
             loss='binary_crossentropy',
             metrics=['accuracy'])

history = model.fit(X_train_reduced, y_train, epochs=40, batch_size=512, validation_data=(X_val_reduced, y_val))


y_pred_prob = model.predict(X_test_reduced)
y_pred = (y_pred_prob > 0.5).astype(int)

f1 = f1_score(y_test, y_pred)

print("F1 Score:", f1)
```

Paso 12. Grafica los resultados del modelo.

```python
def plot_ann_decision_boundary(X, y, model, steps=1000):
    xmin, xmax = X[:,0].min() - 1, X[:,0].max() + 1
    ymin, ymax = X[:,1].min() - 1, X[:,1].max() + 1
    steps = 1000

    x_span = np.linspace(xmin, xmax, steps)
    y_span = np.linspace(ymin, ymax, steps)
    xx, yy = np.meshgrid(x_span, y_span)

    labels = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = labels.reshape(xx.shape)

    plt.contourf(xx, yy, Z, cmap="RdBu", alpha=0.5)
    plt.plot(X[:, 0][y==0], X[:, 1][y==0], 'k.', markersize=2)
    plt.plot(X[:, 0][y==1], X[:, 1][y==1], 'r.', markersize=2)


plt.figure(figsize=(14, 6))
plot_ann_decision_boundary(X_train_reduced.values, y_train, model)
plt.xlabel("V10", fontsize=14)
plt.ylabel("V14", fontsize=14)
plt.show()
```

Paso 13. Visualiza las predicciones que hace el modelo.

```python
y_pred_prob = model.predict(X_train_reduced)
y_pred = (y_pred_prob > 0.5).astype(int)

plt.figure(figsize=(14, 6))
plt.plot(X_train_reduced[y_pred==1]["V10"], X_train_reduced[y_pred==1]["V14"], 'go', markersize=4)
plot_ann_decision_boundary(X_train_reduced.values, y_train, model)
plt.xlabel("V10", fontsize=14)
plt.ylabel("V14", fontsize=14)
plt.show()
```

Paso 14. También, evalúa la precisión del modelo.

```python
y_pred_prob = model.predict(X_test_reduced)
y_pred = (y_pred_prob > 0.5).astype(int)


print("F1 Score:", f1_score(y_test, y_pred))
```

## Resultado esperado:

![imagen resultado](/images/cap3_resultado.png) 
