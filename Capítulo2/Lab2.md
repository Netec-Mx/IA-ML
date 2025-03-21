# Primera Red Neuronal

## Objetivo de la práctica:
Crear una red neuronal desde cero que separe dos grupos de datos.

## Objetivo Visual 

![Objetivo visual](images\cap2_obj_vis.png)

## Duración aproximada:

- 50 minutos.

## Instrucciones 

### **CONFIGURACIÓN DEL ENTORNO DE TRABAJO**

Dentro de Google Drive, seleccionar `Nuevo`

![imagen resultado](images\conf_1.png)

Dar clic en `Más` y `Conectar con más aplicaciones`

![imagen resultado](images\conf_2.png)

En el buscador escribir `Colab` y seleccionar el que dice `Colaboratory`

![imagen resultado](images\conf_3.png)

Dar clic en `Instalar`

![imagen resultado](images\conf_4.png)

En `Nuevo`, seleccionar `Colaboratory`
![imagen resultado](images\conf_5.png)

Cuando se abra un nuevo archivo, seleccionar `Entorno de ejecución`
![imagen resultado](images\conf_6.png)

Seleccionar `Cambiar tipo de entorno de ejecución`

![imagen resultado](images\conf_7.png)

Seleccionar `T4 GPU` y dar clic en `Guardar`
![imagen resultado](images\conf_8.png)

Finalmente, conectarse a los recursos seleccionados

![imagen resultado](images\conf_9.png)

### Tarea 1. **Modificar una red neuronal de manera interactiva**

Paso 1. Ingresar al siguiente [enlace](https://playground.tensorflow.org/#activation=sigmoid&batchSize=10&dataset=circle&regDataset=reg-plane&learningRate=0.03&regularizationRate=0&noise=0&networkShape=7,2&seed=0.24502&showTestData=false&discretize=true&percTrainData=50&x=true&y=true&xTimesY=false&xSquared=false&ySquared=false&cosX=false&sinX=false&cosY=false&sinY=false&collectStats=false&problem=classification&initZero=false&hideText=true)

Paso 2. Colocar los siguientes parametros:

**Parámetros Generales**

- **Ratio de entrenamiento a prueba:** 50%
- **Ruido:** 0
- **Tamaño del batch:** 10

**Selección de Características (Features)**

Activar solo las siguientes características de entrada:
- ✅ `X₁`
- ✅ `X₂`
- ❌ No activar ninguna otra característica adicional (como `X₁²`, `X₂²`, `X₁X₂`, `sin(X₁)`, `sin(X₂)`).

**Arquitectura de la Red Neuronal**

- **Capas ocultas:** 2
  - Primera capa oculta: **7 neuronas**
  - Capa de salida: **2 neuronas**

**Configuración en la parte superior**

- **Learning rate:** `0.1`
- **Función de activación:** `Sigmoid`
- **Regularización:** `None`
- **Tasa de regularización:** `0`
- **Tipo de problema:** `Clasificación`

![tensorflow interface](/images/cap2_1.png) 

### Tarea 2. **Cargar el dataset**

Paso 3. Importar las siguientes librerias y ejecutar la celda

```python
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
```

![imagen resultado](images/cap2_2.png) 

Paso 4. Insertar y ejecutar el siguiente codigo en una celda nueva para insertar los datos circulares

```python
n = 500
p = 2

X,Y = make_circles(n_samples = n, factor = 0.5, noise = 0.05)

Y = Y[:,np.newaxis]

plt.scatter(X[:,0], X[:,1])

plt.show()

print(Y)
```

Paso 5. Ahora que ya se han insertado los datos, es posible visualizarlos y colocarles un color para diferenciarlos

```python
plt.scatter(X[Y[:,0] == 0,0],X[Y[:,0] == 0,1], c = "gray")
plt.scatter(X[Y[:,0] == 1,0],X[Y[:,0] == 1,1], c = "red")

plt.axis("equal")
plt.show()
```

### Tarea 3, **Visualizar las funciones de activacion**

Paso 6. Se definiran de manera manual las funciones de activacion, aunque para aplicaciones reales es recomendable usar las que vienen por defecto en las librerias de redes neuronales.

```python
import numpy as np
import matplotlib.pyplot as plt

# Función Sigmoide
sigm = lambda x: 1 / (1 + np.e**(-x))

# ReLU
relu = lambda x: np.maximum(0, x)

_x = np.linspace(-5, 5, 100)

plt.figure() 
plt.plot(_x, sigm(_x))
plt.title('Sigmoid Function')
plt.xlabel('x')
plt.ylabel('sigmoid(x)')
plt.grid(True)

plt.figure()
plt.plot(_x, relu(_x))
plt.title('ReLU Function')
plt.xlabel('x')
plt.ylabel('ReLU(x)')
plt.grid(True)

# Derivada de la sigmoide
sigm = (lambda x: 1 /(1 + np.e**(-x)),
        lambda x: x*(1-x))

plt.show()
```

Paso 7. Definiremos una arquitectura para la red neuronal

```python
l0 = neural_layer(p, 4, sigm)
l1 = neural_layer(4, 2, sigm)
l2 = neural_layer(2, 1, sigm)

topology = [p, 4, 2, 1]

l0 = neural_layer(p,4,sigm)
l1 = neural_layer(p,8,sigm)

topology = [p, 4, 8, 1]

def create_nn(topology,act_f):
  nn = []
  for i,layer in enumerate(topology[:-1]):

    nn.append(neural_layer(topology[i],topology[i + 1], act_f))
  return nn

neural_net = create_nn(topology,sigm)
neural_net

l2_cost = (lambda Yp,Yr : np.mean((Yp - Yr)**2),
          lambda Yp, Yr: (Yp - Yr))

def train(neural_net, X, Y, l2_cost, lr = 0.5):
  z = X @ neural_net[0].W + neural_net[0].b
  a = neural_net[0].act_f(z)
```

Paso 8. Se realizara una entrenamiento de dicha red neuronal en funcion a la arquitectura definida

```python
def train(neural_net, X, Y, l2_cost, lr = 0.5,train = True):

  out = [(None,X)]
  for i, layer in enumerate(neural_net):

    z = out[-1][1] @ neural_net[i].w + neural_net[i].b
    a = neural_net[i].act_f[0](z)

    out.append((z,a))

  print(l2_cost[0](out[-1][1], Y))

  if train:
    deltas = []

    for i in reversed(range(0,len(neural_net))):

      z = out[i+1][0]
      a = out[i+1][1]

      if i == len(neural_net) - 1:

        deltas.insert(0,l2_cost[1](a, Y)*neural_net[i].act_f[1](a))

      else:

        deltas.insert(0, deltas[0] @ _W.T*neural_net[i].act_f[1](a))

      _W = neural_net[i].w

      neural_net[i].b = neural_net[i].b - np.mean(deltas[0], axis = 0, keepdims = True) * lr

      neural_net[i].w = neural_net[i].w - out[i][1].T @ deltas[0]*lr

  return out[-1][1]


train(neural_net, X, Y, l2_cost, 0.5)

```

### Tarea 3. **Visualizar el entrenamiento**

Paso 10. Finalmente, se graficara el resultado del entrenamiento a lo largo del tiempo, a la vez que se verifica mediante una grafica, la reduccion del error.

```python
import time
from IPython.display import clear_output

neural_n = create_nn(topology,sigm)

loss = []

for i in range(10000):
  pY = train(neural_n, X, Y, l2_cost, lr = 0.04)

  if i % 25 == 0:
    loss.append(l2_cost[0](pY,Y))

    res = 50

    _x0 = np.linspace(-1.5,1.5,res)
    _x1 = np.linspace(-1.5,1.5,res)

    _Y = np.zeros((res,res))

    for i0, x0 in enumerate(_x0):
      for i1,x1 in enumerate(_x1):
        _Y[i0, i1] = train(neural_n, np.array([[x0, x1]]), Y, l2_cost, train = False)[0][0]
    plt.pcolormesh(_x0, _x1, _Y, cmap = "coolwarm")
    plt.axis("equal")

    plt.scatter(X[Y[:,0] == 0,0], X[Y[:,0] == 0,1], c = "gray")
    plt.scatter(X[Y[:,0] == 1,0], X[Y[:,0] == 1,1], c = "red")

    clear_output(wait = True)
    plt.show()
    plt.plot(range(len(loss)),loss)
    plt.show()
    time.sleep(0.5)
```

### Resultado esperado
![imagen resultado](images/cap2_resultado_1.png) 

![imagen resultado](images/cap2_resultado_2.png) 
