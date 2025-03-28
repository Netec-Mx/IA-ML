# API GPT: DALL-E 3

## Objetivo de la práctica:

Aprender cómo utilizar la API de GPT para realizar llamadas al modelo de generación de imágenes DALL-E 3.


## Objetivo Visual 

![Objetivo visual](/images/cap5_obj_vis.png) 

## Duración aproximada:

- 45 minutos.

## Instrucciones 

### **CONFIGURACIÓN DEL ENTORNO DE TRABAJO**

Dentro de Google Drive, seleccionar `Nuevo`

![imagen resultado](/images/conf_1.png)

Dar clic en `Más` y `Conectar con más aplicaciones`

![imagen resultado](/images/conf_2.png)

En el buscador escribir `Colab` y seleccionar el que dice `Colaboratory`

![imagen resultado](/images/conf_3.png)

Dar clic en `Instalar`

![imagen resultado](/images/conf_4.png)

En `Nuevo`, seleccionar `Colaboratory`
![imagen resultado](/images/conf_5.png)

Cuando se abra un nuevo archivo, seleccionar `Entorno de ejecución`
![imagen resultado](/images/conf_6.png)

Seleccionar `Cambiar tipo de entorno de ejecución`

![imagen resultado](/images/conf_7.png)

Seleccionar `T4 GPU` y dar clic en `Guardar`
![imagen resultado](/images/conf_8.png)

Finalmente, conectarse a los recursos seleccionados

![imagen resultado](/images/conf_9.png)

### Tarea 1. **Crear imagenes con DALL-E 3**

Paso 1. Ejecutar el siguiente comando para usar Dall-E 3

```python
!pip install openai==0.28
```

Paso 2. Ir al siguiente [enlace](https://platform.openai.com/api-keys) y autenticarse para crear una API-Key.

Paso 3. Una vez dentro, ir a la seccion de `API Keys` y crear una nueva API-Key dando click en `Create new secret key`.

![imagen resultado](/images/cap5_3.png)
![imagen resultado](/images/cap5_4.png)

Paso 4. Colocar un nombre para la API y dar click en `Create secret key`.

![imagen resultado](/images/cap5_5.png)

Paso 5. Copiar el valor de la API-Key y pegarlo en un bloc de notas al que puedan acceder facilmente.

![imagen resultado](/images/cap5_6.png)

Paso 6. Ejecutar el siguiente codigo en Colab y dar clic en `Elegir archivos` para cargar el bloc de notas donde se encuentra la API-Key.

```python
from google.colab import files
import openai
from IPython.display import Image

uploaded = files.upload()

with open("api_key.txt", "r") as file:
    api_key = file.read().strip()

import openai
openai.api_key = api_key
```

![imagen resultado](/images/cap5_7.png)
![imagen resultado](/images/cap5_8.png)

Paso 7. Ejecutar el siguiente codigo, donde en la variable llamada `prompt` se escribira el texto que se le pide a DALL-E 3 para generar la imagen.

```python
response = openai.Image.create(
  model="dall-e-3",
  prompt="Un gato tocando el piano al atardecer",
  n=1,
  size="1024x1024"
)
image_url = response.data[0].url
```

Paso 8. Finalmente, ejecute el siguiente codigo para visualizar la imagen creada.

```python
from IPython.display import Image

Image(url=image_url)
```

### Tarea 2. **Editar una imagen**

Paso 9. Ingresar al siguiente [enlace](https://drive.google.com/file/d/1IWtCHoxJgBAfFzFOchZT4aYSKB4UQd8O/view?usp=sharing) y descargar la imagen.

![imagen resultado](images/cap5_1.png)

Paso 10. Ir al siguiente [enlace](https://platform.openai.com/docs/guides/images?api-mode=chat) y bajar hasta la seccion de requerimientos de la imagen de entrada.

![imagen resultado](images/cap5_2.png)

Paso 11. Ingresar al siguiente [enlace](https://www.photopea.com/) y arrastrar en la seecion de `Suelta cualquier archivo aqui` la imagen que se descargo previamente. 

Paso 12. Seleccionar la opcion de borrador

![imagen resultado](images/cap5_9.png)

Paso 13. Iniciar con el borrado de la carretera usando el mouse

![imagen resultado](images/cap5_10.png)

Paso 14. Ir a `Archivo`, luego a `Exportar`, y exportar la imagen a `png`.

![imagen resultado](images/cap5_11.png)

Paso 15. Colocarle un nombre y seleccionar `Guardar`.

![imagen resultado](images/cap5_12.png)

Paso 16. Cargar esa imagen descargada a Colab, y colocar el siguiente codigo

```python
response = openai.Image.create_edit(
  image=open("/content/carretera.png", "rb"),
  mask=open("/content/carretera_mask.png", "rb"),
  prompt="ferrocarril en un paisaje",
  n=1,
  size="512x512"
)
image_url = response['data'][0]['url']
```

Paso 17. Para la nueva imagen cargada, dar click derecho y seleccionar `copiar ruta`.

![imagen resultado](images/cap5_13.png)

Paso 18. Dicha ruta, colocarla en la linea de codigo de la varible `mask`

```python
mask=open("/content/carretera_mask.png", "rb"),
```
![imagen resultado](images/cap5_14.png)

Paso 19. Finalmente, ejecutar la celda de codigo para visualizar la imagen editada.

### Resultado esperado

![imagen resultado](images/cap5_resultado_esperado.png)
