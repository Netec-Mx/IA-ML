# Demostración: IA en Programación: GitHub Copilot

## Objetivo de la práctica:

Al finalizar la práctica, serás capaz de:

- Aprender a crear una red neuronal para detectar valores atípicos en un conjunto de datos generado con GitHub Copilot.

## Objetivo visual:

![Objetivo visual](../images/cap6_obj_vis.png) 

## Duración aproximada:

- 50 minutos.

---

**[⬅️ Atrás](https://netec-mx.github.io/IA-ML/Cap%C3%ADtulo5/Lab5.html)** | **[Lista General](https://netec-mx.github.io/IA-ML/)** 

---

## Instrucciones:

### Tarea 1. Creación del archivo de trabajo.

Paso 1. Crea un nuevo archivo seleccionando "File".

![imagen resultado](../images/cap6_1.png)

Paso 2. Selecciona "Open Folder".

![imagen resultado](../images/cap6_2.png)

Paso 3. Elige una carpeta de trabajo.

![imagen resultado](../images/cap6_3.png)

Paso 4. Abre el Explorador y crear un nuevo archivo.

![imagen resultado](../images/cap6_4.png)

Paso 5. Crea un archivo llamado `Red_neuronal.py`, además, verifica que tenga la extensión `.py`.

![imagen resultado](../images/cap6_5.png)

### Tarea 2. Generación del conjunto de datos.

Paso 6. Abre el chat de Copilot.

![imagen resultado](../images/cap6_6.png)

Paso 7. Coloca el siguiente prompt en la caja de texto:

   `Proporciona una guía detallada, paso a paso, para desarrollar un algoritmo basado en redes neuronales capaz de detectar valores atípicos en un conjunto de datos. Incluye la preparación de datos, la arquitectura de la red, el proceso de entrenamiento y la evaluación del modelo.`

![imagen resultado](../images/cap6_7.png)

Paso 8. Lee el proceso recomendado.

![imagen resultado](../images/cap6_8.png)

Paso 9. Solicita la generación de un conjunto de datos con valores atípicos usando el siguiente prompt:

   `Crea un conjunto de datos sintéticos utilizando Python, asegurándote de incluir características con valores atípicos. Guarda este conjunto de datos en un archivo CSV. Proporcióname el código necesario para realizar esta tarea.`

![imagen resultado](../images/cap6_9.png)

Paso 10. Selecciona `Más opciones`.

![imagen resultado](../images/cap6_10.png)

Paso 11. Selecciona `Insert into New File`.

![imagen resultado](../images/cap6_11.png)

Paso 12. Guarda el archivo usando la combinación de teclas `Ctrl + S`, asígnale un nombre y almacénalo.

![imagen resultado](../images/cap6_12.png)

Paso 13. Ejecuta el código y verifica la creación del archivo CSV en el Explorador de Archivos.

![imagen resultado](../images/cap6_13.png)

### Tarea 3. Creación y entrenamiento de la red neuronal.

Paso 14. Regresa al chat de Copilot y solicita la implementación del modelo de red neuronal con el siguiente prompt:

   "Proporciona un código en Python que utilice un conjunto de datos previamente generados para entrenar una red neuronal y detectar valores atípicos. El código debe incluir:

   a. Importación de bibliotecas necesarias.
   b. Carga y preprocesamiento de los datos desde un archivo CSV.
   c. Definición y compilación del modelo de red neuronal adecuado para la detección de anomalías.
   d. Procedimiento de entrenamiento del modelo.
   e. Evaluación del modelo y ajuste de hiperparámetros si es necesario.
   f. Detección y marcado de valores atípicos en los datos de prueba.
   g. Almacenamiento de los resultados en un archivo CSV y visualización de los valores atípicos."

![imagen resultado](../images/cap6_14.png)

Paso 15. Dirígete al archivo `Red_neuronal.py` e inserta el código en el archivo de la red neuronal al seleccionar `Insert at Cursor`.

![imagen resultado](../images/cap6_15.png)

Paso 16. Ejecuta el código y visualiza los resultados.

![imagen resultado](../images/cap6_16.png)

### Tarea 4. Visualización y análisis de resultados.

Paso 17. Para generar una visualización gráfica de los valores atípicos detectados, solicita a Copilot el siguiente código:

   "Después de entrenar la red neuronal, genera una visualización gráfica de los valores atípicos detectados en el conjunto de datos. Incluye el código en Python necesario para crear esta gráfica y explica cómo interpretar los resultados mostrados."

![imagen resultado](../images/cap6_17.png)

Paso 18. Inserta el nuevo código generado en la parte final del script.

![imagen resultado](../images/cap6_18.png)

Paso 19. Ejecuta el código y verifica que la predicción de valores atípicos se realiza correctamente.

![imagen resultado](../images/cap6_19.png)

Paso 20. Revisa el Explorador de Archivos para confirmar la creación del archivo con los valores atípicos.

![imagen resultado](../images/cap6_20.png)

Paso 21. Verifica que el archivo contiene una columna adicional indicando los valores atípicos detectados.

![imagen resultado](../images/cap6_21.png)

Paso 22. Si aún observas valores atípicos no detectados, se recomienda mejorar la red neuronal mediante pruebas con distintas arquitecturas y ajustes de hiperparámetros. El uso de GitHub Copilot permite optimizar este proceso, reduciendo el tiempo de implementación y prueba.

## Resultado esperado:

![imagen resultado](../images/cap6_resultado_final.png)
