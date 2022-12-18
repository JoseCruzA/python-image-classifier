import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import math
import numpy as np

TAMAÑO_LOTE = 32

datos, metadatos = tfds.load('fashion_mnist', as_supervised = True , with_info = True)

datos_entrenamiento, datos_pruebas = datos['train'], datos['test']
nombres_clases = metadatos.features['label'].names
num_ej_entrenamiento = metadatos.splits['train'].num_examples
num_ej_pruebas = metadatos.splits['test'].num_examples

#Normalizar los datos ( Pasar de 0-255 a 0-1)
def normalizar(imagenes, etiquetas):
    imagenes = tf.cast(imagenes, tf.float32)
    imagenes /= 255 #Pasa de 0-255 a 0-1
    return imagenes, etiquetas

#Normalizar los datos de entrenamiento y pruebas con la funcion que hicimos
datos_entrenamiento = datos_entrenamiento.map(normalizar)
datos_pruebas = datos_pruebas.map(normalizar)
    
#Agregar a cache (usar memoria en lugar de disco, entrenamiento más rápido)
datos_entrenamiento = datos_entrenamiento.cache()
datos_pruebas = datos_pruebas.cache()

#Mostrar una imagen de los datos de pruebas, de momentos mostremos la primera
"""for imagen, etiqueta in datos_entrenamiento.take(1):
    break
imagen = imagen.numpy().reshape((28,28))

#Dibujar la imagen
plt.figure()
plt.imshow(imagen, cmap = plt.cm.binary)
plt.colorbar()
plt.grid(False)
plt.show()

#Muestra las 25 primeras imagenes con la clase
plt.figure(figsize=(10,10))
for i, (imagen,etiqueta) in enumerate(datos_entrenamiento.take(25)):
    imagen = imagen.numpy().reshape((28,28))
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(imagen, cmap=plt.cm.binary)
    plt.xlabel(nombres_clases[etiqueta])
plt.show()"""

#Crear el modelo
modelo =  tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28,1)), # 1- blanco y negro
    tf.keras.layers.Dense(50,activation=tf.nn.relu),
    tf.keras.layers.Dense(50,activation=tf.nn.relu),
    tf.keras.layers.Dense(10,activation=tf.nn.softmax) #Para redes de clasificación
])

#Compilar el modelo
modelo.compile(
    optimizer = 'adam',
    loss = tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics = ['accuracy']
)

#Entrenar el modelo
datos_entrenamiento = datos_entrenamiento.repeat().shuffle(num_ej_entrenamiento).batch(TAMAÑO_LOTE)
datos_pruebas = datos_pruebas.batch(TAMAÑO_LOTE)


#Entrenamiento
historial = modelo.fit(datos_entrenamiento, epochs = 5, steps_per_epoch = math.ceil(num_ej_entrenamiento/TAMAÑO_LOTE))

#Gráfica de la magnitud de perdida
"""plt.xlabel('Epoca')
plt.ylabel('Magnitud de perdida')
plt.plot(historial.history['loss'])
plt.show()"""

for imagenes_prueba, etiquetas_prueba in datos_pruebas.take(1):
    imagenes_prueba = imagenes_prueba.numpy()
    etiquetas_prueba = etiquetas_prueba.numpy()
    predicciones =modelo.predict(imagenes_prueba)
    
def graficar_imagen(i, arr_predicciones, etiquetas_reales, imagenes):
    arr_predicciones, etiquetas_reales, imagen = arr_predicciones[i], etiquetas_reales[i], imagenes[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(imagen[...,0], cmap=plt.cm.binary)
    
    predicciones_etiqueta = np.argmax(arr_predicciones)
    if predicciones_etiqueta == etiquetas_reales:
        color = 'blue' #si le atino
    else:
        color = 'red'  #fallo
    
    plt.xlabel("{} {:2.0f}% ({})".format(
        nombres_clases[predicciones_etiqueta],
        100*np.max(arr_predicciones),
        nombres_clases[etiquetas_reales],
        color=color
    ))

def graficar_valor_arreglo(i, arr_predicciones, etiqueta_real):
    arr_predicciones, etiqueta_real = arr_predicciones[i],etiqueta_real[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    grafica = plt.bar(range(10), arr_predicciones, color="#777777")
    plt.ylim([0,1])

    etiqueta_prediccion = np.argmax(arr_predicciones)

    grafica[etiqueta_prediccion].set_color('red')
    grafica[etiqueta_real].set_color('blue')

num_filas = 5
num_columnas = 5
num_imagenes = num_filas*num_columnas
plt.figure(figsize=(2*2*num_columnas, 2*num_filas))

for i in range(num_imagenes):
    plt.subplot(num_filas, 2*num_columnas, 2*i+1)
    graficar_imagen(i, predicciones, etiquetas_prueba, imagenes_prueba)
    plt.subplot(num_filas, 2*num_columnas, 2*i+2)
    graficar_valor_arreglo(i, predicciones, etiquetas_prueba)
    
plt.show()


imagen = imagenes_prueba[5]
imagen = np.array([imagen])
prediccion = modelo.predict(imagen)
print("Prediccion: " + nombres_clases[np.argmax(prediccion[0])] )

#Exportacion del modelo a h5
modelo.save('modelo_exportado.h5')