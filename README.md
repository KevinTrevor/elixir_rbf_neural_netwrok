# RBF Project - Asperger Spectrum Classificator

Proyecto universitario para la UDONE con la finalidad de clasificar tests de Asperger de la página EspectroAutista.info a través de una Red Neuronal de Base Radial.

### ¿Cómo es la estructura de este proyecto?

Empecemos por las carpetas. Tenemos tres carpetas con archivos .csv:
  1. base: Esta carpeta posee los archivos que actúan como la base de conocimiento.
     - hidden_layer.csv: Este archivo posee los datos de las neuronas de la capa oculta de la red neuronal.
     - output_layer.csv: Este archivo posee los datos de las neuronas de la capa de salida de la red neuronal.
  2. input: Esta carpeta posee el archivo que actúa como interfaz entre la red neuronal y el usuario.
     - inputs.csv: Este archivo posee datos de entrada (tests) para la red neuronal.
  3. training: Esta carpeta posee los archivos que se utilizan para el entrenamiento de la red.
     - dataset.csv: Este archivo posee los patrones de ejemplo para entrenar la red neuronal.
     - expected.csv: Este archivo posee las salidas esperadas de los patrones de ejemplo.
    
Es importante tener esta organización de archivos para que el algoritmo funcione.

### ¿Cómo usar este proyecto?

Una vez tenemos todo organizado, podemos empezar a ejecutar el proyecto siguiendo los siguientes pasos:

1. Ejecutar el comando **elixirc radial.ex** estando en la raíz del proyecto.
2. Ejecutar el comando **iex**.
3. Una vez dentro del Shell de Elixir, podemos llamar al módulo NetworkApplication; en específico a su método run(), de la siguiente forma: **NetworkApplication.run()**
4. Tras ejecutar exitosamente, obtendrás una respuesta **:ok** y se generará un archivo **output.csv** en la carpeta raíz, donde se encontrarán las respuesta de la red neuronal en base a los datos de entrada del archivo **inputs.csv**
