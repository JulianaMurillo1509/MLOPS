# MLOPS_PUJ
Ejemplo de feature store usando feast

El objetivo principal de un Feature Store es proporcionar una única fuente de verdad para las características utilizadas en el ciclo de vida de un modelo de machine learning. Esto implica almacenar y gestionar las características de manera centralizada, garantizando la consistencia y calidad de los datos, y permitiendo su fácil acceso y reutilización.

un Feature Store en machine learning y MLOps sirve para gestionar y organizar las características utilizadas en los modelos de ML, asegurando de cierta forma su calidad, facilitando su acceso ya que no requiere conexion a base de datos "off-line" y reutilización, y mejorando la eficiencia y la colaboración en el desarrollo y despliegue de modelos

en conclusion es una capa adicional de abstraccion de datos.

# Instrucciones para ver el ejemplo de feature store


- entrar a la carpeta /home/estudiante/repo/MLOPS/taller2
- levantar jupyterlab> jupyter lab --ip 0.0.0.0 --port=8181
- ingresar a la url http://10.43.102.111:8181/lab
- borrar la carpeta "my_project" si existe y correr el notebook
- ingresar a la UI de feast http://10.43.102.111:8888/p/my_project/data-source
