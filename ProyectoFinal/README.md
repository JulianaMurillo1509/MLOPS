# MLOPS_PUJ
Repositorio con ejercicios y proyectos de la case de tópicos avanzados de tópicos de IA

# Estructura del repositorio
* ubicarse en la carpeta Proyectofinal

- Proyecto : Proyecto despliega airflow y servicios en arquitectura micoservicios
= entrar a la carpeta airflow/simple_dag
- correr docker compose up -d
- volver a la carpeta principal ProyectoFinal
- si el sevicio de mlflow no esta registrado correr:   sudo systemctl enable /home/estudiante/repo/MLOPS/ProyectoFinal/mlflow.service
- levantar servicio sudo systemctl start mlflow.service 
- correr docker compose up -d
