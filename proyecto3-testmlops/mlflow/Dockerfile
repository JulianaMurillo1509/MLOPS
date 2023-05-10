FROM python:3.7.4
ARG RUN_ID
ARG MODEL_NAME
ENV SERVER_PORT 5000
ENV SERVER_HOST 0.0.0.0
ENV FILE_STORE /opt/mlflow/fileStore
ENV ARTIFACT_STORE /opt/mlflow/artifactStore
ENV PYTHONPATH /opt/mlflow/utils
RUN mkdir -p /opt/mlflow/scripts \
&& mkdir -p ${FILE_STORE} \
&& mkdir -p ${ARTIFACT_STORE}
RUN pip install pandas\
&& pip install mlflow \
&& apt-get update
RUN mkdir /work
WORKDIR /work
COPY ./mlflow .
RUN pip3 install --no-cache-dir --upgrade -r requirements_mlflow.txt
# Copy over artifact and code

COPY ./mlflow /opt/mlflow/scripts/
RUN chmod +x /opt/mlflow/scripts/run.sh
ENTRYPOINT ["/usr/bin/env"]
CMD ["bash", "/opt/mlflow/scripts/run.sh"]