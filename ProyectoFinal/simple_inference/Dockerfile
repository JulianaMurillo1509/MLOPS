FROM python:3.9
RUN mkdir /work
WORKDIR /work
COPY ./simple_inference .
RUN ls -l
RUN pip3 install --no-cache-dir --upgrade -r requirements_api.txt
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "83"]