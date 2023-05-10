FROM python:3.9
RUN mkdir /work
WORKDIR /work
COPY ./simple_UI .
RUN ls -l
RUN pip3 install --no-cache-dir --upgrade -r requirements_api.txt

COPY . .

EXPOSE 81

CMD ["streamlit", "run", "main.py"]