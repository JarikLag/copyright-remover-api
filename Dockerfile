FROM python:3.7

EXPOSE 8083


COPY ./requirements.txt /inference-api/requirements.txt
RUN pip install -r /inference-api/requirements.txt

COPY ./models.zip /inference-api/
RUN unzip /inference-api/models.zip
RUN rm /inference-api/models.zip

COPY ./application/* /inference-api/application/
COPY server.py /inference-api/server.py
COPY ./config.yml /inference-api/config.yml

WORKDIR /inference-api/

CMD [ "python", "/inference-api/server.py" ]
