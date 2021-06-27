# syntax=docker/dockerfile:1

FROM python:3.9.5-slim-buster

ARG DOCKER_USER_HOME=/code
WORKDIR ${DOCKER_USER_HOME}

COPY . .

COPY requirements.txt /code/requirements.txt
RUN pip3 install -r requirements.txt

ENV PYTHONPATH /code

ENTRYPOINT ["python3"]

CMD ["game2048/show.py"]
