# syntax=docker/dockerfile:1

FROM python:3.9.5-slim-buster

WORKDIR /code
COPY requirements.txt ./requirements.txt
RUN pip3 install -r requirements.txt
COPY . .

ENV S3_URL="AWS"

ENV PYTHONPATH /code
RUN export PYTHONPATH=.
ENTRYPOINT ["python3"]
CMD ["main.py"]
