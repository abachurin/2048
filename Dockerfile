FROM python:3.11-slim-buster

WORKDIR /code
COPY requirements.txt ./requirements.txt
RUN apt-get update -y && apt-get install -y gcc
RUN pip3 install -r requirements.txt
COPY . .

ENV S3_URL="AWS"

ENV PYTHONPATH /code
RUN export PYTHONPATH=.
ENTRYPOINT ["python3"]
CMD ["application.py"]
