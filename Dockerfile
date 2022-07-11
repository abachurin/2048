FROM python:3.10.5-slim-buster

WORKDIR /code
COPY requirements.txt ./requirements.txt
RUN pip3 install -r requirements.txt
COPY . .

ENV S3_URL="AWS"

ENV PYTHONPATH /code
RUN export PYTHONPATH=.
ENTRYPOINT ["python3"]
CMD ["main.py"]
