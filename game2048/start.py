from datetime import datetime, timedelta
import numpy as np
import time
import sys
from pprint import pprint
import random
import pickle
import json
from collections import deque
import os
import time
import boto3
import base64
from multiprocessing import Process
from threading import Thread
import psutil
from dateutil import parser

working_directory = os.path.dirname(os.path.realpath(__file__))
with open(os.path.join(working_directory, 'config.json'), 'r') as f:
    CONF = json.load(f)
LOCAL = os.environ.get('S3_URL', 'local')
dash_intervals = CONF['intervals']
dash_intervals['refresh'] = dash_intervals['refresh_sec'] * 1000
dash_intervals['check_run'] = dash_intervals['refresh_sec'] * 2
dash_intervals['vc'] = dash_intervals['vc_sec'] * 1000
dash_intervals['next'] = dash_intervals['refresh_sec'] + 180
LOWEST_SPEED = 50

GAME_PANE = {}
AGENT_PANE = {}
RUNNING = {}

s3_bucket_name = 'ab2048'
if LOCAL == 'local':
    s3_credentials = CONF['s3_credentials']
    with open(s3_credentials, 'r') as f:
        df = json.load(f)
    s3_engine = boto3.resource(
        service_name='s3',
        region_name='eu-west-1',
        aws_access_key_id=df['access_key'],
        aws_secret_access_key=df['secret_key']
    )
    s3_bucket = s3_engine.Bucket(s3_bucket_name)
elif LOCAL == 'AWS':
    s3_engine = boto3.resource('s3')
    s3_bucket = s3_engine.Bucket(s3_bucket_name)
    LOWEST_SPEED = 100
else:
    print('Unknown environment. Only show.py script is functional here. Check "Environment" notes in readme.md file')


def time_suffix(precision=1):
    return ''.join([v for v in str(datetime.utcnow()) if v.isnumeric()])[4:-precision]


def next_time():
    return str(datetime.utcnow() + timedelta(seconds=dash_intervals['next']))


def temp_local_name(name):
    body, ext = name.split('.')
    return f'temp{time_suffix()}.{ext}', ext


def list_names_s3():
    return [o.key for o in s3_bucket.objects.all()]


def is_data_there(name):
    return name in list_names_s3()


def copy_inside_s3(src, dst):
    s3_bucket.copy({'Bucket': s3_bucket_name, 'Key': src}, dst)


def delete_s3(name):
    if is_data_there(name):
        s3_engine.Object(s3_bucket_name, name).delete()


def load_s3(name):
    if not name or (not is_data_there(name)):
        return
    temp, ext = temp_local_name(name)
    s3_bucket.download_file(name, temp)
    if ext == 'json':
        with open(temp, 'r', encoding='utf-8') as f:
            result = json.load(f)
    elif ext == 'txt':
        with open(temp, 'r') as f:
            result = f.read()
    elif ext == 'pkl':
        with open(temp, 'rb') as f:
            result = pickle.load(f)
    else:
        result = None
    os.remove(temp)
    return result


def save_s3(data, name):
    temp, ext = temp_local_name(name)
    if ext == 'json':
        with open(temp, 'w') as f:
            json.dump(data, f)
    elif ext == 'txt':
        with open(temp, 'w') as f:
            f.write(data)
    elif ext == 'pkl':
        with open(temp, 'wb') as f:
            pickle.dump(data, f, -1)
    else:
        return 0
    s3_bucket.upload_file(temp, name)
    os.remove(temp)
    return 1


def add_status(key, value, parent):
    status: dict = load_s3('status.json')
    status[key][value] = {
        'parent': parent,
        'finish': next_time()
    }
    save_s3(status, 'status.json')


def memory_usage_line():
    memo = psutil.virtual_memory()
    mb = 1 << 20
    return f'{str(datetime.now())[11:]} | total: {int(memo.total / mb)} | used: {int(memo.used / mb)} | ' \
           f'available: {int(memo.available / mb)}\n'


def add_to_memo(text):
    memo_text = load_s3('memory_usage.txt')
    memo_text += text
    save_s3(memo_text, 'memory_usage.txt')


class Logger:
    msg = {
        'welcome': "Welcome! Let's do something interesting. Choose MODE of action!",
        'collapse': 'Current process collapsed!'
    }

    def __init__(self, log_file):
        self.file = log_file
        if self.file not in list_names_s3():
            save_s3('', self.file)

    def add(self, text):
        if text:
            now = load_s3(self.file) or ''
            save_s3(now + '\n' + str(text), self.file)
