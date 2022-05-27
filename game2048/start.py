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
with open(working_directory + '/config.json', 'r') as f:
    CONF = json.load(f)
LOCAL = os.environ.get('S3_URL', 'local')
dash_intervals = CONF['intervals']
dash_intervals['refresh'] = dash_intervals['refresh_sec'] * 1000
dash_intervals['next'] = dash_intervals['refresh_sec'] + 180
LOWEST_SPEED = 50

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
    LOWEST_SPEED = 75
else:
    print('Unknown environment. Only show.py script is functional here. Check "Environment" notes in readme.md file')


def time_suffix():
    return ''.join([v for v in str(datetime.now()) if v.isnumeric()])[4:]


def next_time():
    return str(datetime.utcnow() + timedelta(seconds=dash_intervals['next']))


def temp_local_name(name):
    body, ext = name.split('.')
    if body[1] == '/':
        body = body[2:]
    return f'{body}_{time_suffix()}.{ext}', ext


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


class Logger:
    msg = {
        'welcome': "Welcome! Let's do something interesting. Choose MODE of action!",
        'stop': 'Process terminated by user',
        'training': 'Current process: training agent',
        'testing': 'Current process: collecting agent statistics'
    }

    def __init__(self, log_file):
        self.file = log_file
        if self.file not in list_names_s3():
            save_s3('', self.file)

    def add(self, text):
        if text:
            now = load_s3(self.file) or ''
            save_s3(now + '\n' + str(text), self.file)


def make_empty_status():
    save_s3({'logs': {}, 'proc': {}, 'occupied_agents': []}, 'status.json')


def add_status(key, value):
    status: dict = load_s3('status.json')
    status[key][value] = {
        'parent': str(os.getpid()),
        'finish': next_time()
    }
    save_s3(status, 'status.json')


def delete_status(key, value):
    status: dict = load_s3('status.json')
    status[key].pop(value, None)
    save_s3(status, 'status.json')


def kill_process(data, delete=True):
    if not data:
        return
    pid = data['pid']
    if pid and psutil.pid_exists(int(pid)):
        psutil.Process(int(pid)).kill()
    if delete:
        status: dict = load_s3('status.json')
        status['proc'].pop(pid, None)
        save_s3(status, 'status.json')


def vacuum_cleaner(parent):
    time.sleep(dash_intervals['vc'])
    while True:
        status: dict = load_s3('status.json')
        my_tags = 0
        now = datetime.utcnow()

        for key in status:
            to_delete = []
            for value in status[key]:
                finish = parser.parse(status[key][value]['finish'])
                if status[key][value]['parent'] == parent:
                    my_tags += 1
                if now > finish:
                    if key == 'logs':
                        delete_s3(value)
                    elif key == 'proc' and status[key][value]['parent'] == parent:
                        kill_process({'pid': value}, delete=False)
                    to_delete.append(value)
            for v in to_delete:
                if v in status[key]:
                    del status[key][v]

        save_s3(status, 'status.json')
        if not my_tags:
            sys.exit()
        time.sleep(dash_intervals['vc'])
