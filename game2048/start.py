import numpy as np
import time
import sys
from pprint import pprint
import random
import pickle
import json
import matplotlib.pyplot as plt
from functools import partial
from collections import deque
import os
import dash
from dash import no_update as NUP
import dash_auth
from dash import dash_table, dcc, html
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
from dash_extensions.enrich import DashProxy, MultiplexerTransform, Output, Input, State
from dash_extensions import Monitor
from dash.dash_table.Format import Format, Scheme, Trim
import plotly.express as px
import flask
import time
import boto3
from botocore.errorfactory import ClientError
from botocore.client import Config

working_directory = os.path.dirname(os.path.realpath(__file__))
if os.environ.get('S3_URL', 'local') == 'local':
    with open(os.path.join(working_directory, 'config.json'), 'r') as f:
        s3_credentials = json.load(f)['s3_credentials']
    with open(s3_credentials, 'r') as f:
        df = json.load(f)
    s3_engine = boto3.resource(
        service_name='s3',
        region_name='eu-west-1',
        aws_access_key_id=df['access_key'],
        aws_secret_access_key=df['secret_key']
    )
    s3_bucket_name = 'ab2048'
else:
    pass
s3_bucket = s3_engine.Bucket(s3_bucket_name)


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
    if not is_data_there(name):
        return 'no file'
    ext = name.split('.')[1]
    temp = f'temp.{ext}'
    s3_bucket.download_file(name, temp)
    if ext == 'json':
        with open(temp, 'r', encoding='utf-8') as f:
            result = json.load(f)
    elif ext == 'txt':
        with open(temp, 'r') as f:
            result = f.readlines()
    elif ext == 'pkl':
        with open(temp, 'rb') as f:
            result = pickle.load(f)
    else:
        result = '?'
    os.remove(temp)
    return result


def save_s3(data, name):
    ext = name.split('.')[1]
    temp = f'temp.{ext}'
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
