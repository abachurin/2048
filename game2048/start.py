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
import boto3
from botocore.errorfactory import ClientError
from botocore.client import Config

if os.environ.get('S3_URL', 'local') == 'local':
    with open('config.json', 'r') as f:
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

rows = [1, 1, 2, 2, 0]
cols = [0, 1, 2, 3, 4]
x = np.random.random((4, 10))
y = x.tolist()

now = time.time()
for _ in range(100000):
    rows = [0, 1, 2, 3]
    cols = [7, 3, 2, 5]
    s = sum([y[i][f] for i, f in enumerate(cols)])
print(time.time() - now)

now = time.time()
for _ in range(100000):
    rows = [0, 1, 2, 3]
    cols = [7, 3, 2, 5]
    s = sum([y[i][cols[i]] for i in range(4)])
print(time.time() - now)

