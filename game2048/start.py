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
