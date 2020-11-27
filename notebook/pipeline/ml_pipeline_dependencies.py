import boto3
import time
import re
import uuid
import argparse
from datetime import datetime
import os, urllib.request

import stepfunctions
from stepfunctions.inputs import ExecutionInput
from stepfunctions.steps.sagemaker import *
from stepfunctions.steps.states import *
from stepfunctions.steps.compute import *
from stepfunctions.workflow import Workflow
from stepfunctions.steps import *
from IPython.display import display, HTML, Javascript

import sagemaker
from sagemaker import get_execution_role
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.tuner import IntegerParameter, CategoricalParameter, ContinuousParameter, HyperparameterTuner
from sagemaker.analytics import ExperimentAnalytics
from sagemaker.inputs import TrainingInput

from smexperiments.experiment import Experiment
from smexperiments.trial import Trial
from smexperiments.trial_component import TrialComponent
from smexperiments.tracker import Tracker
from smexperiments.search_expression import Filter, Operator, SearchExpression

session = boto3.Session()
sm = session.client('sagemaker')
sagemaker_session = sagemaker.Session()
sagemaker_execution_role = get_execution_role()

region = session.region_name
account_id = session.client('sts').get_caller_identity().get('Account')

TRAINED_MODEL_URI = "https://df4l9poikws9t.cloudfront.net/model/xgboost-direct-marketing/model.tar.gz"
S3_KEY_TRAINED_MODEL = "sagemaker/model/model.tar.gz"

def setup_trained_model(bucket_name, s3_key_trained_model):
    # upload existing model artifact to working bucket
    s3 = boto3.client('s3')

    os.makedirs('model', exist_ok=True)
    urllib.request.urlretrieve(TRAINED_MODEL_URI, 'model/model.tar.gz')
    s3.upload_file('model/model.tar.gz', bucket_name, s3_key_trained_model)
    
def main(bucket_name):
    setup_trained_model(bucket_name, S3_KEY_TRAINED_MODEL)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load parameters")
    parser.add_argument("--bucket-name", required = True)
    args = vars(parser.parse_args())
    print("args: {}".format(args))
    main(**args)
    