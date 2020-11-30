import boto3
import time
import re
import uuid
import argparse
from datetime import datetime, date, timedelta
import os, urllib.request
import pandas as pd
import numpy as np

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

def display_state_machine_advice(workflow_name, execution_id):
    display(HTML(f'''<br>The Step Function workflow "{workflow_name}" is now executing... 
            <br>To view state machine in the console click 
            <a target="_blank" href="https://{region}.console.aws.amazon.com/states/home?region={region}#/statemachines/view/arn:aws:states:ap-southeast-2:{account_id}:stateMachine:{workflow_name}">State Machine</a> 
            <br>To view execution in the console click 
            <a target="_blank" href="https://{region}.console.aws.amazon.com/states/home?region={region}#/executions/details/arn:aws:states:ap-southeast-2:{account_id}:execution:{workflow_name}:{execution_id}">Execution</a>.
        '''))

def display_training_job_advice(training_job_name):
    display(HTML(f'''<br>The training job "{training_job_name}" is now running. 
        To view it in the console click 
        <a target="_blank" href="https://console.aws.amazon.com/sagemaker/home?region={region}#/jobs">here</a>.
    '''))  
    
def display_codepipeline_advice(code_pipeline_name):
    display(HTML(f'''<br>CodePipeline process "{code_pipeline_name}" will be kicked off shortly. 
        To view it in the console click 
        <a target="_blank" href="https://{region}.console.aws.amazon.com/codesuite/codepipeline/pipelines/{code_pipeline_name}/view?region={region}">here</a>.
    '''))  
    

    
def main(bucket_name):
    setup_trained_model(bucket_name, S3_KEY_TRAINED_MODEL)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load parameters")
    parser.add_argument("--bucket-name", required = True)
    args = vars(parser.parse_args())
    print("args: {}".format(args))
    main(**args)
    