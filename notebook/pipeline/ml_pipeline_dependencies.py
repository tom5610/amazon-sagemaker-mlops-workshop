import boto3
import time
import re
import uuid
import argparse
from datetime import datetime

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
bucket_name = f'xgboost-direct-marketing-{account_id}-{region}'
