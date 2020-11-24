import boto3
import logging
import json

logger = logging.getLogger()
logger.setLevel(logging.INFO)
sm_client = boto3.client('sagemaker')

#Retrieve Hyperparameters Tuning Job infomation.
def lambda_handler(event, context):

    if ('HpoJobName' in event):
        hpo_job_name = event['HpoJobName']

    else:
        raise KeyError('HpoJobName key not found in function input!'+
                      ' The input received was: {}.'.format(json.dumps(event)))

    #Query boto3 API to check training status.
    response = sm_client.describe_hyper_parameter_tuning_job(HyperParameterTuningJobName = hpo_job_name)
    best_training_job = response['BestTrainingJob']
    final_hpo_objective_metric = best_training_job['FinalHyperParameterTuningJobObjectiveMetric']

    response = sm_client.describe_training_job(TrainingJobName = best_training_job['TrainingJobName'])
    logger.info("HPO Job:{} 's best training job':{}.".format(hpo_job_name, best_training_job['TrainingJobName']))
    


    return {
        'statusCode': 200,
        'bestTrainingJob': {
            'trainingJobName': response['TrainingJobName'],
            'finalHyperParameterTuningJobObjectiveMetric': final_hpo_objective_metric,
            'modelArtifacts': response['ModelArtifacts'],
            'hyperParameters': response['HyperParameters']
        }
    }