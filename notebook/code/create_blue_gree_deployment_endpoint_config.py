import boto3
import logging
import json

logger = logging.getLogger()
logger.setLevel(logging.INFO)
sm_client = boto3.client('sagemaker')

#Retrieve endpoint existence and status info.
def lambda_handler(event, context):
    logger.info(f"Input parameters: {json.dumps(event)}")

    endpoint_config_name = get_value("EndpointConfigName", event)

    dev_endpoint_name = get_value('DevEndpointName', event)
    dev_endpoint_config = get_endpoint_config(dev_endpoint_name)
    dev_model_name = dev_endpoint_config['ProductionVariants'][0]['ModelName']

    prd_endpoint_name = get_value('PrdEndpointName', event)
    prd_endpoint_config = get_endpoint_config(prd_endpoint_name)
    prd_model_name = prd_endpoint_config['ProductionVariants'][0]['ModelName']

    sm_client.create_endpoint_config(
        EndpointConfigName = endpoint_config_name,
        ProductionVariants = [
            {
                'VariantName': 'blue-variant',
                'ModelName': prd_model_name,
                'InitialInstanceCount': 1,
                'InstanceType': 'ml.m5.large',
                'InitialVariantWeight': 9
            },    
            {
                'VariantName': 'green-variant',
                'ModelName': dev_model_name,
                'InitialInstanceCount': 1,
                'InstanceType': 'ml.m5.large',
                'InitialVariantWeight': 1
        }
        ]
    )

    return {
        'dev_prd_model_in_sync': (dev_model_name == prd_model_name),
        'blue_model_name': prd_model_name,
        'green_model_name': dev_model_name
    }

def get_value(key, event):
    if (key in event):
        return event[key]
    else:
        raise KeyError(f'{key} key not found in function input!'+
                      f' The input received was: {json.dumps(event)}.')

def get_endpoint_config(endpoint_name):
    response = sm_client.describe_endpoint(EndpointName = endpoint_name)
    endpoint_config_name = response['EndpointConfigName']
    return sm_client.describe_endpoint_config(EndpointConfigName = endpoint_config_name)
