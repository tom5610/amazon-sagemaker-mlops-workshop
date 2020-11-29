import boto3
import logging
import json

logger = logging.getLogger()
logger.setLevel(logging.INFO)
sm_client = boto3.client('sagemaker')

#Retrieve endpoint existence and status info.
def lambda_handler(event, context):

    endpoint_config_name = get_value("EndpointConfigName", event)

    dev_endpoint_name = get_value('DevEndpointName', event)
    dev_endpoint_config = get_endpoint_config(dev_endpoint_name)
    dev_model_name = dev_endpoint_config['ProductionVariants'][0]['ModelName']

    prd_endpoint_name = get_value('PrdEndpointName', event)
    prd_endpoint_config = get_endpoint_config(prd_endpoint_name)
    prd_model_name = prd_endpoint_config['ProductionVariants'][0]['ModelName']

    sm.create_endpoint_config(
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
        'endpoint_existed': endpoint_existed,
        'endpoint_status': endpoint_status
    }

def get_value(key, event):
    if (key in event):
        dev_endpoint_name = event[key]
    else:
        raise KeyError(f'{key} key not found in function input!'+
                      ' The input received was: {}.'.format(json.dumps(event)))

def get_endpoint_config(endpoint_name):
    response = sm_client.describe_endpoint(EndpointName = endpoint_name)
    endpoint_config_name = response['EndpointConfigName']
    return sm_client.describe_endpoint_config(EndpointConfigName = endpoint_config_name)
