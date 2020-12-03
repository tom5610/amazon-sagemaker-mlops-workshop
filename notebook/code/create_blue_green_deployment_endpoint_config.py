import boto3
import logging
import json

logger = logging.getLogger()
logger.setLevel(logging.INFO)
sm_client = boto3.client('sagemaker')
BLUE_VARIANT = 'blue-variant'
GREEN_VARIANT = 'green-variant'

#Retrieve endpoint existence and status info.
def lambda_handler(event, context):
    logger.info(f"Input parameters: {json.dumps(event)}")

    endpoint_config_name = get_value("EndpointConfigName", event)

    dev_endpoint_name = get_value('DevEndpointName', event)
    dev_model_name = get_model_on_endpoint(dev_endpoint_name)

    prd_endpoint_name = get_value('PrdEndpointName', event)
    prd_model_name = get_model_on_endpoint(prd_endpoint_name)
    
    logger.info(f"prd_model_name: {prd_model_name}; dev_model_name: {dev_model_name}")
    production_variants = get_production_variants(prd_model_name, dev_model_name)
    logger.info(f"length of production_variants: {len(production_variants)}")
    
    if len(production_variants) == 0:
        raise Exception("Dev or Prod Endpoint does not exist, Blue/Green deployment aborted!")


    sm_client.create_endpoint_config(
        EndpointConfigName = endpoint_config_name,
        ProductionVariants = production_variants
    )

    return {
        'dev_prd_model_in_sync': (dev_model_name == prd_model_name),
        'blue_model_name': prd_model_name,
        'green_model_name': dev_model_name
    }

def get_production_variant(variant_name, model_name, initial_variant_weight):
    return None if model_name == None else {
        'VariantName': variant_name,
        'ModelName': model_name,
        'InitialInstanceCount': 1,
        'InstanceType': 'ml.m5.large',
        'InitialVariantWeight': initial_variant_weight
    }

def get_production_variants(prd_model_name, dev_model_name):
    production_variants = []
    # always set production environment model as the first one.
    if prd_model_name:
        production_variants.append(get_production_variant(BLUE_VARIANT, prd_model_name, 9)) # setup weight as 9 for initial blue/green setting
    if dev_model_name:
        production_variants.append(get_production_variant(GREEN_VARIANT, dev_model_name, 1))
    return production_variants  

def get_value(key, event):
    if (key in event):
        return event[key]
    else:
        raise KeyError(f'{key} key not found in function input!'+
                      f' The input received was: {json.dumps(event)}.')

def get_model_on_endpoint(endpoint_name):
    response = sm_client.list_endpoints(NameContains = endpoint_name)
    if len(response['Endpoints']) == 0:
        model_name = None
    else:
        response = sm_client.describe_endpoint(EndpointName = endpoint_name)
        endpoint_config_name = response['EndpointConfigName']
        endpoint_config = sm_client.describe_endpoint_config(EndpointConfigName = endpoint_config_name)
        model_name = endpoint_config['ProductionVariants'][0]['ModelName']
    return model_name