import boto3
import logging
import json

logger = logging.getLogger()
logger.setLevel(logging.INFO)
sm_client = boto3.client('sagemaker')

#Retrieve transform job name from event and return transform job status.
def lambda_handler(event, context):

    if ('EndpointName' in event):
        endpoint_name = event['EndpointName']

    else:
        raise KeyError('EndpointName key not found in function input!'+
                      ' The input received was: {}.'.format(json.dumps(event)))

    #Query boto3 API to check training status.
    endpoint_existed = False
    endpoint_status = None
    try:
        response = sm_client.describe_endpoint(EndpointName = endpoint_name)
        endpoint_existed = True
        endpoint_status = response['EndpointStatus']
        logger.info("Endpoint:{} has status:{}.".format(endpoint_name, endpoint_status))
    except Exception as e:
        response = ('Failed to read endpoint info!'+ 
                    ' The endpoint may not exist or the endpoint name may be incorrect.'+ 
                    ' Check SageMaker to confirm the endpoint name.')
        print(e)
        print('{} Attempted to read endpoint name: {}.'.format(response, endpoint_name))

    return {
        'statusCode': 200,
        'endpoint_existed': endpoint_existed,
        'endpoint_status': endpoint_status
    }