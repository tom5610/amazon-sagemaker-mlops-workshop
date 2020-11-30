import json
from datetime import datetime
import argparse

def create_worflow_input_file(
    dev_endpoint_name,
    prd_endpoint_name,
    query_endpoint_lambda_function_name,
    create_bg_deployment_endpoint_config_lambda_function_name,
    input_file_path
):

    # execute workflow
    suffix = datetime.now().strftime("%y%m%d-%H%M")
    endpoint_config_name = f"dm-prd-endpoint-config-{suffix}"

    inputs = {
        "EndpointConfigName": endpoint_config_name,
        "DevEndpointName": dev_endpoint_name,
        "PrdEndpointName": prd_endpoint_name,
        "LambdaFunctionNameOfQueryEndpoint": query_endpoint_lambda_function_name,
        "LambdaFunctionNameOfCreateBlueGreenDeploymentEndpointConfig": create_bg_deployment_endpoint_config_lambda_function_name
    }

    with open(input_file_path, "w") as outfile:  
        json.dump(inputs, outfile) 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load parameters")
    parser.add_argument("--dev-endpoint-name", required = True)
    parser.add_argument("--prd-endpoint-name", required = True)
    parser.add_argument("--query-endpoint-lambda-function-name", required = True)
    parser.add_argument("--create-bg-deployment-endpoint-config-lambda-function-name", required = True)
    parser.add_argument("--input-file-path", required = True)
    
    args = vars(parser.parse_args())
    print("args: {}".format(args))
    create_worflow_input_file(**args)