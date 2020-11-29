import json
from datetime import datetime
import argparse

def create_worflow_input_file(
    require_hpo,
    require_model_training,
    query_endpoint_lambda_function_name,
    query_hpo_job_lambda_function_name,
    input_file_path
):

    suffix = datetime.now().strftime("%y%m%d-%H%M%S")
    # execution input parameter values
    preprocessing_job_name = f"dm-preprocessing-{suffix}"
    tuning_job_name = f"dm-tuning-{suffix}"
    training_job_name = f"dm-training-{suffix}"
    model_job_name = f"dm-model-{suffix}"
    endpoint_config_name = f"dm-endpoint-config-{suffix}"
    endpoint_job_name = f"direct-marketing-endpoint-dev"

    inputs = {
        "PreprocessingJobName": preprocessing_job_name,
        "ToDoHPO": str(require_hpo).lower() in ['true', '1', 'yes', 't'],
        "ToDoTraining": str(require_model_training).lower() in ['true', '1', 'yes', 't'],
        "TrainingJobName": training_job_name,
        "TuningJobName": tuning_job_name,
        "ModelName": model_job_name,
        "EndpointConfigName": endpoint_config_name,
        "EndpointName": endpoint_job_name,
        "LambdaFunctionNameOfQueryEndpoint": query_endpoint_lambda_function_name,
        "LambdaFunctionNameOfQueryHpoJob": query_hpo_job_lambda_function_name
    }

    with open(input_file_path, "w") as outfile:  
        json.dump(inputs, outfile) 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load parameters")
    parser.add_argument("--require-hpo", required = True)
    parser.add_argument("--require-model-training", required = True)
    parser.add_argument("--query-endpoint-lambda-function-name", required = True)
    parser.add_argument("--query-hpo-job-lambda-function-name", required = True)
    parser.add_argument("--input-file-path", required = True)
    
    args = vars(parser.parse_args())
    print("args: {}".format(args))
    create_worflow_input_file(**args)