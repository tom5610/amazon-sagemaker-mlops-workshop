from ml_pipeline_dependencies import *

def create_lambda_create_blue_gree_deployment_endpoint_config_step():
    lambda_step = LambdaStep(
        'Create Blue/Green Deployment Endpoint Config',
        parameters = {  
            "FunctionName.$": "$$.Execution.Input['LambdaFunctionNameOfCreateBlueGreenDeploymentEndpointConfig']",
            'Payload':{
                "EndpointConfigName.$": "$$.Execution.Input['EndpointConfigName']",
                "DevEndpointName.$": "$$.Execution.Input['DevEndpointName']",
                "PrdEndpointName.$": "$$.Execution.Input['PrdEndpointName']"
            }
        }
    )   
    return lambda_step

def create_lambda_query_endpoint_step():
    query_endpoint_lambda_step = LambdaStep(
        'Query Endpoint Info',
        parameters = {  
            "FunctionName.$": "$$.Execution.Input['LambdaFunctionNameOfQueryEndpoint']",
            'Payload':{
                "EndpointName.$": "$$.Execution.Input['PrdEndpointName']"
            }
        }
    )
    return query_endpoint_lambda_step

def create_endpoint_step(endpoint_name_placeholder, endpoint_config_name_placeholder, update = False):
    endpoint_step = EndpointStep(
        "Update Endpoint" if update else "Create Endpoint",
        endpoint_name = endpoint_name_placeholder,
        endpoint_config_name = endpoint_config_name_placeholder,
        update = update
    )
    return endpoint_step

def create_query_endpoint_deployment_lambda_step():
    query_endpoint_deployment_lambda_step = LambdaStep(
        'Query Endpoint Deployment Status',
        parameters = {  
            "FunctionName.$": "$$.Execution.Input['LambdaFunctionNameOfQueryEndpoint']",
            'Payload':{
                "EndpointName.$": "$$.Execution.Input['PrdEndpointName']"
            }
        }
    )
    return query_endpoint_deployment_lambda_step

def create_check_endpoint_status_choice_step(
    query_endpoint_lambda_step,
    endpoint_update_step
):
    check_endpoint_status_step = Choice('Endpoint is ready for deployment?')

    endpoint_in_service_rule = ChoiceRule.StringEquals(variable = query_endpoint_lambda_step.output()['Payload']['endpoint_status'], value = 'InService')
    check_endpoint_status_step.add_choice(rule = endpoint_in_service_rule, next_step = endpoint_update_step)
    
    # in case endpoint is in 'failed' state, we allow it to update so as to trigger exception.
    endpoint_failed_rule = ChoiceRule.StringEquals(variable = query_endpoint_lambda_step.output()['Payload']['endpoint_status'], value = 'Failed')
    check_endpoint_status_step.add_choice(rule = endpoint_failed_rule, next_step = endpoint_update_step)

    wait_step = Wait(state_id = f"Wait until Endpoint is ready", seconds = 20)
    wait_step.next(query_endpoint_lambda_step)
    check_endpoint_status_step.default_choice(next_step = wait_step)  

    return check_endpoint_status_step  

def create_check_endpoint_existence_choice_step(
    query_endpoint_lambda_step,
    check_endpoint_status_step,
    endpoint_creation_step
):
    check_endpoint_existence_step = Choice('Endpoint Existed?')

    endpoint_existed_rule = ChoiceRule.BooleanEquals(variable = query_endpoint_lambda_step.output()['Payload']['endpoint_existed'], value = True)
    check_endpoint_existence_step.add_choice(rule = endpoint_existed_rule, next_step = check_endpoint_status_step)

    check_endpoint_existence_step.default_choice(next_step = endpoint_creation_step)
    return check_endpoint_existence_step

def create_check_endpoint_is_deploying_choice_step(
    query_endpoint_deployment_lambda_step,
    success_notification_step
):
    # check endpoint readiness
    deployed_endpoint_updating_step = Choice('Endpoint is deploying?')

    wait_deployment_step = Wait(state_id = "Wait Until Deployment is Completed...", seconds = 20)
    wait_deployment_step.next(query_endpoint_deployment_lambda_step)

    deployed_endpoint_updating_rule = ChoiceRule.StringEquals(variable = query_endpoint_deployment_lambda_step.output()['Payload']['endpoint_status'], value = 'InService')
    deployed_endpoint_updating_step.add_choice(rule = deployed_endpoint_updating_rule, next_step = success_notification_step)
    
    deployed_endpoint_updating_step.default_choice(next_step = wait_deployment_step)

    return deployed_endpoint_updating_step

def create_success_notification_step(topic_arn, subject):
    success_sns_step = SnsPublishStep(
        state_id = 'SNS Notification - Pipeline Succeeded',
        parameters = {
            'TopicArn': topic_arn,
            'Message.$': "$$.Execution.Id",
            'Subject': subject
        }
    )    
    return success_sns_step 

def create_failure_notification_step(
    topic_arn
):
    failure_sns_step = SnsPublishStep(
        state_id = 'SNS Notification - Pipeline Failure',
        parameters = {
            'TopicArn': topic_arn,
            'Message.$': "$",
            'Subject': '[ML Pipeline] Execution failed...'
        }
    )    
    return failure_sns_step

def get_state_machine_arn(workflow_name, region, account_id):
    return f"arn:aws:states:{region}:{account_id}:stateMachine:{workflow_name}"

def is_workflow_existed(workflow_role_arn):
    try:
        sfn_client = boto3.client('stepfunctions')
        response = sfn_client.describe_state_machine(
            stateMachineArn = workflow_role_arn
        )
        return True
    except: 
        return False
    
def create_workflow(
    topic_name,
    workflow_name,
    region, 
    account_id,
    workflow_execution_role
):
    # Workflow Execution parameters
    execution_input = ExecutionInput(
        schema = {
            "EndpointConfigName": str,
            "DevEndpointName": str,
            "PrdEndpointName": str,
            "LambdaFunctionNameOfQueryEndpoint": str,
            "LambdaFunctionNameOfCreateBlueGreenDeploymentEndpointConfig": str
        }
    )

    # create the steps
    blue_green_endpoint_config_step = create_lambda_create_blue_gree_deployment_endpoint_config_step()
    query_endpoint_lambda_step = create_lambda_query_endpoint_step()
    endpoint_update_step = create_endpoint_step(execution_input["PrdEndpointName"], execution_input["EndpointConfigName"], True)
    endpoint_creation_step = create_endpoint_step(execution_input["PrdEndpointName"], execution_input["EndpointConfigName"], False)
    
    # create the choice steps
    check_endpoint_status_choice_step = create_check_endpoint_status_choice_step(query_endpoint_lambda_step, endpoint_update_step)
    check_endpoint_existence_choice_step = create_check_endpoint_existence_choice_step(
        query_endpoint_lambda_step,
        check_endpoint_status_choice_step,
        endpoint_creation_step
    )
    
    
    query_endpoint_deployment_lambda_step = create_query_endpoint_deployment_lambda_step()
    topic_arn = f"arn:aws:sns:{region}:{account_id}:{topic_name}"
    success_notification_step = create_success_notification_step(topic_arn, "[ML Pipeline] Blue/Green Deployment Endpoint is ready.")
    check_endpoint_is_deploying_choice_step = create_check_endpoint_is_deploying_choice_step(
        query_endpoint_deployment_lambda_step,
        success_notification_step
    )
    endpoint_creation_step.next(Chain([query_endpoint_deployment_lambda_step, check_endpoint_is_deploying_choice_step]))
    endpoint_update_step.next(Chain([query_endpoint_deployment_lambda_step, check_endpoint_is_deploying_choice_step]))
    
    deployment_path = Chain(
        [
            blue_green_endpoint_config_step, 
            query_endpoint_lambda_step,
            check_endpoint_existence_choice_step            
        ]
    )

    # catch execution exception
    failed_state_sagemaker_pipeline_step_failure = Fail(
        "ML Workflow Failed", cause = "SageMakerPipelineStepFailed"
    )
    failure_notification_step = create_failure_notification_step(topic_arn)
    
    catch_state_processing = Catch(
        error_equals = ["States.TaskFailed"],
        next_step = Chain([failure_notification_step, failed_state_sagemaker_pipeline_step_failure])
    )
    blue_green_endpoint_config_step.add_catch(catch_state_processing)
    endpoint_update_step.add_catch(catch_state_processing)
    query_endpoint_deployment_lambda_step.add_catch(catch_state_processing)
    
    # Create Workflow
    workflow_arn = get_state_machine_arn(workflow_name, region, account_id)
    workflow_existed = is_workflow_existed(workflow_arn)
    if workflow_existed:
        # To update SFN workflow, need to do 'attach' & 'update' together.
        workflow = Workflow.attach(state_machine_arn = workflow_arn)
        workflow.update(definition = deployment_path, role = workflow_execution_role) 
        # Wait for 10s so that the update is completed before executing workflow
        time.sleep(10)
    else:
        workflow = Workflow(
            name = workflow_name,
            definition = deployment_path,
            role = workflow_execution_role
        )
        workflow.create()
    return workflow

def main(
    workflow_name,
    workflow_execution_role,
    topic_name
):
    # bucket_name is created in ml_pipeline_dependencies.py, which is imported at the beginning.
    workflow = create_workflow(
        topic_name,
        workflow_name,
        region, 
        account_id,
        workflow_execution_role
    )    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load parameters")
    parser.add_argument("--workflow-name", required = True)
    parser.add_argument("--workflow-execution-role", required = True)
    parser.add_argument("--topic-name", required = True)
    
    args = vars(parser.parse_args())
    args['region'] = region
    args['account_id'] = account_id
    print("args: {}".format(args))
    main(**args)