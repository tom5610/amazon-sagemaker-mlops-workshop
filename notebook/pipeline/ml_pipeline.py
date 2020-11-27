from ml_pipeline_dependencies import *

def upload_preprocess_code(bucket_name):
    PREPROCESSING_SCRIPT_LOCATION = "./pipeline/preprocessing.py"
    input_code_uri = sagemaker_session.upload_data(
        PREPROCESSING_SCRIPT_LOCATION,
        bucket = bucket_name,
        key_prefix = "preprocessing/code",
    )
    return input_code_uri

def create_experiment(experiment_name):
    experiment = Experiment.create(
        experiment_name = experiment_name, 
        description = "Classification of target direct marketing", 
        sagemaker_boto_client = sm
    )
    return experiment

def create_trial(experiment_name, trial_name):
    trial = Trial.create(
        trial_name = trial_name, 
        experiment_name = experiment_name,
        sagemaker_boto_client = sm,
    )
    return trial        

def create_preprocessing_step(
    processing_job_placeholder,
    input_code_uri,
    bucket_name,
    data_file,
    experiment_name,
    trial_name,
    sagemaker_execution_role
):
    preprocessing_processor = SKLearnProcessor(
        framework_version='0.20.0',
        role = sagemaker_execution_role,
        instance_count = 1,
        instance_type = 'ml.m5.xlarge',
        max_runtime_in_seconds = 1200
    )

    processing_input_data = f's3://{bucket_name}/preprocessing/input/{data_file}'
    inputs = [
        ProcessingInput(
            input_name = "code",
            source = input_code_uri,
            destination = "/opt/ml/processing/input/code"
        ),
        ProcessingInput(
            input_name = "input_data",
            source = processing_input_data,
            destination='/opt/ml/processing/input'
        )
    ]

    processing_output_data = f"s3://{bucket_name}/preprocessing/output"
    outputs = [
        ProcessingOutput(
            output_name = "train_data",
            source = "/opt/ml/processing/output/train",
            destination = f"{processing_output_data}/train"
        ),
        ProcessingOutput(
            output_name = "validation_data",
            source = "/opt/ml/processing/output/validation",
            destination = f"{processing_output_data}/validation"
        ),
        ProcessingOutput(
            output_name = "test_data",
            source = "/opt/ml/processing/output/test",
            destination = f"{processing_output_data}/test"
        )
    ]

    processing_step = ProcessingStep(
        "Preprocessing",
        processor = preprocessing_processor,
        job_name = processing_job_placeholder,
        inputs = inputs,
        outputs = outputs,
        container_arguments = ["--data-file", data_file],
        container_entrypoint = ["python3", "/opt/ml/processing/input/code/preprocessing.py"],
        experiment_config = {
            "TrialName": trial_name,
            "TrialComponentDisplayName": "Processing",
        }
    )    

    return processing_step
    
def create_hpo_step(
    tuning_job_name_placeholder, 
    image_uri, 
    bucket_name, 
    sagemaker_execution_role,
    ml_instance_count = 1,
    ml_instance_type = 'ml.m5.xlarge',
    objective_metric_name = 'validation:auc'
):
    tuning_output_path = f's3://{bucket_name}/tuning/output'

    tuning_estimator = sagemaker.estimator.Estimator(
        image_uri,
        sagemaker_execution_role, 
        instance_count = ml_instance_count, 
        instance_type = ml_instance_type,
        output_path = tuning_output_path,
        sagemaker_session = sagemaker_session
    )    
    hpo = dict(
        max_depth = 5,
        eta = 0.2,
        gamma = 4,
        min_child_weight = 6,
        subsample = 0.8,
        silent = 0,
        objective = 'binary:logistic',
        num_round = 100
    ) 
    tuning_estimator.set_hyperparameters(**hpo)
    
    hyperparameter_ranges = {
        'eta': ContinuousParameter(0, 1),
        'min_child_weight': ContinuousParameter(1, 10),
        'alpha': ContinuousParameter(0, 2),
        'max_depth': IntegerParameter(1, 10)
    }
    hpo_tuner = HyperparameterTuner(
        tuning_estimator,
        objective_metric_name,
        hyperparameter_ranges,
        max_jobs = 20,
        max_parallel_jobs = 3
    )

    processing_output_data = f"s3://{bucket_name}/preprocessing/output"
    s3_input_train = TrainingInput(s3_data = f'{processing_output_data}/train', content_type = 'csv')
    s3_input_validation = TrainingInput(s3_data = f'{processing_output_data}/validation', content_type = 'csv')
    hpo_data = dict(
        train = s3_input_train,
        validation = s3_input_validation
    )

    # as long as HPO is selected, wait for completion.
    tuning_step = TuningStep(
        "HPO Step",
        tuner = hpo_tuner,
        job_name = tuning_job_name_placeholder,
        data = hpo_data,
        wait_for_completion = True
    )

    return tuning_step

def create_lambda_query_hpo_job_step(lambda_function_name_query_hpo_job_placeholder):
    query_hpo_job_lambda_step = LambdaStep(
        'Query HPO Job',
        parameters = {  
            "FunctionName": lambda_function_name_query_hpo_job_placeholder,
            'Payload':{
                "HpoJobName.$": "$$.Execution.Input['TuningJobName']"
            }
        }
    )   
    return query_hpo_job_lambda_step

def create_hpo_job_sns_notification_step(
    topic_arn,
    query_hpo_job_lambda_step
):
    hpo_job_sns_step = SnsPublishStep(
        state_id = 'SNS Notification - HPO Job',
        parameters = {
            'TopicArn': topic_arn,
            'Message': query_hpo_job_lambda_step.output()['Payload']['bestTrainingJob']
        }
    )    
    return hpo_job_sns_step

def create_training_step(
    training_job_name_placeholer, 
    image_uri, 
    bucket_name, 
    experiment_name,
    trial_name,
    role, 
    ml_instance_count = 1,
    ml_instance_type = 'ml.m5.xlarge',
):
    training_output_path = f's3://{bucket_name}/training/output'
    training_estimator = sagemaker.estimator.Estimator(
        image_uri,
        role, 
        instance_count = ml_instance_count, 
        instance_type = ml_instance_type,
        output_path = training_output_path,
        sagemaker_session = sagemaker_session
    )
        
    hpo = dict(
        max_depth = 5,
        eta = 0.2,
        gamma = 4,
        min_child_weight = 6,
        subsample = 0.8,
        silent = 0,
        objective = 'binary:logistic',
        num_round = 100
    )
    training_estimator.set_hyperparameters(**hpo) 
    
    processing_output_data = f"s3://{bucket_name}/preprocessing/output"
    s3_input_train = sagemaker.inputs.TrainingInput(s3_data=f'{processing_output_data}/train', content_type='csv')
    s3_input_validation = sagemaker.inputs.TrainingInput(s3_data=f'{processing_output_data}/validation', content_type='csv')

    training_data = dict(
        train = s3_input_train,
        validation = s3_input_validation
    )

    training_step = TrainingStep(
        "Model Training",
        estimator = training_estimator,
        data = training_data,
        job_name = training_job_name_placeholer,
        wait_for_completion = True,
        experiment_config = {
            "TrialName": trial_name,
            "TrialComponentDisplayName": "Training",
        },
    )    
    return training_step

def create_model_step(
    model_name_placeholder, 
    training_step
):
    model_step = ModelStep(
        "Save Model",
        model = training_step.get_expected_model(),
        model_name = model_name_placeholder
    )
    return model_step

def create_existing_model_step(
    model_name_placeholder, 
    existing_model_name,
    image_uri, 
    existing_model_uri,
    sagemaker_execution_role
):
    # for deploying existing model
    existing_model = Model(
        model_data = existing_model_uri,
        image_uri = image_uri,
        role = sagemaker_execution_role,
        name = existing_model_name
    )
    existing_model_step = ModelStep(
        "Using Existing Model",
        model = existing_model,
        model_name = model_name_placeholder
    )
    return existing_model_step

def create_endpoint_configurgation_step(
    endpoint_config_name_placeholder, 
    model_name_placeholder, 
    ml_instance_count = 1,
    ml_instance_type = 'ml.m5.xlarge'
):
    endpoint_config_step = EndpointConfigStep(
        "Create Endpoint Config",
        endpoint_config_name = endpoint_config_name_placeholder,
        model_name = model_name_placeholder,
        initial_instance_count = ml_instance_count,
        instance_type = ml_instance_type
    )
    return endpoint_config_step

def create_lambda_query_endpoint_step(lambda_function_name_query_endpoint_placeholder):
    query_endpoint_lambda_step = LambdaStep(
        'Query Endpoint Info',
        parameters = {  
            "FunctionName": lambda_function_name_query_endpoint_placeholder,
            'Payload':{
                "EndpointName.$": "$$.Execution.Input['EndpointName']"
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

def create_query_endpoint_deployment_lambda_step(lambda_function_name_query_endpoint_placeholder):
    query_endpoint_deployment_lambda_step = LambdaStep(
        'Query Endpoint Deployment Status',
        parameters = {  
            "FunctionName": lambda_function_name_query_endpoint_placeholder,
            'Payload':{
                "EndpointName.$": "$$.Execution.Input['EndpointName']"
            }
        }
    )
    return query_endpoint_deployment_lambda_step

def create_check_endpoint_status_choice_step(
    query_endpoint_lambda_step,
    endpoint_update_step
):
    check_endpoint_status_step = Choice('Endpoint is InService?')

    endpoint_in_service_rule = ChoiceRule.StringEquals(variable = query_endpoint_lambda_step.output()['Payload']['endpoint_status'], value = 'InService')
    endpoint_in_service_rule = ChoiceRule.StringEquals(variable = query_endpoint_lambda_step.output()['Payload']['endpoint_status'], value = 'Failed')
    check_endpoint_status_step.add_choice(rule = endpoint_in_service_rule, next_step = endpoint_update_step)

    wait_step = Wait(state_id = f"Wait Until Endpoint becomes InService", seconds = 20)
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
    query_endpoint_deployment_lambda_step
):
    # check endpoint readiness
    deployed_endpoint_updating_step = Choice('Endpoint is deploying?')

    wait_deployment_step = Wait(state_id = "Wait Until Deployment is Completed...", seconds = 20)
    wait_deployment_step.next(query_endpoint_deployment_lambda_step)

    final_step = Pass(state_id = 'Pass Step')
    deployed_endpoint_updating_rule = ChoiceRule.StringEquals(variable = query_endpoint_deployment_lambda_step.output()['Payload']['endpoint_status'], value = 'InService')
    deployed_endpoint_updating_step.add_choice(rule = deployed_endpoint_updating_rule, next_step = final_step)
    
    deployed_endpoint_updating_step.default_choice(next_step = wait_deployment_step)

    return deployed_endpoint_updating_step

def create_to_do_hpo_choice_step(
    tuning_path,
    training_choice
):
    to_do_hpo_choice = Choice("To Do HPO?")

    to_do_hpo_choice.add_choice(
        rule = ChoiceRule.BooleanEquals(variable = "$$.Execution.Input['ToDoHPO']", value = True),
        next_step = tuning_path                 
    )
    to_do_hpo_choice.add_choice(
        rule = ChoiceRule.BooleanEquals(variable = "$$.Execution.Input['ToDoHPO']", value = False),
        next_step = training_choice
    )
    return to_do_hpo_choice

def create_to_do_training_choice_step(
    training_path,
    deploy_existing_model_path
):
    to_do_training_choice = Choice("To Do Model Training?")

    to_do_training_choice.add_choice(
        rule = ChoiceRule.BooleanEquals(variable = "$$.Execution.Input['ToDoTraining']", value = True),
        next_step = training_path
    )
    to_do_training_choice.add_choice(
        rule = ChoiceRule.BooleanEquals(variable = "$$.Execution.Input['ToDoTraining']", value = False),
        next_step = deploy_existing_model_path
    )
    return to_do_training_choice

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
    bucket_name, 
    data_file,
    topic_name,
    experiment_name,
    workflow_name,
    region, 
    account_id,
    workflow_execution_role,
    sagemaker_execution_role
):
    suffix = datetime.now().strftime("%y%m%d-%H%M")

    # Workflow Execution parameters
    execution_input = ExecutionInput(
        schema = {
            "PreprocessingJobName": str,
            "ToDoHPO": bool,
            "ToDoTraining": bool,
            "TrainingJobName": str,
            "TuningJobName": str,
            "ModelName": str,
            "EndpointConfigName": str,
            "EndpointName": str,
            "LambdaFunctionNameOfQueryEndpoint": str,
            "LambdaFunctionNameOfQueryHpoJob": str
        }
    )
    image_uri = sagemaker.image_uris.retrieve(region = region, framework='xgboost', version='latest')

    # create the steps
    trial = create_trial(experiment_name, f"xgb-processing-job-{suffix}")
    input_code_uri = upload_preprocess_code(bucket_name)
    processing_step = create_preprocessing_step(
        execution_input["PreprocessingJobName"], 
        input_code_uri, 
        bucket_name,
        data_file, 
        experiment_name,
        trial.trial_name,
        sagemaker_execution_role
    )
    
    tuning_step = create_hpo_step(execution_input["TuningJobName"], image_uri, bucket_name, sagemaker_execution_role)
    query_hpo_job_lambda_step = create_lambda_query_hpo_job_step(execution_input['LambdaFunctionNameOfQueryHpoJob'])
    topic_arn = f"arn:aws:sns:{region}:{account_id}:{topic_name}"
    hpo_job_sns_notification_step = create_hpo_job_sns_notification_step(topic_arn, query_hpo_job_lambda_step)
    training_trial = create_trial(experiment_name, f"xgb-training-job-{suffix}")
    training_step = create_training_step(execution_input["TrainingJobName"], image_uri, bucket_name, experiment_name, training_trial.trial_name, sagemaker_execution_role)
    model_step = create_model_step(execution_input["ModelName"], training_step)
    existing_model_uri = f"s3://{bucket_name}/{S3_KEY_TRAINED_MODEL}"
    existing_model_step = create_existing_model_step(execution_input["ModelName"], f"dm-model-{suffix}", image_uri, existing_model_uri, sagemaker_execution_role)
    query_endpoint_lambda_step = create_lambda_query_endpoint_step(execution_input['LambdaFunctionNameOfQueryEndpoint'])
    endpoint_config_step = create_endpoint_configurgation_step(
        execution_input["EndpointConfigName"], 
        execution_input["ModelName"]
    )
    endpoint_creation_step = create_endpoint_step(execution_input["EndpointName"], execution_input["EndpointConfigName"], False)
    endpoint_update_step = create_endpoint_step(execution_input["EndpointName"], execution_input["EndpointConfigName"], True)
    query_endpoint_deployment_lambda_step = create_query_endpoint_deployment_lambda_step(execution_input['LambdaFunctionNameOfQueryEndpoint'])

    # create the choice steps
    check_endpoint_status_choice_step = create_check_endpoint_status_choice_step(query_endpoint_lambda_step, endpoint_update_step)
    check_endpoint_existence_choice_step = create_check_endpoint_existence_choice_step(
        query_endpoint_lambda_step,
        check_endpoint_status_choice_step,
        endpoint_creation_step
    )
    check_endpoint_is_deploying_choice_step = create_check_endpoint_is_deploying_choice_step(
        query_endpoint_deployment_lambda_step
    )

    query_endpoint_deployment_lambda_step.next(check_endpoint_is_deploying_choice_step)
    endpoint_creation_step.next(query_endpoint_deployment_lambda_step)
    endpoint_update_step.next(query_endpoint_deployment_lambda_step)
    
    training_path = Chain(
        [
            training_step, 
            model_step, 
            endpoint_config_step, 
            query_endpoint_lambda_step, 
            check_endpoint_existence_choice_step
        ]
    )
    deploy_existing_model_path = Chain(
        [
            existing_model_step, 
            endpoint_config_step, 
            query_endpoint_lambda_step, 
            check_endpoint_existence_choice_step
        ]
    )
    tuning_path = Chain([tuning_step, query_hpo_job_lambda_step, hpo_job_sns_notification_step])

    to_do_training_choice_step = create_to_do_training_choice_step(training_path, deploy_existing_model_path)
    to_do_hpo_choice_step = create_to_do_hpo_choice_step(tuning_path, to_do_training_choice_step)

    # catch execution exception
    failed_state_sagemaker_pipeline_step_failure = Fail(
        "ML Workflow Failed", cause = "SageMakerPipelineStepFailed"
    )
    catch_state_processing = Catch(
        error_equals = ["States.TaskFailed"],
        next_step = failed_state_sagemaker_pipeline_step_failure   
    )
    processing_step.add_catch(catch_state_processing)
    tuning_step.add_catch(catch_state_processing)
    training_step.add_catch(catch_state_processing)
    model_step.add_catch(catch_state_processing)
    endpoint_config_step.add_catch(catch_state_processing)
    endpoint_creation_step.add_catch(catch_state_processing)
    endpoint_update_step.add_catch(catch_state_processing)
    existing_model_step.add_catch(catch_state_processing)
    
    workflow_graph = Chain([processing_step, to_do_hpo_choice_step])
#     workflow_graph = Chain([to_do_hpo_choice_step])

    # Create Workflow
    workflow_arn = get_state_machine_arn(workflow_name, region, account_id)
    workflow_existed = is_workflow_existed(workflow_arn)
    if workflow_existed:
        # To update SFN workflow, need to do 'attach' & 'update' together.
        workflow = Workflow.attach(state_machine_arn = workflow_arn)
        workflow.update(definition = workflow_graph, role = workflow_execution_role) 
        # Wait for 10s so that the update is completed before executing workflow
        time.sleep(10)
    else:
        workflow = Workflow(
            name = workflow_name,
            definition = workflow_graph,
            role = workflow_execution_role
        )
        workflow.create()
    return workflow
    
    
def main(
    require_hpo,
    require_model_training,
    bucket_name, 
    data_file,
    topic_name,
    workflow_name,
    region, 
    account_id,
    workflow_execution_role,
    sagemaker_execution_role
):
    
    suffix = datetime.now().strftime("%y%m%d-%H%M%S")
    experiment = create_experiment(f"xgboost-target-direct-marketing-{suffix}")
    # bucket_name is created in ml_pipeline_dependencies.py, which is imported at the beginning.
    workflow = create_workflow(
        bucket_name, 
        data_file,
        topic_name,
        experiment.experiment_name,
        workflow_name, 
        region, 
        account_id,
        workflow_execution_role,
        sagemaker_execution_role
    )
    
    # execute workflow
    # execution input parameter values
    preprocessing_job_name = f"dm-preprocessing-{suffix}"
    tuning_job_name = f"dm-tuning-{suffix}"
    training_job_name = f"dm-training-{suffix}"
    model_job_name = f"dm-model-{suffix}"
    endpoint_config_name = f"dm-endpoint-config-{suffix}"
    endpoint_job_name = f"direct-marketing-endpoint"
    lambda_function_query_endpoint = 'query_endpoint'
    lambda_function_query_hpo_job = 'query_hpo_job'

    execution = workflow.execute(
        inputs = {
            "PreprocessingJobName": preprocessing_job_name,
            "ToDoHPO": str(require_hpo).lower() in ['true', '1', 'yes', 't'],
            "ToDoTraining": str(require_model_training).lower() in ['true', '1', 'yes', 't'],
            "TrainingJobName": training_job_name,
            "TuningJobName": tuning_job_name,
            "ModelName": model_job_name,
            "EndpointConfigName": endpoint_config_name,
            "EndpointName": endpoint_job_name,
            "LambdaFunctionNameOfQueryEndpoint": lambda_function_query_endpoint,
            "LambdaFunctionNameOfQueryHpoJob": lambda_function_query_hpo_job
        }
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load parameters")
    parser.add_argument("--workflow-name", required = True)
    parser.add_argument("--workflow-execution-role", required = True)
    parser.add_argument("--data-file", required = True)
    parser.add_argument("--topic-name", required = True)
    parser.add_argument("--bucket-name", required = True)
    parser.add_argument("--require-hpo", required = True)
    parser.add_argument("--require-model-training", required = True)
    
    args = vars(parser.parse_args())
    args['region'] = region
    args['sagemaker_execution_role'] = sagemaker_execution_role
    args['account_id'] = account_id
    print("args: {}".format(args))
    main(**args)