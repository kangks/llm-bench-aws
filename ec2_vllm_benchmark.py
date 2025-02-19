import boto3
import time
import logging
from botocore.exceptions import ClientError
import base64

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def get_or_create_security_group(ec2, vpc_id, group_name):
    try:
        # Try to find existing security group
        response = ec2.describe_security_groups(
            Filters=[
                {'Name': 'group-name', 'Values': [group_name]},
                {'Name': 'vpc-id', 'Values': [vpc_id]}
            ]
        )
        if response['SecurityGroups']:
            logger.info(f"Using existing security group: {group_name}")
            return response['SecurityGroups'][0]['GroupId']
    except ClientError as e:
        if e.response['Error']['Code'] != 'InvalidGroup.NotFound':
            raise

    # Create new security group if not found
    logger.info(f"Creating new security group: {group_name}")
    sg_response = ec2.create_security_group(
        GroupName=group_name,
        Description='Security group for Batch EFS mount'
    )
    security_group_id = sg_response['GroupId']

    # Allow NFS traffic
    ec2.authorize_security_group_ingress(
        GroupId=security_group_id,
        IpPermissions=[{
            'FromPort': 2049,
            'ToPort': 2049,
            'IpProtocol': 'tcp',
            'UserIdGroupPairs': [{'GroupId': security_group_id}]
        }]
    )
    return security_group_id

def get_or_create_efs(efs, ec2, subnet_ids, security_group_id):
    try:
        # List existing filesystems
        filesystems = efs.describe_file_systems()['FileSystems']
        for fs in filesystems:
            if any(tag['Key'] == 'Name' and tag['Value'] == '/models' for tag in fs.get('Tags', [])):
                logger.info("Using existing EFS filesystem")
                file_system_id = fs['FileSystemId']
                break
        else:
            raise ClientError({'Error': {'Code': 'FileSystemNotFound'}}, 'describe_file_systems')
    except ClientError:
        # Create new filesystem if not found
        logger.info("Creating new EFS filesystem")
        efs_response = efs.create_file_system(
            PerformanceMode='generalPurpose',
            ThroughputMode='bursting',
            Tags=[{'Key': 'Name', 'Value': '/models'}]
        )
        file_system_id = efs_response['FileSystemId']

        # Wait for EFS to be available
        logger.info("Waiting for EFS to become available...")
        waiter = efs.get_waiter('file_system_available')
        waiter.wait(FileSystemId=file_system_id)

    # Check existing mount targets
    mount_targets = efs.describe_mount_targets(FileSystemId=file_system_id)['MountTargets']
    existing_subnets = {mt['SubnetId'] for mt in mount_targets}

    # Create mount targets only in subnets that don't have them
    for subnet_id in subnet_ids:
        if subnet_id not in existing_subnets:
            logger.info(f"Creating mount target in subnet: {subnet_id}")
            efs.create_mount_target(
                FileSystemId=file_system_id,
                SubnetId=subnet_id,
                SecurityGroups=[security_group_id]
            )

    return file_system_id

def get_or_create_launch_template(ec2, name, file_system_id, security_group_id):
    try:
        # Check if launch template exists
        response = ec2.describe_launch_templates(
            LaunchTemplateNames=[name]
        )
        if response['LaunchTemplates']:
            logger.info(f"Using existing launch template: {name}")
            return response['LaunchTemplates'][0]['LaunchTemplateId']
    except ClientError as e:
        if e.response['Error']['Code'] != 'InvalidLaunchTemplateName.NotFoundException':
            raise

    # Create MIME-based user data for EFS mounting
    mime_userdata = f"""MIME-Version: 1.0
Content-Type: multipart/mixed; boundary="==MYBOUNDARY=="

--==MYBOUNDARY==
Content-Type: text/cloud-config; charset="us-ascii"

packages:
- amazon-efs-utils

runcmd:
- file_system_id_01={file_system_id}
- efs_directory=/mnt/efs/fs1

- mkdir -p /mnt/efs/fs1
- echo "{file_system_id}:/ /mnt/efs/fs1 efs tls,_netdev" >> /etc/fstab
- mount -a -t efs defaults

--==MYBOUNDARY==--"""

    # Create launch template
    logger.info(f"Creating new launch template: {name}")
    response = ec2.create_launch_template(
        LaunchTemplateName=name,
        VersionDescription='Initial version',
        LaunchTemplateData={
            'UserData': base64.b64encode(mime_userdata.encode('utf-8')).decode(encoding="utf-8"),
            'SecurityGroupIds': [security_group_id],
            'MetadataOptions': {
                'HttpTokens': 'required',
                'HttpEndpoint': 'enabled'
            }
        }
    )
    return response['LaunchTemplate']['LaunchTemplateId']

def get_or_create_iam_role(iam, role_name, service_principal, policy_arn):
    try:
        iam.get_role(RoleName=role_name)
        logger.info(f"Using existing IAM role: {role_name}")
    except ClientError:
        logger.info(f"Creating new IAM role: {role_name}")
        iam.create_role(
            RoleName=role_name,
            AssumeRolePolicyDocument=f'''{{
                "Version": "2012-10-17",
                "Statement": [
                    {{
                        "Effect": "Allow",
                        "Principal": {{
                            "Service": "{service_principal}"
                        }},
                        "Action": "sts:AssumeRole"
                    }}
                ]
            }}'''
        )
        iam.attach_role_policy(
            RoleName=role_name,
            PolicyArn=policy_arn
        )

def wait_for_compute_environment(batch, compute_env_arn):
    logger.info("Waiting for compute environment to be VALID...")
    max_retries = 30
    retry_count = 0
    while retry_count < max_retries:
        response = batch.describe_compute_environments(computeEnvironments=[compute_env_arn])
        if not response['computeEnvironments']:
            raise Exception("Compute environment not found")
            
        status = response['computeEnvironments'][0]['status']
        state = response['computeEnvironments'][0]['state']
        
        if status == 'VALID' and state == 'ENABLED':
            logger.info("Compute environment is now VALID and ENABLED")
            return True
        elif status == 'INVALID':
            error_msg = f"Compute environment became INVALID: {response['computeEnvironments'][0].get('statusReason', 'No reason provided')}"
            logger.error(error_msg)
            raise Exception(error_msg)
            
        logger.info(f"Current status: {status}, state: {state}. Waiting...")
        time.sleep(20)
        retry_count += 1
    
    error_msg = "Timed out waiting for compute environment to become VALID"
    logger.error(error_msg)
    raise Exception(error_msg)

def wait_for_job_queue(batch, queue_arn):
    logger.info("Waiting for job queue to be VALID...")
    max_retries = 30
    retry_count = 0
    while retry_count < max_retries:
        response = batch.describe_job_queues(jobQueues=[queue_arn])
        if not response['jobQueues']:
            raise Exception("Job queue not found")
            
        status = response['jobQueues'][0]['status']
        state = response['jobQueues'][0]['state']
        
        if status == 'VALID' and state == 'ENABLED':
            logger.info("Job queue is now VALID and ENABLED")
            return True
        elif status == 'INVALID':
            error_msg = f"Job queue became INVALID: {response['jobQueues'][0].get('statusReason', 'No reason provided')}"
            logger.error(error_msg)
            raise Exception(error_msg)
            
        logger.info(f"Current status: {status}, state: {state}. Waiting...")
        time.sleep(20)
        retry_count += 1
    
    error_msg = "Timed out waiting for job queue to become VALID"
    logger.error(error_msg)
    raise Exception(error_msg)

def get_or_create_compute_environment(batch, name, subnet_ids, security_group_id, launch_template_id, instance_type):
    try:
        # Check if compute environment exists
        response = batch.describe_compute_environments(computeEnvironments=[name])
        if response['computeEnvironments']:
            logger.info(f"Using existing compute environment: {name}")
            compute_env_arn = response['computeEnvironments'][0]['computeEnvironmentArn']
            wait_for_compute_environment(batch, compute_env_arn)
            return compute_env_arn
    except ClientError:
        pass

    # Create new compute environment if not found
    logger.info(f"Creating new compute environment: {name}")
    response = batch.create_compute_environment(
        computeEnvironmentName=name,
        type='MANAGED',
        state='ENABLED',
        computeResources={
            'type': 'EC2',
            'maxvCpus': 256,
            'minvCpus': 0,
            'desiredvCpus': 0,
            'instanceTypes': [instance_type],
            'subnets': subnet_ids,
            'securityGroupIds': [security_group_id],
            'instanceRole': 'ec2AdminRole',
            'tags': {'Name': f'Batch-Graviton-{instance_type}'},
            'allocationStrategy': 'BEST_FIT_PROGRESSIVE',
            'launchTemplate': {
                'launchTemplateId': launch_template_id,
                'version': '$Latest'
            }
        },
        serviceRole='AWSBatchServiceRole'
    )
    compute_env_arn = response['computeEnvironmentArn']
    wait_for_compute_environment(batch, compute_env_arn)
    return compute_env_arn

def get_or_create_job_queue(batch, name, compute_env_arn):
    try:
        # Check if job queue exists
        response = batch.describe_job_queues(jobQueues=[name])
        if response['jobQueues']:
            logger.info(f"Using existing job queue: {name}")
            queue_arn = response['jobQueues'][0]['jobQueueArn']
            wait_for_job_queue(batch, queue_arn)
            return queue_arn
    except ClientError:
        pass

    # Create new job queue if not found
    logger.info(f"Creating new job queue: {name}")
    response = batch.create_job_queue(
        jobQueueName=name,
        state='ENABLED',
        priority=1,
        computeEnvironmentOrder=[
            {
                'order': 1,
                'computeEnvironment': compute_env_arn
            }
        ]
    )
    queue_arn = response['jobQueueArn']
    wait_for_job_queue(batch, queue_arn)
    return queue_arn

def monitor_job(batch, job_id, instance_type):
    """Monitor the status of a batch job until it completes or fails."""
    while True:
        response = batch.describe_jobs(jobs=[job_id])
        if not response['jobs']:
            logger.error(f"Job {job_id} ({instance_type}) not found")
            return False

        job = response['jobs'][0]
        status = job['status']
        reason = job.get('statusReason', 'No reason provided')
        
        if status == 'SUCCEEDED':
            logger.info(f"Job {job_id} ({instance_type}) completed successfully")
            return True
        elif status in ['FAILED', 'CANCELLED']:
            logger.error(f"Job {job_id} ({instance_type}) {status.lower()}: {reason}")
            # Log container details if available
            if 'container' in job and 'logStreamName' in job['container']:
                logger.info(f"Log stream: {job['container']['logStreamName']}")
            return False
        
        logger.info(f"Job {job_id} ({instance_type}) status: {status}")
        time.sleep(30)

def monitor_all_jobs(batch, job_ids):
    """Monitor all jobs concurrently until they complete or fail."""
    active_jobs = job_ids.copy()
    results = {}

    while active_jobs:
        for instance_type, job_id in list(active_jobs.items()):
            response = batch.describe_jobs(jobs=[job_id])
            if not response['jobs']:
                logger.error(f"Job {job_id} ({instance_type}) not found")
                results[instance_type] = False
                active_jobs.pop(instance_type)
                continue

            job = response['jobs'][0]
            status = job['status']
            reason = job.get('statusReason', 'No reason provided')

            if status == 'SUCCEEDED':
                logger.info(f"Job {job_id} ({instance_type}) completed successfully")
                results[instance_type] = True
                active_jobs.pop(instance_type)
            elif status in ['FAILED', 'CANCELLED']:
                logger.error(f"Job {job_id} ({instance_type}) {status.lower()}: {reason}")
                if 'container' in job and 'logStreamName' in job['container']:
                    logger.info(f"Log stream: {job['container']['logStreamName']}")
                results[instance_type] = False
                active_jobs.pop(instance_type)
            else:
                logger.info(f"Job {job_id} ({instance_type}) status: {status}")

        if active_jobs:
            time.sleep(30)

    return results

def create_batch_environment():
    batch = boto3.client('batch')
    efs = boto3.client('efs')
    ec2 = boto3.client('ec2')
    iam = boto3.client('iam')

    # Get default VPC and subnets
    vpc_response = ec2.describe_vpcs(
        Filters=[{'Name': 'isDefault', 'Values': ['true']}]
    )
    vpc_id = "vpc-00426987e51164259"
    
    subnet_response = ec2.describe_subnets(
        Filters=[{'Name': 'vpc-id', 'Values': [vpc_id]}]
    )
    subnet_ids = [subnet['SubnetId'] for subnet in subnet_response['Subnets']]

    # Get or create security group
    security_group_id = get_or_create_security_group(ec2, vpc_id, 'BatchEFSSecurityGroup')

    # Get or create EFS filesystem and mount targets
    file_system_id = get_or_create_efs(efs, ec2, subnet_ids, security_group_id)

    # Get or create Launch Template
    launch_template_id = get_or_create_launch_template(ec2, 'BatchEFSLaunchTemplate', file_system_id, security_group_id)

    # Get or create IAM roles
    get_or_create_iam_role(
        iam,
        'AWSBatchServiceRole',
        'batch.amazonaws.com',
        'arn:aws:iam::aws:policy/service-role/AWSBatchServiceRole'
    )

    # Create execution role for job definition with CloudWatch Logs permissions
    get_or_create_iam_role(
        iam,
        'BatchJobExecutionRole',
        'ecs-tasks.amazonaws.com',
        'arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy'
    )

    # Attach additional CloudWatch Logs permissions to the execution role
    try:
        iam.put_role_policy(
            RoleName='BatchJobExecutionRole',
            PolicyName='CloudWatchLogsPolicy',
            PolicyDocument='''{
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Action": [
                            "logs:CreateLogGroup",
                            "logs:CreateLogStream",
                            "logs:PutLogEvents"
                        ],
                        "Resource": "*"
                    }
                ]
            }'''
        )
        logger.info("Added CloudWatch Logs permissions to BatchJobExecutionRole")
    except ClientError as e:
        if e.response['Error']['Code'] != 'EntityAlreadyExists':
            raise
        logger.info("CloudWatch Logs permissions already exist for BatchJobExecutionRole")

    # Instance types and their environments
    instance_configs = [
        ('r8g.4xlarge', 'GravitonR8gEnvironment', 'GravitonR8gQueue'),
        ('c8g.4xlarge', 'GravitonC8gEnvironment', 'GravitonC8gQueue'),
        ('m8g.4xlarge', 'GravitonM8gEnvironment', 'GravitonM8gQueue')
    ]

    compute_envs = {}
    job_queues = {}

    # Create compute environments and job queues for each instance type
    for instance_type, env_name, queue_name in instance_configs:
        compute_env_arn = get_or_create_compute_environment(
            batch,
            env_name,
            subnet_ids,
            security_group_id,
            launch_template_id,
            instance_type
        )
        compute_envs[instance_type] = compute_env_arn

        queue_arn = get_or_create_job_queue(batch, queue_name, compute_env_arn)
        job_queues[instance_type] = queue_arn

    # Create CloudWatch log group if it doesn't exist
    logs_client = boto3.client('logs')
    try:
        logs_client.create_log_group(logGroupName='awslogs-vllm-batch')
        logger.info("Created CloudWatch log group: awslogs-vllm-batch")
    except ClientError as e:
        if e.response['Error']['Code'] != 'ResourceAlreadyExistsException':
            raise
        logger.info("Using existing CloudWatch log group: awslogs-vllm-batch")

    # Register job definition
    job_def_response = batch.register_job_definition(
        jobDefinitionName='vllm-offline-job',
        type='container',
        containerProperties={
            'image': "654654616949.dkr.ecr.us-east-1.amazonaws.com/vllm/vllm-arm:latest",
            'vcpus': 2,
            'memory': 64,
            'jobRoleArn': f"arn:aws:iam::{boto3.client('sts').get_caller_identity()['Account']}:role/BatchJobExecutionRole",
            'privileged': True,
            'mountPoints': [
                {
                    'sourceVolume': 'efs',
                    'containerPath': '/mnt/efs/fs1',
                    'readOnly': False
                }
            ],
            'volumes': [
                {
                    'name': 'efs',
                    "host": {
                        "sourcePath": "/mnt/efs/fs1"
                    },
                }
            ],            
            "logConfiguration": {
                "logDriver": "awslogs",
                "options": {
                    "awslogs-group": "awslogs-vllm-batch",
                    "awslogs-stream-prefix": "vllm-offline-14b"
                }
            }
        },
        
    )

    return {
        'compute_environments': compute_envs,
        'job_queues': job_queues,
        'job_definition': job_def_response['jobDefinitionArn'],
        'file_system_id': file_system_id,
        'launch_template_id': launch_template_id
    }

def submit_batch_job(job_queue, job_definition,
                     MODEL='deepseek-ai/DeepSeek-R1-Distill-Qwen-14B',
                     VLLM_CPU_KVCACHE_SPACE=20,
                     VLLM_CPU_OMP_THREADS_BIND="all"):
    batch = boto3.client('batch')
    
    try:
        logger.info(f"Submitting job to queue: {job_queue}")
        response = batch.submit_job(
            jobName='vllm-offline-test',
            jobQueue=job_queue,
            jobDefinition=job_definition,
            containerOverrides={
                'environment': [
                    {
                        'name': 'MODEL',
                        'value': MODEL
                    },
                    {
                        'name': 'VLLM_CPU_KVCACHE_SPACE',
                        'value': VLLM_CPU_KVCACHE_SPACE
                    },
                    {
                        'name': 'VLLM_CPU_OMP_THREADS_BIND',
                        'value': VLLM_CPU_OMP_THREADS_BIND
                    }
                ],
                "resourceRequirements":[
                    {
                        'type': 'MEMORY',
                        'value': '64000'
                    }
                ]
            }
        )
        job_id = response['jobId']
        logger.info(f"Successfully submitted job with ID: {job_id}")
        return job_id
        
    except ClientError as e:
        logger.error(f"Failed to submit job: {str(e)}")
        raise

if __name__ == '__main__':
    # Create the batch environment
    logger.info("Creating AWS Batch environment...")
    resources = create_batch_environment()
    
    logger.info("\nCreated/Found resources:")
    for key, value in resources.items():
        if isinstance(value, dict):
            logger.info(f"{key}:")
            for k, v in value.items():
                logger.info(f"  {k}: {v}")
        else:
            logger.info(f"{key}: {value}")
    
    # Submit test jobs to all queues concurrently
    logger.info("\nSubmitting test jobs to all queues...")
    job_ids = {}
    for instance_type, queue_arn in resources['job_queues'].items():
        logger.info(f"\nSubmitting job to {instance_type} queue...")
        job_id = submit_batch_job(
            queue_arn,
            resources['job_definition'],
            MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
            VLLM_CPU_KVCACHE_SPACE="20"
        )
        job_ids[instance_type] = job_id
        logger.info(f"Submitted job ID for {instance_type}: {job_id}")

    # Monitor all jobs concurrently
    logger.info("\nMonitoring all jobs...")
    results = monitor_all_jobs(boto3.client('batch'), job_ids)
    
    # Print final results
    logger.info("\nFinal results:")
    for instance_type, success in results.items():
        logger.info(f"{instance_type}: {'Succeeded' if success else 'Failed'}")