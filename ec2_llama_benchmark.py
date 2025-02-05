import boto3
import logging
import time
import paramiko
import os
import yaml
from datetime import datetime

class EC2LlamaBenchmark:
    def __init__(self, 
                 download_instance_type: str = "t4g.medium", 
                 volume_id: str = None
                 ):
        
        self.download_instance_type = download_instance_type
        self.volume_id = volume_id  # If provided, skip download phase
        
        self.ec2_client = boto3.client('ec2')
        self.ec2_resource = boto3.resource('ec2')
        self.key_name = 'llama-benchmark-key'
        self.instance_id = None
        self.instance = None
        self.elastic_ip = None
        self.elastic_ip_allocation_id = None
        self.ssh_user="ec2-user"
        self.ask_before_terminate = True
        
        # Get default VPC
        vpcs = list(self.ec2_resource.vpcs.filter(
            Filters=[{'Name': 'isDefault', 'Values': ['true']}]
        ))
        self.vpc = vpcs[0] if vpcs else None

    def create_and_attach_volume(self, instance_id: str, device_name: str = '/dev/sdh') -> str:
        """Create a new EBS volume and attach it to the instance"""
        logging.info(f"Creating new EBS volume for instance {instance_id}")
        instance = self.ec2_resource.Instance(instance_id)
        
        logging.info(f"Creating 500GB gp3 volume in availability zone {instance.placement['AvailabilityZone']}")
        volume = self.ec2_client.create_volume(
            AvailabilityZone=instance.placement['AvailabilityZone'],
            Size=500,
            VolumeType='gp3',
            TagSpecifications=[
                {
                    'ResourceType': 'volume',
                    'Tags': [
                        {
                            'Key': 'Name',
                            'Value': 'llama-model-volume'
                        },
                    ]
                },
            ]
        )
        volume_id = volume['VolumeId']
        
        # Wait for volume to be available
        logging.info(f"Waiting for volume {volume_id} to become available...")
        waiter = self.ec2_client.get_waiter('volume_available')
        waiter.wait(VolumeIds=[volume_id])
        logging.info(f"Volume {volume_id} is now available")
        
        # Attach volume
        logging.info(f"Attaching volume {volume_id} to instance {instance_id} at {device_name}")
        self.ec2_client.attach_volume(
            Device=device_name,
            InstanceId=instance_id,
            VolumeId=volume_id
        )
        
        return volume_id

    def detach_volume(self, volume_id: str):
        """Detach an EBS volume"""
        waiter = self.ec2_client.get_waiter('volume_available')
        self.ec2_client.detach_volume(VolumeId=volume_id)
        waiter.wait(VolumeIds=[volume_id])

    def download_model(self, models: list) -> str:
        """Launch a small instance and download the model to EBS"""
        if self.volume_id:
            logging.info(f"Using existing volume {self.volume_id}")
            return self.volume_id

        logging.info(f"Initiating model download process using instance type {self.download_instance_type}")
        # Launch a small instance for downloading
        self.instance_id = self._launch_instance(self.download_instance_type, instanceName='llama-download')
        volume_id = self.create_and_attach_volume(self.instance_id)
        
        logging.info("Preparing volume and initiating model download...")
        # Prepare the volume and download the model
        user_data = f"""#!/bin/bash -x
sudo mkfs -t xfs /dev/sdh
sudo mkdir /data
sudo mount /dev/sdh /data
sudo chown {self.ssh_user}:{self.ssh_user} /data
sudo yum install -y python3-pip
pip install "huggingface_hub[hf_transfer]" hf_transfer
"""
        self._run_remote_commands(user_data)

        for model in models:
            model_name = model['Name']
            model_url = model['URL']
            model_repoid = model['RepoId']
            model_filename = model['Filename']
            download_command = f"""#!/bin/bash -x
HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download {model_repoid} {model_filename} --local-dir /data
"""
            logging.info(f"Downloading model {model_filename} from {model_repoid}")
            self._run_remote_commands(download_command)

        # Detach volume and terminate instance
        logging.info(f"Model download completed. Detaching volume {volume_id}")
        self.detach_volume(volume_id)
        logging.info("Terminating download instance while keeping the volume")
        self.terminate_instance(keep_volume=True)
        
        logging.info(f"Model download process completed successfully. Volume ID: {volume_id}")
        return volume_id

    def setup_benchmark_instance(self, instance_type: str, volume_id: str):
        """Setup the benchmark instance with the model volume attached"""
        logging.info(f"Setting up benchmark instance with volume {volume_id}")
        self.instance_id = self._launch_instance(instance_type, instanceName='llama-benchmark')
        self.ec2_client.attach_volume(
            Device='/dev/sdh',
            InstanceId=self.instance_id,
            VolumeId=volume_id
        )
        logging.info(f"Benchmark instance {self.instance_id} setup completed successfully")
        logging.info(f"Volume attached successfully. Installing requirements and starting benchmark...")
        # Install requirements and run benchmark
        user_data = f"""#!/bin/bash
sudo mkdir /data
sudo mount /dev/sdh /data
cd /home/{self.ssh_user}
sudo yum update
sudo yum -y groupinstall "Development Tools"
sudo yum install -y python3-pip git cmake
git clone https://github.com/ggerganov/llama.cpp.git
cd ~/llama.cpp/ && cmake -B build -DCMAKE_CXX_FLAGS="-mcpu=native" -DCMAKE_C_FLAGS="-mcpu=native" && cmake --build build -v --config Release -j $(nproc)
"""
        return self._run_remote_commands(user_data)
    
    def run_benchmark(self, 
                      prompt: str, 
                      tokens: int, 
                      models: list):
        """Launch the benchmark instance and run the test"""
        if not self.volume_id:
            raise ValueError("No volume_id specified for benchmark")

        for model in models:
            model_name = model['Name']
            user_data = f"""#!/bin/bash
~/llama.cpp/build/bin/llama-cli -m /data/{model_name} -p "{prompt}" -n {tokens} -t $(nproc) -no-cnv
"""
        return self._run_remote_commands(user_data)

    def _launch_instance(self, instance_type: str, instanceName: str):
        """Launch EC2 instance with basic setup"""
        logging.info(f"Launching new EC2 instance of type {instance_type}")
        security_group_id = self.create_security_group()
        
        # Get public subnet
        subnets = self.ec2_client.describe_subnets(
            Filters=[{'Name': 'vpc-id', 'Values': [self.vpc.id]}]
        )
        public_subnet = None

        # Get main route table of the VPC
        main_route_table = self.ec2_client.describe_route_tables(
            Filters=[
                {'Name': 'vpc-id', 'Values': [self.vpc.id]},
                {'Name': 'association.main', 'Values': ['true']}
            ]
        )['RouteTables'][0]
        
        # Check if main route table has internet gateway
        main_has_igw = any(route.get('GatewayId', '').startswith('igw-') for route in main_route_table['Routes'])
        
        for subnet in subnets['Subnets']:
            # First check explicit route table associations
            filter=[
                {'Name': 'association.subnet-id', 'Values': [subnet['SubnetId']]},
                {'Name': 'vpc-id', 'Values': [self.vpc.id]}
            ]
            logging.info(f"Checking subnet {subnet['SubnetId']} in VPC {self.vpc.id} using filter {filter}")
            route_tables = self.ec2_client.describe_route_tables(
                Filters=filter
            )
            
            is_public = False
            if route_tables['RouteTables']:
                # Subnet has explicit route table association
                logging.info(f"Found subnet {subnet['SubnetId']} with explicit route table association")
                for route_table in route_tables['RouteTables']:
                    logging.info(f"Checking route table {route_table['RouteTableId']} for public subnet")
                    if any(route.get('GatewayId', '').startswith('igw-') for route in route_table['Routes']):
                        is_public = True
                        break
            else:
                # Subnet uses main route table
                logging.info(f"Subnet {subnet['SubnetId']} uses main route table")
                is_public = main_has_igw
            
            if is_public:
                public_subnet = subnet['SubnetId']
                break

        if not public_subnet:
            raise ValueError(f"No public subnet found in the default VPC {self.vpc.id}") 

        logging.info("Creating EC2 instance with Amazon LInux 2023")
        response = self.ec2_client.run_instances(
            ImageId='ami-0b29c89c15cfb8a6d',  # Amazon LInux 2023
            InstanceType=instance_type,
            KeyName=self.key_name,
            MinCount=1,
            MaxCount=1,
            SecurityGroupIds=[security_group_id],
            SubnetId=public_subnet,
            BlockDeviceMappings=[{
                'DeviceName': '/dev/sda1',
                'Ebs': {
                    'VolumeSize': 150,
                    'VolumeType': 'gp3'
                }
            }],
            TagSpecifications=[
                {
                    'ResourceType': 'instance',
                    'Tags': [
                        {
                            'Key': 'Name',
                            'Value': instanceName
                        },
                    ]
                },
            ]
        )
        
        instance_id = response['Instances'][0]['InstanceId']
        self.instance = self.ec2_resource.Instance(instance_id)
        
        logging.info(f"Waiting for instance {instance_id} to be running...")
        self.instance.wait_until_running()
        
        # Wait for status checks
        waiter = self.ec2_client.get_waiter('instance_status_ok')
        waiter.wait(InstanceIds=[instance_id])
        
        return instance_id

    def _run_remote_commands(self, commands: str) -> str:
        """Execute commands on the remote instance"""
        logging.info("Preparing to execute remote commands")
        key = paramiko.RSAKey.from_private_key_file(f"{self.key_name}.pem")
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        
        # Get instance public IP
        instance = self.ec2_resource.Instance(self.instance_id)
        public_ip = instance.public_ip_address
        
        logging.info(f"Waiting for SSH to become available on {public_ip}...")
        # Wait for SSH to be available
        time.sleep(30)
        
        try:
            ssh.connect(hostname=public_ip, username=self.ssh_user, pkey=key)
            
            # Execute commands
            for command in commands.split('\n'):
                if command.strip():
                    logging.info(f"Executing SSH command: {command}")
                    stdin, stdout, stderr = ssh.exec_command(command)
                    output = stdout.read().decode('utf-8')
                    error = stderr.read().decode('utf-8')
                    if error:
                        logging.error(f"Error executing SSH {command}: {error}")
                    logging.info(f"SSH Output: {output}")
            
            return output
        finally:
            ssh.close()

    def create_security_group(self):
        """Create security group for SSH access"""
        logging.info("Setting up security group for SSH access")
        try:
            logging.info("Creating new security group 'llama-benchmark-sg'")
            response = self.ec2_client.create_security_group(
                GroupName='llama-benchmark-sg',
                Description='Security group for Llama benchmark'
            )
            security_group_id = response['GroupId']
            
            self.ec2_client.authorize_security_group_ingress(
                GroupId=security_group_id,
                IpPermissions=[
                    {
                        'IpProtocol': 'tcp',
                        'FromPort': 22,
                        'ToPort': 22,
                        'IpRanges': [{'CidrIp': '0.0.0.0/0'}]
                    }
                ]
            )
            return security_group_id
        except self.ec2_client.exceptions.ClientError as e:
            if e.response['Error']['Code'] == 'InvalidGroup.Duplicate':
                security_groups = self.ec2_client.describe_security_groups(
                    Filters=[{'Name': 'group-name', 'Values': ['llama-benchmark-sg']}]
                )
                return security_groups['SecurityGroups'][0]['GroupId']
            raise

    def create_key_pair(self):
        """Create SSH key pair"""
        logging.info(f"Checking for existing key pair '{self.key_name}'")
        try:
            key_pair = self.ec2_client.create_key_pair(KeyName=self.key_name)
            with open(f"{self.key_name}.pem", 'w') as f:
                f.write(key_pair['KeyMaterial'])
            os.chmod(f"{self.key_name}.pem", 0o400)
            logging.info(f"Created and saved new key pair '{self.key_name}'")
        except self.ec2_client.exceptions.ClientError as e:
            if e.response['Error']['Code'] != 'InvalidKeyPair.Duplicate':
                logging.error(f"Failed to create key pair: {str(e)}")
                raise
            logging.info(f"Using existing key pair '{self.key_name}'")

    def terminate_instance(self, keep_volume: bool = False):
        if self.ask_before_terminate:
            confirmation = input("Are you sure you want to terminate the instance and cleanup all resources? (yes/no): ")
            if confirmation.lower() != 'yes':
                logging.info("Operation cancelled by user")
                return

        """Terminate instance and cleanup resources"""
        if self.instance_id:
            self.ec2_client.terminate_instances(InstanceIds=[self.instance_id])
            logging.info(f"Terminated instance {self.instance_id}")
            waiter = self.ec2_client.get_waiter('instance_terminated')
            waiter.wait(InstanceIds=[self.instance_id])
            
        if not keep_volume and self.volume_id:
            self.ec2_client.delete_volume(VolumeId=self.volume_id)
            logging.info(f"Deleted volume {self.volume_id}")
            
        self.instance_id = None
        if not keep_volume:
            self.volume_id = None

def main():
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(f'llama_benchmark_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
            logging.StreamHandler()
        ]
    )
    logging.getLogger("paramiko").setLevel(logging.INFO) # Set paramiko logging level
    
    # Load config file
    with open('benchmark.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize the benchmark with existing volume ID if available
    volume_id = os.getenv('LLAMA_VOLUME_ID')
    volume_id='vol-05ae89dfb0e0d1325'
    logging.info(f"Starting LLaMa benchmark with volume_id={volume_id}")
        
    benchmark = EC2LlamaBenchmark(
        volume_id=volume_id
    )
    
    try:
        # Create SSH key pair
        benchmark.create_key_pair()
        
        # Download model if needed
        if not benchmark.volume_id:
            models = config['Models']
            logging.info(f"Downloading model with {models} models")
            volume_id = benchmark.download_model(models)
            logging.info(f"Created volume {volume_id} with model")
            benchmark.volume_id = volume_id

        commands=config["Commands"]
        instances=config["Instances"]
        for instance in instances['CPU']:
            logging.info(f"Setting up benchmark instance of type {instance}")
            benchmark.setup_benchmark_instance(
                instance_type=instance,
                volume_id=benchmark.volume_id
            )
            for command in commands:
                logging.info(f"Running benchmark {command}")
                output = benchmark.run_benchmark(
                    prompt=command['Prompt'],
                    tokens=command['Tokens'],
                    models=config['Models']
                )
                print("Benchmark results:", output)
        
    finally:
        # Cleanup but keep the volume
        benchmark.terminate_instance(keep_volume=True)
        print(f"Model volume ID: {benchmark.volume_id}")

if __name__ == '__main__':
    main()