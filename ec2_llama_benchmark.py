import boto3
import paramiko
import time
import os
import logging
from datetime import datetime
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'llama_benchmark_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logging.getLogger("paramiko").setLevel(logging.DEBUG) # for example

class EC2LlamaBenchmark:
    def __init__(self, instance_family: str = "t4g", instance_size: str = "medium"):
        self.elastic_ip_allocation_id = None
        self.elastic_ip = None
        self.ec2_client = boto3.client('ec2')
        self.ec2_resource = boto3.resource('ec2')
        self.instance_type = f"{instance_family}.{instance_size}"
        self.key_name = "llama-benchmark-key"
        self.instance = None
        self.instance_id = None
        self.vpc = None
        
    def create_key_pair(self):
        """Create a key pair for SSH access"""
        logging.info("Creating SSH key pair...")
        try:
            key_pair = self.ec2_client.create_key_pair(KeyName=self.key_name)
            # Save private key
            with open(f"{self.key_name}.pem", 'w') as f:
                f.write(key_pair['KeyMaterial'])
            os.chmod(f"{self.key_name}.pem", 0o400)
        except Exception as e:
            logging.info(f"Key pair already exists: {e}")

    def create_security_group(self):
        """Create security group with SSH access"""
        logging.info("Creating security group...")
        try:
            self.vpc = list(self.ec2_resource.vpcs.all())[0]
            security_group = self.ec2_resource.create_security_group(
                GroupName='llama-benchmark-sg',
                Description='Security group for Llama benchmark',
                VpcId=self.vpc.id
            )
            security_group.authorize_ingress(
                IpProtocol='tcp',
                FromPort=22,
                ToPort=22,
                CidrIp='0.0.0.0/0'
            )
            logging.info("Security group created successfully")
            return security_group.id
        except Exception as e:
            logging.warning(f"Security group might already exist: {e}")
            security_groups = self.ec2_client.describe_security_groups(
                Filters=[{'Name': 'group-name', 'Values': ['llama-benchmark-sg']}]
            )
            return security_groups['SecurityGroups'][0]['GroupId']

    def launch_instance(self):
        """Launch EC2 instance with llama-cpp installation"""
        security_group_id = self.create_security_group()
        
        # User data script to install llama-cpp and llmperf
        user_data = """#!/bin/bash
sudo apt-get update
sudo apt-get install -y python3-pip git cmake
"""

        # Get subnet in the same VPC that's public (has route to internet gateway)
        # vpc = list(self.ec2_resource.vpcs.all()).sort()[0]
        vpc = self.vpc
        subnets = self.ec2_client.describe_subnets(
            Filters=[{'Name': 'vpc-id', 'Values': [vpc.id]}]
        )
        public_subnet = None
        for subnet in subnets['Subnets']:
            route_tables = self.ec2_client.describe_route_tables(
                Filters=[{'Name': 'association.subnet-id', 'Values': [subnet['SubnetId']]}]
            )
            for route_table in route_tables['RouteTables']:
                for route in route_table['Routes']:
                    if route.get('GatewayId', '').startswith('igw-'):
                        public_subnet = subnet['SubnetId']
                        break

        response = self.ec2_client.run_instances(
            ImageId='ami-0a7a4e87939439934',  # Ubuntu 22.04 LTS AMI
            InstanceType=self.instance_type,
            KeyName=self.key_name,
            MinCount=1,
            MaxCount=1,
            SecurityGroupIds=[security_group_id],
            SubnetId=public_subnet,
            UserData=user_data,
            BlockDeviceMappings=[{
                'DeviceName': '/dev/sda1',  # Root volume
                'Ebs': {
                    'VolumeSize': 300,
                    'VolumeType': 'gp3'
                }
            }]
        )
        
        self.instance_id = response['Instances'][0]['InstanceId']
        self.instance = self.ec2_resource.Instance(self.instance_id)
        
        logging.info(f"Waiting for instance {self.instance_id} to be running...")
        self.instance.wait_until_running()
        
        # Wait for status checks
        waiter = self.ec2_client.get_waiter('instance_status_ok')
        waiter.wait(InstanceIds=[self.instance_id])
        
        # Allocate and associate Elastic IP
        allocation = self.ec2_client.allocate_address(Domain='vpc')
        self.elastic_ip_allocation_id = allocation['AllocationId']
        self.elastic_ip = allocation['PublicIp']
        
        self.ec2_client.associate_address(
            InstanceId=self.instance_id,
            AllocationId=self.elastic_ip_allocation_id
        )
        
        return self.elastic_ip

    def run_benchmark(self, ip_address: str):
        """Run llmperf benchmark via SSH"""
        logging.info("Initializing SSH connection for benchmark...")
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        
        # Wait for instance to be ready
        logging.info("Waiting 60 seconds for instance setup to complete...")
        time.sleep(60)  # Wait for user data script to complete
        
        try:
            ssh.connect(ip_address, username='ubuntu', key_filename=f"{self.key_name}.pem")
            
            # Install required packages
            commands = [
                'sudo apt-get update',
                'sudo apt-get install -y python3-pip git cmake pipx',
                'git clone https://github.com/ggerganov/llama.cpp.git',
                'mkdir -p ~/llama.cpp/build && cd ~/llama.cpp/build && cmake .. -DCMAKE_CXX_FLAGS="-mcpu=native" -DCMAKE_C_FLAGS="-mcpu=native" && cmake --build . -v --config Release -j $nproc',
                'mkdir -p ~/llama.cpp/bin && cd ~/llama.cpp/bin && wget https://huggingface.co/bartowski/DeepSeek-R1-Distill-Llama-70B-GGUF/resolve/main/DeepSeek-R1-Distill-Llama-70B-Q4_0.gguf',
                'pipx install install "huggingface_hub[cli]"',
                'mkdir -p ~/llama.cpp/bin && cd ~/llama.cpp/bin && ~/.local/bin/huggingface-cli download bartowski/DeepSeek-R1-Distill-Llama-70B-GGUF DeepSeek-R1-Distill-Llama-70B-Q4_0.gguf',
                'cd ~/llama.cpp/bin && ./llama-cli -m DeepSeek-R1-Distill-Llama-70B-Q4_0.gguf -p "Building a visually appealing website can be done in ten simple steps:/" -n 512 -t 64 -no-cnv',
                'pip3 install llmperf'
            ]
            
            output = []
            for cmd in commands:
                logging.info(f"Executing command: {cmd}")
                stdin, stdout, stderr = ssh.exec_command(cmd)
                stdout_str = stdout.read().decode()
                stderr_str = stderr.read().decode()
                if stderr_str:
                    logging.warning(f"Command stderr: {stderr_str}")
                logging.info(f"Command stdout: {stdout_str}")
                output.append(stdout_str)
                
            # Run benchmark
            logging.info("Starting llmperf benchmark...")
            stdin, stdout, stderr = ssh.exec_command('cd llama.cpp && llmperf benchmark')
            stdout_str = stdout.read().decode()
            stderr_str = stderr.read().decode()
            if stderr_str:
                logging.warning(f"Benchmark stderr: {stderr_str}")
            logging.info(f"Benchmark stdout: {stdout_str}")
            output.append(stdout_str)
            
            # Combine all output
            output = '\n'.join(output)
            with open('benchmark_results.txt', 'w') as f:
                f.write(output)
                
            logging.info("Benchmark results saved to benchmark_results.txt")
            
        finally:
            ssh.close()

    def terminate_instance(self):
        """Terminate the EC2 instance and cleanup all associated resources"""
        # Ask for user confirmation before cleanup
        confirmation = input("Are you sure you want to terminate the instance and cleanup all resources? (yes/no): ")
        if confirmation.lower() != 'yes':
            logging.info("Operation cancelled by user")
            return
            
        if self.elastic_ip_allocation_id:
            try:
                # Disassociate the Elastic IP from the instance
                self.ec2_client.disassociate_address(AssociationId=self.elastic_ip_allocation_id)
                # Release the Elastic IP
                self.ec2_client.release_address(AllocationId=self.elastic_ip_allocation_id)
                logging.info(f"Released Elastic IP {self.elastic_ip}")
            except Exception as e:
                logging.info(f"Error cleaning up Elastic IP: {str(e)}")
            
        if self.instance_id:
            self.ec2_client.terminate_instances(InstanceIds=[self.instance_id])
            logging.info(f"Terminated instance {self.instance_id}")
            # Wait for instance to terminate before cleaning up security group
            waiter = self.ec2_client.get_waiter('instance_terminated')
            waiter.wait(InstanceIds=[self.instance_id])
            
        # Clean up security group
        try:
            security_groups = self.ec2_client.describe_security_groups(
                Filters=[{'Name': 'group-name', 'Values': ['llama-benchmark-sg']}]
            )
            if security_groups['SecurityGroups']:
                sg_id = security_groups['SecurityGroups'][0]['GroupId']
                self.ec2_client.delete_security_group(GroupId=sg_id)
                logging.info(f"Deleted security group {sg_id}")
        except Exception as e:
            logging.info(f"Error cleaning up security group: {str(e)}")
            
        # Clean up key pair
        try:
            self.ec2_client.delete_key_pair(KeyName=self.key_name)
            logging.info(f"Deleted key pair {self.key_name}")
        except Exception as e:
            logging.info(f"Error cleaning up key pair: {str(e)}")
            
        self.elastic_ip = None
        self.elastic_ip_allocation_id = None

def main():
    logging.info("Starting Llama benchmark process...")
    # Create benchmark instance
    benchmark = EC2LlamaBenchmark(instance_family="t4g", instance_size="medium")
    
    try:
        # Setup and launch instance
        benchmark.create_key_pair()
        ip_address = benchmark.launch_instance()
        logging.info(f"Instance launched with IP: {ip_address}")
        
        # Run benchmark
        benchmark.run_benchmark(ip_address)
        
    finally:
        # Cleanup
        logging.info("Cleaning up resources...")
        benchmark.terminate_instance()
        logging.info("Benchmark process completed")

if __name__ == "__main__":
    main()