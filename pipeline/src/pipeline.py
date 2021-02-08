import kfp
import os
print(kfp.__version__)
import boto3
import kfp.components as comp
from kfp import dsl
# Load the component by calling load_component_from_file or load_component_from_url
# To load the component, the pipeline author only needs to have access to the component.yaml file.
# The Kubernetes cluster executing the pipeline needs access to the container image specified in the component.
#dummy_op = kfp.components.load_component_from_file(os.path.join(component_root, 'component.yaml')) 
from pathlib import Path

####################################################################################
#component_root = "https://raw.githubusercontent.com/Benjamin-Etheredge/kubeflow_pipelines/master/components"
# TODO why did changing the github name have no affect on this?
component_root = "https://raw.githubusercontent.com/Benjamin-Etheredge/pipeline_components/master/pipeline/components"
def load_component(dir_name):
    filename = os.path.join(component_root, dir_name, 'component.yaml')
    print(f"Loading {filename}")
    return kfp.components.load_component_from_url(filename)

####################################################################################

def load_local_component(dir_name):
    filename = os.path.join('component.yaml')
    print(f"Loading {filename}")
    return kfp.components.load_component_from_url(filename)

####################################################################################

from kubernetes.client.models import V1EnvVar
def add_envs(op, vars):
    out_op = op
    for var in vars:
        out_op = add_env(out_op, var)
    return out_op

####################################################################################

def add_env(op, var):
    return op.add_env_variable(V1EnvVar(name=var, value=os.getenv(var)))

####################################################################################

def is_available(bucket) -> int:

    import boto3
    import botocore
    from botocore.client import Config
    import os
    s3 = boto3.resource('s3',
                    endpoint_url=os.getenv('S3_ENDPOINT'),
                    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
                    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
                    config=Config(signature_version='s3v4'),
                    region_name='us-east-1')
    count = len(list(iter(s3.Bucket(bucket).objects.all())))
    print(f"count: {count}")
    return count

def upload_data(bucket):
    import boto3
    import botocore
    from botocore.client import Config
    import os
    s3 = boto3.resource('s3',
                    endpoint_url=os.getenv('S3_ENDPOINT'),
                    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
                    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
                    config=Config(signature_version='s3v4'),
                    region_name='us-east-1')
    
    print(f"count: {count}")
    return count


def combine_dirs(dir1: str, dir2: str, out_dir: str):
    from pathlib import Path
    import shutil
    from os.path import basename
    dir1 = Path(dir1)
    dir2 = Path(dir2)
    
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exists_ok=True)
    for file in [*dir1.glob('*'), *dir2.glob('*')]:
        shutil.copyfile(str(file), out_dir/basename(file))
    
    

####################################################################################

BUCKET = 'gloves-clean'
BUCKET_URL = f's3://{BUCKET}'
# Define a pipeline and create a task from a component:
@kfp.dsl.pipeline(
    name='Exploring Reusability pipeline', 
    description='Pipeline to try out reusable components')
def my_pipeline(
    data_url="https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz",
    dense_nodes: int = 1024,
    epochs: int = 1,
    batch_size: int = 32,
    lr: float = 0.0003, 
    optimizer: str = "adam",
    transfer_learning: bool = False,
    verbose: int = 2,
    test_split: float = 0.2,
    #left_over_split: float = 0.8
    val_split: float = 0.2,
    
    random_seed: int = 4
):
    

    ##########################################################################
    
    is_available_op = comp.create_component_from_func(
        is_available, output_component_file='is_aval_component.yaml',
        packages_to_install=['boto3==1.16.57']) 
    
    
    is_available_task = is_available_op(bucket=BUCKET)
    add_envs(is_available_task, [
            'S3_ENDPOINT', 'AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY'])
    
    ##########################################################################
    
    with kfp.dsl.Condition(is_available_task.output <= 0):
        get_data_op = load_component("wget_url")
        get_data_task = get_data_op(
            data_url=data_url
            #output=Path("data"),
        )

        extract_data_op = load_component("untar_data")
        extract_data_task = extract_data_op(
            path_to_tar_file=get_data_task.outputs['output_file'],
            tar_file_name='images.tar.gz'
        )
   
        clean_data_op = load_component("clean_oxford_pet_data")
        clean_data_task = clean_data_op(
            data_dir=extract_data_task.outputs['data_dir'])
        
        s3_data_upload_task = kfp.dsl.ContainerOp(
            name='s3 uploader',
            image='amazon/aws-cli',
            arguments=[
                's3',
                '--endpoint-url', os.getenv('S3_ENDPOINT'),
                'sync', 
                clean_data_task.outputs['cleaned_data_dir'],
                BUCKET_URL,
            ]
        )
  
    s3_data_task = kfp.dsl.ContainerOp(
        name='s3 getter',
        image='amazon/aws-cli',
        arguments=[
            's3',
            '--endpoint-url', os.getenv('S3_ENDPOINT'),
            'sync', 
            BUCKET_URL,
            '/output'
        ],
        file_outputs={
            'output': '/output',
        }
    ).after(s3_data_upload_task)
    add_envs(s3_data_task, [
        'S3_ENDPOINT', 'AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY'])
    # TODO why does this work^ it's not assigning return?

    ##########################################################################
    
    split_op = load_component("split_oxford_pet_data")
    #test_split_task = split_op(
        #data_dir=s3_data_task.outputs['output'],
        #split_ratio=test_split,
        #random_seed=random_seed,
    #)
    val_split_task = split_op(
        data_dir=s3_data_task.outputs['output'],
        #data_dir=test_split_task.outputs['train_dir'],
        split_ratio=val_split, # I would like it to be val_split of whole, but oh well
        random_seed=random_seed,
    )

    ##########################################################################
    
    s3_my_pets_data_task = kfp.dsl.ContainerOp(
        name='my pets getter',
        image='amazon/aws-cli',
        arguments=[
            's3',
            '--endpoint-url', os.getenv('S3_ENDPOINT'),
            'sync', 
            's3://my-pets',
            '/output'
        ],
        file_outputs={
            'output': '/output',
        }
    )
    s3_my_pets_data_task = add_envs(s3_my_pets_data_task, [
        'S3_ENDPOINT', 'AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY'])
    '''
    fuzer_op = comp.create_component_from_func(
        combine_dirs, output_component_file='combine_component.yaml',
        #packages_to_install=['boto3==1.16.57']
    ) 
    fuzer_task = fuzer_op(
        dir1=s3_my_pets_data_task.outputs['output'],
        dir2=val_split_task.outputs['train_dir'],
        out_dir='/output')
    
    data_fuzer_task = kfp.dsl.ContainerOp(
        name='data fuzer',
        image='alpine',
        command='cp',
        arguments=[
            '-r',
            s3_my_pets_data_task.outputs['output'],
            val_split_task.outputs['train_dir'],
            '/output'
        ],
        artifact_argument_paths=[
            
        ]
        file_outputs={
            'output': '/output',
        }
    )
    '''
    ##########################################################################
    
    join_op = load_component("data_joiner")
    join_task = join_op(
        dir_1=val_split_task.outputs['train_dir'],
        dir_2=s3_my_pets_data_task.outputs['output'],
    ).set_display_name('join-val-mine-oxford-data')
    # TODO: it seems I can't use the same output twice as args?
    join_all_task = join_op(
        dir_1=s3_data_task.outputs['output'],
        dir_2=s3_my_pets_data_task.outputs['output'],
    ).set_display_name('join-all-mine-oxford-data')
    
    ##########################################################################
    
    # can't use files since it can't reach it later on...
    train_op = kfp.components.load_component_from_url(
        "https://raw.githubusercontent.com/Benjamin-Etheredge/Gloves/master/pipeline/component/component.yaml")
    data_sets = [
        [
            join_task.outputs['out_dir'],
            val_split_task.outputs['test_dir'],
            join_all_task.outputs['out_dir']
        ],
        [
            val_split_task.outputs['train_dir'],
            val_split_task.outputs['test_dir'],
            s3_data_task.outputs['output']
        ]
    ]
    for train_dir, test_dir, all_dir in data_sets:
        train_task = train_op(
            train_dir=train_dir,
            test_dir=test_dir,
            #all_dir=join_task.outputs['out_dir'],
            all_dir=all_dir,
            #all_dir=test_split_task.outputs['train_dir'],
            #metrics_file_name="metrics.yaml",
            model_filename="model.h5",
            # hypers
            dense_nodes=dense_nodes,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            optimizer=optimizer,
            #transfer_learning=transfer_learning,
        ).set_gpu_limit(1)
        train_task = add_envs(train_task, [
            'S3_ENDPOINT', 'MLFLOW_S3_ENDPOINT_URL', 'MLFLOW_TRACKING_URI',
            'MLFLOW_TRACKING_USERNAME', 'MLFLOW_TRACKING_PASSWORD', 
            'TF_FORCE_GPU_ALLOW_GROWTH', 'AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY'])
            #os.environ['PYTHONHASHSEED']=str(4)
            
    ##########################################################################
    
    label_split_op = load_component("split_off_label")
    mittens_data_op = label_split_op(
        data_dir=s3_my_pets_data_task.outputs['output'],
        label='mittens',
    ).set_display_name('split-mittens-data')
    #mittens_data_op.set_display_name('split-mittens-data')
    dave_data_task = label_split_op(
        data_dir=s3_my_pets_data_task.outputs['output'],
        label='dave',
    ).set_display_name('split-dave-data')
    maine_coon_task = label_split_op(
        data_dir=s3_data_task.outputs['output'],
        label='Maine_Coon'
    ).set_display_name('split-maine-coon-data')
    boxer_task = label_split_op(
        data_dir=s3_data_task.outputs['output'],
        label='boxer'
    ).set_display_name('split-boxer-data')
    
    mittens_classifier_data_task =join_op(
        dir_1=maine_coon_task.outputs['label_dir'],
        dir_2=mittens_data_op.outputs['label_dir']
    ).set_display_name('combine-mittens-maine-coon-data')
    
    dave_classifier_data_task =join_op(
        dir_1=boxer_task.outputs['label_dir'],
        dir_2=dave_data_task.outputs['label_dir']
    ).set_display_name('combine-dave-boxer-data')
    
    
    ##########################################################################
    tf_data_op = load_component("oxford_pet_tf_dataset")
    tf_data_task = tf_data_op(
        #data_dir=mittens_classifier_data_task.outputs['out_dir'],
        data_dir=s3_data_task.outputs['output'],
        height=224,
        width=224,
    ).set_display_name('create-tf-data')
    cls_op = load_component("imagenet_classifier")
    cls_task = cls_op(
        data_dir=tf_data_task.outputs['tf_data_dir'],
        label_dir=tf_data_task.outputs['tf_label_dir'],
        height=224,
        width=224
    ).set_display_name('oxford-imagenet-classifier')
        
        
    ##########################################################################
            

if __name__ == "__main__":
    # This pipeline can be compiled, uploaded and submitted for execution.
    #kfp.Client().create_run_from_pipeline_func(my_pipeline, arguments={})
    import kfp.compiler as compiler
    # TODO cleanup absolute path
    compiler.Compiler().compile(pipeline_func=my_pipeline, package_path="/pipeline/gloves_pipeline.tar.gz")

