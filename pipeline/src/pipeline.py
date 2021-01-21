import kfp
import os
print(kfp.__version__)
# Load the component by calling load_component_from_file or load_component_from_url
# To load the component, the pipeline author only needs to have access to the component.yaml file.
# The Kubernetes cluster executing the pipeline needs access to the container image specified in the component.
#dummy_op = kfp.components.load_component_from_file(os.path.join(component_root, 'component.yaml')) 
from pathlib import Path
component_root = "https://raw.githubusercontent.com/Benjamin-Etheredge/kubeflow_pipelines/master/components"
def load_component(dir_name):
    filename = os.path.join(component_root, dir_name, 'component.yaml')
    print(f"Loading {filename}")
    return kfp.components.load_component_from_url(filename)

def load_local_component(dir_name):
    filename = os.path.join('component.yaml')
    print(f"Loading {filename}")
    return kfp.components.load_component_from_url(filename)

from kubernetes.client.models import V1EnvVar
def add_envs(op, vars):
    out_op = op
    for var in vars:
        out_op = add_env(out_op, var)
    return out_op

def add_env(op, var):
    return op.add_env_variable(V1EnvVar(name=var, value=os.getenv(var)))

# Define a pipeline and create a task from a component:
@kfp.dsl.pipeline(
    name='Exploring Reusability pipeline', 
    description='Pipeline to try out reusable components')
def my_pipeline(
    data_url="https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz",
    dense_nodes: int = 1024,
    epochs: int = 100,
    batch_size: int = 32,
    lr: float = 0.0003, 
    optimizer: str = "adam",
    transfer_learning: bool = True,
    verbose: int = 2,
    split_ratio=0.1
):
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
  
    split_op = load_component("split_oxford_pet_data")
    split_task = split_op(
        data_dir=clean_data_task.outputs['cleaned_data_dir'],
        split_ratio=split_ratio
    )

    # can't use files since it can't reach it later on...
    train_op = kfp.components.load_component_from_file("/pipeline/component.yaml")
    train_op = kfp.components.load_component_from_url(
        "https://raw.githubusercontent.com/Benjamin-Etheredge/Gloves/master/component.yaml")
    train_task = train_op(
        train_dir=split_task.outputs['train_dir'],
        test_dir=split_task.outputs['test_dir'],
        all_dir=clean_data_task.outputs['cleaned_data_dir'],
        metrics_file_name="metrics.yaml",
        model_file_name="model.h5",
        # hypers
        dense_nodes=dense_nodes,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        optimizer=optimizer,
        #transfer_learning=transfer_learning,
    )
    train_task = add_envs(train_task, [
        'S3_ENDPOINT', 'MLFLOW_S3_ENDPOINT_URL', 'MLFLOW_TRACKING_URI',
        'MLFLOW_TRACKING_USERNAME', 'MLFLOW_TRACKING_PASSWORD', 
        'TF_FORCE_GPU_ALLOW_GROWTH', 'AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY'])
        #os.environ['PYTHONHASHSEED']=str(4)
      

if __name__ == "__main__":
    # This pipeline can be compiled, uploaded and submitted for execution.
    #kfp.Client().create_run_from_pipeline_func(my_pipeline, arguments={})
    import kfp.compiler as compiler
    # TODO cleanup absolute path
    compiler.Compiler().compile(pipeline_func=my_pipeline, package_path="/pipeline/gloves_pipeline.tar.gz")

