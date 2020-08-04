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


# Load two more components for importing and exporting the data:
#download_from_gcs_op = kfp.components.load_component_from_url('http://....../component.yaml')
#upload_to_gcs_op = kfp.components.load_component_from_url('http://....../component.yaml')

# dummy_op is now a "factory function" that accepts the arguments for the component's inputs
# and produces a task object (e.g. ContainerOp instance).
# Inspect the dummy_op function in Jupyter Notebook by typing "dummy_op(" and pressing Shift+Tab
# You can also get help by writing help(dummy_op) or dummy_op? or dummy_op??
# The signature of the dummy_op function corresponds to the inputs section of the component.
# Some tweaks are performed to make the signature valid and pythonic:
# 1) All inputs with default values will come after the inputs without default values
# 2) The input names are converted to pythonic names (spaces and symbols replaced
#    with underscores and letters lowercased).

#output_model_uri_template = os.path.join(, kfp.dsl.EXECUTION_ID_PLACEHOLDER, 'output_model_uri', 'data')
# Define a pipeline and create a task from a component:
@kfp.dsl.pipeline(name='Exploring Reusability pipeline', description='Pipeline to try out reusable components')
def my_pipeline():
    get_data_op = load_component("wget_url")
    get_data_task = get_data_op(
        # Input name "Input 1" is converted to pythonic parameter name "input_1"
        data_url=r"https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz",
        #output=Path("data"),
    )

    extract_data_op = load_component("untar_data")
    extract_data_task = extract_data_op(
        tar_file=get_data_task.outputs['output_file']
    )

    clean_data_op = load_component("clean_oxford_pet_data")
    clean_data_task = clean_data_op(
        data_dir=extract_data_task.outputs['data_dir']
    )
    

# This pipeline can be compiled, uploaded and submitted for execution.
#kfp.Client().create_run_from_pipeline_func(my_pipeline, arguments={})
import kfp.compiler as compiler
compiler.Compiler().compile(pipeline_func=my_pipeline, package_path="/pipeline/gloves_pipeline.tar.gz")