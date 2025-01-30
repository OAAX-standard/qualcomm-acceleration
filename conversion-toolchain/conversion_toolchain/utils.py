from onnx import load, save
from os.path import splitext
import hashlib
from onnxsim import simplify

from pathlib import Path
import tempfile
from zipfile import ZipFile
import uuid
from lgg import logger

from .config_utils import InputMetadata

def get_random_temp_path():
    # Create a temporary directory
    temp_dir = tempfile.gettempdir()
    output_file = Path(temp_dir) / (str(uuid.uuid4()) + ".onnx")
    return output_file

def simplify_onnx(onnx_path: Path) -> Path:
    logger.info('Simplifying ONNX model')
    
    model, check = simplify(onnx_path.as_posix(), check_n=1)
    if not check:
        logger.debug('Failed to simplify ONNX model')
        return onnx_path
    
    output_path = get_random_temp_path()
    save(model, output_path)

    return output_path

def md5_hash(file_path):
    with open(file_path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()
    
def extract_zip(zip_path: Path) -> Path:
    # Create a temporary directory
    temp_dir = tempfile.gettempdir()
    output_dir = Path(temp_dir) / str(uuid.uuid4())
    output_dir.mkdir(parents=True, exist_ok=True)
    # Extract the zip file
    with ZipFile(zip_path.as_posix(), 'r') as zip_ref:
        zip_ref.extractall(output_dir)
    return output_dir

def create_inventory(folder_path: Path, recursive=True):
    """Creates an inventory of files in a folder.
    
    Args:
        folder_path (str): The path to the folder to inventory.
        recursive (bool): Whether to include files in subdirectories.
    
    Returns:
        dict: A dictionary with file extensions as keys and full paths as values.
    """
    # Create an empty dictionary to store the inventory
    inventory = {}
    # Determine the function to use for iterating over the files
    if recursive:
        function = folder_path.rglob
    else:
        function = folder_path.iterdir
    # Iterate over the files in the folder
    for file_path in function('*'):
        if file_path.is_file():
            extension = file_path.suffix
            if extension not in inventory:
                inventory[extension] = []
            inventory[extension].append(file_path)
    return inventory


def has_dynamic_shapes(onnx_path: Path) -> bool:
    """Check if the ONNX model has dynamic shapes.

    Args:
        onnx_path (str): Path to ONNX model.

    Returns:
        bool: True if the ONNX model has dynamic shapes.
    """
    logger.info('Checking dynamic shapes')
    model = load(onnx_path)
    for i in model.graph.input:
        # check if the input has a dynamic axis
        for d in i.type.tensor_type.shape.dim:
            if d.HasField('dim_param'):
                logger.debug(f'Input {i.name} has dynamic shape')
                return True
    return False


def get_input_metadata(onnx_path: Path) -> InputMetadata:
    """Get the input shapes of an ONNX model.

    Args:
        onnx_path (str): Path to ONNX model.

    Returns:
        InputMetadata: A dataclass with input 
    """
    logger.info('Getting ONNX input metadata')
    model = load(onnx_path)
    
    # Check if the model has no inputs
    if len(model.graph.input) == 0:
        logger.debug('Model has no inputs')
        return None
    
    # Check if the model has multiple inputs
    if len(model.graph.input) > 1:
        logger.debug('Model has multiple inputs')
        return None
    
    # Read the only input tensor
    input_tensor = model.graph.input[0]
    input_shape = [int(d.dim_value) for d in input_tensor.type.tensor_type.shape.dim]

    # Ensure input shape is 4D
    if len(input_shape) != 4:
        logger.debug('Input shape is not 4D')
        return None
    
    # Read input name and dimensions    
    Name = input_tensor.name
    # Extract order of dimensions: NCHW or NHWC
    if input_shape[1] == 1 or input_shape[1] == 3:      # NCHW
        NCHW = True
        Grayscale = input_shape[1] == 1
        Height = input_shape[2]
        Width = input_shape[3]
    elif input_shape[-1] == 3 or input_shape[-1] == 1:   # NHWC
        NCHW = False
        Height = input_shape[1]
        Width = input_shape[2]
        Grayscale = input_shape[-1] == 1
    else:
        logger.debug('Input shape is not in NCHW or NHWC format: ' + str(input_shape))
        return None
        
    return InputMetadata(Name=Name, Width=Width, Height=Height, Grayscale=Grayscale, NCHW=NCHW)