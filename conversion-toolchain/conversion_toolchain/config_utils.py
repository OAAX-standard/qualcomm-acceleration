from pydantic import BaseModel, ValidationError
from lgg import logger
from pathlib import Path
import json


class InputMetadata(BaseModel):
    Name: str
    Width: int
    Height: int
    Grayscale: bool
    NCHW: bool
    
class ModelConfig(BaseModel):
    Means: list[float]
    Stds: list[float]
    
    
def load_config(json_path: Path) -> ModelConfig:
    """Load config from a JSON file.
    
    Args:
        json_path (str): Path to JSON file.
    
    Returns:
        ModelConfig: The configuration.
    """
    logger.info(f"Loading config from JSON file")
    with open(json_path, 'r') as f:
        js = json.load(f)
    try:
        config = ModelConfig(**js)
    except ValidationError as e:
        logger.debug(f"Encountered an error while loading the JSON file: {e}")
        return None
    return config
