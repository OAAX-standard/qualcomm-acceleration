import numpy as np
from PIL import Image
from onnxruntime.quantization import CalibrationDataReader
from lgg import logger

from ..config_utils import ModelConfig, InputMetadata


class DataReader(CalibrationDataReader):
    def __init__(self, images_path: list[str], config: ModelConfig, input_metadata: InputMetadata):
        self.enum_data = None

        # Load images 
        images = self.load_images(images_path, input_metadata.Width, input_metadata.Height, input_metadata.Grayscale)
        # Normalize images
        inputs = self.normalize_images(images, config.Means, config.Stds)
        # Reshape images
        inputs = self.reshape_images(inputs, input_metadata.NCHW)
        
        # build input dict 
        self.data_list = [{input_metadata.Name: array} for array in inputs]

        self.datasize = len(self.data_list)
        
        logger.debug(f"Loaded {self.datasize} images for calibration")

    def get_next(self):
        if self.enum_data is None:
            self.enum_data = iter(
                self.data_list
            )
        return next(self.enum_data, None)

    def rewind(self):
        self.enum_data = None
        
    @staticmethod
    def load_images(images_path: list[str], width: int, height: int, grayscale: bool):
        images = []
        for image_path in images_path:
            try:
                image = Image.open(image_path)
                image = image.convert("RGB").resize((width, height))
                if grayscale:
                    image = image.convert("L")
                image = np.array(image)
            except Exception as e:
                logger.warning(f"Failed to load image: {e}")
                continue
            images.append(image)
        return images
    
    @staticmethod
    def normalize_images(images: list[np.array], means: list[float], stds: list[float]):
        means = np.array(means)
        stds = np.array(stds)
        images = [(image - means) / stds for image in images]
        images = [image.astype('float32') for image in images]
        return images

    @staticmethod
    def reshape_images(images: list[np.array], nchw: bool):
        images = [np.expand_dims(image, axis=0) for image in images]
        if nchw:
            images = [np.transpose(image, (0, 3, 1, 2)) for image in images]
        return images