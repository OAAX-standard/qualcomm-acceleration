SUPPORTED_IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png"]

def cli():
    import argparse
    from pathlib import Path
    from lgg import logger
    import traceback as tb
    import shutil
    
    from .logger import Logs
    from .utils import extract_zip, create_inventory, has_dynamic_shapes, \
                        md5_hash, simplify_onnx, get_input_metadata
    from .config_utils import load_config
    
    from .quantize import DataReader, quantize_onnx
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Quantize an ONNX model')
    parser.add_argument('-z', '--zip-path', required=True, help='Path to the Zip file')
    parser.add_argument('-o', '--output-dir', required=True, help='Output directory')
    args = parser.parse_args()
    zip_path = Path(args.zip_path).resolve()
    output_dir = Path(args.output_dir).resolve()

    # check that zip_path exists
    if not zip_path.exists():
        logger.error(f'Zip file not found: {zip_path}')
        return
    # check that output_dir doesn't exist
    if output_dir.exists():
        # check if it's empty
        if list(output_dir.glob('*')):
            logger.error(f'Output directory is not empty: {output_dir}')
            return
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
            
    # Initialize logs
    logs = Logs()        
    
    try:
        # Extract the zip file
        extracted_dir = extract_zip(zip_path)
        
        # Create an inventory of the extracted files
        inventory = create_inventory(extracted_dir, recursive=True)
        
        # Ensure that there's exactly am ONNX, a JSON and more than one image in the inventory
        if ".onnx" not in inventory:
            logger.error('ONNX file not found in the Zip file')
            return
        if ".json" not in inventory:
            logger.error('JSON file not found in the Zip file')
            return
        if len(inventory[".onnx"]) > 1:
            logger.error('More than one ONNX file found in the Zip file')
            return
        if len(inventory[".json"]) > 1:
            logger.error('More than one JSON file found in the Zip file')
            return
        image_files = []
        for ext in SUPPORTED_IMAGE_EXTENSIONS:
            if ext in inventory:
                image_files.extend(inventory[ext])
        number_of_images = len(image_files)
        if number_of_images <= 1 :
            logger.error('Less than two images found in the Zip file')
            return
        
        # Log the inventory
        logs.add_message('Inventory of files in the archive', inventory)
        
        # Check ONNX model
        onnx_path = Path(inventory[".onnx"][0])
        if has_dynamic_shapes(onnx_path):
            logger.error('Dynamic shapes are not supported')
            return
        
        # Simplify the onnx
        onnx_path = simplify_onnx(onnx_path)
        
        # Load the JSON file
        json_path = Path(inventory[".json"][0])
        config = load_config(json_path)
        if config is None:
            logger.error('Failed to load the config JSON file')
            return
        
        # Extract input metadata from the ONNX model
        input_metadata = get_input_metadata(onnx_path)
        if input_metadata is None:
            logger.error('Failed to extract input metadata from the ONNX model')
            return
        
        # Load data reader
        data_reader = DataReader(image_files, config, input_metadata)
        
        # Quantize the model
        output_onnx = quantize_onnx(onnx_path, data_reader, config)
        
        # copy quantized ONNX in output directory
        output_path = output_dir / 'model.onnx'
        shutil.copy(output_onnx, output_path)
        
        logs.add_message('Quantization completed successfully',
                         {'Output ONNX': output_path,
                          "MD5 Hash": md5_hash(output_path)})
        
        logger.info("Conversion completed successfully")
    except Exception as e:
        tb.print_exc()
        logs.add_message('Failed to convert the model', {'Error': str(e)})
        logger.error(f'Failed to convert the model: {e}')
    finally:
        # Save the logs as a JSON file in the output directory
        logs.save_as_json(output_dir / 'logs.json')

