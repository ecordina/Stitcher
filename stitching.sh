#!/bin/bash
#SBATCH --job-name=Stitching # Specify the number of GPUs needed per job
#SBATCH --time=1:00:00
#SBATCH --mail-type=NONE
#SBATCH --mem=50g

# Arguments
FOLDER=""
XML_FILE="/gpfs/commons/home/ecordina/ecordina_innovation/dragonfly_stitching/old/2025-04-11/ims/MP_Chao_250415-F1_cs_13.31.15.xml"

# Default Arguments
OUTPUT_PATH="./data/output"        # Folder where to save the output
SAVE_FORMAT="tif" 		   # Default to 'tif' if not specified, accepts tif and zarrs

# Optional flags
IMAGE_PREFIX="" 		   # In case the image prefix is different than the one in the metadata /!\, has to be complete as the code will look for image_prefix+"^_F\d+$"
				   # ie: For MP_20250411-C1_14.38.46_F000.ims, IMAGE_PREFIX = "MP_20250411-C1_14.38.46" 
MAX_PROJECT="--max_project"        # Add --max_project to Z-Max Projection
OVERLAP="" 			   # Leave Empty to scrap overlap from the xml, else specify the overlap
REFINE_OVERLAP="--refine_overlap"  # Take it off if you want to force either the given overlap value or the overlap from the xml file
CROP_OVERLAP="--crop_overlap"      # Leave empty to use max projection on the overlap, add --crop_overlap if you want to use crop overlap
flat_field_path=""		   # Flat Field Image if you want to perform Flat-Field Correction, leave empty if not. Be careful that the Flat Field Tile has to be the same dimensions as the image tiles
NO_BACKGROUND=""                   # Leave empty to enable background subtraction, else add --no_background_subtraction
attrs="{'Instrument':'Dragonfly'}" # Attributes to attach to the zarr, only used when save_format == zarr
logger="" 			   # Add --disable_logger to disable logger

# Run the Python script
python ./src/stitching.py \
  --folder "$FOLDER" \
  --xml_file "$XML_FILE" \
  --output_path "$OUTPUT_PATH" \
  --overlap "$OVERLAP" \
  --attrs "$attrs" \
  --image_prefix "$IMAGE_PREFIX" \
  --flat_field_path "$flat_field_path" \
  --save_format "$SAVE_FORMAT" \
  $MAX_PROJECT \
  $REFINE_OVERLAP \
  $CROP_OVERLAP \
  $NO_BACKGROUND \
  $logger \

