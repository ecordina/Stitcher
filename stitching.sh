#!/bin/bash
#SBATCH --job-name=Stitching # Specify the number of GPUs needed per job
#SBATCH --time=1:00:00
#SBATCH --mail-type=NONE
#SBATCH --mem=50g

# Arguments
FOLDER="/gpfs/commons/home/ecordina/ecordina_innovation/dragonfly_stitching/Stitcher/data/tiles/"
XML_FILE="/gpfs/commons/home/ecordina/ecordina_innovation/dragonfly_stitching/Stitcher/data/tiles/MP_Chao_250415-F1_cs_13.31.15.xml"

# Default Arguments
OUTPUT_PATH="./data/output" #Save in folder where the bash script is
SAVE_FORMAT="tiff" # default to 'tiff' if not specified

# Optional flags
IMAGE_PREFIX="" # In case the image prefix is different than the one in the metadata /!\, has to be complete as the code will look for image_prefix+"_F\d+"
		# ie: For MP_20250411-C1_14.38.46_F000.ims, IMAGE_PREFIX = "MP_20250411-C1_14.38.46" 
MAX_PROJECT="--max_project"  # add --max_project to Z-Max Projection
OVERLAP="" #Leave Empty to scrap overlap from the xml
REFINE_OVERLAP="--refine_overlap" # take it off if you want to force either the given overlap value or the overlap from the xml file
CROP_OVERLAP="--crop_overlap" # leave empty to use max projection on the overlap, add --crop_overlap if you want to use crop overlap
flat_field_path="/gpfs/commons/home/ecordina/ecordina_innovation/dragonfly_stitching/Stitcher/data/flat_field.tiff"
NO_BACKGROUND=""          # leave empty to enable background subtraction
attrs="{'Instrument':'Dragonfly'}" # Attributes to attach to the zarr, only used when save_format == zarr
logger="" # Add --disable_logger to disable logger

# Run the Python script
python /gpfs/commons/home/ecordina/ecordina_innovation/dragonfly_stitching/Stitcher/src/stitching.py \
  --folder "$FOLDER" \
  --xml_file "$XML_FILE" \
  --output_path "$OUTPUT_PATH" \
  --overlap "$OVERLAP" \
  --attrs "$attrs" \
  --image_prefix "$IMAGE_PREFIX" \
  --flat_field_path "$flat_field_path" \
  $MAX_PROJECT \
  $REFINE_OVERLAP \
  $CROP_OVERLAP \
  $NO_BACKGROUND \
  $logger \
  --save_format "$SAVE_FORMAT"
