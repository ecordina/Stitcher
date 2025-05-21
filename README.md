# Stitcher

This script stitches tiled TIFF or IMS images exported from **Dragonfly** using tile position metadata provided in an associated XML file. It supports z-stack max projection, background subtraction, overlap handling (cropping or blending), and output to TIFF or Zarr.

---

## Features

- Supports **TIFF** and **IMS (Imaris)** formats
- Reads tile positions from **Dragonfly-generated XML** metadata
- Optional **Z-axis max projection**
- Optional **background subtraction**
- Handles **tile overlaps**:
  - Crop overlapping regions
  - Max-intensity blending of overlaps
- Output to **TIFF** or **Zarr**
- Generates a **thumbnail preview** of the stitched result

---

## Installation

### Dependencies

Install dependencies with `mamba` and `pip`:

```bash
mamba install matplotlib tqdm scikit-image python zarr xarray
pip install imaris-ims-file-reader
