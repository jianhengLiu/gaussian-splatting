#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import logging
from argparse import ArgumentParser
import shutil

from colmap_parser import bin2txt

# This Python script is based on the shell converter script provided in the MipNerF 360 repository.
parser = ArgumentParser("Colmap converter")
parser.add_argument("--no_gpu", action="store_true")
parser.add_argument("--skip_matching", action="store_true")
parser.add_argument("--source_path", "-s", required=True, type=str)
parser.add_argument("--camera", default="OPENCV", type=str)
parser.add_argument("--colmap_executable", default="", type=str)
parser.add_argument("--resize", action="store_true")
parser.add_argument("--magick_executable", default="", type=str)
parser.add_argument(
    "--camera_params", type=str, help="camera parameters ('fx,fy,cx,cy,k1,k2,p1,p2')"
)
args = parser.parse_args()
colmap_command = (
    '"{}"'.format(args.colmap_executable)
    if len(args.colmap_executable) > 0
    else "colmap"
)
magick_command = (
    '"{}"'.format(args.magick_executable)
    if len(args.magick_executable) > 0
    else "magick"
)
use_gpu = 1 if not args.no_gpu else 0

# https://colmap.github.io/cli.html
# https://www.cnblogs.com/phillee/p/14335034.html
if not args.skip_matching:
    os.makedirs(args.source_path + "/distorted/sparse", exist_ok=True)

    ## Feature extraction
    if args.camera_params:
        # https://www.cnblogs.com/phillee/p/14335034.html
        # https://github.com/colmap/colmap/blob/ff8842e7d9e985bd0dd87169f61d5aaeb309ab32/src/colmap/sensor/models.h#L239C7-L239C7
        feat_extracton_cmd = (
            colmap_command + " feature_extractor "
            "--database_path " + args.source_path + "/distorted/database.db "
            "--image_path " + args.source_path + "/input "
            "--ImageReader.single_camera 1 "
            "--ImageReader.camera_model " + args.camera + " "
            "--ImageReader.camera_params " + '"' + args.camera_params + '"' + " "
            "--SiftExtraction.use_gpu " + str(use_gpu)
        )
    else:
        feat_extracton_cmd = (
            colmap_command + " feature_extractor "
            "--database_path " + args.source_path + "/distorted/database.db "
            "--image_path " + args.source_path + "/input "
            "--ImageReader.single_camera 1 "
            "--ImageReader.camera_model " + args.camera + " "
            "--SiftExtraction.use_gpu " + str(use_gpu)
        )
    exit_code = os.system(feat_extracton_cmd)
    if exit_code != 0:
        logging.error(f"Feature extraction failed with code {exit_code}. Exiting.")
        exit(exit_code)

    ## Feature matching
    feat_matching_cmd = (
        colmap_command
        + " exhaustive_matcher \
        --database_path "
        + args.source_path
        + "/distorted/database.db \
        --SiftMatching.use_gpu "
        + str(use_gpu)
    )
    exit_code = os.system(feat_matching_cmd)
    if exit_code != 0:
        logging.error(f"Feature matching failed with code {exit_code}. Exiting.")
        exit(exit_code)

    ## Bundle adjustment

    if args.camera_params:
        ## Bundle adjustment with no refinement of intrinsics
        # https://github.com/colmap/colmap/issues/1919
        mapper_cmd = (colmap_command + " mapper \
            --database_path " + args.source_path + "/distorted/database.db \
            --image_path " + args.source_path + "/input \
            --output_path " + args.source_path + "/distorted/sparse \
            --Mapper.ba_global_function_tolerance=0.000001 \
            --Mapper.ba_refine_focal_length 0 \
            --Mapper.ba_refine_principal_point 0 \
            --Mapper.ba_refine_extra_params 0")
    else:
        # The default Mapper tolerance is unnecessarily large,
        # decreasing it speeds up bundle adjustment steps.
        mapper_cmd = (colmap_command + " mapper \
            --database_path " + args.source_path + "/distorted/database.db \
            --image_path "  + args.source_path + "/input \
            --output_path "  + args.source_path + "/distorted/sparse \
            --Mapper.ba_global_function_tolerance=0.000001")

    exit_code = os.system(mapper_cmd)
    if exit_code != 0:
        logging.error(f"Mapper failed with code {exit_code}. Exiting.")
        exit(exit_code)

## Image undistortion

if args.camera_params:
    # Not to warp images, but only to structure files

    files = os.listdir(args.source_path + "/input")
    if not os.path.exists(args.source_path + "/images"):
        os.makedirs(args.source_path + "/images")
    for file in files:
        source_file = os.path.join(args.source_path, "input", file)
        destination_file = os.path.join(args.source_path, "images", file)
        shutil.copy2(source_file, destination_file)

    if not os.path.exists(args.source_path + "/sparse"):
        os.makedirs(args.source_path + "/sparse")
    img_undist_cmd = (
        "cp -r " + args.source_path + "/distorted/sparse " + args.source_path
    )
    exit_code = os.system(img_undist_cmd)
    if exit_code != 0:
        logging.error(f"Mapper failed with code {exit_code}. Exiting.")
        exit(exit_code)

    img_undist_cmd = (
        "cp -r " + args.source_path + "/distorted/sparse " + args.source_path
    )
    exit_code = os.system(img_undist_cmd)
    if exit_code != 0:
        logging.error(f"Mapper failed with code {exit_code}. Exiting.")
        exit(exit_code)

    bin2txt(args.source_path + "/sparse/0", args.source_path + "/sparse/0")
    # Modify the camera parameters
    with open(args.source_path + "/sparse/0/cameras.txt", "r") as fid:
        lines = fid.readlines()
        # skip #
        for i in range(len(lines)):
            if lines[i][0] != "#":
                lines[i] = lines[i].replace("OPENCV", "PINHOLE")
    with open(args.source_path + "/sparse/0/cameras.txt", "w") as fid:
        fid.writelines(lines)

    img_undist_cmd = (
        "rm "
        + args.source_path
        + "/sparse/0/cameras.bin"
        + " "
        + args.source_path
        + "/sparse/0/images.bin "
        + args.source_path
        + "/sparse/0/points3D.bin"
        + args.source_path
        + "/sparse/0/project.ini"
    )
    
else:
    ## We need to undistort our images into ideal pinhole intrinsics.
    img_undist_cmd = (
        colmap_command
        + " image_undistorter \
        --image_path "
        + args.source_path
        + "/input \
        --input_path "
        + args.source_path
        + "/distorted/sparse/0 \
        --output_path "
        + args.source_path
        + "\
        --output_type COLMAP"
    )

exit_code = os.system(img_undist_cmd)
if exit_code != 0:
    logging.error(f"Mapper failed with code {exit_code}. Exiting.")
    exit(exit_code)

files = os.listdir(args.source_path + "/sparse")
os.makedirs(args.source_path + "/sparse/0", exist_ok=True)
# Copy each file from the source directory to the destination directory
for file in files:
    if file == "0":
        continue
    source_file = os.path.join(args.source_path, "sparse", file)
    destination_file = os.path.join(args.source_path, "sparse", "0", file)
    shutil.move(source_file, destination_file)

if args.resize:
    print("Copying and resizing...")

    # Resize images.
    os.makedirs(args.source_path + "/images_2", exist_ok=True)
    os.makedirs(args.source_path + "/images_4", exist_ok=True)
    os.makedirs(args.source_path + "/images_8", exist_ok=True)
    # Get the list of files in the source directory
    files = os.listdir(args.source_path + "/images")
    # Copy each file from the source directory to the destination directory
    for file in files:
        source_file = os.path.join(args.source_path, "images", file)

        destination_file = os.path.join(args.source_path, "images_2", file)
        shutil.copy2(source_file, destination_file)
        exit_code = os.system(
            magick_command + " mogrify -resize 50% " + destination_file
        )
        if exit_code != 0:
            logging.error(f"50% resize failed with code {exit_code}. Exiting.")
            exit(exit_code)

        destination_file = os.path.join(args.source_path, "images_4", file)
        shutil.copy2(source_file, destination_file)
        exit_code = os.system(
            magick_command + " mogrify -resize 25% " + destination_file
        )
        if exit_code != 0:
            logging.error(f"25% resize failed with code {exit_code}. Exiting.")
            exit(exit_code)

        destination_file = os.path.join(args.source_path, "images_8", file)
        shutil.copy2(source_file, destination_file)
        exit_code = os.system(
            magick_command + " mogrify -resize 12.5% " + destination_file
        )
        if exit_code != 0:
            logging.error(f"12.5% resize failed with code {exit_code}. Exiting.")
            exit(exit_code)

print("Done.")