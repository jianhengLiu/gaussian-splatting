from argparse import ArgumentParser
import os
import open3d as o3d
import numpy as np


def convert_pcd_to_txt(dataset_dir, downsample_factor=1):
    # Read PCD file
    dataset_name = dataset_dir.split("/")[-1]
    pcd_path = os.path.join(dataset_dir, "points.pcd")
    txt_path = os.path.join(
        dataset_dir, "train/" + dataset_name + "/sparse/0/points3D.txt"
    )

    pcd = o3d.io.read_point_cloud(pcd_path)
    xyzs = np.asarray(pcd.points)
    rgbs = np.asarray(pcd.colors) * 255
    try:
        errors = np.asarray(pcd.errors)
    except:
        errors = np.zeros((xyzs.shape[0], 1))

    # Downsample data
    if downsample_factor > 1:
        xyzs = xyzs[::downsample_factor]
        rgbs = rgbs[::downsample_factor]
        errors = errors[::downsample_factor]

    print("downsampled points number:", xyzs.shape[0])

    # Write TXT file
    with open(txt_path, "w") as fid:
        fid.write("# 3D point list with one line of data per point:\n")
        fid.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR\n")
        for i in range(xyzs.shape[0]):
            fid.write(
                f"{i} {xyzs[i,0]:.6f} {xyzs[i,1]:.6f} {xyzs[i,2]:.6f} {rgbs[i,0]:.0f} {rgbs[i,1]:.0f} {rgbs[i,2]:.0f} {errors[i,0]:.6f}\n"
            )


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--downsample_factor", type=int, required=True)
    args = parser.parse_args()
    convert_pcd_to_txt(args.dataset_dir,args.downsample_factor)
