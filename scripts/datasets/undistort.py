from argparse import ArgumentParser
import cv2
import numpy as np
import os

def undistort_images(dataset_dir):
    image_dir = os.path.join(dataset_dir, "input")
    out_dir = os.path.join(dataset_dir, "undistort")
    cam_params_path = os.path.join(dataset_dir, "cameras.txt")
    
    # copy cameras.txt
    train_dir = os.path.join(dataset_dir, "train/" + dataset_dir.split("/")[-1] + "/sparse/0")
    os.system(f"cp {cam_params_path} {train_dir}")

    # Load the camera parameters
    with open(cam_params_path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                fx = float(elems[4])
                fy = float(elems[5])
                cx = float(elems[6])
                cy = float(elems[7])
                k1 = float(elems[8])
                k2 = float(elems[9])
                p1 = float(elems[10])
                p2 = float(elems[11])
            
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    # Load the camera matrix and distortion coefficients
    D = np.array([k1, k2, p1, p2])

    # Loop over all images in the 'input' directory and undistort them
    for filename in os.listdir(image_dir):
        if filename.endswith('.png'):
            # Load the image and undistort it
            img = cv2.imread(os.path.join(image_dir, filename))
            undistorted_img = cv2.undistort(img, K, D)
            
            # mkdir
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)

            # Save the undistorted image to disk with the same filename as before
            cv2.imwrite(os.path.join(out_dir, filename), undistorted_img)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, required=True)
    args = parser.parse_args()
    undistort_images(args.dataset_dir)
