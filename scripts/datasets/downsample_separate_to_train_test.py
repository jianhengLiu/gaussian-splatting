from argparse import ArgumentParser
import cv2
import os

import numpy as np

# python downsample_datasets.py --dataset_dir /home/chrisliu/Projects/LidarNeRF/datasets/hkust_livo_origin/livo_hkust_red_bird


def downsample_dataset(dataset_dir):
    dataset_name = dataset_dir.split("/")[-1]
    image_dir = os.path.join(dataset_dir, "undistort")
    poses_path = os.path.join(dataset_dir, "images.txt")

    out_train_img_dir = os.path.join(dataset_dir, "train/" + dataset_name + "/images")
    out_train_pose_dir = os.path.join(
        dataset_dir, "train/" + dataset_name + "/sparse/0"
    )
    out_test_img_dir = os.path.join(dataset_dir, "test/" + dataset_name + "/images")
    out_test_pose_dir = os.path.join(dataset_dir, "test/" + dataset_name + "/sparse/0")

    input_frame_rate = 10
    expect_frame_rate = 2
    train_frame_skip_step = int(input_frame_rate / expect_frame_rate)
    test_frame_skip_step = int(train_frame_skip_step * 4.5)

    poses = {}
    with open(poses_path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                image_id = int(elems[0])
                q_cw = np.array(tuple(map(float, elems[1:5])))
                t_cw = np.array(tuple(map(float, elems[5:8])))
                poses[image_id] = np.concatenate([q_cw, t_cw])

    images_name = {}
    for filename in os.listdir(image_dir):
        for filename in os.listdir(image_dir):
            if filename.endswith(".png"):
                images_name[int(filename.split(".")[0])] = filename

    images_name = dict(sorted(images_name.items()))

    # mkdir
    if not os.path.exists(out_train_img_dir):
        os.makedirs(out_train_img_dir)
    if not os.path.exists(out_train_pose_dir):
        os.makedirs(out_train_pose_dir)
    if not os.path.exists(out_test_img_dir):
        os.makedirs(out_test_img_dir)
    if not os.path.exists(out_test_pose_dir):
        os.makedirs(out_test_pose_dir)

    with open(os.path.join(out_train_pose_dir, "images.txt"), "w") as train_fid, open(
        os.path.join(out_test_pose_dir, "images.txt"), "w"
    ) as test_fid:
        for idx, image_id in enumerate(images_name):
            filename = images_name[image_id]
            key = filename.split(".")[0]
            if image_id % train_frame_skip_step == 0:
                img = cv2.imread(os.path.join(image_dir, filename))

                cv2.imwrite(os.path.join(out_train_img_dir, filename), img)

                pose = poses[image_id]
                train_fid.write(
                    f"{key} {pose[0]} {pose[1]} {pose[2]} {pose[3]} {pose[4]} {pose[5]} {pose[6]}\n"
                )
            elif image_id % test_frame_skip_step == 0:
                img = cv2.imread(os.path.join(image_dir, filename))

                cv2.imwrite(os.path.join(out_test_img_dir, filename), img)

                pose = poses[image_id]
                test_fid.write(
                    f"{key} {pose[0]} {pose[1]} {pose[2]} {pose[3]} {pose[4]} {pose[5]} {pose[6]}\n"
                )


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, required=True)
    args = parser.parse_args()
    downsample_dataset(args.dataset_dir)
