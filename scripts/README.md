
# Pipeline

1. Prepare dataset sturcture like
   ```
   <location>
   |---input
   |   |---<image 0>
   |   |---<image 1>
   |   |---...
   |---images.txt
   |---cameras.txt
   |---points.pcd
   ```
    wherein, 
    - images.txt: `IMAGE_ID QW QX QY QZ TX TY TZ`
    - cameras.txt: `CAMERA_ID MODEL WIDTH HEIGHT fx fy, cx cy k1 k2 p1 p2`
  
2. undistort images
   ```
   python datasets/undistort.py --dataset_dir /home/chrisliu/Projects/LidarNeRF/datasets/hkust_livo_origin/livo_hkust_red_bird
   ```

3. downsample and separate the dataset into train and test set
   ```bash
   python datasets/downsample_separate_to_train_test.py --dataset_dir /home/chrisliu/Projects/LidarNeRF/datasets/hkust_livo_origin/livo_hkust_red_bird
   ```

4. convert pointcloud
   
   adjust `downsample_factor` to control the number of points in the pointcloud
   ```
   python datasets/pcd_convert_txt.py --dataset_dir /home/chrisliu/Projects/LidarNeRF/datasets/hkust_livo_origin/livo_hkust_red_bird --downsample_factor 100
   ```

**The following for colmap pipeline:**


```
mkdir /home/chrisliu/Projects/LidarNeRF/datasets/hkust_livo_origin/livo_hkust_red_bird/colmap
cp -r /home/chrisliu/Projects/LidarNeRF/datasets/hkust_livo_origin/livo_hkust_red_bird/train/livo_hkust_red_bird/images /home/chrisliu/Projects/LidarNeRF/datasets/hkust_livo_origin/livo_hkust_red_bird/colmap/input
python convert.py -s /home/chrisliu/Projects/LidarNeRF/datasets/hkust_livo_origin/livo_hkust_red_bird/colmap --camera_params "646.294856087, 646.155780233, 313.423980506, 262.903356268, -0.07581043635239915, 0.1276695531950238, -0.0004921162736736936, 3.265638595122261e-05"
```

5. run 3gs
   