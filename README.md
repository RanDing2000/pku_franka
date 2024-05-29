# PKU Franka

## 1. Hand-eye Camera Calibration

```py
# 1.1 create scene_id and save to config.json
python /home/hyperpanda/Haoran/1_1_create_scene_name.py
# now open config.json to make sure that the `scene_id` is updated
# 1.2
# first attach pattern on hand, prepare the realsense (eye on base), and make the franka robot in blue light mode
# second activate FCI  refer to: https://github.com/iamlab-cmu/frankapy
cd ~/frankapy
conda deactivate
roslaunch easy_handeye panda_realsense_eyeonbase.launch
# in another terminal
cd ~/frankapy
conda deactivate
rqt
# operate, refer to: https://github.com/IFL-CAMP/easy_handeye/blob/master/README.md, 
# 1.3
python '/home/hyperpanda/Haoran/1_3_save_handeye_transformation.py'
```

## 2. Transformation between camera and board

```py
conda activate qt_env
python 2_1_get_board.py
# change `raw_npz_path` in plane_cam.py according to the get_board.py output
conda activate qt_env
# python 2_2_plane_cam_marker.py
python 2_2_plane_cam_pnp.py
```

## 3. Arrange Scenes

```py
conda activate qt_env
## crop the long-tail noise if possible
## the first click is the target object
python 3_inference_pipeline.py
# after pick_motion is done:
python 3_1_arrange_single.py
python 3_2_arrange_clutter.py
```

## 4. Sim2Real data convertion

```py
python 4_sim2real_pointcloud.py
```

## 5. Model Prediction

```py
cd /home/hyperpanda/GraspInClutter/realworld
conda activate gic
## if GIGA
python giga_model_prediction.py 
## if ours w/ sc
python ours_model_prediction.py
## if ours w/o sc
python ours_model_no_sc.py
```

## 6. Franka Control

```py
cd ~/frankapy
conda deactivate
bash ./bash_scripts/start_control_pc.sh -i localhost

cd ~/frankapy
conda deactivate
cd ../Haoran
/bin/python3 6_franka_control.py
```

## Camera View

```py
realsense-viewer
# Cloris's code: refer to /Home/Cloris/manipsop.py
```
