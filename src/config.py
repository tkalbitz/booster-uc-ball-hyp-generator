from pathlib import Path

scale_factor = 4
scale_factor_f = float(scale_factor)
path_count_w = 4
path_count_h = 4
patch_width = 640 // scale_factor // path_count_w
patch_height = 480 // scale_factor // path_count_h
img_scaled_width = 640 // scale_factor
img_scaled_height = 480 // scale_factor
image_dir = Path('/home/tkalbitz/temp/BallImages/')
testset_csv_collection = Path("/home/tkalbitz/temp/BallImages/uc-ball-30.txt")
trainingset_csv_collection = Path("/home/tkalbitz/temp/BallImages/uc-ball-70.txt")