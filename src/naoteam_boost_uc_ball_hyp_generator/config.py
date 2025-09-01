from pathlib import Path

scale_factor: int = 4
scale_factor_f: float = float(scale_factor)
path_count_w: int = 4
path_count_h: int = 4
patch_width: int = 640 // scale_factor // path_count_w
patch_height: int = 480 // scale_factor // path_count_h
img_scaled_width: int = 640 // scale_factor
img_scaled_height: int = 480 // scale_factor
image_dir: Path = Path('/home/tkalbitz/temp/BallImages/')
testset_csv_collection: Path = Path("/home/tkalbitz/temp/BallImages/uc-ball-30.txt")
trainingset_csv_collection: Path = Path("/home/tkalbitz/temp/BallImages/uc-ball-70.txt")