from pathlib import Path

scale_factor: int = 4
scale_factor_f: float = float(scale_factor)
patch_width: int = 40
patch_height: int = 30
image_dir: Path = Path("/home/tkalbitz/data/BallImages/")
testset_csv_collection: Path = Path("/home/tkalbitz/data/BallImages/uc-ball-30.txt")
trainingset_csv_collection: Path = Path("/home/tkalbitz/data/BallImages/uc-ball-70.txt")
