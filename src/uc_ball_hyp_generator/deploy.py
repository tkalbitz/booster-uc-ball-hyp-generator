import shutil
import sys

import torch
import torch.onnx

import uc_ball_hyp_generator.models as models
from uc_ball_hyp_generator.config import patch_height, patch_width

# Force CPU for deployment
device: torch.device = torch.device("cpu")
print(f"Using device: {device}")


deploy_dir: str = "/home/tkalbitz/naoTeamRepo/firmware_5.0/deploy/data/tflite/"
detector_name: str = "uc-ball-hyp-generator"
weight_file: str = "/home/tkalbitz/PycharmProjects/uc-ball-hyp-generator/model/yuv_2021-05-22-08-48-01/weights.balls.264-0.954385-1.292296.pth"

if len(sys.argv) == 2:
    weight_file = sys.argv[1]

print(f"Weight file: {weight_file}")

model = models.create_network_v2(patch_height, patch_width)
model.load_state_dict(torch.load(weight_file, map_location=device))
model.to(device)
model.eval()


model_file = deploy_dir + "/" + detector_name + ".pth"

with open(deploy_dir + "/" + detector_name + ".txt", "w") as fp:
    fp.write("File: {}\n\n".format(weight_file))
    fp.write(str(model))

shutil.copy(weight_file, model_file)

# Export to ONNX for deployment
dummy_input = torch.randn(1, 3, patch_height, patch_width)
onnx_path = deploy_dir + "/" + detector_name + ".onnx"

torch.onnx.export(
    model,
    (dummy_input,),
    onnx_path,
    export_params=True,
    opset_version=11,
    do_constant_folding=True,
    input_names=["input"],
    output_names=["output"],
)

print(f"Saved ONNX model to {onnx_path}")
