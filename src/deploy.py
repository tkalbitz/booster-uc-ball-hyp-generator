import shutil
import sys

import torch
import torch.onnx

from scale import unscale_x, unscale_y

# Force CPU for deployment
device: torch.device = torch.device('cpu')
print(f"Using device: {device}")

from config import patch_height, patch_width, image_dir, testset_csv_collection
from csv_label_reader import load_csv_collection
from dataset_handling import create_dataset


deploy_dir: str = '/home/tkalbitz/naoTeamRepo/firmware_5.0/deploy/data/tflite/'
detector_name: str = 'uc-ball-hyp-generator'
weight_file: str = '/home/tkalbitz/PycharmProjects/uc-ball-hyp-generator/model/yuv_2021-05-22-08-48-01/weights.balls.264-0.954385-1.292296.pth'

if len(sys.argv) == 2:
    weight_file = sys.argv[1]

print(f"Weight file: {weight_file}")


def create_dataset():
    png_files = {f.name: str(f) for f in image_dir.glob("**/*.png")}

    test_img, test_labels, skipped_testset = load_csv_collection(testset_csv_collection, png_files)

    print(f"Testset contains {len(test_img)} and we removed {skipped_testset} balls.")

    batch_size = 256
    steps = len(test_img) // batch_size + 1
    ds = create_dataset(test_img, test_labels, 1, trainset=False)


    def representative_dataset():
        for data in ds.take(1000):
            a = data[0]
            yield [a]

    return representative_dataset()


def distance_loss(y_pred, y_true):
    """PyTorch version of distance loss."""
    def _logcosh(x):
        return x + torch.nn.functional.softplus(-2. * x) - torch.log(torch.tensor(2.))

    y_t = torch.stack([unscale_x(y_true[:,0]), unscale_y(y_true[:,1])], dim=1)
    y_p = torch.stack([unscale_x(y_pred[:,0]), unscale_y(y_pred[:,1])], dim=1)
    r = _logcosh(y_p - y_t)
    e = torch.exp(3. / y_true[:,2])
    r = r * e.unsqueeze(1)
    return torch.mean(r)


import models
model = models.create_network_v2(patch_height, patch_width)
model.load_state_dict(torch.load(weight_file, map_location=device))
model.to(device)
model.eval()


tflite_file = deploy_dir + "/" + detector_name + ".tflite"
model_file = deploy_dir + "/" + detector_name + ".pth"

with open(deploy_dir + "/" + detector_name + ".txt", "w") as fp:
    fp.write("File: {}\n\n".format(weight_file))
    fp.write(str(model))

shutil.copy(weight_file, model_file)

# Export to ONNX for deployment
dummy_input = torch.randn(1, 3, patch_height, patch_width)
onnx_path = deploy_dir + "/" + detector_name + ".onnx"

torch.onnx.export(model, dummy_input, onnx_path,
                  export_params=True,
                  opset_version=11,
                  do_constant_folding=True,
                  input_names=['input'],
                  output_names=['output'])

print(f"Saved ONNX model to {onnx_path}")
