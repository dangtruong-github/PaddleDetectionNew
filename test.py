"""
CUDA_VISIBLE_DEVICES=0 python -m paddle.distributed.launch --gpus 0 tools/train.py -c configs/deformable_detr/deformable_detr_r50_1x.yml -r output_old/best_model.pdparams --fleet --amp --eval

CUDA_VISIBLE_DEVICES=0 python -m paddle.distributed.launch --gpus 0 tools/train.py -c configs/deformable_detr/deformable_detr_r50_1x.yml -r output/1.pdparams --fleet --amp --eval

CUDA_VISIBLE_DEVICES=0 python tools/eval.py -c configs/deformable_detr/deformable_detr_r50_1x.yml -o weights=output/best_model.pdparams

CUDA_VISIBLE_DEVICES=0 python tools/infer.py -c configs/deformable_detr/deformable_detr_r50_1x.yml -o weights=output_old/best_model.pdparams --infer_dir=dataset/naver/original/images/test

python -m paddle.distributed.launch --gpus 0 tools/train.py -c configs/deformable_detr/deformable_detr_r50_1x_coco.yml --fleet --amp --eval

kaggle datasets download -d dangtop4sure/dataset/valid

CUDA_VISIBLE_DEVICES=0 python tools/infer.py -c configs/deformable_detr/deformable_detr_r50_1x_coco.yml -o weights=output/best_model --infer_dir=dataset/coco/val2017
"""
"""
import gdown

# URL of the Google Drive file
id = "1SOIdd5kgUU27aOeIBjIixp_u8uaGlFKv"
url = f'https://drive.google.com/uc?id={id}'
# Path to save the downloaded file
output = './dataset/naver/new_data.zip'

gdown.download(url, output, quiet=False)"

export LD_LIBRARY_PATH=../.env/naver/lib:$LD_LIBRARY_PATH
conda create --prefix naver python=3.12
conda install -c conda-forge cudatoolkit cudnn
"""

import paddle
import paddle.nn.functional as F

# Input image tensor and scale factor
image = paddle.rand([1, 3, 750, 1333])  # Simulated input image
scale_factor = [1, 2]  # Scale factor for height and width

# 1. Scaling
height, width = image.shape[2], image.shape[3]
scaled_height = int(height * scale_factor[0])
scaled_width = int(width * scale_factor[1])
scaled_image = F.interpolate(image, size=(scaled_height, scaled_width), mode='bilinear')

# 2. Rotation
def rotate_image(img, degrees):
    if degrees == 90:
        return img.transpose([0, 1, 3, 2]).flip(axis=[2])  # Transpose and flip height
    elif degrees == 180:
        return img.flip(axis=[2, 3])  # Flip both height and width
    elif degrees == 270:
        return img.transpose([0, 1, 3, 2]).flip(axis=[3])  # Transpose and flip width
    else:
        return img  # No rotation for 0°

rotated_image_90 = rotate_image(scaled_image, 90)
rotated_image_180 = rotate_image(scaled_image, 180)
rotated_image_270 = rotate_image(scaled_image, 270)

# 3. Flip
horizontal_flip = scaled_image.flip(axis=[3])  # Flip horizontally
vertical_flip = scaled_image.flip(axis=[2])  # Flip vertically

# Output examples
print("Original Image Shape:", image.shape)
print("Scaled Image Shape:", scaled_image.shape)
print("90° Rotated Image Shape:", rotated_image_90.shape)
print("Horizontally Flipped Image Shape:", horizontal_flip.shape)
print("Vertically Flipped Image Shape:", vertical_flip.shape)
