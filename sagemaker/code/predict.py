import io
import os


import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet50


def model_fn(model_dir):
    device = torch.device("cuda")
    model = resnet50()
    with open(os.path.join(model_dir, "model.ckpt"), "rb") as f:
        model.load_state_dict(torch.load(f))
    model.eval()
    model = model.to(device)
    return model


def preprocess_image_batch(images: np.ndarray) -> np.ndarray:
    preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    torch_tensor = torch.Tensor(images.transpose((0, 3, 1, 2)))
    torch_tensor /= 255
    preprocessed_images = preprocess(torch_tensor).numpy()
    return preprocessed_images


def load_parquet_from_bytearray(request_body):
    reader = pa.BufferReader(request_body)
    df = pq.read_table(reader)
    numpy_batch = np.stack(df["image"].to_numpy())
    reshaped_images = numpy_batch.reshape(-1, 256, 256, 3)
    preprocessed_images = preprocess_image_batch(reshaped_images)
    return preprocessed_images


def load_from_bytearray(request_body):
    image_as_bytes = io.BytesIO(request_body)
    image = Image.open(image_as_bytes)
    image_tensor = np.expand_dims(np.asarray(image), 0)
    preprocessed_images = preprocess_image_batch(image_tensor)
    return preprocessed_images


def input_fn(request_body, request_content_type):
    if request_content_type == "application/x-parquet":
        image_tensor = load_parquet_from_bytearray(request_body)
    elif request_content_type == "application/x-image":
        image_tensor = load_from_bytearray(request_body)
    else:
        raise ValueError(f"Unexpected request content type: {request_content_type}.")
    return image_tensor


# Perform prediction on the deserialized object, with the loaded model
def predict_fn(input_object, model):
    with torch.inference_mode():
        output = model(torch.as_tensor(input_object, device="cuda"))
    return output.cpu().numpy()


def output_fn(predictions, content_type):
    if content_type == "application/json":
        return "{}"
    else:
        raise ValueError(f"Unexpected response content type: {content_type}.")
