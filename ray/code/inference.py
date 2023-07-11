import enum
import logging
import os
import sys
import time
from abc import ABC, abstractmethod
from typing import Dict, Optional


import fire
import numpy as np
import ray
import torch
from ray.data import ActorPoolStrategy
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights


logging.basicConfig(
    level=logging.getLevelName(os.getenv("SAGEMAKER_CONTAINER_LOG_LEVEL", logging.INFO)),
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


class RayBatchInferenceExecutor(ABC):
    def __init__(self):
        self._num_gpus = _get_number_of_available_gpus()
        logger.debug("Number of GPUs available: %s", self._num_gpus)

    @abstractmethod
    def _read_data(self, path: str, read_parallelism: int) -> ray.data.Dataset:
        return NotImplemented

    def _preprocess_data(
            self,
            dataset: ray.data.Dataset,
            batch_size: Optional[int],
    ) -> ray.data.Dataset:
        return dataset.map_batches(
            _preprocess,
            batch_size=batch_size,
            batch_format="numpy",
        )

    def _inference_model(
            self,
            model: ray.ObjectRef,
            dataset: ray.data.Dataset,
            batch_size: Optional[int],
            max_concurrency: int,
    ) -> ray.data.Dataset:
        return dataset.map_batches(
            _Actor,
            batch_size=batch_size,
            compute=ActorPoolStrategy(size=self._num_gpus),
            num_gpus=1,  # 1 GPU per actor
            batch_format="numpy",
            fn_constructor_kwargs={"model": model},
            max_concurrency=max_concurrency,  # actor concurrency level
        )

    def _materialize_predictions(self, dataset: ray.data.Dataset, batch_size: Optional[int]) -> None:
        for _ in dataset.iter_batches(batch_size=batch_size, batch_format="numpy"):
            pass

    def run(
            self,
            model: torch.nn.Module,
            data_path: str,
            data_read_parallelism: int,
            preprocessing_batch_size: Optional[int],
            inference_batch_size: Optional[int],
            materialize_batch_size: Optional[int],
            inference_concurrency: int = 2,
    ) -> None:
        if ray.is_initialized():
            ray.shutdown()

        model_ref = _add_to_ray_object_store(model)
        start_time = time.time()
        dataset = self._read_data(data_path, data_read_parallelism)

        start_time_without_metadata_fetching = time.time()
        dataset = self._preprocess_data(dataset, preprocessing_batch_size)

        dataset = self._inference_model(model_ref, dataset, inference_batch_size, inference_concurrency)
        self._materialize_predictions(dataset, materialize_batch_size)
        end_time = time.time()

        logger.info("Total time: %s", end_time - start_time)
        logger.info("Total time w/o metadata fetching: %s", end_time - start_time_without_metadata_fetching)
        logger.info(dataset.stats())


class RayImageBatchInferenceExecutor(RayBatchInferenceExecutor):

    def _read_data(self, path: str, read_parallelism: int) -> ray.data.Dataset:
        return ray.data.read_images(path, parallelism=read_parallelism, mode="RGB")


class RayParquetBatchInferenceExecutor(RayBatchInferenceExecutor):

    def _read_data(self, path: str, read_parallelism: int) -> ray.data.Dataset:
        return ray.data.read_parquet(path, parallelism=read_parallelism)


class ExecutorType(enum.Enum):
    IMAGE = enum.auto()
    PARQUET = enum.auto()


def run_inference(executor_type: ExecutorType, **kwargs) -> None:
    resnet_model = _get_model()
    registered_executors = {
        ExecutorType.IMAGE: RayImageBatchInferenceExecutor,
        ExecutorType.PARQUET: RayParquetBatchInferenceExecutor,
    }
    executor_cls = registered_executors[executor_type]
    executor = executor_cls()
    executor.run(model=resnet_model, **kwargs)


def run_inference_on_images(
        data_path: str,
        data_read_parallelism: int = -1,
        preprocessing_batch_size: Optional[int] = None,  # 4096 is default value in Ray
        inference_batch_size: Optional[int] = None,
        materialize_batch_size: Optional[int] = None,  # 256 is default value in Ray
        inference_concurrency: int = 2,
) -> None:
    run_inference(
        executor_type=ExecutorType.IMAGE,
        data_path=data_path,
        data_read_parallelism=data_read_parallelism,
        preprocessing_batch_size=preprocessing_batch_size,
        inference_batch_size=inference_batch_size,
        materialize_batch_size=materialize_batch_size,
        inference_concurrency=inference_concurrency,
    )


def run_inference_on_parquets(
        data_path: str,
        data_read_parallelism: int = -1,
        preprocessing_batch_size: Optional[int] = None,  # 4096 is default value in Ray
        inference_batch_size: Optional[int] = None,
        materialize_batch_size: Optional[int] = None,  # 256 is default value in Ray
        inference_concurrency: int = 2,
) -> None:
    run_inference(
        executor_type=ExecutorType.PARQUET,
        data_path=data_path,
        data_read_parallelism=data_read_parallelism,
        preprocessing_batch_size=preprocessing_batch_size,
        inference_batch_size=inference_batch_size,
        materialize_batch_size=materialize_batch_size,
        inference_concurrency=inference_concurrency,
    )


class _Actor:
    def __init__(self, model):
        self.model = ray.get(model)
        self.model.eval()
        self.model.to("cuda")

    def __call__(self, batch):
        with torch.inference_mode():
            output = self.model(torch.as_tensor(batch["image"], device="cuda"))
            return {"predictions": output.cpu().numpy()}


def _get_model() -> torch.nn.Module:
    return resnet50(weights=ResNet50_Weights.DEFAULT)


def _add_to_ray_object_store(obj) -> ray.ObjectRef:
    return ray.put(obj)


def _get_number_of_available_gpus() -> int:
    return torch.cuda.device_count()


def _preprocess(image_batch: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    torch_tensor = torch.Tensor(image_batch["image"].transpose((0, 3, 1, 2)))
    torch_tensor /= 255
    preprocessed_images = preprocess(torch_tensor).numpy()
    return {"image": preprocessed_images}


if __name__ == '__main__':
    fire.Fire({
        "run_inference_on_images": run_inference_on_images,
        "run_inference_on_parquets": run_inference_on_parquets,
    })
