"""BasePreprocessing: standard pipeline that mirrors model1's get_transforms()."""
from __future__ import annotations

from PIL import Image
from torchvision import transforms

IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class BasePreprocessing:
    """Standard RGB preprocessing pipeline (same behaviour as model1).

    Accepts a PIL image and returns a normalized (3, H, W) float tensor.
    Augmentation (flip, rotation, colour jitter) is applied only when
    train=True.
    """

    def __init__(
        self,
        image_size: int = IMAGE_SIZE,
        mean: list[float] = IMAGENET_MEAN,
        std: list[float] = IMAGENET_STD,
    ) -> None:
        self.image_size = image_size

        def _ensure_rgb(img: Image.Image) -> Image.Image:
            return img.convert("RGB") if img.mode != "RGB" else img

        base = [
            transforms.Lambda(_ensure_rgb),
            transforms.Resize((image_size, image_size)),
        ]

        self._train_transform = transforms.Compose(
            base
            + [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

        self._eval_transform = transforms.Compose(
            base
            + [
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

    def __call__(self, image: Image.Image, train: bool = True):
        if train:
            return self._train_transform(image)
        return self._eval_transform(image)
