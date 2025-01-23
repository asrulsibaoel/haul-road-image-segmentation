from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union, Any
import numpy as np
from ultralytics import YOLO
import torch
from torch.utils.data import Dataset
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from dataclasses import dataclass
import json
import yaml
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('aerial_detection.log'),
        logging.StreamHandler()
    ]
)


@dataclass
class AerialModelConfig:
    img_size: int = 1024          # Larger size for aerial details
    tile_size: int = 1024         # Size for tiling large images
    tile_overlap: int = 128       # Overlap between tiles
    batch_size: int = 8           # Reduced due to larger images
    num_epochs: int = 100
    learning_rate: float = 0.01
    num_classes: int = 1
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    model_type: str = "yolov8l.pt"  # Larger model for complex features
    min_visibility: float = 0.15   # Minimum object visibility threshold
    cache_images: bool = True      # Cache images in memory for faster training
    save_period: int = 10          # Save checkpoint every N epochs
    project_name: str = "aerial_haul_road_detection"
    experiment_name: str = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


class ImageProcessor:
    """Handle preprocessing of aerial/satellite images"""

    @staticmethod
    def enhance_contrast(image: np.ndarray) -> np.ndarray:
        """Apply CLAHE contrast enhancement"""
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    @staticmethod
    def tile_image(
        image: np.ndarray,
        tile_size: int,
        overlap: int
    ) -> List[Tuple[np.ndarray, Tuple[int, int]]]:
        """Split large images into overlapping tiles"""
        tiles = []
        h, w = image.shape[:2]

        for y in range(0, h-overlap, tile_size-overlap):
            for x in range(0, w-overlap, tile_size-overlap):
                end_y = min(y + tile_size, h)
                end_x = min(x + tile_size, w)
                tile = image[y:end_y, x:end_x]

                # Pad if tile is smaller than tile_size
                if tile.shape[0] != tile_size or tile.shape[1] != tile_size:
                    padded_tile = np.zeros(
                        (tile_size, tile_size, 3), dtype=np.uint8)
                    padded_tile[:tile.shape[0], :tile.shape[1], :] = tile
                    tile = padded_tile

                tiles.append((tile, (x, y)))

        return tiles

    @staticmethod
    def merge_predictions(
        tiles_predictions: List[Dict[str, Any]],
        original_size: Tuple[int, int],
        tile_size: int,
        overlap: int
    ) -> Dict[str, Any]:
        """Merge predictions from tiles back to original image size"""
        merged_boxes = []
        merged_scores = []
        merged_classes = []

        for pred, (x_offset, y_offset) in tiles_predictions:
            if pred.boxes.xyxy.shape[0] > 0:
                # Adjust coordinates based on tile position
                boxes = pred.boxes.xyxy.cpu().numpy()
                boxes[:, [0, 2]] += x_offset
                boxes[:, [1, 3]] += y_offset

                # Add predictions
                merged_boxes.extend(boxes)
                merged_scores.extend(pred.boxes.conf.cpu().numpy())
                merged_classes.extend(pred.boxes.cls.cpu().numpy())

        # Perform NMS on merged predictions
        if merged_boxes:
            merged_boxes = np.array(merged_boxes)
            merged_scores = np.array(merged_scores)
            merged_classes = np.array(merged_classes)

            # Convert to YOLO format for NMS
            merged_predictions = {
                'boxes': torch.from_numpy(merged_boxes),
                'scores': torch.from_numpy(merged_scores),
                'classes': torch.from_numpy(merged_classes)
            }
        else:
            merged_predictions = {
                'boxes': torch.zeros((0, 4)),
                'scores': torch.zeros(0),
                'classes': torch.zeros(0)
            }

        return merged_predictions


class AerialDataset(Dataset):
    """Dataset class for aerial/satellite imagery"""

    def __init__(
        self,
        image_dir: Union[str, Path],
        label_dir: Union[str, Path],
        config: AerialModelConfig,
        transform: Optional[A.Compose] = None,
        cache_images: bool = False
    ) -> None:
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.transform = transform
        self.config = config
        self.cache_images = cache_images
        self.image_files = sorted(list(self.image_dir.glob("*.jpg")))
        self.cache = {}

        if self.cache_images:
            self._cache_images()

    def _cache_images(self) -> None:
        """Cache images in memory"""
        logging.info("Caching images...")
        for idx in tqdm(range(len(self.image_files))):
            img_path = self.image_files[idx]
            image = cv2.imread(str(img_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.cache[idx] = image

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str]]:
        img_path = self.image_files[idx]
        label_path = self.label_dir / f"{img_path.stem}.txt"

        # Load image
        if self.cache_images and idx in self.cache:
            image = self.cache[idx].copy()
        else:
            image = cv2.imread(str(img_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply contrast enhancement
        image = ImageProcessor.enhance_contrast(image)

        # Read YOLO format labels
        labels = []
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    class_id, x, y, w, h = map(float, line.strip().split())
                    labels.append([class_id, x, y, w, h])

        labels = np.array(labels) if labels else np.zeros((0, 5))

        if self.transform:
            transformed = self.transform(
                image=image, bboxes=labels[:, 1:], class_labels=labels[:, 0])
            image = transformed['image']
            if len(transformed['bboxes']) > 0:
                labels = np.column_stack([
                    transformed['class_labels'],
                    transformed['bboxes']
                ])
            else:
                labels = np.zeros((0, 5))

        return {
            'image': image,
            'labels': torch.from_numpy(labels).float(),
            'image_path': str(img_path)
        }


def create_aerial_transforms(config: AerialModelConfig) -> Tuple[A.Compose, A.Compose]:
    """Create augmentation pipelines for aerial imagery"""
    train_transform = A.Compose([
        A.Resize(config.img_size, config.img_size),
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0), p=1),
            A.GaussianBlur(blur_limit=(3, 7), p=1),
            A.MotionBlur(blur_limit=(3, 7), p=1),
        ], p=0.2),
        A.OneOf([
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=1
            ),
            A.HueSaturationValue(
                hue_shift_limit=20,
                sat_shift_limit=30,
                val_shift_limit=20,
                p=1
            ),
        ], p=0.3),
        A.Normalize(mean=[0, 0, 0], std=[255, 255, 255]),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

    val_transform = A.Compose([
        A.Resize(config.img_size, config.img_size),
        A.Normalize(mean=[0, 0, 0], std=[255, 255, 255]),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

    return train_transform, val_transform


class AerialTrainer:
    """Training class for aerial imagery detection"""

    def __init__(self, config: AerialModelConfig) -> None:
        self.config = config
        self.model = YOLO(config.model_type)
        self.save_dir = Path(config.project_name) / config.experiment_name
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def train(self, data_yaml_path: str) -> None:
        """Train the model with aerial-specific settings"""
        logging.info(f"Starting training with config: {self.config}")

        results = self.model.train(
            data=data_yaml_path,
            epochs=self.config.num_epochs,
            imgsz=self.config.img_size,
            batch=self.config.batch_size,
            device=self.config.device,
            project=self.config.project_name,
            name=self.config.experiment_name,
            lr0=self.config.learning_rate,
            patience=50,
            save_period=self.config.save_period,
            # Aerial-specific parameters
            mosaic=0.75,
            mixup=0.25,
            degrees=180,
            scale=0.5,
            fliplr=0.5,
            flipud=0.5,
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            verbose=True
        )

        # Save training results
        self._save_results(results)

    def _save_results(self, results: Any) -> None:
        """Save training results and metrics"""
        results_dict = {
            'train/box_loss': results.results_dict['train/box_loss'],
            'train/cls_loss': results.results_dict['train/cls_loss'],
            'val/box_loss': results.results_dict['val/box_loss'],
            'val/cls_loss': results.results_dict['val/cls_loss'],
            'metrics/precision': results.results_dict['metrics/precision'],
            'metrics/recall': results.results_dict['metrics/recall'],
            'metrics/mAP50': results.results_dict['metrics/mAP50'],
            'metrics/mAP50-95': results.results_dict['metrics/mAP50-95']
        }

        # Save metrics
        with open(self.save_dir / 'results.json', 'w') as f:
            json.dump(results_dict, f, indent=4)

        # Plot training results
        self._plot_results(results_dict)

    def _plot_results(self, results: Dict[str, List[float]]) -> None:
        """Plot training metrics"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Plot losses
        axes[0, 0].plot(results['train/box_loss'], label='Train Box Loss')
        axes[0, 0].plot(results['val/box_loss'], label='Val Box Loss')
        axes[0, 0].set_title('Box Loss')
        axes[0, 0].legend()

        axes[0, 1].plot(results['train/cls_loss'], label='Train Cls Loss')
        axes[0, 1].plot(results['val/cls_loss'], label='Val Cls Loss')
        axes[0, 1].set_title('Classification Loss')
        axes[0, 1].legend()

        # Plot metrics
        axes[1, 0].plot(results['metrics/precision'], label='Precision')
        axes[1, 0].plot(results['metrics/recall'], label='Recall')
        axes[1, 0].set_title('Precision & Recall')
        axes[1, 0].legend()

        axes[1, 1].plot(results['metrics/mAP50'], label='mAP50')
        axes[1, 1].plot(results['metrics/mAP50-95'], label='mAP50-95')
        axes[1, 1].set_title('mAP')
        axes[1, 1].legend()

        plt.tight_layout()
        plt.savefig(self.save_dir / 'training_plots.png')
        plt.close()


class Predictor:
    """Prediction class for aerial imagery"""

    def __init__(self, model_path: str, config: AerialModelConfig) -> None:
        self.model = YOLO(model_path)
        self.config = config

    def predict_large_image(
        self,
        image_path: str,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45
    ) -> Dict[str, Any]:
        """Handle prediction for large aerial images using tiling"""
        # Load and preprocess image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_size = image.shape[:2]

        # Enhanced contrast
        image = ImageProcessor.enhance_contrast(image)

        # Generate tiles
        tiles = ImageProcessor.tile_image(
            image,
            self.config.tile_size,
            self.config.tile_overlap
        )

        # Predict on each tile
        tiles_predictions = []
        for tile, (x, y) in tiles:
            prediction = self.model.predict(
                tile,
                conf=conf_threshold,
                iou=iou_threshold,
                verbose=False
            )[0]

            tiles_predictions.append((prediction, (x, y)))

        # Merge predictions
        merged_predictions = ImageProcessor.merge_predictions(
            tiles_predictions,
            original_size,
            self.config.tile_size,
            self.config.tile_overlap
        )

        return merged_predictions

    def visualize_predictions(
        self,
        image_path: str,
        predictions: Dict[str, torch.Tensor],
        output_path: str,
        class_names: List[str]
    ) -> None:
        """Visualize predictions on the image"""
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Draw boxes
        for box, score, class_id in zip(
            predictions['boxes'],
            predictions['scores'],
            predictions['classes']
        ):
            x1, y1, x2, y2 = box.cpu().numpy().astype(int)
            class_id = int(class_id)

            # Draw box
            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            # Add label
            label = f"{class_names[class_id]} {score:.2f}"
            cv2.putText(
                image,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )

        # Save result
        plt.figure(figsize=(20, 20))
        plt.imshow(image)
        plt.axis('off')
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.close()


def create_dataset_yaml(
    train_path: str,
    val_path: str,
    class_names: List[str],
    output_path: str = "dataset.yaml"
) -> None:
    """Create YAML configuration file for training"""
    data_yaml = {
        'train': train_path,
        'val': val_path,
        'names': {i: name for i, name in enumerate(class_names)},
        'nc': len(class_names)
    }

    with open(output_path, 'w') as f:
        yaml.dump(data_yaml, f, sort_keys=False)


def main() -> None:
    """Main training pipeline"""
    # Configuration
    config = AerialModelConfig()

    # Setup logging
    logging.info(
        f"Starting aerial haul road detection pipeline with config: {config}")

    # Create transforms
    train_transform, val_transform = create_aerial_transforms(config)

    # Create datasets
    train_dataset = AerialDataset(
        image_dir="path/to/train/images",
        label_dir="path/to/train/labels",
        config=config,
        transform=train_transform,
        cache_images=config.cache_images
    )

    val_dataset = AerialDataset(
        image_dir="path/to/val/images",
        label_dir="path/to/val/labels",
        config=config,
        transform=val_transform,
        cache_images=config.cache_images
    )

    # Create YAML config
    class_names = ["haul_road"]
    create_dataset_yaml(
        train_path=str(Path("path/to/train/images").absolute()),
        val_path=str(Path("path/to/val/images").absolute()),
        class_names=class_names,
        output_path="aerial_dataset.yaml"
    )

    # Initialize trainer and train
    trainer = AerialTrainer(config)
    trainer.train(data_yaml_path="aerial_dataset.yaml")

    # Example of prediction on a large image
    predictor = Predictor(
        model_path=f"{
            config.project_name}/{config.experiment_name}/weights/best.pt",
        config=config
    )

    # Predict on a test image
    test_image_path = "path/to/test/image.jpg"
    predictions = predictor.predict_large_image(
        test_image_path,
        conf_threshold=0.25,
        iou_threshold=0.45
    )

    # Visualize results
    predictor.visualize_predictions(
        test_image_path,
        predictions,
        output_path=f"""{
            config.project_name}/{config.experiment_name}/predictions/test_prediction.jpg""",
        class_names=class_names
    )


if __name__ == "__main__":
    main()
