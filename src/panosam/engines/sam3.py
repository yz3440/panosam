"""SAM3 Engine using HuggingFace Transformers.

This module provides the SAM3Engine for segmentation using Meta's SAM3 model.
Requires the [sam] extra: pip install "panosam[sam]"
"""

from typing import List, Optional, Tuple, TYPE_CHECKING

import numpy as np
from PIL import Image

from ..sam.models import FlatMaskResult

# Type hints for optional dependencies
if TYPE_CHECKING:
    import torch


def _check_sam_dependencies():
    """Check if SAM3 dependencies are installed and provide helpful error message."""
    missing = []

    try:
        import torch
    except ImportError:
        missing.append("torch")

    try:
        from transformers import Sam3Processor, Sam3Model
    except ImportError:
        missing.append("transformers (with SAM3 support)")

    if missing:
        raise ImportError(
            f"SAM3 dependencies not installed: {', '.join(missing)}\n\n"
            "Install with:\n"
            "  pip install 'panosam[sam]'\n\n"
            "Then login to HuggingFace (required for SAM3 model access):\n"
            "  huggingface-cli login\n\n"
            "You must also accept the SAM3 license at:\n"
            "  https://huggingface.co/facebook/sam3"
        )


class SAM3Engine:
    """SAM3 segmentation engine using HuggingFace Transformers.

    This engine uses the facebook/sam3 model for Promptable Concept Segmentation (PCS)
    on images. It supports text prompts to segment all instances of a concept.

    Attributes:
        model: The SAM3 model.
        processor: The SAM3 processor for pre/post-processing.
        device: The device to run inference on (cuda, mps, or cpu).

    Note:
        Requires SAM3 dependencies. Install with: pip install "panosam[sam]"
        Also requires HuggingFace login: huggingface-cli login
    """

    def __init__(
        self,
        model_id: str = "facebook/sam3",
        device: Optional[str] = None,
        dtype: "torch.dtype" = None,
    ):
        """Initialize the SAM3 engine.

        Args:
            model_id: HuggingFace model ID for SAM3.
            device: Device to use. If None, auto-detects (cuda > mps > cpu).
            dtype: Data type for model weights. Defaults to torch.float32.

        Raises:
            ImportError: If SAM3 dependencies are not installed.
        """
        # Check dependencies first with helpful error message
        _check_sam_dependencies()

        import torch
        from transformers import Sam3Processor, Sam3Model

        # Set default dtype after torch import
        if dtype is None:
            dtype = torch.float32

        # Auto-detect device
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self.device = device
        self.dtype = dtype

        print(f"Loading SAM3 model on {device}...")
        self.model = Sam3Model.from_pretrained(model_id).to(device)
        self.processor = Sam3Processor.from_pretrained(model_id)
        print("SAM3 model loaded successfully.")

    def segment(
        self,
        image: Image.Image,
        text_prompt: str,
        threshold: float = 0.5,
        mask_threshold: float = 0.5,
        simplify_tolerance: float = 0.005,
        return_raw_masks: bool = False,
    ) -> List[FlatMaskResult] | Tuple[List[FlatMaskResult], List[np.ndarray]]:
        """Segment objects in an image using a text prompt.

        Args:
            image: Input image as PIL Image.
            text_prompt: Text describing the objects to segment (e.g., "car", "person").
            threshold: Confidence threshold for detections (0-1).
            mask_threshold: Threshold for binary mask generation (0-1).
            simplify_tolerance: Tolerance for polygon simplification (0-1).
            return_raw_masks: If True, also return raw binary masks for visualization.

        Returns:
            List of FlatMaskResult objects containing segmentation masks.
            If return_raw_masks=True, returns tuple of (flat_results, raw_masks).
        """
        import torch

        # Ensure RGB
        if image.mode != "RGB":
            image = image.convert("RGB")

        original_size = image.size  # (width, height)

        # Process inputs
        inputs = self.processor(images=image, text=text_prompt, return_tensors="pt").to(
            self.device
        )

        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Post-process results
        results = self.processor.post_process_instance_segmentation(
            outputs,
            threshold=threshold,
            mask_threshold=mask_threshold,
            target_sizes=[list(reversed(original_size))],  # (height, width)
        )[0]

        # Convert to FlatMaskResult
        flat_results = []
        raw_masks = []
        masks = results.get("masks", [])
        scores = results.get("scores", [])

        for i, (mask, score) in enumerate(zip(masks, scores)):
            # Convert mask to numpy
            mask_np = mask.cpu().numpy() if isinstance(mask, torch.Tensor) else mask

            # Ensure 2D mask
            if mask_np.ndim > 2:
                mask_np = mask_np.squeeze()

            # Create FlatMaskResult from binary mask
            flat_result = FlatMaskResult.from_binary_mask(
                mask=mask_np,
                score=float(score),
                label=text_prompt,
                mask_id=f"{text_prompt}_{i}",
                simplify_tolerance=simplify_tolerance,
            )

            # Only include masks with at least one valid polygon (3+ points)
            if flat_result.polygons and any(len(p) >= 3 for p in flat_result.polygons):
                flat_results.append(flat_result)
                if return_raw_masks:
                    raw_masks.append(mask_np)

        if return_raw_masks:
            return flat_results, raw_masks
        return flat_results

    def segment_with_boxes(
        self,
        image: Image.Image,
        boxes: List[Tuple[int, int, int, int]],
        box_labels: Optional[List[int]] = None,
        threshold: float = 0.5,
        mask_threshold: float = 0.5,
        simplify_tolerance: float = 0.005,
    ) -> List[FlatMaskResult]:
        """Segment objects in an image using bounding box prompts.

        Args:
            image: Input image as PIL Image.
            boxes: List of bounding boxes in (x1, y1, x2, y2) pixel format.
            box_labels: List of labels (1 for positive, 0 for negative). Defaults to all positive.
            threshold: Confidence threshold for detections (0-1).
            mask_threshold: Threshold for binary mask generation (0-1).
            simplify_tolerance: Tolerance for polygon simplification (0-1).

        Returns:
            List of FlatMaskResult objects containing segmentation masks.
        """
        import torch

        if len(boxes) == 0:
            return []

        # Ensure RGB
        if image.mode != "RGB":
            image = image.convert("RGB")

        original_size = image.size  # (width, height)

        # Default to positive boxes
        if box_labels is None:
            box_labels = [1] * len(boxes)

        # Format for processor: [batch, num_boxes, 4]
        input_boxes = [[list(box) for box in boxes]]
        input_boxes_labels = [box_labels]

        # Process inputs
        inputs = self.processor(
            images=image,
            input_boxes=input_boxes,
            input_boxes_labels=input_boxes_labels,
            return_tensors="pt",
        ).to(self.device)

        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Post-process results
        results = self.processor.post_process_instance_segmentation(
            outputs,
            threshold=threshold,
            mask_threshold=mask_threshold,
            target_sizes=[list(reversed(original_size))],
        )[0]

        # Convert to FlatMaskResult
        flat_results = []
        masks = results.get("masks", [])
        scores = results.get("scores", [])

        for i, (mask, score) in enumerate(zip(masks, scores)):
            mask_np = mask.cpu().numpy() if isinstance(mask, torch.Tensor) else mask

            if mask_np.ndim > 2:
                mask_np = mask_np.squeeze()

            flat_result = FlatMaskResult.from_binary_mask(
                mask=mask_np,
                score=float(score),
                label=f"box_{i}",
                mask_id=f"box_{i}",
                simplify_tolerance=simplify_tolerance,
            )

            # Only include masks with at least one valid polygon (3+ points)
            if flat_result.polygons and any(len(p) >= 3 for p in flat_result.polygons):
                flat_results.append(flat_result)

        return flat_results

    def get_raw_masks(
        self,
        image: Image.Image,
        text_prompt: str,
        threshold: float = 0.5,
        mask_threshold: float = 0.5,
    ) -> Tuple[List[np.ndarray], List[float]]:
        """Get raw binary masks without polygon conversion.

        Useful when you need the full mask data rather than simplified polygons.

        Args:
            image: Input image as PIL Image.
            text_prompt: Text describing the objects to segment.
            threshold: Confidence threshold for detections (0-1).
            mask_threshold: Threshold for binary mask generation (0-1).

        Returns:
            Tuple of (masks, scores) where masks are numpy arrays.
        """
        import torch

        if image.mode != "RGB":
            image = image.convert("RGB")

        original_size = image.size

        inputs = self.processor(images=image, text=text_prompt, return_tensors="pt").to(
            self.device
        )

        with torch.no_grad():
            outputs = self.model(**inputs)

        results = self.processor.post_process_instance_segmentation(
            outputs,
            threshold=threshold,
            mask_threshold=mask_threshold,
            target_sizes=[list(reversed(original_size))],
        )[0]

        masks = []
        scores = []

        for mask, score in zip(results.get("masks", []), results.get("scores", [])):
            mask_np = mask.cpu().numpy() if isinstance(mask, torch.Tensor) else mask
            if mask_np.ndim > 2:
                mask_np = mask_np.squeeze()
            masks.append(mask_np)
            scores.append(float(score))

        return masks, scores
