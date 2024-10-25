import numpy as np
import rasterio
import torch

from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection, AutoModelForMaskGeneration
from typing import List, Optional


# Defined functions
def detect_segment(
    image_path: str,
    labels: List[List[str]],
    confidence: float = 0.1,
    detector_id: Optional[str] = None,
    segmenter_id: Optional[str] = None,
    transform: Optional[bool] = False
):
    image = Image.open(image_path)
    if transform:
        src = rasterio.open(image_path)

    det_model = AutoModelForZeroShotObjectDetection.from_pretrained(detector_id)
    det_processor = AutoProcessor.from_pretrained(detector_id)

    seg_model = AutoModelForMaskGeneration.from_pretrained(segmenter_id)
    seg_processor = AutoProcessor.from_pretrained(segmenter_id)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    det_model = det_model.to(device)
    seg_model = seg_model.to(device)

    preds = []
    with torch.no_grad():

        # Target image sizes (height, width) to rescale box predictions [batch_size, 2]
        target_sizes = torch.tensor([image.size[::-1]])

        inputs = det_processor(text=labels, images=image, return_tensors='pt')
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        outputs = det_model(**inputs)

        # Convert outputs (bounding boxes and class logits) to Pascal VOC format (xmin, ymin, xmax, ymax)
        detections = det_processor.post_process_object_detection(
            outputs, 
            threshold=confidence, 
            target_sizes=target_sizes
        )
        detections = detections[0]

        inputs = seg_processor(images=image, input_boxes=[detections['boxes'].tolist()], return_tensors='pt')
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        outputs = seg_model(**inputs)

        masks = seg_processor.post_process_masks(
            masks=outputs['pred_masks'],
            original_sizes=inputs['original_sizes'],
            reshaped_input_sizes=inputs['reshaped_input_sizes']
        )
        masks = masks[0]

        for i, mask in enumerate(refine_masks(masks)):

            xmin, ymin, xmax, ymax = detections['boxes'][i].int().tolist()
            if transform:
                xmin, ymin = src.xy(xmin, ymin)
                xmax, ymax = src.xy(xmax, ymax)

                xindex, yindex = np.where(mask == 1)
                xindex, yindex = src.xy(xindex, yindex)
                mask = np.hstack((xindex[..., np.newaxis], yindex[..., np.newaxis]))

            preds.append({
                'confidence': detections['scores'][i].item() * 100,
                'label': labels[0][detections['labels'][i].item()],
                'box': {'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax},
                'mask': mask.tolist()
            })

        return preds


def refine_masks(masks: torch.BoolTensor):
    masks = masks.permute(0, 2, 3, 1)
    masks = masks.prod(axis=-1)
    masks = (masks.cpu() > 0).numpy()
    return masks.astype(np.uint8)
