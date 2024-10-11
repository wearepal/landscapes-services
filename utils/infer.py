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
    threshold: float = 0.3,
    detector_id: Optional[str] = None,
    segmenter_id: Optional[str] = None,
    transform: Optional[bool] = False
) -> List[dict]:
    if not image_path.endswith('.tif'):
        raise ValueError('The image must be a GeoTIFF file.')

    image = Image.open(image_path, 'r')

    if transform:
        with rasterio.open(image_path) as src:
            transform = src.transform

        width, height = image.size
        cols, rows = np.meshgrid(np.arange(width), np.arange(height))

        long, lat = rasterio.transform.xy(src.transform, rows, cols)
        long, lat = np.array(long), np.array(lat)

    det_model = AutoModelForZeroShotObjectDetection.from_pretrained(detector_id)
    det_processor = AutoProcessor.from_pretrained(detector_id)

    seg_model = AutoModelForMaskGeneration.from_pretrained(segmenter_id)
    seg_processor = AutoProcessor.from_pretrained(segmenter_id)

    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        det_model = det_model.to(device)
        seg_model = seg_model.to(device)

    except:
        pass

    preds = []
    with torch.no_grad():

        # Target image sizes (height, width) to rescale box predictions [batch_size, 2]
        target_sizes = torch.tensor([image.size[::-1]])

        inputs = det_processor(text=labels, images=image, return_tensors='pt')
        outputs = det_model(**inputs)

        # Convert outputs (bounding boxes and class logits) to Pascal VOC format (xmin, ymin, xmax, ymax)
        detections = det_processor.post_process_object_detection(
            outputs, 
            threshold=threshold, 
            target_sizes=target_sizes
        )
        detections = detections[0]

        inputs = seg_processor(images=image, input_boxes=[detections['boxes'].tolist()], return_tensors='pt')
        outputs = seg_model(**inputs)

        masks = seg_processor.post_process_masks(
            masks=outputs['pred_masks'],
            original_sizes=inputs['original_sizes'],
            reshaped_input_sizes=inputs['reshaped_input_sizes']
        )
        masks = masks[0]

        masks = refine_masks(masks)
        for i, mask in enumerate(masks):

            xmin, ymin, xmax, ymax = detections['boxes'][i].int().tolist()

            if transform:
                xmin, ymin = rasterio.transform.xy(transform, xmin, ymin)
                xmax, ymax = rasterio.transform.xy(transform, xmax, ymax)

                xmask, ymask = np.where(mask==1)
                mask = np.hstack((long[xmask][..., np.newaxis], lat[ymask][..., np.newaxis]))

            preds.append({
                'score': detections['scores'][i].item(),
                'label': labels[0][detections['labels'][i].item()],
                'box': {'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax},
                'mask': mask.tolist()
            })

        return preds


def refine_masks(masks: torch.BoolTensor) -> List[np.ndarray]:
    masks = masks.cpu().float()
    masks = masks.permute(0, 2, 3, 1)
    masks = masks.mean(axis=-1)
    masks = (masks > 0).int()
    masks = masks.numpy().astype(np.uint8)
    return list(masks)