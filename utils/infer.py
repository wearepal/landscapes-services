import numpy as np
import rasterio
import torch
import torch.nn.functional as F

from collections import defaultdict
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm
from transformers import (
    AutoProcessor, 
    AutoModelForZeroShotObjectDetection, 
    AutoTokenizer, 
    BlipForImageTextRetrieval
)
from typing import List, Optional

from model.segment_anything.utils.transforms import ResizeLongestSide


# Defined functions
@torch.no_grad()
def detect_segment(
    image_path: str,
    labels: List[List[str]],
    det_conf: float = 0.05,
    clf_conf: float = 0.90,
    n_repeats: int = 5,
    detector_id: Optional[str] = None,
    segmenter_id: Optional[str] = None,
    classifier_id: Optional[str] = None,
    transform: Optional[bool] = False
):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Target image sizes (height, width) to rescale box predictions [batch_size, 2]
    image = Image.open(image_path)
    image = np.array(image.convert('RGB'))
    target_height, target_width, _ = image.shape

    all_preds = []
    for n in tqdm(range(n_repeats)):

        preds = []

        if n > 0:
            det_model.to(device)
        else:
            det_processor = AutoProcessor.from_pretrained(detector_id)
            det_model = AutoModelForZeroShotObjectDetection.from_pretrained(detector_id)
    
            det_model = det_model.to(device)
            det_model.eval()

        inputs = det_processor(text=labels, images=image, return_tensors='pt')
        outputs = det_model(**inputs.to(device))

        # Convert outputs (bounding boxes and class logits) to Pascal VOC format (xmin, ymin, xmax, ymax)
        detections = det_processor.post_process_object_detection(
            outputs, 
            threshold=det_conf, 
            target_sizes=[(target_height, target_width)]
        )
        detections = detections[0]

        if len(detections['boxes']) < 1:
            break

        if n < n_repeats - 1:
            det_model.to('cpu')
        else:
            del det_model
            torch.cuda.empty_cache()

        tokenizer = AutoTokenizer.from_pretrained(segmenter_id)

        if n > 0:
            seg_model.to(device)
        else:
            if 'sam2' in segmenter_id:
                from model.evf_sam2 import EvfSam2Model
                seg_model = EvfSam2Model.from_pretrained(segmenter_id, low_cpu_mem_usage=True)
                model_type = "sam2"
            else:
                from model.evf_sam import EvfSamModel
                seg_model = EvfSamModel.from_pretrained(segmenter_id, low_cpu_mem_usage=True)
                model_type = "ori"
    
            seg_model = seg_model.to(device)
            seg_model.eval()
    
        all_roi = defaultdict(list)
        for i, box in enumerate(detections['boxes'].int().tolist()):
    
            # preprocess
            xmin, ymin, xmax, ymax = box
            if xmin < 0 or ymin < 0 or xmax <= xmin or ymax <= ymin:
                continue
            if xmax > target_width or ymax > target_height:
                continue
            if (xmax - xmin) * (ymax - ymin) <= 0:
                continue
    
            roi = image[ymin:ymax, xmin:xmax]
    
            image_beit = beit3_preprocess(roi, 224).to(dtype=seg_model.dtype, device=seg_model.device)
            image_sam, resize_shape = sam_preprocess(roi, model_type=model_type)
            image_sam = image_sam.to(dtype=seg_model.dtype, device=seg_model.device)
    
            prompt = labels[0][detections['labels'][i].item()]
            input_ids = tokenizer(f'[semantic] {prompt}', return_tensors='pt')
            input_ids = input_ids['input_ids'].to(seg_model.device)
    
            # infer
            mask = seg_model.inference(
                image_sam.unsqueeze(0),
                image_beit.unsqueeze(0),
                input_ids,
                resize_list=[resize_shape],
                original_size_list=[roi.shape[:2]],
            )
    
            mask = mask.detach().cpu().numpy()[0]
            mask = mask > 0
    
            if clf_conf > 0:
                all_roi['image'].append(roi * mask[:, :, np.newaxis])
                all_roi['text'].append(prompt)
    
            full_mask = np.zeros((target_height, target_width), dtype=np.uint8)
            full_mask[ymin:ymax, xmin:xmax] = mask
    
            preds.append({
                'confidence': detections['scores'][i].item() * 100,
                'box': {'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax},
                'label': prompt,
                'mask': full_mask.tolist()
            })

        if n < n_repeats - 1:
            det_model.to('cpu')
        else:
            del seg_model
            torch.cuda.empty_cache()

        if clf_conf > 0:

            if n > 0:
                clf_model.to(device)
            else:
                clf_processor = AutoProcessor.from_pretrained(classifier_id)
                clf_model = BlipForImageTextRetrieval.from_pretrained(classifier_id)
        
                clf_model = clf_model.to(device)
                clf_model.eval()

            inputs = clf_processor(images=all_roi['image'], text=all_roi['text'], return_tensors='pt')
            logits_per_image = clf_model(**inputs.to(device)).itm_score

            probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
            indices = torch.nonzero(probs[:, 1] >= clf_conf).flatten()
            preds = [preds[i] for i in indices]

            if n < n_repeats - 1:
                det_model.to('cpu')
            else:
                del clf_model
                torch.cuda.empty_cache()

        if len(preds) < 1:
            break
        all_preds.extend(preds)

        if n_repeats > 1:
            inverted_mask = [p['mask'] for p in preds]
            inverted_mask = ~np.logical_or.reduce(inverted_mask)
            image *= inverted_mask[..., np.newaxis]

    if transform:

        src = rasterio.open(image_path)
        for pred in all_preds:

            xmin, ymin, xmax, ymax = pred['box'].values()
            xmin, ymin = src.xy(xmin, ymin)
            xmax, ymax = src.xy(xmax, ymax)
            pred['box'] = {'xmin': xmin, 'ymin': ymax, 'xmax': xmax, 'ymax': ymin}

            full_mask = np.array(pred['mask'])
            xindex, yindex = np.where(full_mask == 1)
            xindex, yindex = src.xy(xindex, yindex)
            full_mask = np.hstack((xindex[..., np.newaxis], yindex[..., np.newaxis]))
            pred['mask'] = full_mask.tolist()

    return all_preds


def sam_preprocess(
    x: np.ndarray,
    pixel_mean=torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1),
    pixel_std=torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1),
    img_size=1024,
    model_type="ori") -> torch.Tensor:
    '''
    preprocess of Segment Anything Model, including scaling, normalization and padding.  
    preprocess differs between SAM and Effi-SAM, where Effi-SAM use no padding.
    input: ndarray
    output: torch.Tensor
    '''
    assert img_size==1024, \
        "both SAM and Effi-SAM receive images of size 1024^2, don't change this setting unless you're sure that your employed model works well with another size."

    # Normalize colors
    if model_type=="ori":
        x = ResizeLongestSide(img_size).apply_image(x)
        h, w = resize_shape = x.shape[:2]
        x = torch.from_numpy(x).permute(2,0,1).contiguous()
        x = (x - pixel_mean) / pixel_std
        # Pad
        padh = img_size - h
        padw = img_size - w
        x = F.pad(x, (0, padw, 0, padh))
    else:
        x = torch.from_numpy(x).permute(2,0,1).contiguous()
        x = F.interpolate(x.unsqueeze(0), (img_size, img_size), mode="bilinear", align_corners=False).squeeze(0)
        x = (x - pixel_mean) / pixel_std
        resize_shape = None

    return x, resize_shape


def beit3_preprocess(x: np.ndarray, img_size=224) -> torch.Tensor:
    '''
    preprocess for BEIT-3 model.
    input: ndarray
    output: torch.Tensor
    '''
    beit_preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((img_size, img_size), interpolation=InterpolationMode.BICUBIC, antialias=None), 
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    return beit_preprocess(x)
