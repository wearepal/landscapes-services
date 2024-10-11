import cv2
import matplotlib.pyplot as plt
import numpy as np

from PIL import Image
from typing import List, Optional, Union


# Defined functions
def plot_predictions(
    image_path: str,
    detections: List[dict],
    save_name: Optional[str] = None
) -> None:
    if not image_path.endswith('.tif'):
        raise ValueError('The image must be a GeoTIFF file.')

    image = Image.open(image_path, 'r')
    annotated_image = annotate(image, detections)

    plt.imshow(annotated_image)
    plt.axis('off')
    if save_name:
        plt.savefig(save_name, bbox_inches='tight')

    plt.show()
    plt.close()


def annotate(image: Union[Image.Image, np.ndarray], detection_results: List[dict]) -> np.ndarray:
    # Convert PIL Image to OpenCV format
    image_cv2 = np.array(image) if isinstance(image, Image.Image) else image
    image_cv2 = cv2.cvtColor(image_cv2, cv2.COLOR_RGB2BGR)

    # Iterate over detections and add bounding boxes and masks
    for detection in detection_results:
        score = detection['score']
        label = detection['label']
        box = detection['box']
        mask = detection['mask']

        # Sample a random color for each detection
        color = np.random.randint(0, 256, size=3)

        # Draw bounding box
        cv2.rectangle(image_cv2, (box['xmin'], box['ymin']), (box['xmax'], box['ymax']), color.tolist(), 2)
        cv2.putText(image_cv2, f'{label}: {score:.2f}', (box['xmin'], box['ymin'] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color.tolist(), 2)

        # If mask is available, apply it
        if mask is not None:
            # Convert mask to uint8
            mask_uint8 = (mask * 255).astype(np.uint8)
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(image_cv2, contours, -1, color.tolist(), 2)

    return cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)
