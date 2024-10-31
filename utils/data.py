import json

from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm


# Defined functions
def expand2square(image, size):
    width, height = image.size
    background_color = (0, 0, 0)

    if (width < size) and (height < size):
        result = Image.new(image.mode, (size, size), background_color)
        result.paste(image)

    elif width == height:
        result = image

    elif width > height:
        result = Image.new(image.mode, (width, width), background_color)
        result.paste(image)

    else:
        result = Image.new(image.mode, (height, height), background_color)
        result.paste(image)

    return result


# Defined classes
class DetectionDataset(Dataset):

    def __init__(self, ann_path, split):
        self.data, self.label2id = dict(), dict()
        for ann in tqdm(json.load(open(ann_path))):

            if ann['split'] != split:
                continue

            image_id = ann['image_id']
            _ = self.label2id.setdefault(ann['label'], len(self.label2id))

            if image_id not in self.data:
                self.data[image_id] = {
                    'label': ann['label'],
                    'image': ann['image'],
                    'box': ann['box']
                }

            else:
                self.data[image_id]['label'].append(ann['label'])
                self.data[image_id]['box'].append(ann['box'])

        self.data = list(self.data.values())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        return {
            'image': Image.open(sample['image']),
            'class_label': [self.label2id[l] for l in sample['label']],
            'box': sample['box']
        }
