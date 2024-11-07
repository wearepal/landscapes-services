import json
import os
import torch

from PIL import Image
from torch.utils.data import Dataset
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from transformers import Trainer, AutoProcessor, AutoModelForZeroShotObjectDetection
from transformers.image_transforms import corners_to_center_format
from tqdm import tqdm

from loss import DeformableDetrLoss, DeformableDetrHungarianMatcher


# Defined functions
def train_model(detector_id, args, train_data, val_data=None):
    det_processor = AutoProcessor.from_pretrained(detector_id)
    det_model = AutoModelForZeroShotObjectDetection.from_pretrained(detector_id)

    def data_collator(features):
        batch = {}
        encoding = det_processor(
            text=[[*[f'A photo of a {label.strip()}' for label in train_data.label2id.keys()], '']],
            images=[f['image'] for f in features],
            return_tensors='pt'
        )
        for k, v in encoding.items():
            batch[k] = v

        batch['labels'] = []
        for f in features:

            w, h = f['image'].size
            box = torch.tensor(f['box'])

            if (box[:, 2:] >= box[:, :2]).all():
                box = corners_to_center_format(box)

            batch['labels'].append({
                'target_sizes': torch.tensor([w, h], dtype=torch.long),
                'class_labels': torch.tensor(f['class_label'], dtype=torch.long),
                'boxes': box / torch.tensor([w, h, w, h])
            })

        batch['return_loss'] = True
        return batch

    metric = MeanAveragePrecision(box_format='cxcywh')
    metric.warn_on_many_detections = False

    def compute_metrics(eval_preds, compute_result):
        predictions, label_ids = eval_preds

        logits = predictions[0]
        pred_boxes = predictions[2]

        target_sizes = label_ids[0]['target_sizes']
        class_labels = label_ids[0]['class_labels']
        boxes = label_ids[0]['boxes']

        probs = torch.max(logits[..., :-1], dim=-1)
        scores = torch.sigmoid(probs.values)
        labels = probs.indices

        # Rescale coordinates, image is padded to square for inference,
        # that is why we need to scale boxes to the max size
        img_h, img_w = target_sizes.reshape(-1, 2).unbind(1)
        size = torch.max(img_h, img_w)
        scale_fct = torch.stack([size, size, size, size], dim=1)

        preds = [
            dict(
                boxes=(pred_boxes * scale_fct[:, None, :]).squeeze(0).cpu(),
                scores=scores.squeeze(0).cpu(),
                labels=labels.squeeze(0).cpu()
            )
        ]
        target = [
            dict(
                boxes=(boxes * scale_fct[:, None, :]).squeeze(0).cpu(),
                labels=class_labels.cpu()
            )
        ]

        metric.update(preds, target)
        if compute_result:
            return {k: v for k, v in metric.compute().items() if 'map' in k}

    trainer = DetectionTrainer(
        model=det_model,
        args=args,
        data_collator=data_collator,
        train_dataset=train_data,
        eval_dataset=val_data if val_data else None,
        compute_metrics=compute_metrics
    )
    trainer.train()

    # Save the model
    trainer.save_model(args.output_dir)

    with open(os.path.join(args.output_dir, 'loss.txt'), 'w') as file:
        for obj in trainer.state.log_history:
            file.write(json.dumps(obj))
            file.write('\n\n')


# Defined classes
class DetectionTrainer(Trainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.model.config.model_type not in ['owlvit', 'owlv2']:
            raise ValueError('The model must be an owlvit or owlv2 model type.')

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        # First, sent images through DETR base model to obtain encoder + decoder outputs
        labels = inputs.pop('labels')
        _ = inputs.pop('return_loss')
        outputs = model(**inputs)

        # class logits + predicted bounding boxes
        logits = outputs.logits
        pred_boxes = outputs.pred_boxes

        loss, loss_dict = None, None
        if labels is not None:
            # First: create the matcher
            matcher = DeformableDetrHungarianMatcher(
                class_cost=self.args.class_cost, 
                bbox_cost=self.args.bbox_cost, 
                giou_cost=self.args.giou_cost,
                focal_loss=self.args.focal_loss,
                focal_alpha=self.args.focal_alpha,
                focal_gamma=self.args.focal_gamma,
            )
            # Second: create the criterion
            losses = ["labels", "boxes"]
            criterion = DeformableDetrLoss(
                matcher=matcher,
                num_classes=logits.shape[-1] - 1,
                focal_alpha=self.args.focal_alpha,
                focal_gamma=self.args.focal_gamma,
                losses=losses,
            )
            criterion.to(model.device)
            # Third: compute the losses, based on outputs and labels
            outputs_loss = {}
            outputs_loss["logits"] = logits
            outputs_loss["pred_boxes"] = pred_boxes

            loss_dict = criterion(outputs_loss, labels)
            # Fourth: compute total loss, as a weighted sum of the various losses
            weight_dict = {"loss_ce": 1, "loss_bbox": 1}
            weight_dict["loss_giou"] = 1
            loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss


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
