import json
import os
import torch

from peft import get_peft_model
from peft import LoraConfig
from transformers import Trainer, AutoProcessor, AutoModelForZeroShotObjectDetection

from data import expand2square
from loss import DeformableDetrLoss, DeformableDetrHungarianMatcher


# Defined functions
def train_model(detector_id, args, train_data, val_data=None):
    det_processor = AutoProcessor.from_pretrained(detector_id)
    det_model = AutoModelForZeroShotObjectDetection.from_pretrained(detector_id)

    def data_collator(features):
        batch = {}
        encoding = det_processor(
            text=[*list(train_data.label2id.keys()), ""],
            images=[
                expand2square(f['image'], det_model.config.vision_config.image_size) for f in features
            ],
            return_tensors='pt'
        )
        for k, v in encoding.items():
            batch[k] = v

        batch['labels'] = []
        for f in features:

            batch['labels'].append({
                'class_labels': torch.tensor(f['class_label'], dtype=torch.long),
                'boxes': torch.tensor(f['box'], dtype=torch.float)
            })

        batch['return_loss'] = True
        return batch

    if hasattr(args, 'r'):
        det_model = get_peft_model(
            det_model,
            peft_config=LoraConfig(
                r=args.r,
                lora_alpha=(args.r * 2),
                target_modules=['q_proj', 'v_proj']
            )
        )

    trainer = DetectionTrainer(
        model=det_model,
        args=args,
        data_collator=data_collator,
        train_dataset=train_data,
        eval_dataset=val_data if val_data else None
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
