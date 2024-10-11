import argparse
import json
import os
import sys

from transformers import set_seed

sys.path.append(os.path.join('utils'))

from utils.infer import detect_segment


def main():

    # Define the arguments
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--image_path',
        type=str,
        required=True,
        help='The image path to use.'
    )
    parser.add_argument(
        '--labels',
        nargs='+',
        type=str,
        required=True,
        help='The labels to use.'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.1,
        help='The threshold to use.'
    )
    parser.add_argument(
        '--detector_id',
        type=str,
        default='google/owlv2-base-patch16',
        help='The detector to use.'
    )
    parser.add_argument(
        '--segmenter_id',
        type=str,
        default='facebook/sam-vit-base',
        help='The segmentor to use.'
    )

    parser.add_argument(
        '--save_path',
        type=str,
        required=True,
        help='The save path to use.'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=0,
        help='The seed to use.'
    )

    args = parser.parse_args()

    # Set the hyperparameters
    IMAGE_PATH = args.image_path
    LABELS = args.labels
    THRESHOLD = args.threshold
    DETECTOR_ID = args.detector_id
    SEGMENTER_ID = args.segmenter_id

    SAVE_PATH = args.save_path
    SEED = args.seed


    # Set the seed
    set_seed(SEED)

    # Run the pipeline
    predictions = detect_segment(
        image_path=IMAGE_PATH,
        labels=[LABELS],
        threshold=THRESHOLD,
        detector_id=DETECTOR_ID,
        segmenter_id=SEGMENTER_ID,
        transform=True
    )

    # Save the predictions
    with open(os.path.join(SAVE_PATH, 'predictions.json'), 'w') as f:
        json.dump(predictions, f)


if __name__ == '__main__':
    main()
