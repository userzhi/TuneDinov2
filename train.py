from  argparse import ArgumentParser

from model.fg_classifier import FgClassifier
from utils_dir.backbones_utils import load_backbone, extract_backbone_features
def get_argparse():
    parser = ArgumentParser()
    parser.add_argument('--train_root_dir', type=str)
    parser.add_argument('--train_annotations_file', type=str)

    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lr_step_size', type=int, default=30)
    parser.add_argument('--lr_decay', type=float, default=0.1)
    parser.add_argument('--epoch', type=int, default=50)

    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=8)

    parser.add_argument('--save_dir', type=str, default=None)
    parser.add_argument('--backbone_type', type=str, default='dinov2')
    parser.add_argument('--num_layers', type=int, default=2)

    args = parser.parse_args()

    return args


def train(args):

    # load backbone
    dinov2_backbone = load_backbone(args.backbone_type)
    fg_classifier = FgClassifier()

    # load dataset    


    pass





    