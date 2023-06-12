import os, torch

# If you wanna use a keypoint detector based on SIFT/FAST/SURF, please set detector as None.
# Detection and descriptor extraction methods are encapsulated in the same Baseline class.
def det_dict(name=None):
    if name is None:
        return None
    elif name == 'RidgeDet':
        from models import RidgeDetector
        return RidgeDetector(ks=11, c=4)

def desp_dict(name):
    if name in ['SIFT', 'FAST', 'SURF']:
        from models import DespBaseline
        return DespBaseline(name)
    elif name == 'SOSNet':
        from models import SOSNet
        return SOSNet('models/weights/sosnet-32x32-liberty.pth')
    elif name == 'TFeat':
        from models import TFeat
        return TFeat('models/weights/tfeat-liberty.params')


if __name__ == '__main__':
    from utils.evaluator import Evaluator
    from utils.class_related import Matcher
    from utils.utils import getConfig
    import argparse
    
    root = os.path.dirname(os.path.abspath(__file__))
    data_root = os.path.join(root, 'data')

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--pair_file', type=str, default=os.path.join('data', 'pairs_meta', f'cross_dataset.txt'), help='File path of the input-pair meta info.')
    parser.add_argument('-s', '--save_path', type=str, default='./results', help='Path of the log & result files to save.')
    parser.add_argument('-desp', '--descriptor', type=str, default='SOSNet', help='Options [\'SOSNet\', \'TFeat\', \'SIFT\', \'FAST\', \'SURF\']')
    parser.add_argument('-det', '--detector', type=str, default='RidgeDet', help='Options [\'RidgeDet\', \'None\'].If you wanna use a keypoint detector absed on SIFT/FAST/SURF, please set detector as \'None\'')
    args = parser.parse_args()
    
    log_suffix = os.path.basename(args.pair_file).replace('.txt', '')
    log_name = f'{args.descriptor}-{log_suffix}'
    device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
    detector = det_dict(args.detector)
    descriptor = desp_dict(args.descriptor)
    matcher = Matcher(dist_thresh=descriptor.dist_thresh)

    evaluator = Evaluator(descriptor, matcher, data_root, device, args.pair_file, detector)
    evaluator.data_config = getConfig(os.path.join(data_root, 'config.yaml'))

    log_file = f'{args.save_path}/{log_name}.txt'
    evaluator(save_path=args.save_path, log_file=log_file)