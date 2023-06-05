import os, torch
from utils.eval import Evaluator
from models import SOSNet, TFeat, Baseline
from utils.class_related import Matcher
from utils.detector import RidgeDetector
from utils.utils import getConfig


# If you wanna use a keypoint detector based on SIFT/FAST/SURF, please set detector as None.
# Detection and descriptor extraction methods are encapsulated in the same Baseline class.
det_dict = {
    'None': None,
    'RidgeDet': RidgeDetector(ks=11, c=4)
}

desp_dict = {
    'SIFT': Baseline('RootSIFT'),
    'FAST': Baseline('FAST'),
    'SURF': Baseline('SURF'),
    'SOSNet': SOSNet('models/weights/sosnet-32x32-liberty.pth'),
    'TFeat': TFeat('models/weights/tfeat-liberty.params')
}


if __name__ == '__main__':
    import argparse
    root = os.path.dirname(os.path.abspath(__file__))
    data_root = os.path.join(root, 'data')
    db = 'SDUMLA'
    protocol = 'FVC'

    parser = argparse.ArgumentParser()
    parser.add_argument('-db', '--db_name', type=str, default=db, help='Name of the database in the data root dir.')
    parser.add_argument('-p', '--pair_file', type=str, default=os.path.join(data_root, *['pairs', f'{db}-{protocol}.txt']), help='File path of the input-pair meta info.')
    parser.add_argument('-s', '--save_path', type=str, default='./results', help='Path of the log & result files to save.')
    parser.add_argument('-desp', '--descriptor', type=str, default='SOSNet', help='Options [\'SOSNet\', \'TFeat\', \'SIFT\', \'FAST\', \'SURF\']')
    parser.add_argument('-det', '--detector', type=str, default='RidgeDet', help='Options [\'RidgeDet\', \'None\'].If you wanna use a keypoint detector absed on SIFT/FAST/SURF, please set detector as \'None\'')
    args = parser.parse_args()
    data_path = os.path.join(data_root, args.db_name)

    log_prefix = args.descriptor # A prefix of the log name, which can be randomly defined.

    device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
    print('device: ', device)
    detector = det_dict[args.detector]
    descriptor = desp_dict[args.descriptor]
    matcher = Matcher(dist_thresh=descriptor.dist_thresh)

    # res_file = 'results-'+prefix+'.txt'
    kargs = {
        'im_file': 'seg-thu',
        'thi_file': 'thi',
        'pair_file': args.pair_file,
        'prefix': log_prefix
    }
    evaluator = Evaluator(descriptor, matcher, data_path , device, detector, **kargs)
    evaluator.data_config = getConfig(os.path.join(data_root, 'config.yaml'))[args.db_name]

    log_file = f'{args.save_path}/{log_prefix}-{args.db_name}-{protocol}.txt'
    evaluator(save_path=args.save_path, log_file=log_file)