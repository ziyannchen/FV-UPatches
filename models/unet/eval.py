import torch
import os
from glob import glob
import cv2
from torchvision import transforms
from operator import add
from tqdm import tqdm
import numpy as np
import time
from utils import calculate_metrics


if __name__ == '__main__':
    from unet import UNet
    import argparse
    from pathlib import Path
    PROJ_ROOT = Path(__file__).resolve().parents[2]
    print(PROJ_ROOT)

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_path', type=str, default=f'{PROJ_ROOT}/data/SDUMLA/roi', help='Name of the database in the data root dir.')
    parser.add_argument('-w', '--model_weight', type=str, default=f'{PROJ_ROOT}/models/unet/weights/fv_unet-v1.pth')
    parser.add_argument('-s', '--save_path', type=str, default=f'{PROJ_ROOT}/data/SDUMLA/gass-thu')
    parser.add_argument('-l', '--label_path', type=str, default=None)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet(weight_file=args.model_weight, train=False, device=device)
    save_im = False

    os.makedirs(args.save_path, exist_ok=True)
    trim = [10, 0, 10, 0]
    print(device)

    test_x = sorted(glob(f'{args.data_path}/*'))
    print(len(test_x))
    if args.label_path is not None:
        test_y = sorted(glob(f"{args.label_path}/*"))
        print(len(test_y))

        assert len(test_x) == len(test_y), 'Data files from roi and test not equal!'

    metrics_score = [0.0, 0.0, 0.0, 0.0, 0.0]
    time_taken = []
    ss = (256, 128)
    for i, im_path in tqdm(enumerate(test_x), total=len(test_x)):
        image = cv2.imread(im_path, 0)
        ori_h, ori_w = image.shape
        image = cv2.resize(image, (model.W, model.H))
        x = transforms.ToTensor()(image)
        x = x.unsqueeze(0).to(device)

        if args.label_path:
            y = test_y[i]
            """ Reading mask """
            mask = cv2.imread(y, cv2.IMREAD_GRAYSCALE)  ## (512, 512)
            mask[:trim[0], :] = 0
            mask[-1 - trim[2]:, :] = 0
            mask = cv2.resize(mask, ss)
            y = transforms.ToTensor()(mask)
            y = y.unsqueeze(0).to(device)

        with torch.no_grad():
            start_time = time.time()
            pred_y = model(x)
            pred_y = torch.sigmoid(pred_y)
            total_time = time.time() - start_time
            time_taken.append(total_time)
            pred_y[..., :trim[0], :] = 0
            pred_y[..., -1 - trim[2]:, :] = 0

            if args.label_path:
                score = calculate_metrics(y, pred_y)
                metrics_score = list(map(add, metrics_score, score))
            pred_y = pred_y[0].cpu().numpy()  ## (1, 512, 512)
            pred_y = np.squeeze(pred_y, axis=0)  ## (512, 512)
            #         pred_y = pred_y > 0.5    # ReLu-style Binalization
            pred_y = np.array(pred_y * 255, dtype=np.uint8)

        im_name = str(Path(im_path).resolve()).split(os.sep)[-1]

        if save_im:
            cv2.imwrite(f'{args.save_path}/{im_name}', cv2.resize(pred_y, (ori_w, ori_h)))
            if args.label_path:
                save_combine_path = f'{args.save_path}/combine'
                os.makedirs(save_combine_path, exist_ok=True)
                plot_size = (ori_w, ori_h)
                cat_images = np.concatenate(
                    [cv2.resize(image, plot_size), cv2.resize(mask, plot_size), cv2.resize(pred_y, plot_size)],
                    axis=0
                )
                cv2.imwrite(f"{save_combine_path}/{im_name}", cat_images)

    if args.label_path:
        jaccard = metrics_score[0] / len(test_x)
        f1 = metrics_score[1] / len(test_x)
        recall = metrics_score[2] / len(test_x)
        precision = metrics_score[3] / len(test_x)
        acc = metrics_score[4] / len(test_x)
        print(f"Jaccard: {jaccard:1.4f} - F1: {f1:1.4f} - Recall: {recall:1.4f} - Precision: {precision:1.4f} - Acc: {acc:1.4f}\n")

    # To precisely compute the fps, set save_im=False to ignore the influence of extra I/O time.
    fps = 1 / np.mean(time_taken)
    print("FPS: ", fps)

    if save_im:
        print('All results have saved to '+args.save_path)