


if __name__ == '__main__':
    import numpy as np
    import os, cv2
    from tqdm import tqdm
    import argparse
    from pre_process import pre_processing
    db = 'THUFV'

    parser = argparse.ArgumentParser()
    parser.add_argument('-db', '--db_name', type=str, default=db, help='Name of the database in the data root dir.')
    parser.add_argument('-r', '--training_rate', type=float, default=0.85, help='Training data & validation data split ratio.')
    args = parser.parse_args()

    data_root = '../../data'
    targets = ['image', 'mask']
    source = 'gass'
    rate = args.training_rate

    for target in targets:
        name = source if target == 'mask' else 'roi'
        fp = f'{data_root}/{db}/{name}'
        files = sorted(os.listdir(fp), key=lambda i:str(i.split('.')[0]))

        save_train = os.path.join(f'train_data/{source}/train', target)
        save_test = os.path.join(f'train_data/{source}/test', target)
        os.makedirs(save_train, exist_ok=True)
        os.makedirs(save_test, exist_ok=True)

        trim = [10, 10, 10, 10]
        for i, f in enumerate(tqdm(files)):
            if not f.endswith('.bmp'):
                continue

            if i < len(files)*rate:
                save = save_train
            else:
                save = save_test
            dataFile = os.path.join(fp, f)
            im = cv2.imread(dataFile, 0)
            # im = pre_processing(im)
            im = cv2.resize(im, (256, 128))
            if target == 'mask' and source == 'gass':
                im = cv2.GaussianBlur(im, (5, 5), 0)

            cv2.imwrite(os.path.join(save, f), (im).astype(np.float32))
    print('Done')
