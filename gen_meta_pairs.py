
if __name__ is '__main__':
    import random
    from tqdm import tqdm
    import argparse
    from utils.utils import *

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_path', type=str, default='data/SDUMLA')
    parser.add_argument('-cfg', '--config_file', type=str, default='data/config.yaml')
    args = parser.parse_args()

    save = 'txt'
    # modes = ['full', 'short', 'FVC']
    db = args.data_path.split('/')[-1]
    thi_files = os.path.join(args.data_path, 'thi')
    names = os.listdir(thi_files)
    # names = sorted(names, key=lambda i:str(i.split('.')[0]))
    d = getConfig(args.config_file)[db]
    prefix_num, postfix, split, trim, session = d.values()
    select_rate = 0.5
    total_len = 0
    flag = '_flag_'
    genuine = []; imposter = []; imposter_FVC = []
    testi = 0; testj = 0
    for step, i in tqdm(enumerate(names[:-1])):
        if not i.endswith(postfix):
            continue
        name1 = i.replace(postfix, '')
        cur = 0
        select1 = np.random.choice(range(1, 7))
        select2 = np.random.choice(range(1, 7))
        for ind, j in enumerate(names[step+1:]):
            cur += 1
            if not j.endswith(postfix):
                continue

            name2 = j.replace(postfix, '')
            print(name1, name2)
            if db == 'MMCBNU':
                testi = '_'.join(name1.split('_')[:3]) == '054_2_2'
                testj = '_'.join(name2.split('_')[:3]) == '054_2_3'

            tmp = name1 + flag + name2 + flag
            judge = judge_prefix(name1, name2, prefix_num, split)
            if judge or (testi and testj):
                genuine.append(tmp+'1')
            else:
                imposter.append(tmp+'0')
                if not judge_FVC(session, prefix_num, name1, name2, step=step):
                    imposter_FVC.append(tmp+'0')
            total_len += 1

    print('totle_len:', total_len, '; genuine:', len(genuine), ':imposter:', len(imposter))
    classes = {}
    res = []
    # pick up all genuine matching
    for ii in tqdm(genuine):
        res.append(ii)
    gs = len(res)
    res_full = res.copy()
    res_FVC = res.copy()
    res_short = res.copy()
    random.shuffle(imposter)
    ims = 0

    mode = 'FVC'
    res = res_FVC
    for ii in tqdm(imposter_FVC):
        res_FVC.append(ii)
    ims = len(res) - gs
    print(db+'-'+mode, f'total: {len(res)}; genuine: {gs}; imposter: {ims}')
    random.shuffle(res)
    s = '\n'.join(res)
    f_new = open(save+'/'+db+'-'+mode+'.txt', 'w')
    f_new.write(s)
    f_new.close()

    mode = 'full'
    res = res_full
    for ii in tqdm(imposter):
        res.append(ii)
    ims = len(res) - gs
    print(db+'-'+mode, f'total: {len(res)}; genuine: {gs}; imposter: {ims}')
    random.shuffle(res)
    s = '\n'.join(res)
    f_new = open(save+'/'+db+'-'+mode+'.txt', 'w')
    f_new.write(s)
    f_new.close()

    mode = 'short'
    res = res_short
    ims = 0
    while ims < 10*gs:
        res.append(imposter[ims])
        ims += 1
    print(db+'-'+mode, f'total: {len(res)}; genuine: {gs}; imposter: {ims}')
    res = res_short
    random.shuffle(res)
    s = '\n'.join(res)
    f_new = open(save+'/'+db+'-'+mode+'.txt', 'w')
    f_new.write(s)
    f_new.close()