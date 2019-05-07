import argparse
parser = argparse.ArgumentParser(description="configs of the model")


# ========================= Path Configs ==========================
parser.add_argument('--split_mode', type=int, default=1, choices=[1,2,3])
parser.add_argument('--dataset', type=str, default='hmdb51', choices=['hmdb51', 'ucf101'])
parser.add_argument('--source_path', type=str, default='../data_raw/hmdb51')
#parser.add_argument('--clip_len', type=int, default=16)
#parser.add_argument('--video_path', type=str, default='../data_raw')
parser.add_argument('--out_path', type=str, default='../frames')
parser.add_argument('--split_file_path', type=str, default='./testTrainMulti_7030_splits')

parser.add_argument('--subset', type=str, default='train', choices=['train', 'valid', 'test'])


# ========================= Learning Configs ==========================



# ========================= Runtime Configs ==========================








