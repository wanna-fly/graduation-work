import torch
import torchvision
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from torch.utils.data import Dataset
from conf import Path
from sklearn.model_selection import train_test_split

class VideoDataset(Dataset):

    def __init__(self, dataset='hmdb51', split='train', clip_len=30, preprocess=False):
        self.root_dir, self.output_dir = Path.db_dir(dataset)
        folder = os.path.join(self.output_dir, split)
        self.clip_len = clip_len
        self.split = split

        self.resize_height = 224
        self.resize_width = 224
        self.crop_size = 112  ##

        if not self.check_integrity():
            raise RuntimeError('Dataset not found')
        if (not self.check_preprocess()) or preprocess:
            print('preprocessing the dataset ...')
            self.preprocess()

        self.fnames, labels = [], []
        for label in sorted(os.listdir(folder)):
            for fname in os.listdir(os.path.join(folder, label)):
                self.fnames.append(os.path.join(folder, label, fname))
                labels.append(label)

        assert len(labels) == len(self.fnames)
        print('num of {} videos: {:d}'.format(split, len(self.fnames)))

        self.label2index = {label: index for index, label in enumerate(sorted(set(labels)))}
        self.label_array = np.array([self.label2index[label] for label in labels], dtype=int)

        if dataset == "ucf101":
            if not os.path.exists('./ucf_labels.txt'):
                with open('./ucf_labels.txt', 'w') as f:
                    for id, label in enumerate(sorted(self.label2index)):
                        f.writelines(str(id + 1) + ' ' + label + '\n')
        elif dataset == 'hmdb51':
            if not os.path.exists('./hmdb_labels.txt'):
                with open('./hmdb_labels.txt', 'w') as f:
                    for id, label in enumerate(sorted(self.label2index)):
                        f.writelines(str(id) + ' ' + label + '\n')

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, index):
        buffer = self.load_frames(self.fnames[index])
        #buffer = self.crop(buffer, self.clip_len, self.crop_size)
        labels = np.array(self.label_array[index])

        if self.split == 'test':
            buffer = self.randomflip(buffer)
        buffer = self.normalize(buffer)
        buffer = self.to_tensor(buffer)
        return torch.from_numpy(buffer), torch.from_numpy(labels)

    def check_integrity(self):
        if not os.path.exists(self.root_dir):
            return False
        else:
            return True

    def check_preprocess(self):
        if not os.path.exists(self.output_dir):
            return False
        elif not os.path.exists(os.path.join(self.output_dir, 'train')):
            return False

        return True

    def preprocess(self):
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)
            os.mkdir(os.path.join(self.output_dir, 'train'))
            os.mkdir(os.path.join(self.output_dir, 'valid'))
            os.mkdir(os.path.join(self.output_dir, 'test'))

        for file in os.listdir(self.root_dir):
            file_path = os.path.join(self.root_dir, file)
            video_files = [name for name in os.listdir(file_path)]

            train_and_valid, test = train_test_split(video_files, test_size=0.2, random_state=42)
            train, val = train_test_split(train_and_valid, test_size=0.2, random_state=42)

            train_dir = os.path.join(self.output_dir, 'train', file)
            val_dir = os.path.join(self.output_dir, 'valid', file)
            test_dir = os.path.join(self.output_dir, 'test', file)

            if not os.path.exists(train_dir):
                os.mkdir(train_dir)
            if not os.path.exists(val_dir):
                os.mkdir(val_dir)
            if not os.path.exists(test_dir):
                os.mkdir(test_dir)

            for video in train:
                self.process_video(video, file, train_dir)
            for video in val:
                self.process_video(video, file, val_dir)
            for video in test:
                self.process_video(video, file, test_dir)

    def process_video(self, video, action_name, save_dir):
        video_filename = video.split('.')[0]
        if not os.path.exists(os.path.join(save_dir, video_filename)):
            os.mkdir(os.path.join(save_dir, video_filename))

        capture = cv2.VideoCapture(os.path.join(self.root_dir, action_name, video))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        skip = frame_count // self.clip_len
        if skip == 0:
            skip = 1
        count = 0
        i = 0
        success = True
        while (count < frame_count and success):
            success, frame = capture.read()
            if frame is None:
                continue
            if count % skip == 0:
                frame = cv2.resize(frame, (self.resize_width, self.resize_height))
                cv2.imwrite(os.path.join(save_dir, video_filename, '00{}.jpg'.format(str(i))), frame)
                i += 1
            count += 1
        capture.release()

    def load_frames(self, file_dir):
        #print(file_dir)
        with open('./sample_name.json', 'a', encoding='utf-8') as f:
            f.writelines(file_dir+'\n')
        frames = sorted([os.path.join(file_dir, img) for img in os.listdir(file_dir)])
        #frame_count = len(frames)
        buffer = np.empty((self.clip_len, self.resize_height, self.resize_width,3), np.dtype('float32'))
        for i, frame_name in enumerate(frames):
            frame = np.array(cv2.imread(frame_name)).astype(np.float64)
            buffer[i] = frame
            if i == (self.clip_len-1):
                break
        return buffer
    #返回类型timestep*H*W*C

    def randomflip(self, buffer):
        if np.random.random() < 0.5:
            for i, frame in enumerate(buffer):
                frame = cv2.flip(buffer[i], flipCode=1)
                buffer[i] = cv2.flip(frame, flipCode=1)
        return buffer

    def normalize(self, buffer):
        for i, frame in enumerate(buffer):
            frame -= np.array([[[90.0,98.0,102.0]]])
            buffer[i] = frame
        return buffer


    def to_tensor(self, buffer):
        return buffer.transpose((3,0,1,2))
        #返回C*timestep*H*W
        #imread是B*G*R

    def get_map(self):
        return self.label2index

#用来验证label和input的对应关系
def imshow(tensor, title=None):
    unloader = torchvision.transforms.ToPILImage()
    image = tensor.cpu().clone()
    #print(image.size())
    index = [2,0,1]#BGR->RGB
    image = image[index]
    #b,g,r = cv2.split(image)
    #image = cv2.merge([r.g.b])
    #image =image.float()
    image = unloader(image)
    plt.imshow(image)
    plt.pause(0.001)

def get_keys(dict, value):
    for k,v in dict.items():
        if v == value:
            return k

if __name__ == "__main__":
    from torch.utils.data import DataLoader
    train_data =VideoDataset(dataset='hmdb51', split='train', clip_len=30, preprocess=False)
    #print(len(train_data))
    train_loader = DataLoader(train_data, batch_size=10, shuffle=True, num_workers=4)

    my_map = train_data.get_map()
    for i, sample in enumerate(train_loader):
        inputs = sample[0]
        labels = sample[1]
        print(inputs.size())
        print(labels)

        if i == 0:
            break
    # for j in range(inputs.size(0)):
    #     for k in range(inputs.size(2)):
    #         img = inputs[j,:,k,:,:]
    #         #print(img.size())
    #         imshow(img)
    #         label = get_keys(my_map,labels[j].item())
    #         print("class:",label)

