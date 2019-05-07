import torch
import torch.nn as nn
import numpy as np
import argparse
from torch.utils.data import DataLoader
from PIL import Image
import os
import matplotlib.pyplot as plt

from data_reader import VideoDataset
from model import spatio_temp_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(args):
    model = spatio_temp_model(2048, 1024, 256, 51).eval()
    model.to(device)
    #model.load_state_dict(torch.load(args.model_path, map_location={'cuda:4':'cuda:0'}))

    checkpoint = torch.load(args.model_path, map_location={'cuda:4':'cuda:0'})
    # Load all tensors onto the CPU
    model.load_state_dict(checkpoint['state_dict'])
    #optimizer.load_state_dict(checkpoint['opt_dict'])

    sample_data = VideoDataset(dataset='hmdb51', split='test', clip_len=30, preprocess=False)
    sample_loader = DataLoader(sample_data, batch_size=args.batch_size, shuffle=True, num_workers=2)
    for i, (inputs, labels) in enumerate(sample_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            outputs, att= model(inputs)
        att = torch.Tensor.tolist(att)
        index = np.arange(16)
        plt.bar(left=index, height=att)
        plt.show()
        outputs=torch.sum(outputs,1)
        probs = nn.Softmax(dim=1)(outputs)
        print(probs.size())
        preds = torch.max(probs, 1)[1]
        for j in range(len(labels)):
            print('the true label is ', labels[j])
            print('the pred is ', preds[j])
        if i == 1:
            break
    with open('sample_name.json', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    for line in lines:
        for image in os.listdir(line.split('\n')[0]):
            #print(line.split('\n')[0])
            #print(image)
            img = Image.open(os.path.join(line.split('\n')[0], image))
            img.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='test the trained model')
    parser.add_argument('--model_path', type=str, default='model/LSTM-hmdb51_epoch-9.pth.tar', help='path of model to be loaded')
    parser.add_argument('--batch_size', type=int, default=1)
    args = parser.parse_args()
    main(args)