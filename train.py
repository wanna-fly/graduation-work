import timeit
from datetime import datetime
import socket
import os
import glob
import argparse
from tqdm import tqdm

import torch
from tensorboardX import SummaryWriter
from torch import nn, optim
from torch.utils.data import DataLoader

from data_reader import VideoDataset
from model import spatio_temp_model


#device configuration
#os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str,[3]))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device being used:", device)

resume_epoch = 0  # Default is 0, change if want to resume
snapshot = 50 # Store a model every snapshot epochs

#dataset = 'hmdb51' # Options: hmdb51 or ucf101

save_dir_root = os.path.join(os.path.dirname(os.path.abspath(__file__)))#获取当前文件所在目录的路径
exp_name = os.path.dirname(os.path.abspath(__file__)).split('/')[-1]
#获取当前文件的上一级目录文件夹名字，windows下路径间隔为\

if resume_epoch != 0:
    runs = sorted(glob.glob(os.path.join(save_dir_root, 'run', 'run_*')))
    run_id = int(runs[-1].split('_')[-1]) if runs else 0
else:
    runs = sorted(glob.glob(os.path.join(save_dir_root, 'run', 'run_*')))
    run_id = int(runs[-1].split('_')[-1]) + 1 if runs else 0
    #print(runs)
    #print(run_id)

save_dir = os.path.join(save_dir_root, 'run', 'run_' + str(run_id))
#modelName = 'LSTM' # Options: C3D or R2Plus1D or R3D
dataset = 'hmdb51'
saveName = dataset

'''
def to_onehot(label):
    label=label.view(-1,1)
    y = torch.LongTensor(len(label),num_classes).to(device)
	y.zero_()
	y.scatter_(1,label,1)
	return y
'''

def main(args, save_dir=save_dir,save_epoch=snapshot):
    """
        Args:
            num_classes (int): Number of classes in the data
            num_epochs (int, optional): Number of epochs to train for.
    """
    if args.dataset == 'hmdb51':
        num_classes=51
    elif args.dataset == 'ucf101':
        num_classes = 101
    else:
        print('We only implemented hmdb51 and ucf101 datasets.')
        raise NotImplementedError

    model = spatio_temp_model(2048, args.spatio_hidden_size, args.temp_hidden_size, num_classes)
    #(input_size, lstm_hidden_size, rnn_hidden_size, num_classes)
    #train_params = model.parameters()
    criterion = nn.CrossEntropyLoss()  # 使用交叉熵作为loss函数
    #optimizer = optim.SGD(train_params, lr=lr, momentum=0.9, weight_decay=1e-3)
    optimizer= optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    # the scheduler divides the lr by 10 every 10 epochs

    #if resume_epoch == 0:
    #print("Training {} from scratch...".format(modelName))
    '''
    else:
        checkpoint = torch.load(os.path.join(save_dir, 'models', saveName + '_epoch-' + str(resume_epoch - 1) + '.pth.tar'),
                       map_location=lambda storage, loc: storage)   # Load all tensors onto the CPU

        print("Initializing weights from: {}...".format(
            os.path.join(save_dir, 'models', saveName + '_epoch-' + str(resume_epoch - 1) + '.pth.tar')))
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['opt_dict'])
    '''
    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    #model = torch.nn.DataParallel(model,[0,1,3,4])
    model.to(device)
    criterion = criterion.to(device)

    log_dir = os.path.join(save_dir, 'models', datetime.now().strftime('%b%d_%H-%M-%S') + '_' + socket.gethostname())
    writer = SummaryWriter(log_dir=log_dir)

    print('Training model on {} dataset...'.format(args.dataset))
    train_dataloader = DataLoader(VideoDataset(dataset=args.dataset, split='train',clip_len=30), batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_dataloader   = DataLoader(VideoDataset(dataset=args.dataset, split='valid',  clip_len=30), batch_size=args.batch_size, num_workers=4)
    test_dataloader  = DataLoader(VideoDataset(dataset=args.dataset, split='test', clip_len=30), batch_size=args.batch_size, num_workers=4)

    trainval_loaders = {'train': train_dataloader, 'val': val_dataloader}
    trainval_sizes = {x: len(trainval_loaders[x].dataset) for x in ['train', 'val']}
    test_size = len(test_dataloader.dataset)

    for epoch in range(args.epochs):
        # each epoch has a training and validation step
        for phase in ['train', 'val']:
            start_time = timeit.default_timer()
            # reset the running loss and corrects
            running_loss = 0.0
            running_corrects = 0.0

            # set model to train() or eval() mode depending on whether it is trained
            # or being validated. Primarily affects layers such as BatchNorm or Dropout.
            if phase == 'train':
                # scheduler.step() is to be called once every epoch during training
                scheduler.step()
                model.train()
            else:
                model.eval()

            for inputs, labels in tqdm(trainval_loaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()

                if phase == 'train':
                    outputs = model(inputs)
                else:
                    with torch.no_grad():
                        outputs = model(inputs)

                #outputs=torch.sum(outputs,1)
                probs = nn.Softmax(dim=1)(outputs)
                preds = torch.max(probs, 1)[1]
                loss = criterion(outputs, labels)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / trainval_sizes[phase]
            epoch_acc = running_corrects.double() / trainval_sizes[phase]

            if phase == 'train':
                writer.add_scalar('data/train_loss_epoch', epoch_loss, epoch)
                writer.add_scalar('data/train_acc_epoch', epoch_acc, epoch)
            else:
                writer.add_scalar('data/val_loss_epoch', epoch_loss, epoch)
                writer.add_scalar('data/val_acc_epoch', epoch_acc, epoch)

            print("[{}] Epoch: {}/{} Loss: {} Acc: {}".format(phase, epoch+1, args.epochs, epoch_loss, epoch_acc))
            stop_time = timeit.default_timer()
            print("Execution time: " + str(stop_time - start_time) + "\n")

        if epoch % save_epoch == (save_epoch - 1):
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'opt_dict': optimizer.state_dict(),
            }, os.path.join(save_dir, 'models', saveName + '_epoch-' + str(epoch) + '.pth.tar'))
            print("Save model at {}\n".format(os.path.join(save_dir, 'models', saveName + '_epoch-' + str(epoch) + '.pth.tar')))

        if epoch % args.interval == (args.interval - 1):
            model.eval()
            start_time = timeit.default_timer()

            running_loss = 0.0
            running_corrects = 0.0

            for inputs, labels in tqdm(test_dataloader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                with torch.no_grad():
                    outputs = model(inputs)
                #outputs=torch.sum(outputs,1)
                #outputs=torch.div(outputs,30)
                probs = nn.Softmax(dim=1)(outputs)
                preds = torch.max(probs, 1)[1]
                loss = criterion(outputs, labels)

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / test_size
            epoch_acc = running_corrects.double() / test_size

            writer.add_scalar('data/test_loss_epoch', epoch_loss, epoch)
            writer.add_scalar('data/test_acc_epoch', epoch_acc, epoch)

            print("[test] Epoch: {}/{} Loss: {} Acc: {}".format(epoch+1, args.epochs, epoch_loss, epoch_acc))
            stop_time = timeit.default_timer()
            print("Execution time: " + str(stop_time - start_time) + "\n")

    writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='run the train script')
    parser.add_argument('-l', '--learning_rate', type=float, default=0.00001)
    parser.add_argument('-b', '--batch_size', type=int, default=128)
    parser.add_argument('-n', '--num_workers', type=int, default=4)
    parser.add_argument('-d', '--dataset', type=str, default='hmdb51', help='name of dataset to be used: ucf101 or hmdb51')
    parser.add_argument('-e', '--epochs', type=int, default=40, help='numbers of epochs for training')
    parser.add_argument('-s', '--spatio_hidden_size', type=int, default=1024, help='the hidden size of LSTM')
    parser.add_argument('-t', '--temp_hidden_size', type=int, default=256, help='the hidden size of BiRNN')
    parser.add_argument('-i', '--interval', type=int, default=10, help='the epoch interval to test the model')
    parser.add_argument('')
    args = parser.parse_args()
    print(args)
    main(args)
    torch.cuda.empty_cache()
