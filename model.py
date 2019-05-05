import torch
import torch.nn as nn
import torchvision

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")

class lstm_cell(nn.Module):
    def __init__(self, input_num, hidden_num):
        super(lstm_cell, self).__init__()

        self.input_num = input_num
        self.hidden_num = hidden_num

        self.Wxi = nn.Linear(self.input_num, self.hidden_num, bias=True)
        self.Whi = nn.Linear(self.hidden_num, self.hidden_num, bias=False)
        self.Wxf = nn.Linear(self.input_num, self.hidden_num, bias=True)
        self.Whf = nn.Linear(self.hidden_num, self.hidden_num, bias=False)
        self.Wxc = nn.Linear(self.input_num, self.hidden_num, bias=True)
        self.Whc = nn.Linear(self.hidden_num, self.hidden_num, bias=False)
        self.Wxo = nn.Linear(self.input_num, self.hidden_num, bias=True)
        self.Who = nn.Linear(self.hidden_num, self.hidden_num, bias=False)

    def forward(self, xt, ht_1, ct_1): #xt即为输入数据
        xt = xt.to(device)
        ht_1 = ht_1.to(device)
        ct_1 = ct_1.to(device)
        it = torch.sigmoid(self.Wxi(xt) + self.Whi(ht_1))
        ft = torch.sigmoid(self.Wxf(xt) + self.Whf(ht_1))
        ot = torch.sigmoid(self.Wxo(xt) + self.Who(ht_1))
        ct = ft * ct_1 + it * torch.tanh(self.Wxc(xt) + self.Whc(ht_1))
        ht = ot * torch.tanh(ct)
        return  ht, ct


class spatio_att_net(nn.Module):

    def __init__(self, input_num, hidden_num, num_layers,out_num ):
        super(spatio_att_net, self).__init__()
        # Make sure that `hidden_num` are lists having len == num_layers
        hidden_num = self._extend_for_multilayer(hidden_num, num_layers)

        if not len(hidden_num) == num_layers:
            raise ValueError('The length of hidden_num is not consistent with num_layers.')

        self.input_num = input_num
        self.hidden_num = hidden_num
        self.num_layers = num_layers
        self.out_num = out_num

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_num = self.input_num if i == 0 else self.hidden_num[i - 1]
            cell_list.append(lstm_cell(cur_input_num,self.hidden_num[i]))

        self.cell_list = nn.ModuleList(cell_list)
        self.conv=nn.Sequential(*list(torchvision.models.resnet101(pretrained=True).children())[:-2])
        for param in self.conv.parameters():
            param.requires_grad = False
        #self.conv=nn.Sequential(*list(torchvision.models.resnet101(pretrained=True).children())[:-2])
        self.Wha=nn.Linear(self.hidden_num[-1],49)
        self.fc=nn.Linear(self.hidden_num[-1],self.out_num)
        self.softmax=nn.Softmax(dim=1)

    def forward(self, x, hidden_state=None):
        #input model: batchsize x channel x seq_len x height x width
        #input size: 30 x 224 x 224 for googLeNet
        # init -1 time hidden units
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            hidden_state = self._init_hidden(batch_size=x.size(0))

        out_list=[]
        seq_len = x.size(2) #seq输入的长度，即帧数
        feature_cube = []
        for t in range(seq_len):
            output_t = []
            for layer_idx in range(self.num_layers):
                if 0==t:
                    ht_1, ct_1 = hidden_state[layer_idx][0],hidden_state[layer_idx][1]
                    attention_h=hidden_state[-1][0].to(device)
                else:
                    ht_1, ct_1 = hct_1[layer_idx][0],hct_1[layer_idx][1]
                if 0==layer_idx:
                    feature_map=self.conv(x[:,:,t,:,:].to(device))
                    feature_map=feature_map.view(feature_map.size(0),feature_map.size(1),-1)
                    attention_map=self.Wha(attention_h)
                    attention_map=torch.unsqueeze(self.softmax(attention_map),1)
                    #利用spatial attention对提取到的特征进行加权求和
                    attention_feature=attention_map*feature_map #batchsize*2048*49
                    attention_feature=torch.sum(attention_feature,2) #batchsize*2048
                    ht, ct = self.cell_list[layer_idx](attention_feature,ht_1, ct_1)
                    output_t.append([ht,ct])
                else:
                    ht, ct = self.cell_list[layer_idx](output_t[layer_idx-1][0], ht_1, ct_1)
                    output_t.append([ht,ct]) #把ht和ct作为输出存储
            attention_h=output_t[-1][0] #取最后一层的ht作为attention_h
            hct_1=output_t
            feature_cube.append(attention_feature)
            out_list.append(self.fc(output_t[-1][0])) #将最后一层的ht通过全连接层得到输出yt
        #seq_len*batchsize*2048 and
        return torch.stack(feature_cube,0), torch.stack(out_list,1)#在第二个维度上连接out_list中的tensor，把每个time_setp中获得的输出yt堆叠起来作为总体输出

    def _init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append([torch.zeros(batch_size, self.hidden_num[i]),torch.zeros(batch_size, self.hidden_num[i])])
        return init_states

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param


class saliency_evaluater(nn.Module):
    def __init__(self, hidden_size, bidirectional=True):
        super(saliency_evaluater,self).__init__()
        #weight_size: 用于生成attention的全连接层的input_size
        self.weight_size = hidden_size*2 if bidirectional else hidden_size
        self.fc = nn.Linear(self.weight_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attention = self.sigmoid(self.fc(x))
        return attention.squeeze(2)


class temp_att_net(nn.Module):
    def __init__(self,input_size, hidden_size, num_layers, bidirectional=True, dropout=0.0):
        super(temp_att_net,self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.directions = 2 if bidirectional else 1
        self.net = nn.RNN(input_size, hidden_size, num_layers, bidirectional=bidirectional,dropout=dropout)
        #self.dropout = nn.Dropout(dropout)
        self.saliency_evaluater = saliency_evaluater(self.hidden_size, bidirectional)

    def forward(self, x):
        # x:  seq_len*batch_size*input_size
        #初始化h0,其shape为(num_layers*directions)*batch_size*hidden_size
        self.h_0 = torch.zeros(self.num_layers*self.directions, x.size(1), self.hidden_size).to(device)
        #输入到RNN网络中，这里y包括了所有time_step的输出，h_t则只是最后time-step的隐层状态，双向情况下则是两端的隐层状态
        y, h_t = self.net(x, self.h_0)
        attention = self.saliency_evaluater(y)
        return attention


class spatio_temp_model(nn.Module):
    def __init__(self, input_size, lstm_hidden_size, rnn_hidden_size, out_num, num_lstm_layers=1, num_rnn_layers=1):
        super(spatio_temp_model,self).__init__()
        self.spatio_att_net = spatio_att_net(input_size, lstm_hidden_size, num_lstm_layers, out_num)
        self.temp_att_net = temp_att_net(input_size, rnn_hidden_size, num_rnn_layers)

    def forward(self, x):
        # shape: batchsize*channel*seq_len*height*width
        feature, out = self.spatio_att_net(x)
        temp_att = self.temp_att_net(feature)  #seq_len*batchsize
        normalized_temp_att = nn.Softmax(dim=1)(torch.transpose(temp_att, 0, 1)) #normalization
        weighted_pred = torch.mul(normalized_temp_att.unsqueeze(2), out)

        return weighted_pred, temp_att


if __name__ == "__main__":
    #若效果不好尝试attention_weight
    #改变初始化策略
    #多层+dropout
    #数据增强

    #batchsize*channel*seq_len*height*width
    """
    inputs = torch.randn(2, 3, 4, 224, 224)
    net = ALSTM(2048, 256, 1, 51)
    net = net.to(device)
    feature, out = net.forward(inputs)
    print("size of feature",feature.size())
    print("size of out",out.size()) # batchsize*seq_len*class_num

    att_net = temp_att_net(2048,1024,1).to(device)
    attention = att_net(feature) #seq_len*batchsize
    attention = nn.Softmax(dim=1)(torch.transpose(attention, 0, 1)) #nomalize
    weighted_pred = torch.mul(attention.unsqueeze(2), out) #以时间注意力作为权重
    """
    inputs = torch.randn(2, 3, 4, 224, 224).to(device)
    model = spatio_temp_model(2048,256,51).to(device)
    output = model(inputs)
    print(output) #batchsize*seq_len*class_num
