import random
from typing import Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import librosa as lb
import math

class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on:
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, x0, x1, y):
        # euclidian distance
        diff = x0 - x1
        dist_sq = torch.sum(torch.pow(diff, 2), 1)
        dist = torch.sqrt(dist_sq)

        mdist = self.margin - dist
        dist = torch.clamp(mdist, min=0.0)
        loss = y * dist_sq + (1 - y) * torch.pow(dist, 2)
        loss = torch.sum(loss) / 2.0 / x0.size()[0]
        return loss


class MultiBranchConv1D_Batra(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes):
        super(MultiBranchConv1D_Batra, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_sizes[0],stride=2)  
        self.conv2 = nn.Conv1d(in_channels, out_channels, kernel_sizes[1],stride=2)
        self.conv3 = nn.Conv1d(in_channels, out_channels, kernel_sizes[2],stride=2)
        self.dropout = nn.Dropout(0.25)
        
    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.dropout(x1) 

        x2 = self.conv2(x)
        x2 = self.dropout(x2) 
        
        x3 = self.conv3(x)
        x3 = self.dropout(x3) 
        if x1.shape[2] > x2.shape[2]:
            padding_left = (x1.shape[2] - x2.shape[2]) // 2
            padding_right = x1.shape[2] - x2.shape[2] - padding_left
            x2 = nn.functional.pad(x2, (padding_left, padding_right))
        elif x1.shape[2] < x2.shape[2]:
            padding_left = (x2.shape[2] - x1.shape[2]) // 2
            padding_right = x2.shape[2] - x1.shape[2] - padding_left
            x1 = nn.functional.pad(x1, (padding_left, padding_right))
            
        if x2.shape[2] > x3.shape[2]:
            padding_left = (x2.shape[2] - x3.shape[2]) // 2
            padding_right = x2.shape[2] - x3.shape[2] - padding_left
            x3 = nn.functional.pad(x3, (padding_left, padding_right))
        elif x2.shape[2] < x3.shape[2]:
            padding_left = (x3.shape[2] - x2.shape[2]) // 2
            padding_right = x3.shape[2] - x2.shape[2] - padding_left
            x2 = nn.functional.pad(x2, (padding_left, padding_right))
        x = torch.cat((x1, x2, x3), dim=1)

        return x


class Batra_N1_Transformer(nn.Module):
    def __init__(self):
        super(Batra_N1_Transformer, self).__init__()
        self.winSize = int(np.ceil(30e-3*16000)) # in samples
        self.hopLength = int(np.ceil(15e-3*16000))
        self.branch_a = MultiBranchConv1D_Batra(in_channels=1, out_channels=4, kernel_sizes=[7, 11, 17])
        self.branch_b = MultiBranchConv1D_Batra(in_channels=12, out_channels=4, kernel_sizes=[5, 7, 9])
        self.max_pool = nn.MaxPool1d(kernel_size=2)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=12, nhead=2, dim_feedforward = 512, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=2)

        ## for integration with AAISIST
        self.dense_after_concat = nn.Linear(1788,512)
        self.device = torch.device("cuda:5")

        self.out1 = nn.Linear(672,256)
        self.out2 = nn.Linear(256,2)

        self.tanh = nn.Tanh()

    def forward(self, x,x_test_emd):
        x_cpu = x.cpu()
        frames = lb.util.frame(x_cpu, frame_length=self.winSize, hop_length=self.hopLength)
        embeddings = []

        for i in range(0, frames.shape[3] - 4):  # Sliding window approach
            frames_batch = frames[:, :, :, i:i+5]  # Select 5 frames
            frames_batch = torch.tensor(frames_batch, dtype=torch.float32).to(self.device)
            frames_concatenated = torch.cat([frames_batch[:, :, :, j] for j in range(5)], dim=2)  # Concatenate along frame dimension
            x_a = self.branch_a(frames_concatenated)
            x_a = F.relu(x_a)
            x_b = self.branch_b(x_a)
            x_b = F.relu(x_b)
            x = self.max_pool(x_b)
            x = F.relu(x)
            x = self.max_pool(x)
            x = F.relu(x)
            x = x.permute(0, 2, 1)
            x = self.encoder_layer(x)
            embeddings.append(x)
       
        embeddings = torch.stack(embeddings)
        averaged_embedding = torch.mean(embeddings, dim=0)
        concatenated_output = torch.cat([averaged_embedding[:, :, i] for i in range(averaged_embedding.size(2))], dim=1)
        dense_after_concat = self.dense_after_concat(concatenated_output) # project all emb of LPC to 512
        batra_assist_concat = torch.cat((dense_after_concat, x_test_emd), dim=1)
        hidden1 = self.out1(batra_assist_concat)
        hidden1 = self.tanh(hidden1)
        hidden2 = self.out2(hidden1)
        return hidden2

class Batra_Siamese_N2(nn.Module):
    def __init__(self):
        super(Batra_Siamese_N2, self).__init__()
        self.winSize = int(np.ceil(30e-3*16000)) # in samples
        self.hopLength = int(np.ceil(15e-3*16000))
        self.branch_a = MultiBranchConv1D_Batra(in_channels=1, out_channels=4, kernel_sizes=[7, 11, 17])
        self.branch_b = MultiBranchConv1D_Batra(in_channels=12, out_channels=4, kernel_sizes=[5, 7, 9])
        self.max_pool = nn.MaxPool1d(kernel_size=2)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=12, nhead=2, dim_feedforward = 512, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=2)

        ## for integration with AAISIST
        self.dense_after_concat = nn.Linear(1788,512)
        self.device = torch.device("cuda:3")

        self.concat_batra = nn.Linear(1024,512)
        self.out1 = nn.Linear(672,256)
        self.out2 = nn.Linear(256,2)

    def forward_once(self, x):
        x_cpu = x.cpu()
        frames = lb.util.frame(x_cpu, frame_length=self.winSize, hop_length=self.hopLength)
        embeddings = []

        for i in range(0, frames.shape[3] - 4):  # Sliding window approach
            frames_batch = frames[:, :, :, i:i+5]  # Select 5 frames
            frames_batch = torch.tensor(frames_batch, dtype=torch.float32).to(self.device)
            frames_concatenated = torch.cat([frames_batch[:, :, :, j] for j in range(5)], dim=2)  # Concatenate along frame dimension
            x_a = self.branch_a(frames_concatenated)
            x_a = F.relu(x_a)
            x_b = self.branch_b(x_a)
            x_b = F.relu(x_b)
            x = self.max_pool(x_b)
            x = F.relu(x)
            x = self.max_pool(x)
            x = F.relu(x)
            x = x.permute(0, 2, 1)
            x = self.encoder_layer(x)
            embeddings.append(x)
       
        embeddings = torch.stack(embeddings)
        averaged_embedding = torch.mean(embeddings, dim=0)
        concatenated_output = torch.cat([averaged_embedding[:, :, i] for i in range(averaged_embedding.size(2))], dim=1)
        dense_after_concat = self.dense_after_concat(concatenated_output) # project all emb of LPC to 512
        return dense_after_concat
        
    def forward(self, xtest_wav, xtest_res, xe_res):
        output_LPC_xt = self.forward_once(xtest_res)
        output_LPC_xe = self.forward_once(xe_res)
        concatenated_batra = torch.cat((output_LPC_xt, output_LPC_xe), dim=1)
        proj_Batra = self.concat_batra(concatenated_batra) # proj both LPC from 1024 to 512
        batra_assist_concat = torch.cat((xtest_wav, proj_Batra), dim=1)
        hidden1 = self.out1(batra_assist_concat) # 672 to 256
        hidden1 = F.relu(hidden1)
        hidden2 = self.out2(hidden1) # 256 to 2
        return hidden2, output_LPC_xt, output_LPC_xe
        