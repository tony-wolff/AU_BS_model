# %%
from __future__ import unicode_literals, print_function, division
import time
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings("ignore")
import time
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import numpy as np
import sys
from torch.utils.data import DataLoader, RandomSampler
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
data_path = "data/"

#From StackOverflow : https://stackoverflow.com/questions/71998978/early-stopping-in-pytorch
class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss, model, criterion, voice_type):
        if validation_loss < self.min_validation_loss:
            torch.save(model.state_dict(), f"pytorch models/{voice_type}/{model.__class__.__name__}_{criterion.__class__.__name__}_{voice_type}.pth")
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


class GRUNet(nn.Module):
    def __init__(self, input_size=17, hidden_size=128, output_size=61, dropout_p=0.1):
        super(GRUNet, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout = nn.Dropout(dropout_p)
        self.gru = nn.GRU(input_size, self.hidden_size)
        self.linear = nn.Linear(hidden_size, output_size, device=device)
        self.relu = nn.ReLU()

    def forward(self, input_tensor):
        output, _ = self.gru(input_tensor)
        output = self.linear(output)
        output = self.dropout(output)
        output = self.relu(output)
        return output

    def train_epoch(self, dataloader, optimizer, criterion):
        total_loss = 0
        self.train()
        for batch in dataloader:
            input_tensor, target_tensor, l = batch
            optimizer.zero_grad()
            outputs = self.forward(input_tensor)
            loss = self.loss_comp(outputs, target_tensor, criterion, l)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(dataloader)
    
    def loss_comp(self, out, target, criterion, l=None):
        return criterion(out.view(-1), target.view(-1))
    

    def evaluate_epoch(self, valid_dataloader, criterion):
        loss = 0
        self.eval()
        with torch.no_grad():
            for batch in valid_dataloader:
                inp, tgt, l = batch
                out = self.forward(inp)
                loss += self.loss_comp(out, tgt, criterion).item()
        return loss / len(valid_dataloader)
    
    def train_(self, train_dataloader, valid_dataloader, n_epochs, criterion, optimizer, voice_type):
        train_loss = []
        valid_loss = []
        last_epoch = 0
        best_loss = np.inf
        early_stopper = EarlyStopper(patience=10)
        for epoch in range(1, n_epochs + 1):
            tloss = self.train_epoch(train_dataloader, optimizer, criterion)
            vloss = self.evaluate_epoch(valid_dataloader, criterion)
            if vloss < best_loss:
                best_loss = vloss
            train_loss.append(tloss)
            valid_loss.append(vloss)
            last_epoch = epoch
            if early_stopper.early_stop(vloss, self, criterion, voice_type=voice_type):
                break
            print(f"epoch: {epoch}\n   train loss: {tloss}\t valid loss: {vloss}")
        print(f"best vloss: {best_loss}")
        plt.plot(train_loss, label="train loss")    
        plt.plot(valid_loss, label="valid loss")
        plt.legend(loc="upper right")
        plt.xlabel(f"epochs: ({last_epoch})")
        plt.ylabel("loss amplitude")
        plt.title(f"train {criterion.__class__.__name__} | {self.__class__.__name__}")

class GRUNetNeg(GRUNet):
    def __init__(self, input_size=17, hidden_size=128, output_size=61, dropout_p=0.1):
        super().__init__(input_size, hidden_size, output_size, dropout_p)

    def forward(self, input_tensor):
        output, _ = self.gru(input_tensor)
        output = self.linear(output)
        output = self.dropout(output)
        return output

    def loss_comp(self, out, target, criterion, l=None):
            return criterion(out.view(-1), target.view(-1)) + torch.mean(NegRELU(out.view(-1)))

class GRUNetPack(GRUNet):
    def __init__(self, input_size=17, hidden_size=128, output_size=61, dropout_p=0.1):
        super().__init__(input_size, hidden_size, output_size, dropout_p)

    def forward(self, input_tensor):
        gru_output, _ = self.gru(input_tensor)
        pad_output, _ = torch.nn.utils.rnn.pad_packed_sequence(gru_output)
        linear_out = self.linear(pad_output)
        output = self.dropout(linear_out)
        output = self.relu(output)
        return output
    

    def loss_comp(self, output, target, criterion, l):
        loss = 0
        pred_list = torch.nn.utils.rnn.unpad_sequence(output.clone(), l)
        target_list = torch.nn.utils.rnn.unpad_sequence(target, l)
        for pred, tgt in zip(pred_list, target_list):
            loss += criterion(pred.view(-1), tgt.view(-1))
        return loss

    def train_epoch(self, dataloader, optimizer, criterion):
        total_loss = 0
        self.train()
        for batch in dataloader:
            input_tensor, target_tensor, sequence_lengths = batch
            loss = 0
            input_gru = torch.nn.utils.rnn.pack_padded_sequence(input=input_tensor, lengths=sequence_lengths)
            optimizer.zero_grad()
            outputs = self.forward(input_gru)
            loss = self.loss_comp(outputs, target_tensor, criterion, sequence_lengths)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(dataloader)

    def evaluate_epoch(self, valid_dataloader, criterion):
        loss = 0
        self.eval()
        with torch.no_grad():
            for batch in valid_dataloader:
                inp, tgt, l = batch
                
                input_gru = torch.nn.utils.rnn.pack_padded_sequence(input=inp, lengths=l)
                out = self.forward(input_gru)
                loss += self.loss_comp(out, tgt, criterion, l).item()
        return loss / len(valid_dataloader)

class GRUNetSeq(GRUNetPack):
    def __init__(self, input_size=17, hidden_size=128, output_size=61, dropout_p=0.1):
        super().__init__(input_size, hidden_size, output_size, dropout_p=dropout_p)
        self.encoder = nn.GRU(input_size, hidden_size)
        self.decoder = nn.GRU(input_size+hidden_size, hidden_size) #concat input and context vector
        self.linear = nn.Linear(input_size + hidden_size*2, output_size) #concat input, hidden and context vectors
        self.tgr_len = 0
        self.batch_size = 0

    def _init(self, input_pack_tensor):
        #input = [sequence_length, batch_size, input_size]
        #target = [sequence_length, batch_size, output_size]
        input_pad, _ =  torch.nn.utils.rnn.pad_packed_sequence(input_pack_tensor)
        self.batch_size = input_pad.shape[1] 
        self.trg_len = input_pad.shape[0]
        enc_outputs, enc_hid = self.encoder(input_pack_tensor)
        return input_pad, enc_outputs, enc_hid

    def forward(self, input_pack_tensor):
        input_pad, enc_outputs, enc_hid = self._init(input_pack_tensor)
        hidden = torch.zeros(1, self.batch_size, self.hidden_size).to(device) #hidden = [1, batch_size, hidden_size]
        outputs = torch.zeros(self.trg_len, self.batch_size, self.output_size).to(device)

        for t in range(0, self.trg_len):
            inp = input_pad[t] #inp [4, input_size]
            inp = inp.unsqueeze(0) #inp [1, 4, input_size], gru accepts 3D batched tensors, this adds a new dimension at position 0
            input_context = torch.cat((inp, enc_hid), dim = 2) #input_contex = [1, 4, input_size+hidden_size]
            output, hidden = self.forward_step(input_context, hidden, enc_hid, inp)
            #output = [batch_size, output_size]
            outputs[t] = output
        return outputs # outputs = [batch_size, sequence lengths, output size]

    def forward_step(self, input_context, hidden, enc_hid, inp):
        output, hidden = self.decoder(input_context, hidden)
        #hidden = [1, 4, hidden_size]
        fc_input = torch.cat((inp.squeeze(0), hidden.squeeze(0), enc_hid.squeeze(0)), dim=1)
        output = self.linear(fc_input)
        return output, hidden

class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)

    def forward(self, hidden, enc_output):
        hidden = hidden.unsqueeze(1).repeat(1, enc_output.shape[0], 1)
        # hidden = [seq_len, batch_size, hidden]
        enc_output = enc_output.permute(1, 0, 2)
        #encoder output = [seq_len, batch_size, hidden]
        energy = torch.tanh(self.Wa(hidden) + self.Ua(enc_output))  # Calculate energy for each sequence element
        scores = self.Va(energy).squeeze(2)  # Linear transformation and squeeze
        
        weights = F.softmax(scores, dim=1)  # Apply softmax to get attention weights
        context = torch.bmm(weights.unsqueeze(1), enc_output).squeeze(1)  # Calculate context vector

        return context, weights

class GRUNetAtt(GRUNetSeq):
    def __init__(self, input_size, hidden_size, output_size, dropout_p=0.1): #17
        super().__init__(input_size, hidden_size, output_size, dropout_p)
        self.attention = BahdanauAttention(hidden_size)
        self.decoder = nn.GRU(input_size+hidden_size, hidden_size)
        self.linear = nn.Linear(input_size+hidden_size*2, output_size)

    def forward(self, input_pack_tensor):
        input_pad, enc_outputs, enc_hid = self._init(input_pack_tensor)
        hidden = enc_hid #hidden = encoder hidden
        outputs = torch.zeros(self.trg_len, self.batch_size, self.output_size).to(device)
        enc_outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(enc_outputs, batch_first=False) #enc_outputs = [seq_len, batch_size, hidden_size]

        for t in range(0, self.trg_len):
            inp = input_pad[t] #inp [4, input_size]
            inp = inp.unsqueeze(0) #inp [1, 4, input_size]
            output, hidden = self.forward_step(enc_outputs, hidden, inp)
            #output = [batch_size, output_size]
            outputs[t] = output
        return outputs # outputs = [batch_size, sequence lengths, output size]

    def forward_step(self, enc_outputs, hidden, inp):
        context, _ = self.attention(hidden.squeeze(0), enc_outputs)
        context = context.unsqueeze(0) #context = [1, batch_size, hid]
        input_gru = torch.cat((inp, context), dim=2) #input gru = [1, batch_size, input_size + hidden_size]
        output, hidden = self.decoder(input_gru, hidden) #output = hidden = [1, batch_size, hidden_size]
        fc_input = torch.cat((output.squeeze(0), context.squeeze(0), inp.squeeze(0)), dim=1) #fc_input = [1, batch_size, input_size+hidden*3]
        output = self.linear(fc_input) #output = [1, batch_size, output_size]

        return output, hidden

class GRUNetSig(GRUNet):
    def __init__(self, input_size, hidden_size, output_size, dropout_p=0.1):
        super().__init__(input_size, hidden_size, output_size, dropout_p)
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(dropout_p)
        self.gru = nn.GRU(input_size, self.hidden_size)
        self.linear = nn.Linear(hidden_size, output_size, device=device)
        self.linear2 = nn.Linear(input_size, output_size, device=device)
        self.sig = nn.Sigmoid()
        self.bce = nn.BCELoss()

    def forward(self, input_packed):
        gru_output, _ = self.gru(input_packed)
        pad_output, _ = torch.nn.utils.rnn.pad_packed_sequence(gru_output)
        input2, _ = torch.nn.utils.rnn.pad_packed_sequence(input_packed)
        linear_out = self.linear(pad_output)
        activation_probability = self.linear2(input2)
        activation_probability = self.sig(activation_probability)
        return linear_out, activation_probability

    def loss_comp(self, output, target, prob, criterion, l):
        loss = 0
        bce_loss = 0
        out_list = torch.nn.utils.rnn.unpad_sequence(output.clone(), l)
        target_list = torch.nn.utils.rnn.unpad_sequence(target.clone(), l)
        prob_list = torch.nn.utils.rnn.unpad_sequence(prob.clone(), l)
        for p, t, prob in zip(out_list, target_list, prob_list):
            tr = nn.Threshold(0.5, 0)
            prob_threshold = (tr(prob) != 0).float()
            new_p = prob_threshold*p
            loss += criterion(new_p.view(-1), t.view(-1))
            bce_loss += my_bce_loss(prob, t, self.bce)
        return (loss)+(bce_loss*0.1)

    def train_epoch(self, dataloader, optimizer, criterion):
        total_loss=0
        self.train()
        for batch in dataloader:
            optimizer.zero_grad()
            input_tensor, target_tensor, lengths = batch
            input_gru = torch.nn.utils.rnn.pack_padded_sequence(input=input_tensor, lengths=lengths)
            out, out_pred = self.forward(input_gru)
            loss = self.loss_comp(out, target_tensor, out_pred, criterion, lengths)
            loss.backward()
            #scheduler.step()
            optimizer.step()
            total_loss+=loss.item()
        return total_loss / len(dataloader)   

    def evaluate_epoch(self, valid_dataloader, criterion):
        loss = 0
        self.eval()
        with torch.no_grad():
            for batch in valid_dataloader:
                inp, tgt, l = batch
                input_gru = torch.nn.utils.rnn.pack_padded_sequence(input=inp, lengths=l)
                out, prob = self.forward(input_gru)
                loss += self.loss_comp(out, tgt, prob, criterion, l).item()
        return loss / len(valid_dataloader)   

class FCNet(GRUNet):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__(input_size, hidden_size, output_size)
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input_tensor):
        out = self.linear1(input_tensor)
        out = self.linear2(out)
        out = self.out(out)
        out = self.relu(out)
        return out
    
    def loss_comp(self, output, target, criterion, l):
        loss = 0
        pred_list = torch.nn.utils.rnn.unpad_sequence(output.clone(), l)
        target_list = torch.nn.utils.rnn.unpad_sequence(target, l)
        for pred, tgt in zip(pred_list, target_list):
            loss += criterion(pred.view(-1), tgt.view(-1))
        return loss
    
    def evaluate_epoch(self, valid_dataloader, criterion):
        loss = 0
        self.eval()
        with torch.no_grad():
            for batch in valid_dataloader:
                inp, tgt, l = batch
                out = self.forward(inp)
                loss += self.loss_comp(out, tgt, criterion, l).item()
        return loss / len(valid_dataloader)

def NegRELU(tensor): ##Custom loss function to penalize negative values
    relu = nn.ReLU()
    return relu(torch.neg(tensor))

class RMSELoss(nn.Module):
    def __init__(self, reduction="mean"):
        super(RMSELoss, self).__init__()
        self.reduction = reduction

    def forward(self, inp, target):
        if self.reduction == "mean":
            mse = torch.mean((inp - target) ** 2)
            rmse = torch.sqrt(mse + 1e-7)
            return rmse
        elif self.reduction == "none":
            mse = (inp - target) ** 2
            rmse = torch.sqrt(mse+1e-7)
        elif self.reduction == "sum":
            mse = torch.sum((inp - target) ** 2)
            rmse = torch.sqrt(mse + 1e-7)

def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.normal_(param.data, mean=0, std=0.01)

def my_bce_loss(x, y, bce): ###Custom BCE
    activated = (y != 0).float().clamp(min=1e-10)
    unactivated = (y == 0).float().clamp(min=1e-10)
    Tr = nn.Threshold(0.5, 1e-10)
    act_x = Tr(x)
    unact_x = (x - act_x).clamp(min=1e-10)
    term1 = (torch.log(act_x)*activated).clamp(1e-10)
    term2 =  (torch.log(unact_x)*unactivated).clamp(1e-10)
    column_losses = -1.0 * (torch.mean(term1, dim=1) + torch.mean(term2, dim=1)) # Calculate mean along the row dimension
    #loss = bce(x, activated)
    return column_losses.mean()

def evaluate(criterion, model, test_data, loss_by_bs=False, use_pack=True, with_prob=False):
        model.eval()
        plot_loss = []
        predictions = []
        activation_precision = []
        activation_recall = []
        loss_arr = []
        if loss_by_bs:
            criterion = criterion.__class__(reduction="none")
        with torch.no_grad():
            prob=None
            for inp, tgt, sequence_lengths in test_data:
                loss_by_batch = []
                if use_pack:
                    inp = torch.nn.utils.rnn.pack_padded_sequence(input=inp, lengths=sequence_lengths)
                if with_prob:
                    output, prob = model(inp)
                else:
                    output = model(inp)
                compute_loss_function(criterion, output, tgt, sequence_lengths, activation_precision, activation_recall, plot_loss, loss_by_batch, loss_arr, prob, loss_by_bs)

                #Predictions is a list of tuples, output has the shape L*Batch*blendshape counts, sequence lengths will be useful to cut script values
                predictions.append((inp, output, sequence_lengths))
        print(f'mean  {criterion.__class__.__name__}: {np.mean(plot_loss)}')
        print(f"mean activation precision: {np.mean(activation_precision)}")
        print(f"mean activation recall: {np.mean(activation_recall)}")
        plt.plot(plot_loss)
        plt.xlabel(f"scripts ({len(plot_loss)})")
        plt.ylabel("loss amplitude")
        plt.title(f"test {criterion.__class__.__name__} | {model.__class__.__name__}")
        return predictions, loss_arr


def loss_unpad(criterion, pred_list, target_list, plot_loss, loss_by_batch, activation_precision, activation_recall, loss_arr, loss_by_bs):
    for p, t in zip(pred_list, target_list):
        act_p, act_r = (evaluate_true_positive(p, t))
        activation_precision.append(act_p)
        activation_recall.append(act_r)
        if loss_by_bs:
            val_loss = torch.mean(criterion(p, t), dim=0)
            loss_arr.append(val_loss)
            val_loss = val_loss.mean()
        else: 
            val_loss = criterion(p.view(-1), t.view(-1))
        plot_loss.append(val_loss.item())
        loss_by_batch.append(val_loss.item())

def loss_with_prob(criterion, pred_list, target_list, prob_list, plot_loss, loss_by_batch, activation_precision, activation_recall, loss_arr, loss_by_bs):
    for p, t, prob in zip(pred_list, target_list, prob_list):
        tr = nn.Threshold(0.5, 0)
        prob_threshold = (tr(prob) > 1e-5).float()
        new_pred = prob_threshold*p
        act_p, act_r = evaluate_true_positive(new_pred, t)
        activation_precision.append(act_p)
        activation_recall.append(act_r)
        if loss_by_bs:
            val_loss = torch.mean(criterion(p, t), dim=0)
            loss_arr.append(val_loss)
            val_loss = val_loss.mean()
        else:
            val_loss = criterion(p.view(-1), t.view(-1))
        plot_loss.append(val_loss.item())
        loss_by_batch.append(val_loss.item())

def compute_loss_bs_wise(criterion, output, tgt, plot_loss, activation_precision, activation_recall):
    ###Permute dimension and flatten the tensors to have a shape [52, sequence_length*batch_size], allow to compute loss for each 52 blendshapes
    array = criterion(torch.flatten(output.permute(2, 0, 1), start_dim=1), torch.flatten(tgt.permute(2, 0, 1), start_dim=1))
    loss_arr = torch.mean(array, dim=1)
    loss = torch.sum(loss_arr)
    plot_loss.append(loss.item())
    act_p, act_r = (evaluate_true_positive(output, tgt))
    activation_precision.append(act_p)
    activation_recall.append(act_r)

def compute_loss_function(criterion, output, tgt, seq_length, activation_precision, activation_recall, plot_loss, loss_by_batch, loss_arr, prob, loss_by_bs):
    pred_list = torch.nn.utils.rnn.unpad_sequence(output.clone(), seq_length)
    target_list = torch.nn.utils.rnn.unpad_sequence(tgt, seq_length)
    if prob!=None:
        prob_list = torch.nn.utils.rnn.unpad_sequence(prob, seq_length)
        loss_with_prob(criterion, pred_list, target_list, prob_list, plot_loss, loss_by_batch, activation_precision, activation_recall, loss_arr, loss_by_bs)
    else:
        loss_unpad(criterion, pred_list, target_list, plot_loss, loss_by_batch, activation_precision, activation_recall, loss_arr, loss_by_bs)
    

def evaluate_true_positive(prediction, target):
    pred_activated = (prediction > 1e-10)
    activated = (prediction > 1e-10)
    true_positive = torch.count_nonzero(torch.logical_and(prediction, target))
    non_zero_count_predictions = torch.count_nonzero(pred_activated).item()
    non_zero_count_target = torch.count_nonzero(activated).item()
    activation_precision = true_positive.item()
    if non_zero_count_predictions != 0: 
        activation_precision/=non_zero_count_predictions
    activation_recall = true_positive.item()
    if non_zero_count_target != 0:
        activation_recall/=non_zero_count_target
    return activation_precision, activation_recall

def save_results(dataframe, inp_and_out, dic, folder, bs_only=True):
    for au, pred in inp_and_out:
        au_np = au.cpu().data.numpy()
        keys = [k for k, v in dic.items() if v.to_numpy().shape == au_np.shape and np.allclose(v.to_numpy(), au_np, atol=0.00001)]
        columns = dataframe.columns[-61:]
        if bs_only:
            columns = columns[:-9] #remove 9 last columns
        df = pd.DataFrame(columns=columns, data=pred.cpu().data.numpy())
        path_to_folder = "predictions/" + folder
        if not os.path.exists(path_to_folder):
            os.makedirs(path_to_folder)
            print(path_to_folder)
        df.to_csv(path_to_folder +"_pred_"+keys[0]+'.csv', index=False)

        