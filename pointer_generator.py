import os
import csv 
import json
import torch
import random
import argparse
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
from networks import Autoencoder
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

def open_csv_file(file_path):
    ids_list = []
    articles_list = []
    highlights_list = []
    with open(file_path) as f:
        csv_reader = csv.reader(f, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count==0:
                col_names = row
            else:
                if len(row[1])>0 and len(row[2])>0:
                    articles_list.append(row[1])
                    highlights_list.append(row[2])
                    ids_list.append(row[0])
                else:
                    print('row skipped.')

            line_count += 1

    return ids_list, articles_list, highlights_list


def create_dictionary(train_path, valid_path, test_path, dictionary_dir):
    train_ids, train_arts, train_highs = open_csv_file(train_path)
    valid_ids, valid_arts, valid_highs = open_csv_file(valid_path)
    test_ids, test_arts, test_highs = open_csv_file(test_path)
    w2idx = {}
    idx2w = {}
    wcount = {}
    idx = 0 
    print('Process the training set...')
    for i in range(len(train_ids)):
        if train_arts[i] == ' ':
            continue
        else:
            art_tokens = train_arts[i].split(' ')
            high_tokens = train_highs[i].split(' ')
            for token in art_tokens:
                if token not in w2idx:
                    w2idx[token] = idx
                    idx2w[idx] = token
                    wcount[token] = 0
                    idx += 1
                else:
                    wcount[token] += 1

            for token in high_tokens:
                if token not in w2idx:
                    w2idx[token] = idx
                    idx2w[idx] = token
                    wcount[token] = 0
                    idx += 1
                else:
                    wcount[token] += 1
    print('Process the validation set...')
    for i in range(len(valid_ids)):
        if valid_arts[i] == ' ':
            continue
        else:
            art_tokens = valid_arts[i].split(' ')
            high_tokens = valid_highs[i].split(' ')
            for token in art_tokens:
                if token not in w2idx:
                    w2idx[token] = idx
                    idx2w[idx] = token
                    wcount[token] = 0
                    idx += 1
                else:
                    wcount[token] += 1
            for token in high_tokens:
                if token not in w2idx:
                    w2idx[token] = idx
                    idx2w[idx] = token
                    wcount[token] = 0
                    idx += 1
                else:
                    wcount[token] += 1
    print('Process the test set...')
    for i in range(len(test_ids)):
        if test_arts[i] == ' ':
            continue
        else:
            art_tokens = test_arts[i].split(' ')
            high_tokens = test_highs[i].split(' ')
            for token in art_tokens:
                if token not in w2idx:
                    w2idx[token] = idx
                    idx2w[idx] = token
                    wcount[token] = 0
                    idx += 1
                else:
                    wcount[token] += 1
            for token in high_tokens:
                if token not in w2idx:
                    w2idx[token] = idx
                    idx2w[idx] = token
                    wcount[token] = 0
                    idx += 1
                else:
                    wcount[token] += 1
    
    new_w2idx = {'<pad>': 0, '<unk>': 1}
    new_idx2w = {0: '<pad>', 1: '<unk>'}
    idx = 2
    for k, v in wcount.items():
        if v > 100:
            new_w2idx[k] = idx
            new_idx2w[idx] = k
            idx += 1

    #print(new_w2idx, len(new_w2idx))
    # print(new_w2idx['<pad>'], new_w2idx['<unk>'])

    with open(dictionary_dir+'w2idx.json', 'w') as fp:
        json.dump(new_w2idx, fp)
    with open(dictionary_dir+'idx2w.json', 'w') as fp:
        json.dump(new_idx2w, fp)

def collate_fn(data):
    ids, inputs, targets, input_lengths, target_lengths = zip(*data)
    inputs_list = []
    input_leng_list = []
    targets_list = []
    target_leng_list = []
    idx_list = []
    for idx in ids:
        idx_list.append(idx)
    for sentence in inputs:
        inputs_list.append(torch.tensor(sentence).unsqueeze(1).long())
    for sentence in targets:
        targets_list.append(torch.tensor(sentence).unsqueeze(1).long())
    for leng in input_lengths:
        input_leng_list.append(leng)
    for leng in target_lengths:
        target_leng_list.append(leng)

    return idx_list, pad_sequence(inputs_list), pad_sequence(targets_list), \
           torch.tensor(input_leng_list).long(), torch.tensor(target_leng_list).long()


class CNNdataset(Dataset):
    def __init__(self, datapath, dictionary_dir, sr=1, transform=None):
        ids, arts, highs = open_csv_file(datapath)
        self.new_ids = []
        self.new_arts = []
        self.new_highs = []
        self.art_lens = []
        self.high_lens = []
        self.sr = sr

        with open(dictionary_dir+'w2idx.json', 'r') as fp:
            self.w2idx = json.load(fp)
        with open(dictionary_dir+'idx2w.json', 'r') as fp:
            self.idx2w = json.load(fp)

        #print(len(self.w2idx), len(self.idx2w))
        #print(self.w2idx['</s>'], self.w2idx['<s>'], self.w2idx['</eos>'])

        for i in range(len(ids)):
            if arts[i] == ' ':
                continue
            else:
                self.new_ids.append(ids[i])
                self.new_arts.append(arts[i])
                self.new_highs.append(highs[i])

    def __len__(self):
        return len(self.new_ids)//self.sr
    
    def __getitem__(self, idx):
        # return self.input_[idx], self.target_[idx], self.input_len_[idx], self.target_len_[idx]
        art_tokens = self.new_arts[idx].split(' ')
        high_tokens = self.new_highs[idx].split(' ')
        inputs = []
        targets = []
        input_length = len(art_tokens)
        target_length = len(high_tokens)
        for token in art_tokens:
            if token not in self.w2idx:
                inputs.append(self.w2idx['<unk>'])
            else:
                inputs.append(self.w2idx[token])
        for token in high_tokens:
            if token not in self.w2idx:
                targets.append(self.w2idx['<unk>'])
            else:
                targets.append(self.w2idx[token])

        return self.new_ids[idx], inputs, targets, input_length, target_length




class PointerGenerator():
    def __init__(self, args):
        self.args = args
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        train_path = '/mnt/raptor/shihpo/cs6120/project/kaggle/cnn_dailymail/train_processed.csv'
        valid_path = '/mnt/raptor/shihpo/cs6120/project/kaggle/cnn_dailymail/validation_processed.csv'
        test_path = '/mnt/raptor/shihpo/cs6120/project/kaggle/cnn_dailymail/test_processed.csv'
        dictionary_dir = '/mnt/raptor/shihpo/cs6120/project/kaggle/cnn_dailymail/'
        if not os.path.isfile(dictionary_dir+'w2idx.json'):
            print('No dictionaries exist, start generating dictionaries...')
            create_dictionary(train_path, valid_path, test_path, dictionary_dir)
        else:
            print('Dictionaries exist, load from %s and %s'%(dictionary_dir+'w2idx.json', dictionary_dir+'idx2w.json'))
        if args.eval:
            test_dataset = CNNdataset(test_path, dictionary_dir)
            self.w2idx = test_dataset.w2idx
            self.idx2w = test_dataset.idx2w
            self.test_loader = DataLoader(test_dataset, collate_fn=collate_fn, batch_size=args.batch_size, shuffle=False, num_workers=1)
        else:
            train_dataset = CNNdataset(train_path, dictionary_dir, args.sr)
            valid_dataset = CNNdataset(valid_path, dictionary_dir)
            self.w2idx = train_dataset.w2idx
            self.idx2w = train_dataset.idx2w
            self.train_loader = DataLoader(train_dataset, collate_fn=collate_fn, batch_size=args.batch_size, shuffle=True, num_workers=1)
            self.valid_loader = DataLoader(valid_dataset, collate_fn=collate_fn, batch_size=args.batch_size, shuffle=False, num_workers=1)
        self.model = Autoencoder(args, EOS=self.w2idx['</s>'], dict_size=len(self.w2idx), device=self.device).to(self.device)
        # print(self.model)
        print('Dictionary size:', len(self.w2idx))
        if args.optimizer == 'sgd':
            self.optimizer = optim.SGD(self.model.parameters(), lr=args.lr)
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.95)
        elif args.optimizer == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr)
            self.scheduler = None
    
    def show_results(self, input, target, pred, out_list=None, visualized=True):
        idx = 0
        for text in pred:
            pred_text = '<s> ' + self.show_text(text)
            input_text = self.show_text(input[idx])
            target_text = self.show_text(target[idx])
            if visualized:
                print('===========================')
                print('input:\t%s'%(input_text))
                print('target:\t%s'%(target_text))
                print('pred:\t%s'%(pred_text))
            idx += 1
            if out_list is not None:
                out_list.append({
                    'input': input_text,
                    'target': target_text,
                    'pred': pred_text
                })
        if out_list is not None:
            return out_list

    def show_text(self, x):
        string = ''
        for i in x:
            # if i.item() == self.w2idx['<s>']: #or i.item() == self.w2idx['</s>']:
            #     continue
            # elif i.item() == 0:
            #     break
            if i.item() == self.w2idx['<pad>']:
                continue
            string += self.idx2w[str(i.item())]
            string += ' '
        return string
    
    def save_model(self, epoch):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            }, 'weights/'+self.args.checkname+str(epoch)+'.pth')
    
    def load_model(self):
        checkpoint = torch.load('weights/'+self.args.checkname+'.pth')
        self.model.load_state_dict(checkpoint['model_state_dict'])
    
    def criterion(self, out, target, loss_func):
        if self.args.network == 'pointergen' or self.args.network == 'pointergen_output':
            loss = - torch.sum(torch.log(out), dim=(0, 1)) / (out.size(0) * out.size(1))
        else:  
            target_ = torch.cat((target[:, 1:], target[:, target.size(1)-1:target.size(1)]), dim=1)
            target_ = target_.contiguous().view(-1)
            # target_ = target.contiguous().view(-1)
            out = out.contiguous().view(-1, len(self.w2idx))
            loss = loss_func(out,target_)
        return loss

    def train(self):
        cri = nn.CrossEntropyLoss(ignore_index=self.w2idx['<pad>'])
        for epoch in range(self.args.epoch):
            pbar = tqdm(self.train_loader)
            for data in pbar:
                
                self.model.zero_grad()
                idx, art, high, art_len, high_len = data
                art = art.squeeze(2).permute(1, 0).to(self.device)
                high = high.squeeze(2).permute(1, 0).to(self.device)

                out = self.model(art, high, art_len, high_len)
                loss = self.criterion(out, high, cri)
                loss.backward()
                self.optimizer.step()
                pbar.set_description('epoch '+str(epoch))
                pbar.set_postfix({'CE Loss': (loss.item()),}, refresh=True)

            if self.scheduler is not None:
                self.scheduler.step()
            #if epoch % 5 == 0:
            #self.show_results(input, target, out)
            self.save_model(epoch)
            # self.eval()

    def eval(self):
        print('Evaluating...')
        self.model.eval()
        for iter, data in enumerate(self.valid_loader):
            idx, art, high, art_len, high_len = data
            art = art.squeeze(2).permute(1, 0).to(self.device)
            high = high.squeeze(2).permute(1, 0).to(self.device)
            _, out = self.model(art, high, art_len, high_len)
            self.show_results(art, high, out)
            if iter > 2:
                break
        self.model.train()

    def inference(self, visualized=False):
        print('Loading the model weights...')
        self.load_model()
        print('Evaluating on the test set...')
        self.model.eval()
        out_list = []
        for iter, data in enumerate(self.test_loader):
            if iter <= 300:
                continue
            idx, art, high, art_len, high_len = data

            art = art.squeeze(2).permute(1, 0).to(self.device)
            high = high.squeeze(2).permute(1, 0).to(self.device)

            _, out = self.model(art, high, art_len, high_len)
            out_list = self.show_results(art.cpu(), high.cpu(), out.cpu(), out_list, visualized)
            print('[%d/%d]'%(iter, len(self.test_loader)))
            
            
            if iter > 450:
                break
        with open(self.args.checkname+'_split3.json', 'w') as fp:
            json.dump(out_list, fp)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Spelling-Corrector')
    parser.add_argument('--batch-size', default=8, type=int)
    parser.add_argument('--sr', default=10, type=int)
    parser.add_argument('--network', default='vanilla', help='vanilla/attention/pointergen')
    parser.add_argument('--eval', default=False, action='store_true')
    parser.add_argument('--epoch', default=50, type=int)
    parser.add_argument('--rnn', default='lstm', type=str, help='gru/lstm')
    parser.add_argument('--dim', default=256, type=int, help='hidden size of recurrent units')
    parser.add_argument('--bidir', default=True, action='store_true')
    parser.add_argument('--tf', default=1.0, type=float, help='teacher forcing ratio')
    parser.add_argument('--word-drop', default=0.4, type=float, help='dropout technique')
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--optimizer', default='adam', help='sgd/adam')
    parser.add_argument('--checkname', default='test', type=str, help='saving checkname')
    args = parser.parse_args()
    network = PointerGenerator(args)
    if args.eval:
        network.inference()
    else:
        network.train()
    