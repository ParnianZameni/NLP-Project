
import torch
import random
import torch.nn as nn
import numpy as np
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence


class Autoencoder(nn.Module):
    def __init__(self, args, EOS, dict_size, device):
        super(Autoencoder, self).__init__()
        self.args = args
        if args.network == 'vanilla':
            self.encoder = Encoder(args, dict_size)
            self.decoder = Decoder(args, EOS, dict_size, device)
        elif args.network == 'attention':
            self.encoder = AttnEncoder(args, dict_size)
            self.decoder = AttnDecoder(args, EOS, dict_size, device)
        elif args.network == 'attention_output':
            self.encoder = Encoder(args, dict_size)
            self.decoder = AttnOutputDecoder(args, EOS, dict_size, device)
        elif args.network == 'pointergen_output':
            self.encoder = Encoder(args, dict_size)
            self.decoder = PointerGenOutputDecoder(args, EOS, dict_size, device)
        elif args.network == 'pointergen':
            self.encoder = AttnEncoder(args, dict_size)
            self.decoder = PointerGenDecoder(args, EOS, dict_size, device)
    def forward(self, input, target, input_len, target_len):
        out, h = self.encoder(input, input_len)
        if self.training:
            if random.random() < self.args.tf:
                out = self.decoder(target, target_len, (out, h), input, input_len)
            else:
                out, out_seq = self.decoder.serial(target, (out, h), input, input_len)
            return out
        else:
            out, out_seq = self.decoder.serial(target, (out, h), input, input_len)
            return out, out_seq
            
        
class AttnEncoder(nn.Module):
    def __init__(self, args, dict_size):
        super(AttnEncoder, self).__init__()
        self.dim = args.dim
        self.bidir = args.bidir
        self.embedding = nn.Embedding(dict_size, self.dim, padding_idx=0)
        self.rnntype = args.rnn
        self.rnn = nn.LSTM(self.dim, self.dim)
    def forward(self, words, lengths):
        words_embedding = self.embedding(words)
        #batch_size = words_embedding.size(1)
        batch_size = words_embedding.size(0)
        b_h_all_list = []
        b_out_all_list = []
        for i in range(batch_size):
            #sentence = words_embedding[:, i, 0, :]
            sentence = words_embedding[i, :, :]
            hid = None
            for j in range(lengths[i]):
                if hid is not None:
                    out, h = self.rnn(sentence[j:j+1], (hid, cell))
                else:
                    out, h = self.rnn(sentence[j:j+1])
                hid, cell = h
                
                if j == 0:
                    h_all = hid
                else:
                    h_all = torch.cat((h_all, hid), dim=0)
            if self.bidir:
                hid = None
                for j in range(lengths[i]):
                    if hid is not None:
                        out, h = self.rnn(sentence[lengths[i]-j-1:lengths[i]-j], (hid, cell))
                    else:
                        out, h = self.rnn(sentence[lengths[i]-j-1:lengths[i]-j])
                    hid, cell = h
                    
                    if j == 0:
                        h_all_rev = hid
                    else:
                        h_all_rev = torch.cat((h_all_rev, hid), dim=0)
                b_h_all_list.append(torch.cat((h_all, h_all_rev), dim=1))
            else:
                b_h_all_list.append(h_all)
            
        
        b_h_all = pad_sequence(b_h_all_list)
        return out, b_h_all

class Encoder(nn.Module):
    def __init__(self, args, dict_size):
        super(Encoder, self).__init__()
        self.dim = args.dim
        self.bidir = args.bidir
        self.embedding = nn.Embedding(dict_size, self.dim, padding_idx=0)
        self.rnntype = args.rnn
        if args.rnn == 'gru':
            self.rnn = nn.GRU(self.dim, self.dim,num_layers=1,batch_first=True,bidirectional=args.bidir)
        elif args.rnn == 'lstm':
            self.rnn = nn.LSTM(self.dim, self.dim,num_layers=1,batch_first=True,bidirectional=args.bidir)
        else:
            raise Exception('RNN cell implementation error, should be gru/lstm')
    def forward(self, words, lengths): #only serial pass to get the attention weights
        
        words_embedding = self.embedding(words)
        #ordering
        sorted_len, indices = torch.sort(lengths, descending=True)
        _, reverse_sorting = torch.sort(indices)
        words_embedding = words_embedding[indices]
        lengths = lengths[indices]
        #lengths[lengths == 0] = 1
        packed_padded_sequence = pack_padded_sequence(words_embedding,lengths,batch_first=True)
        out, h = self.rnn(packed_padded_sequence)
        out, out_len = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        if self.rnntype == 'gru':
            h = h.permute(1, 0, 2).contiguous()
            h = h.view(-1, self.dim * (2 if self.bidir else 1))
            h = h[reverse_sorting]
        elif self.rnntype == 'lstm':
            hid, ceil = h
            hid = hid.permute(1, 0, 2).contiguous()
            hid = hid.view(-1, self.dim * (2 if self.bidir else 1))
            hid = hid[reverse_sorting]
            ceil = ceil.permute(1, 0, 2).contiguous()
            ceil = ceil.view(-1, self.dim * (2 if self.bidir else 1))
            ceil = ceil[reverse_sorting]
            h = (hid, ceil)
        out = out[reverse_sorting]

        return out, h


class Decoder(nn.Module):
    def __init__(self, args, EOS, dict_size, device):
        super(Decoder, self).__init__()
        self.dim = args.dim * (2 if args.bidir else 1)
        self.bidir = args.bidir
        self.embedding = nn.Embedding(dict_size, self.dim)
        self.rnntype = args.rnn
        self.tf = args.tf
        self.device = device
        self.EOS = torch.tensor([[EOS]]).to(self.device)
        self.PAD = torch.tensor([[0]]).to(self.device)
        self.PAD_p = torch.zeros((1,dict_size)).to(self.device)
        self.PAD_p[0][0] = 1.0
        if args.rnn == 'gru':
            self.rnn = nn.GRU(self.dim, self.dim, num_layers=1, batch_first=True, bidirectional=False)
        elif args.rnn == 'lstm':
            self.rnn = nn.LSTM(self.dim, self.dim, num_layers=1, batch_first=True, bidirectional=False)
        else:
            raise Exception('RNN cell implementation error, should be gru/lstm')
        self.linear = nn.Linear(self.dim, dict_size)
    
    def serial(self, words, h, input=None, input_len=None):
        encoder_output, pre_h = h
        
        word_sequence = []
        out_sequence = []
        for i in range(words.size(0)):
            maxlen = words.size(1)-1
            idx = 0
            token = words[i][0].unsqueeze(0).unsqueeze(1)
            if self.rnntype == 'gru':
                h = pre_h[i].unsqueeze(0).unsqueeze(1)
            elif self.rnntype == 'lstm':
                h = pre_h[0][i].unsqueeze(0).unsqueeze(1)
                ceil = torch.zeros((h.size(0),h.size(1),h.size(2))).to(self.device)
                h = (h,ceil)
            while True:
                #if random.random() < self.tf:
                if idx == 0:
                    token = words[i][0].unsqueeze(0).unsqueeze(1)
                elif self.training and random.random() < self.tf:
                    token = words[i][idx].unsqueeze(0).unsqueeze(1)
                token = token.detach()
                word_embedding = self.embedding(token)
                out, h = self.rnn(word_embedding, h)
                out = self.linear(out)
                #s_out = torch.softmax(out, dim=2)
                #print(s_out[0, 0, 2])
                #_, text = out.topk(1)
                
                #
                softmax_out = torch.softmax(out, dim=2)
                prob_list = softmax_out.view(-1).detach().cpu().numpy()
                offset = 1. - np.sum(prob_list)
                prob_list[1] += offset
                prob_list[prob_list < 0] = 0.
                text = np.random.choice([i for i in range(out.size(2))], 1, p=prob_list)
                text = torch.tensor(np.array([[text]])).to(self.device)

                token = text.squeeze(0)
                out = out.squeeze(0)
                if idx == 0:
                    word_seq = token
                    out_seq = out
                else:
                    word_seq = torch.cat((word_seq, token),dim=0)
                    out_seq = torch.cat((out_seq, out),dim=0)
                idx += 1
                if idx == maxlen or token.item() == self.EOS:
                    break
            while idx < maxlen:
                word_seq = torch.cat((word_seq, self.PAD),dim=0)
                out_seq = torch.cat((out_seq, self.PAD_p),dim=0)
                idx += 1
            word_sequence.append(word_seq)
            out_sequence.append(out_seq)
        word_sequence = pad_sequence(word_sequence, batch_first=True)
        out_sequence = pad_sequence(out_sequence, batch_first=True)
        return out_sequence, word_sequence
    
    def forward(self, words, lengths, h, input=None, input_len=None):
        encoder_output, pre_h = h
        words_embedding = self.embedding(words)
        #ordering
        sorted_len, indices = torch.sort(lengths, descending=True)
        _, reverse_sorting = torch.sort(indices)
        words_embedding = words_embedding[indices]
        lengths = lengths[indices]
        if self.rnntype == 'gru':
            pre_h = pre_h[indices]
            pre_h = pre_h.unsqueeze(0)
        elif self.rnntype == 'lstm':
            hid, _ = pre_h
            hid = hid[indices]
            hid = hid.unsqueeze(0)
            ceil = torch.zeros((hid.size(0),hid.size(1),hid.size(2))).to(self.device)
            pre_h = (hid, ceil)
        packed_padded_sequence = pack_padded_sequence(words_embedding,lengths,batch_first=True)
        out, h = self.rnn(packed_padded_sequence, pre_h)
        out, out_len = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        
        #h = h.permute(1, 0, 2).contiguous()
        #h = h.view(-1, self.dim)
        #h = h[reverse_sorting]
        out = out[reverse_sorting]
        out = self.linear(out)
        return out


class AttnDecoder(nn.Module):
    def __init__(self, args, EOS, dict_size, device):
        super(AttnDecoder, self).__init__()
        self.dim = args.dim * (2 if args.bidir else 1)
        self.bidir = args.bidir
        self.embedding = nn.Embedding(dict_size, self.dim)
        self.rnntype = args.rnn
        self.tf = args.tf
        self.device = device
        self.EOS = torch.tensor([[EOS]]).to(self.device)
        self.PAD = torch.tensor([[0]]).to(self.device)
        self.PAD_p = torch.zeros((1,dict_size)).to(self.device)
        self.PAD_p[0][0] = 1.0
        self.rnn = nn.LSTM(self.dim, self.dim)
        
        self.attn = True
        if self.attn:
            self.W_h = nn.Linear(self.dim, self.dim, bias=False)
            self.W_s = nn.Linear(self.dim, self.dim)
            self.v_t = nn.Linear(self.dim, 1, bias=False)
            self.V = nn.Linear(self.dim * 2, self.dim)
            self.V_prime = nn.Linear(self.dim, dict_size)
        else:
            self.linear = nn.Linear(self.dim, dict_size)
    
    def serial(self, words, h, input, input_len):
        encoder_output, pre_h = h
        # dimension of pre_h: (length, batch, F)
        words_embedding = self.embedding(words)
        batch_size = words_embedding.size(0)
        word_sequence = []
        out_sequence = []
        for i in range(batch_size):
            #sentence = words_embedding[:, i, 0, :]
            token = words_embedding[i, 0:1, :]
            maxlen = words.size(1)-1
            hid = None
            j = 0
            while True:
                # if hid is None:
                #     cell = torch.randn((1, pre_h.size(2))).to(self.device)
                #     hid = pre_h[input_len[i]-1, i, :].unsqueeze(0)
                #     out, h = self.rnn(sentence[j:j+1], (hid, cell))
                # else:
                #     out, h = self.rnn(sentence[j:j+1], (hid, cell))
                ######################
                if hid is None:
                    cell = torch.randn((1, pre_h.size(2))).to(self.device)
                    hid = pre_h[input_len[i]-1, i, :].unsqueeze(0)
                else:
                    token = self.embedding(token).squeeze(0)
                
                out, h = self.rnn(token, (hid, cell))
                ######################
                hid, cell = h

                if self.attn:
                    for k in range(input_len[i]):
                        e_i = self.v_t(torch.tanh(self.W_h(pre_h[k:k+1, i, :]) + self.W_s(cell)))
                        if k == 0:
                            e_all = e_i
                        else:
                            e_all = torch.cat((e_all, e_i), dim=0)
                    a_all = torch.softmax(e_all, dim=0)
                    context = torch.sum(a_all * pre_h[0:input_len[i], i, :], dim=0, keepdim=True)
                    out = self.V_prime(self.V(torch.cat((context, cell), dim=1)))
                else:
                    out = self.linear(out)

                _, token = out.topk(1)
                if j == 0:
                    word_seq = token
                    out_seq = out
                else:
                    word_seq = torch.cat((word_seq, token),dim=0)
                    out_seq = torch.cat((out_seq, out),dim=0)
                j += 1
                #print(j, maxlen, token)
                if j == maxlen or token.item() == self.EOS:
                    break
            word_sequence.append(word_seq)
            out_sequence.append(out_seq)
        word_sequence = pad_sequence(word_sequence)#, batch_first=True)
        out_sequence = pad_sequence(out_sequence)#, batch_first=True)
        return out_sequence, word_sequence

    def forward(self, words, lengths, h, input, input_len):
        encoder_output, pre_h = h
        # dimension of pre_h: (length, batch, F)
        words_embedding = self.embedding(words)
        batch_size = words_embedding.size(0)
        b_out_all_list = []
        for i in range(batch_size):
            #sentence = words_embedding[:, i, 0, :]
            sentence = words_embedding[i, :, :]
            hid = None
            for j in range(lengths[i]):
                if hid is None:
                    cell = torch.randn((1, pre_h.size(2))).to(self.device)
                    hid = pre_h[input_len[i]-1, i, :].unsqueeze(0)
                    out, h = self.rnn(sentence[j:j+1], (hid, cell))
                else:
                    out, h = self.rnn(sentence[j:j+1], (hid, cell))
                hid, cell = h

                if self.attn:
                    for k in range(input_len[i]):
                        e_i = self.v_t(torch.tanh(self.W_h(pre_h[k:k+1, i, :]) + self.W_s(cell)))
                        if k == 0:
                            e_all = e_i
                        else:
                            e_all = torch.cat((e_all, e_i), dim=0)
                    a_all = torch.softmax(e_all, dim=0)
                    context = torch.sum(a_all * pre_h[0:input_len[i], i, :], dim=0, keepdim=True)
                    out = self.V_prime(self.V(torch.cat((context, cell), dim=1)))
                else:
                    out = self.linear(out)

                if j == 0:
                    out_all = out
                else:
                    out_all = torch.cat((out_all, out), dim=0)
            b_out_all_list.append(out_all)
        
        b_out_all = pad_sequence(b_out_all_list)#, batch_first=True)
        return b_out_all


class AttnOutputDecoder(nn.Module):
    def __init__(self, args, EOS, dict_size, device):
        super(AttnOutputDecoder, self).__init__()
        self.dim = args.dim * (2 if args.bidir else 1)
        self.bidir = args.bidir
        self.embedding = nn.Embedding(dict_size, self.dim)
        self.rnntype = args.rnn
        self.tf = args.tf
        self.device = device
        self.EOS = torch.tensor([[EOS]]).to(self.device)
        self.PAD = torch.tensor([[0]]).to(self.device)
        self.PAD_p = torch.zeros((1,dict_size)).to(self.device)
        self.PAD_p[0][0] = 1.0
        self.rnn = nn.LSTM(self.dim, self.dim)
        

        self.W_h = nn.Linear(self.dim, self.dim, bias=False)
        self.W_s = nn.Linear(self.dim, self.dim)
        self.v_t = nn.Linear(self.dim, 1, bias=False)
        self.V = nn.Linear(self.dim * 2, self.dim)
        self.V_prime = nn.Linear(self.dim, dict_size)

        #self.linear = nn.Linear(self.dim, dict_size)
    
    def serial(self, words, h, input, input_len):
        encoder_output, h_c = h
        pre_h, _ = h_c
        # dimension of pre_h: (batch, F)
        words_embedding = self.embedding(words)
        batch_size = words_embedding.size(0)
        word_sequence = []
        out_sequence = []
        for i in range(batch_size):
            #sentence = words_embedding[:, i, 0, :]
            token = words_embedding[i, 0:1, :]
            maxlen = words.size(1)-1
            hid = None
            j = 0
            while True:
                ######################
                if hid is None:
                    cell = torch.randn((1, pre_h.size(1))).to(self.device)
                    hid = pre_h[i, :].unsqueeze(0)
                else:
                    token = self.embedding(token).squeeze(0)

                out, h = self.rnn(token, (hid, cell))
                ######################
                hid, cell = h


                for k in range(input_len[i]):
                    e_i = self.v_t(torch.tanh(self.W_h(encoder_output[i, k:k+1, :]) + self.W_s(out)))
                    if k == 0:
                        e_all = e_i
                    else:
                        e_all = torch.cat((e_all, e_i), dim=0)
                a_all = torch.softmax(e_all, dim=0)
                #print(input_len[i], a_all.size(),encoder_output.size())
                context = torch.sum(a_all * encoder_output[i, 0:input_len[i], :], dim=0, keepdim=True)
                out = self.V_prime(self.V(torch.cat((context, out), dim=1)))

                #_, token = out.topk(1)
                softmax_out = torch.softmax(out, dim=1)
                prob_list = softmax_out.view(-1).detach().cpu().numpy()
                offset = 1. - np.sum(prob_list)
                prob_list[1] += offset
                prob_list[prob_list < 0] = 0.
                text = np.random.choice([i for i in range(out.size(1))], 1, p=prob_list)
                token = torch.tensor(np.array([text])).to(self.device)

                if j == 0:
                    word_seq = token
                    out_seq = out
                else:
                    word_seq = torch.cat((word_seq, token),dim=0)
                    out_seq = torch.cat((out_seq, out),dim=0)
                j += 1
                #print(j, maxlen, token)
                if j == maxlen or token.item() == self.EOS:
                    break
            word_sequence.append(word_seq)
            out_sequence.append(out_seq)
        word_sequence = pad_sequence(word_sequence, batch_first=True)
        out_sequence = pad_sequence(out_sequence, batch_first=True)
        return out_sequence, word_sequence

    def forward(self, words, lengths, h, input, input_len):
        encoder_output, h_c = h
        pre_h, _ = h_c
        words_embedding = self.embedding(words)
        batch_size = words_embedding.size(0)
        cell = torch.randn((1, batch_size, encoder_output.size(2))).to(self.device)
        words_embedding = words_embedding.permute(1, 0, 2)
        pre_h = pre_h.unsqueeze(0)
        out, h = self.rnn(words_embedding, (pre_h, cell))
        final_h, _ = h

        out = out.permute(1, 0, 2)
        wh = self.W_h(encoder_output).unsqueeze(2).repeat(1, 1, out.size(1), 1)
        ws = self.W_s(out).unsqueeze(1)
        e = self.v_t(torch.tanh(wh + ws))
        
        a = torch.softmax(e, dim=1)
        context = torch.sum(encoder_output.unsqueeze(2) * a, dim=1)
        final_out = self.V_prime(self.V(torch.cat((context, out), dim=2)))
        
        # final_out = self.linear(out)
        # final_out = final_out.permute(1, 0, 2)

        return final_out


class PointerGenOutputDecoder(nn.Module):
    def __init__(self, args, EOS, dict_size, device):
        super(PointerGenOutputDecoder, self).__init__()
        self.dim = args.dim * (2 if args.bidir else 1)
        self.bidir = args.bidir
        self.embedding = nn.Embedding(dict_size, self.dim)
        self.rnntype = args.rnn
        self.tf = args.tf
        self.device = device
        self.EOS = torch.tensor([[EOS]]).to(self.device)
        self.PAD = torch.tensor([[0]]).to(self.device)
        self.PAD_p = torch.zeros((1,dict_size)).to(self.device)
        self.PAD_p[0][0] = 1.0
        self.rnn = nn.LSTM(self.dim, self.dim)
        

        self.W_h = nn.Linear(self.dim, self.dim, bias=False)
        self.W_h_prime_t = nn.Linear(self.dim, 1, bias=False)
        self.W_s = nn.Linear(self.dim, self.dim)
        self.W_s_t = nn.Linear(self.dim, 1)
        self.W_x_t = nn.Linear(self.dim, 1, bias=False)
        self.v_t = nn.Linear(self.dim, 1, bias=False)
        self.V = nn.Linear(self.dim * 2, self.dim)
        self.V_prime = nn.Linear(self.dim, dict_size)

        #self.linear = nn.Linear(self.dim, dict_size)
    
    def serial(self, words, h, input, input_len):
        encoder_output, h_c = h
        pre_h, _ = h_c
        # dimension of pre_h: (batch, F)
        words_embedding = self.embedding(words)
        batch_size = words_embedding.size(0)
        word_sequence = []
        out_sequence = []
        for i in range(batch_size):
            #sentence = words_embedding[:, i, 0, :]
            token = words_embedding[i, 0:1, :]
            maxlen = words.size(1)-1
            hid = None
            j = 0
            while True:
                ######################
                if hid is None:
                    cell = torch.randn((1, pre_h.size(1))).to(self.device)
                    hid = pre_h[i, :].unsqueeze(0)
                else:
                    token = self.embedding(token).squeeze(0)

                out, h = self.rnn(token, (hid, cell))
                ######################
                hid, cell = h


                for k in range(input_len[i]):
                    e_i = self.v_t(torch.tanh(self.W_h(encoder_output[i, k:k+1, :]) + self.W_s(out)))
                    if k == 0:
                        e_all = e_i
                    else:
                        e_all = torch.cat((e_all, e_i), dim=0)
                a_all = torch.softmax(e_all, dim=0)
                #print(input_len[i], a_all.size(),encoder_output.size())
                context = torch.sum(a_all * encoder_output[i, 0:input_len[i], :], dim=0, keepdim=True)
                # out = self.V_prime(self.V(torch.cat((context, out), dim=1)))


                # # _, token = out.topk(1)
                # softmax_out = torch.softmax(out, dim=1)
                # prob_list = softmax_out.view(-1).detach().cpu().numpy()
                # offset = 1. - np.sum(prob_list)
                # prob_list[1] += offset
                # prob_list[prob_list < 0] = 0.
                # text = np.random.choice([i for i in range(out.size(1))], 1, p=prob_list)
                # token = torch.tensor(np.array([text])).to(self.device)

                
                final_out = self.V_prime(self.V(torch.cat((context, out), dim=1)))
                wht = self.W_h_prime_t(context)
                wst = self.W_s_t(out)
                #wxt = self.W_x_t(words_embedding[i, j])
                wxt = self.W_x_t(token)
                p_gen = torch.sigmoid(wht + wst + wxt)

                dis_vocab = torch.softmax(final_out, dim=1)

                #spec_p = dis_vocab[i, j, words[i, j]]
                #print(torch.sum(dis_vocab))
                #print(final_out.size(), dis_vocab.size(), a_all.size(), p_gen.size())
                for k in range(input_len[i]):
                    #print(input[i, k], p_gen[0], a_all[k])
                    dis_vocab[0, input[i, k]] = p_gen[0] * dis_vocab[0, input[i, k]] + (1 - p_gen[0]) * a_all[k]
                dis_vocab = torch.softmax(dis_vocab, dim=1)
                _, token = dis_vocab.topk(1)
                
                # prob_list = dis_vocab.view(-1).detach().cpu().numpy()
                # offset = 1. - np.sum(prob_list)
                # prob_list[1] += offset
                # prob_list[prob_list < 0] = 0.
                # text = np.random.choice([i for i in range(dis_vocab.size(1))], 1, p=prob_list)
                # token = torch.tensor(np.array([text])).to(self.device)

                if j == 0:
                    word_seq = token
                    out_seq = final_out
                else:
                    word_seq = torch.cat((word_seq, token),dim=0)
                    out_seq = torch.cat((out_seq, final_out),dim=0)
                j += 1
                #print(j, maxlen, token)
                if j == maxlen or token.item() == self.EOS:
                    break
            word_sequence.append(word_seq)
            out_sequence.append(out_seq)
        word_sequence = pad_sequence(word_sequence, batch_first=True)
        out_sequence = pad_sequence(out_sequence, batch_first=True)
        return out_sequence, word_sequence

    def forward(self, words, lengths, h, input, input_len):
        encoder_output, h_c = h
        pre_h, _ = h_c
        words_embedding = self.embedding(words)
        batch_size = words_embedding.size(0)
        max_len = words_embedding.size(1)
        cell = torch.randn((1, batch_size, encoder_output.size(2))).to(self.device)
        words_embedding = words_embedding.permute(1, 0, 2)
        pre_h = pre_h.unsqueeze(0)
        out, h = self.rnn(words_embedding, (pre_h, cell))
        final_h, _ = h

        out = out.permute(1, 0, 2)
        wh = self.W_h(encoder_output).unsqueeze(2).repeat(1, 1, out.size(1), 1)
        ws = self.W_s(out).unsqueeze(1)
        e = self.v_t(torch.tanh(wh + ws))
        
        a = torch.softmax(e, dim=1)
        context = torch.sum(encoder_output.unsqueeze(2) * a, dim=1)
        final_out = self.V_prime(self.V(torch.cat((context, out), dim=2)))
        
        wht = self.W_h_prime_t(context)
        wst = self.W_s_t(out)
        wxt = self.W_x_t(words_embedding.permute(1, 0, 2))
        p_gen = torch.sigmoid(wht + wst + wxt)

        dis_vocab = torch.softmax(final_out, dim=2)
        
        #print(p_gen.size(), dis_vocab.size())
        for j in range(max_len):
            if j < max_len - 1:
                indices = input == words[:, j+1:j+2] 
            else:
                indices = input == words[:, j:j+1]
            if j == 0:
                indices_all = indices.unsqueeze(2)
            else:
                indices_all = torch.cat((indices_all, indices.unsqueeze(2)), dim=2)
        indices_all = indices_all.unsqueeze(3)
        a_idx_attn = torch.sum(indices_all * a, dim=1)
        #select_P = dis_vocab[words].unsqueeze(2)
        #print(a_idx_attn.size(), select_P.size())
        for i in range(batch_size):
            for j in range(max_len):
                if j < max_len - 1:
                    spec_p = dis_vocab[i, j, words[i, j+1]] 
                else:
                    spec_p = dis_vocab[i, j, words[i, j]]
                if j == 0:
                    select_P = spec_p.unsqueeze(0)
                else:
                    select_P = torch.cat((select_P, spec_p.unsqueeze(0)), dim=0)
            if i == 0:
                select_b_P = select_P.unsqueeze(0)
            else:
                select_b_P = torch.cat((select_b_P, select_P.unsqueeze(0)), dim=0)
        select_b_P = select_b_P.unsqueeze(2)
        #print(a_idx_attn.size(), select_b_P.size(), p_gen.size())
        return p_gen * select_b_P + (1-p_gen) * a_idx_attn

class PointerGenDecoder(nn.Module):
    def __init__(self, args, EOS, dict_size, device):
        super(PointerGenDecoder, self).__init__()
        self.dim = args.dim * (2 if args.bidir else 1)
        self.bidir = args.bidir
        self.embedding = nn.Embedding(dict_size, self.dim)
        self.rnntype = args.rnn
        self.tf = args.tf
        self.device = device
        self.EOS = torch.tensor([[EOS]]).to(self.device)
        self.UNK = torch.tensor([[1]]).to(self.device)
        self.PAD = torch.tensor([[0]]).to(self.device)
        self.PAD_p = torch.zeros((1,dict_size)).to(self.device)
        self.PAD_p[0][0] = 1.0
        self.rnn = nn.LSTM(self.dim, self.dim)
        self.W_h = nn.Linear(self.dim, self.dim, bias=False)
        self.W_h_prime_t = nn.Linear(self.dim, 1, bias=False)
        self.W_s = nn.Linear(self.dim, self.dim)
        self.W_s_t = nn.Linear(self.dim, 1)
        self.W_x_t = nn.Linear(self.dim, 1, bias=False)
        self.v_t = nn.Linear(self.dim, 1, bias=False)
        self.V = nn.Linear(self.dim * 2, self.dim)
        self.V_prime = nn.Linear(self.dim, dict_size)


    def serial(self, words, h, input, input_len):
        encoder_output, pre_h = h
        # dimension of pre_h: (length, batch, F)
        words_embedding = self.embedding(words)
        batch_size = words_embedding.size(0)
        word_sequence = []
        out_sequence = []
        for i in range(batch_size):
            # get <s> token
            #token = words_embedding[:, i, 0, :]
            token = words_embedding[i, 0:1, :]
            maxlen = words.size(1)-1
            hid = None
            j = 0
            while True:
                ######################
                if hid is None:
                    cell = torch.randn((1, pre_h.size(2))).to(self.device)
                    hid = pre_h[input_len[i]-1, i, :].unsqueeze(0)
                else:
                    token = self.embedding(token).squeeze(0)
                
                #print(token.size())
                out, h = self.rnn(token, (hid, cell))
                ######################
                hid, cell = h

                for k in range(input_len[i]):
                    e_i = self.v_t(torch.tanh(self.W_h(pre_h[k:k+1, i, :]) + self.W_s(cell)))
                    if k == 0:
                        e_all = e_i
                    else:
                        e_all = torch.cat((e_all, e_i), dim=0)
                a_all = torch.softmax(e_all, dim=0)
                context = torch.sum(a_all * pre_h[0:input_len[i], i, :], dim=0, keepdim=True)
                out = self.V_prime(self.V(torch.cat((context, cell), dim=1)))

                # pointer generator
                # dis_vocab = torch.softmax(out, dim=1)
                # _, token_before = out.topk(1)
                # p_gen = torch.sigmoid(self.W_h_prime_t(context) + self.W_s_t(cell) + self.W_x_t(token_before))
                # a_idx_attn = 0
                # for k in range(input_len[i]):
                #     if input[k, i, 0] == words[j, i, 0]:
                #         a_idx_attn += a_all[k, 0]
                
                # #P_vocab = dis_vocab[0, words[j+1, i, 0].item()]
                # P_vocab = dis_vocab[0, token_before[0, 0].item()]

                # #P_vocab: scalar, a_idx_attn: scalar
                # final_out = p_gen * P_vocab + (1 - p_gen) * (0 if a_idx_attn == 0 else a_idx_attn)


                # _, token = final_out.topk(1)
                _, token = out.topk(1)
                if j == 0:
                    word_seq = token
                    out_seq = out
                else:
                    word_seq = torch.cat((word_seq, token),dim=0)
                    out_seq = torch.cat((out_seq, out),dim=0)
                j += 1
                
                if j == maxlen or token.item() == self.EOS:
                    break
            word_sequence.append(word_seq)
            out_sequence.append(out_seq)
        word_sequence = pad_sequence(word_sequence, batch_first=True)
        out_sequence = pad_sequence(out_sequence, batch_first=True)
        return out_sequence, word_sequence
    
    def forward(self, words, lengths, h, input, input_len):
        encoder_output, pre_h = h
        # dimension of pre_h: (length, batch, F)
        words_embedding = self.embedding(words)
        batch_size = words_embedding.size(0)
        b_out_all_list = []
        for i in range(batch_size):
            #sentence = words_embedding[:, i, 0, :]
            sentence = words_embedding[i, :, :]
            hid = None
            for j in range(lengths[i]-1):
                if hid is None:
                    cell = torch.randn((1, pre_h.size(2))).to(self.device)
                    hid = pre_h[input_len[i]-1, i, :].unsqueeze(0)
                    out, h = self.rnn(sentence[j:j+1], (hid, cell))
                else:
                    out, h = self.rnn(sentence[j:j+1], (hid, cell))
                hid, cell = h
                for k in range(input_len[i]):
                    e_i = self.v_t(torch.tanh(self.W_h(pre_h[k:k+1, i, :]) + self.W_s(cell)))
                    if k == 0:
                        e_all = e_i
                    else:
                        e_all = torch.cat((e_all, e_i), dim=0)
                
                a_all = torch.softmax(e_all, dim=0)
                context = torch.sum(a_all * pre_h[0:input_len[i], i, :], dim=0, keepdim=True)
                out = self.V_prime(self.V(torch.cat((context, cell), dim=1)))

                # pointer generator
                dis_vocab = torch.softmax(out, dim=1)
                p_gen = torch.sigmoid(self.W_h_prime_t(context) + self.W_s_t(cell) + self.W_x_t(sentence[j+1:j+2]))
                a_idx_attn = 0
                for k in range(input_len[i]):
                    if input[i, k] == words[i, j]:
                        a_idx_attn += a_all[k, 0]
                P_vocab = dis_vocab[0, words[i, j+1].item()]

                #P_vocab: scalar, a_idx_attn: scalar
                #print(p_gen, P_vocab, a_idx_attn)
                final_out = p_gen * P_vocab + (1 - p_gen) * (0 if a_idx_attn == 0 else a_idx_attn)
                if j == 0:
                    out_all = final_out
                else:
                    out_all = torch.cat((out_all, final_out), dim=0)
            b_out_all_list.append(out_all)
        
        b_out_all = pad_sequence(b_out_all_list, batch_first=True)
        #print(b_out_all.size())
        return b_out_all