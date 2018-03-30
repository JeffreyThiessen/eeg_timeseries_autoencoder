# Neural net code from the tutorials found here
# http://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html

# To use this code please install pytorch, and mne
# http://pytorch.org
# MNE can be installed alongside other useful tools by installing braindecode
# https://robintibor.github.io/braindecode/index.html

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable

import mne
import random
import numpy as np
from mne import concatenate_raws

import unicodedata
import string
import re


# TOOLS
import time
import math
def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)
def timeSince(since, percent):
    now = time.time()
    s = now - since
    if percent is not 0:
        es = s / (percent)
        rs = es - s
        return '%s (- %s)' % (asMinutes(s), asMinutes(rs))
    else:
        return '%s (- ?)' % (asMinutes(s))

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
def showPlot(points, show_plot):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
    if show_plot:
        plt.show()



use_cuda = torch.cuda.is_available()
use_cuda = False # Overriding cuda because my graphics card only has cuda compatibility 3.0


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.linear = nn.Linear(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        linear1 = self.linear(input).view(1, 1, -1)
        output = linear1
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.linear = nn.Linear(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        output = self.linear(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.out(output[0])
        return output, hidden

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result

#teacher_forcing_ratio = 0 # use for non sequence to sequence
teacher_forcing_ratio = 0.5 # use for sequence to sequence 

def train(input_variable, target_variable, encoder, decoder,
            encoder_optimizer, decoder_optimizer, 
            criterion, max_length):
    encoder_hidden = encoder.initHidden()

    if encoder_optimizer is not None:
        encoder_optimizer.zero_grad()
    if decoder_optimizer is not None:
        decoder_optimizer.zero_grad()

    input_length = input_variable.size()[0]
    target_length = target_variable.size()[0]

    #catalog of encoder outputs
    encoder_outputs = Variable(torch.zeros(input_length, encoder.hidden_size))
    encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_variable[ei], encoder_hidden)
        #We are not using encoder outputs, just final hidden state, but collect data anyway
        encoder_outputs[ei] = encoder_output[0][0]

    decoder_input = Variable(torch.zeros(1,max_length))
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    # Decoders first hidden is the final hidden from the encoder
    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden)
            loss += criterion(decoder_output, target_variable[di])
            decoder_input = target_variable[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next imput
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden)

            decoder_input = decoder_output
            decoder_input = decoder_input.cuda() if use_cuda else decoder_input

            loss += criterion(decoder_output, target_variable[di])

    loss.backward()

    if encoder_optimizer is not None:
        encoder_optimizer.step()
    if decoder_optimizer is not None:
        decoder_optimizer.step()

    return (loss.data[0] / target_length)


def trainIters(encoder, decoder, criterion, pairs, n_iters, print_every=1000, plot_every=100, show_plot=False, learning_rate=0.01, max_length = 100):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    # using random.choice allows us to train more than the 
    # amount of input we have by randomly picking from range over and over
    training_pairs = [random.choice(pairs) for i in range(n_iters)] 

    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        input_variable = training_pair[0]
        target_variable = training_pair[1]

        loss = train(input_variable, target_variable, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, 
                     criterion, max_length)
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, float(iter) / float(n_iters)),
                                         iter, float(iter) / float(n_iters) * 100, print_loss_avg))

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    print("Done!")
    showPlot(plot_losses, show_plot)




def get_data(file_name, channel):
    data_path = '/home/jeff/Documents/pytorch/ae_ts/data/'
    single_data_file = data_path + file_name

    raw = mne.io.read_raw_edf(single_data_file, preload=True, stim_channel='auto')

    # crop it if data too big
    #raw.crop(0,5000)

    # pick a channel
    #>>> raw.ch_names
    #[u'ROC-LOC', u'LOC-ROC', u'F2-F4', u'F4-C4', u'C4-P4', u'P4-O2', 
    # u'F1-F3', u'F3-C3', u'C3-P3', u'P3-O1', u'C4-A1', u'EMG1-EMG2', 
    # u'ECG1-ECG2', u'TERMISTORE', u'TORACE', u'ADDOME', u'Dx1-DX2', 
    # u'SX1-SX2', u'Posizione', u'HR', u'SpO2']

    # Can only pick one channel at a time with this setup
    raw.pick_channels([channel])

    # this makes a numpy array
    # use data.shape to get shape
    data = raw.get_data()

    data_points_per_second = raw.n_times / raw.times.max()

    # free memory
    del raw

    return data, data_points_per_second


def pairs_from_slices(slices):
    #The expected (target) variable should NOT require gradient
    if use_cuda:
        data_pairs = [(Variable(slices[i], requires_grad=True).cuda(),
                        Variable(slices[i], requires_grad=False).cuda()) 
                        for i in range(0,len(slices))]
    else:
        data_pairs = [(Variable(slices[i], requires_grad=True),
                        Variable(slices[i], requires_grad=False)) 
                        for i in range(0,len(slices))]

    return data_pairs

def make_simple_data_pairs(data, max_length):
    # Use a tensor
    tdata = torch.from_numpy(data)

    # Convert from double to float so stuff works
    tdata = tdata.float()

    #if use_cuda:
    #   tdata = tdata.cuda()

    # Since we only have one sample, if we had more we would have them be the other axis
    # samples x channels x data
    # the other way of doing it is by using sequences, as we do in the next function
    tdata = tdata.unsqueeze(0)

    # make data more human readable,
    # (almost all) data point x is -1 < x < 1
    tdata = tdata * 10000

    var = Variable(tdata, requires_grad=True)

    slices = torch.split(tdata, max_length, 2)
    slices = slices[:-1] #last item may not match expected length of MAX_LENGTH

    return pairs_from_slices(slices)

def make_sequenced_data_pairs(data, max_length, sequence_count = 30):
    #split the data into MAX_LENGTH log chunks
    split_sections = int(data.size / max_length)
    data = np.stack(np.array_split(data,split_sections,1)[:-1])

    tdata = torch.from_numpy(data)
    tdata = tdata.float()

    # make data more human readable,
    # (almost all) data point x is -1 < x < 1
    tdata = tdata * 10000

    # sequence count (number of 1 second chunks in a sequence)
    slices = torch.split(tdata, sequence_count, 0)
    slices = slices[:-1]

    return pairs_from_slices(slices)



data, data_points_per_second = get_data('n1.edf', 'F2-F4')
#MAX_LENGTH is the length of a section of data we are observing (feeding into the alg)
MAX_LENGTH = int(data_points_per_second / 4)
#MAX_LENGTH = 512

# Hidden layer size will be a percentage of the data size
hidden_layer_percentage = 0.1
hidden_size = int(MAX_LENGTH * hidden_layer_percentage)

#SC = 30
SC = 4

def run_seq_single(): #used just to see if encoder/decoder is working
    pairs = make_sequenced_data_pairs(data, MAX_LENGTH)
    return run_train_only(pairs, hidden_size, MAX_LENGTH)

def run_seq():
    pairs = make_sequenced_data_pairs(data, MAX_LENGTH, SC)
    return run_train_iters(pairs, hidden_size, MAX_LENGTH)

def run_single(): #used just to see if encoder/decoder is working
    pairs = make_simple_data_pairs(data, MAX_LENGTH)
    return run_train_only(pairs, hidden_size, MAX_LENGTH)

def run():
    pairs = make_simple_data_pairs(data, MAX_LENGTH)
    return run_train_iters(pairs, hidden_size, MAX_LENGTH)



def setup_encoder_decoder(hidden_size, max_length):
    encoder1 = EncoderRNN(MAX_LENGTH, hidden_size)
    decoder1 = DecoderRNN(hidden_size, MAX_LENGTH)
    if use_cuda:
        encoder1 = encoder1.cuda()
        decoder1 = decoder1.cuda()

    criterion1 = nn.L1Loss()
    return encoder1, decoder1, criterion1

def run_train_only(data_pairs, hidden_size, max_length):
    encoder1, decoder1, criterion1 = setup_encoder_decoder(hidden_size, max_length)
    return train(data_pairs[0][0], data_pairs[0][1], 
            encoder1, decoder1,
            None, None, 
            criterion1, max_length)

def run_train_iters(data_pairs, hidden_size, max_length):
    encoder1, decoder1, criterion1 = setup_encoder_decoder(hidden_size, max_length)
    return trainIters(encoder1, decoder1, criterion1, 
            data_pairs, n_iters=len(data_pairs)*5, 
            print_every=50, plot_every=10, show_plot=True,
            learning_rate=0.01, max_length=max_length)
