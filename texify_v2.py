input_files = ['/kaggle/input/crohme2019/crohme2019_train.txt',
               '/kaggle/input/crohme2019/crohme2019_valid.txt',
               '/kaggle/input/crohme2019/crohme2019_test.txt']

vocab = set()

for input_file in input_files:
    for line in open(input_file).readlines():
        if len(line.strip().split('\t')) == 2:
            vocab.update(line.strip().split('\t')[1].split())

vocab_syms = [v for v in vocab if v not in ['Above', 'Below', 'Inside', 'NoRel', 'Right', 'Sub', 'Sup']]

with open('crohme_vocab.txt', 'w') as f:
    f.writelines([c + '\n' for c in sorted(vocab_syms)])
    f.writelines([c + '\n' for c in ['Above', 'Below', 'Inside', 'NoRel', 'Right', 'Sub', 'Sup']])


# vocab class
class Vocab(object):
    def __init__(self, vocab_file=None):
        self.word2index = {}
        self.index2word = {}

        if vocab_file:
            self.load_vocab(vocab_file)

    def load_vocab(self, vocab_file):
        # load vocab from file
        with open(vocab_file, 'r') as f:
            for i, line in enumerate(f):
                word = line.strip()
                self.word2index[word] = i
                self.index2word[i] = word
        # add blank word
        self.word2index['<blank>'] = len(self.word2index)
        self.index2word[self.word2index['<blank>']] = '<blank>'


vocab = Vocab(vocab_file='/kaggle/working/crohme_vocab.txt')
vocab.index2word
## test vocab
vocab = Vocab('crohme_vocab.txt')
input = '- Right \\sqrt Inside 2'.split()
output = [vocab.word2index[word] for word in input]
output
assert output == [4, 105, 66, 103, 9]

# Task 1.2: write dataset class
# handling inkml files
import numpy as np

import xml.etree.ElementTree as ET


class Segment(object):
    """Class to reprsent a Segment compound of strokes (id) with an id and label."""
    __slots__ = ('id', 'label', 'strId')

    def __init__(self, *args):
        if len(args) == 3:
            self.id = args[0]
            self.label = args[1]
            self.strId = args[2]
        else:
            self.id = "none"
            self.label = ""
            self.strId = set([])


class Inkml(object):
    """Class to represent an INKML file with strokes, segmentation and labels"""
    __slots__ = ('fileName', 'strokes', 'strkOrder', 'segments', 'truth', 'UI')

    NS = {'ns': 'http://www.w3.org/2003/InkML', 'xml': 'http://www.w3.org/XML/1998/namespace'}

    def __init__(self, *args):
        self.fileName = None
        self.strokes = {}
        self.strkOrder = []
        self.segments = {}
        self.truth = ""
        self.UI = ""
        if len(args) == 1:
            self.fileName = args[0]
            self.loadFromFile()

    def fixNS(self, ns, att):
        """Build the right tag or element name with namespace"""
        return '{' + Inkml.NS[ns] + '}' + att

    def loadFromFile(self):
        """load the ink from an inkml file (strokes, segments, labels)"""
        tree = ET.parse(self.fileName)
        # # ET.register_namespace();
        root = tree.getroot()
        for info in root.findall('ns:annotation', namespaces=Inkml.NS):
            if 'type' in info.attrib:
                if info.attrib['type'] == 'truth':
                    self.truth = info.text.strip()
                if info.attrib['type'] == 'UI':
                    self.UI = info.text.strip()
        for strk in root.findall('ns:trace', namespaces=Inkml.NS):
            self.strokes[strk.attrib['id']] = strk.text.strip()
            self.strkOrder.append(strk.attrib['id'])
        segments = root.find('ns:traceGroup', namespaces=Inkml.NS)
        if segments is None or len(segments) == 0:
            return
        for seg in (segments.iterfind('ns:traceGroup', namespaces=Inkml.NS)):
            id = seg.attrib[self.fixNS('xml', 'id')]
            label = seg.find('ns:annotation', namespaces=Inkml.NS).text
            strkList = set([])
            for t in seg.findall('ns:traceView', namespaces=Inkml.NS):
                strkList.add(t.attrib['traceDataRef'])
            self.segments[id] = Segment(id, label, strkList)

    def getTraces(self, height=256):
        traces_array = [np.array([p.strip().split()
                                  for p in self.strokes[id].split(',')], dtype='float')
                        for id in self.strkOrder]

        ratio = height / ((np.concatenate(traces_array, 0).max(0) - np.concatenate(traces_array, 0).min(0))[1] + 1e-6)
        return [(trace * ratio).astype(int).tolist() for trace in traces_array]


# visualize inkml traces
import matplotlib.pyplot as plt

def visualize_inkml(ink_obj: Inkml):
    ink = ink_obj

    plt.figure(figsize=(16, 4))
    plt.axis("off")
    for trace in ink.getTraces():
        trace_arr = np.array(trace)
        plt.plot(trace_arr[:, 0], -trace_arr[:, 1])  # invert y coordinate


path = '/kaggle/input/crohme2019/crohme2019/crohme2019/valid/18_em_0.inkml'
ink = Inkml(path)
visualize_inkml(ink)

# ink object from an InkML file
ink = Inkml('/kaggle/input/crohme2019/crohme2019/crohme2019/valid/18_em_0.inkml')

# get traces from the ink object
[np.array(trace).shape for trace in ink.getTraces()]

# build dataset
from torch.utils.data import Dataset


class InkmlDataset(Dataset):
    def __init__(self, annotation, root_dir, vocab):
        """
        Arguments:
            annotation (string): annotation file
            root_dir (string): Directory with all the images.
        """
        self.annotation = annotation
        self.root_dir = root_dir
        self.vocab = vocab

        # load annotations
        self.ink_paths = []
        self.labels = []

        # your code for create self.inks and self.labels from the annotation file

    def __len__(self):
        # This code should return the number of samples in the dataset
        return len(self.labels)

    def __getitem__(self, idx):
        # This code should return the idx-th sample in the dataset

        inkfile = self.ink_paths[idx]
        label = self.labels[idx]

        ### feature
        # read inkml sample --> traces

        # remove consecutive duplicated points

        # extract features (delta_x / distance, delta_y / distance, distance, pen_up)

        ### label
        # convert tokens of label into vocab indexes

        return feature, label, input_len, label_len


# feature representation
def remove_duplicate(trace):
    # your implementation to remove consecutive duplicated points
    return new_trace


ink = Inkml('/kaggle/input/crohme2019/crohme2019/crohme2019/valid/18_em_0.inkml')
traces = ink.getTraces()
traces = [remove_duplicate(trace) for trace in traces]

assert list(map(len, traces)) == [82, 16, 21, 78, 82, 15, 18, 18, 28, 58, 15, 19, 56, 70, 18, 21]


def feature_extraction(traces):
    # your implementation
    return feature


ink = Inkml('/kaggle/input/crohme2019/crohme2019/crohme2019/valid/18_em_0.inkml')
traces = ink.getTraces()
traces = [remove_duplicate(trace) for trace in traces]
feature = feature_extraction(traces)

import numpy.testing as npt

assert feature.shape == (614, 3)
npt.assert_allclose(feature.mean(axis=0), np.array([0.17677799, 0.29519369, 11.35540311]))
npt.assert_allclose(feature.var(axis=0), np.array([3.75390418e-01, 5.06219812e-01, 4.19017361e+02])
                    )


def feature_extraction(traces):
    # your implementation
    return feature


ink = Inkml('/kaggle/input/crohme2019/crohme2019/crohme2019/valid/18_em_0.inkml')
traces = ink.getTraces()
traces = [remove_duplicate(trace) for trace in traces]
feature = feature_extraction(traces)

import numpy.testing as npt

assert feature.shape == (614, 4)
npt.assert_array_equal(np.where(feature[:, 3] == 1)[0],
                       np.array([81, 97, 118, 196, 278, 293, 311, 329, 357, 415, 430, 449, 505,
                                 575, 593]))

dataset = InkmlDataset(annotation='/kaggle/input/crohme2019/crohme2019_valid.txt',
                       root_dir='/kaggle/input/crohme2019/crohme2019', vocab=Vocab('crohme_vocab.txt'))
feature, label, input_len, label_len = dataset.__getitem__(0)

import numpy.testing as npt

assert type(feature) == torch.Tensor
assert type(label) == torch.Tensor
assert feature.shape == (231, 4)
npt.assert_array_equal(label, np.array([59, 105, 1, 105, 59, 105, 1, 105, 87, 105, 2, 105, 2]))
assert input_len == 231
assert label_len == 13


# dataloader
def collate_fn(batch):
    """Create batch"""
    # your code here
    # features, labels, input_lens, label_lens should be torch.tensor
    return features, labels, input_lens, label_lens


features, labels, input_lens, label_lens = collate_fn([dataset[0], dataset[1]])

import numpy.testing as npt

assert type(input_lens) == torch.Tensor
assert type(label_lens) == torch.Tensor

assert features.shape == (2, 231, 4)
assert labels.shape == (2, 13)
npt.assert_array_equal(input_lens.numpy(), np.array([231, 102]))
npt.assert_array_equal(label_lens, np.array([13, 5]))

# test collate_fn with data loader
from torch.utils.data import DataLoader

data_loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)

features, labels, input_lens, label_lens = next(iter(data_loader))
assert features.shape == (2, 231, 4)
assert labels.shape == (2, 13)
npt.assert_array_equal(input_lens.numpy(), np.array([231, 102]))
npt.assert_array_equal(label_lens, np.array([13, 5]))

# pytorch lightning datamodule
import pytorch_lightning as pl
from torch.utils.data import DataLoader


class CROHMEDatamodule(pl.LightningDataModule):
    def __init__(self, root_dir, train_annotation, validation_annotation, test_annotation, vocab_file, batch_size,
                 num_workers):
        super().__init__()
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.train_annotation = train_annotation
        self.validation_annotation = validation_annotation
        self.test_annotation = test_annotation
        self.vocab = Vocab(vocab_file)
        self.num_workers = num_workers

    def setup(self, stage: str):
        self.train_dataset = InkmlDataset(self.train_annotation, self.root_dir, self.vocab)
        self.val_dataset = InkmlDataset(self.validation_annotation, self.root_dir, self.vocab)
        self.test_dataset = InkmlDataset(self.test_annotation, self.root_dir, self.vocab)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True,
                          collate_fn=collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False,
                          collate_fn=collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False,
                          collate_fn=collate_fn)


# task2: modules, loss functions
# build a model of Bidirectional LSTM 3 x 128
import torch.nn as nn


class LSTM_TemporalClassification(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        # declare layers
        super(LSTM_TemporalClassification, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        # write calculation process here

        return x


# test your implementation
model = LSTM_TemporalClassification(4, 128, 3, 109)
assert model.forward(torch.rand((10, 100, 4))).shape == (10, 100, 109)

# lost ctc
import pytorch_lightning as pl
import torch


class MathOnlineModel(pl.LightningModule):
    def __init__(self, lr=0.001, input_size=4, output_size=109, hidden_size=128, num_layers=3, decoder=None):
        super().__init__()
        self.model = LSTM_TemporalClassification(input_size, hidden_size, num_layers, output_size)
        self.criterion = nn.CTCLoss(blank=output_size - 1)
        self.lr = lr
        self.decoder = decoder

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y, x_lens, y_lens = batch

        # your code to calculate loss

        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, x_lens, y_lens = batch

        # your code to calculate loss

        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y, x_lens, y_lens = batch
        # your code to calculate loss

        self.log('test_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)


# task 3: training with trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer

trainer = Trainer(
    callbacks=[
        LearningRateMonitor(logging_interval='step'),
        ModelCheckpoint(filename='{epoch}-{val_loss:.4f}', save_top_k=5, monitor='val_loss', mode='min'),
    ],
    logger=TensorBoardLogger('lightning_logs'),
    check_val_every_n_epoch=1,
    fast_dev_run=False,
    default_root_dir='checkpoint',
    deterministic=False,
    max_epochs=20,
    log_every_n_steps=50,
    devices=1,
)
model = MathOnlineModel()
# model = MathOnlineModel.load_from_checkpoint('lightning_logs/lightning_logs/version_4/checkpoints/epoch=2-val_loss=2.1620.ckpt')

dm = CROHMEDatamodule(root_dir='/kaggle/input/crohme2019/crohme2019',
                      train_annotation='/kaggle/input/crohme2019/crohme2019_train.txt',
                      validation_annotation='/kaggle/input/crohme2019/crohme2019_valid.txt',
                      test_annotation='/kaggle/input/crohme2019/crohme2019_test.txt',
                      vocab_file='crohme_vocab.txt',
                      batch_size=32,
                      num_workers=4
                      )

trainer.fit(model, dm)
# run test on validation dataset
# load your best val_loss model

from pytorch_lightning import Trainer

trainer = Trainer(
    devices=1,
)

# Load the model from a checkpoint
model = MathOnlineModel.load_from_checkpoint(
    'path/to/your_checkpoint.ckpt',
)

# Initialize the data module
dm = CROHMEDatamodule(
    root_dir='/kaggle/input/crohme2019/crohme2019',
    train_annotation='/kaggle/input/crohme2019/crohme2019_train.txt',
    validation_annotation='/kaggle/input/crohme2019/crohme2019_valid.txt',
    test_annotation='/kaggle/input/crohme2019/crohme2019_valid.txt',
    vocab_file='crohme_vocab.txt',
    batch_size=32,
    num_workers=4
)

# Test the model
trainer.test(model, datamodule=dm)

# task 4: decode output
from typing import List
import torch


class GreedyCTCDecoder(torch.nn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.vocab = vocab
        self.blank = vocab.word2index['<blank>']

    def forward(self, emission: torch.Tensor) -> List[str]:
        """Given a sequence emission over labels, get the best path
        Args:
          emission (Tensor): Logit tensors. Shape `[seq_len, num_label]`.

        Returns:
          List[str]: The resulting transcript
        """
        # your implementation
        return output_seq_list


# test code
model = MathOnlineModel.load_from_checkpoint('path/to/your_checkpoint.ckpt')
model.eval()

dataset = InkmlDataset(annotation='/kaggle/input/crohme2019/crohme2019_valid.txt',
                       root_dir='/kaggle/input/crohme2019/crohme2019', vocab=Vocab('crohme_vocab.txt'))
feature, label, input_len, label_len = dataset.__getitem__(0)

vocab = Vocab('crohme_vocab.txt')
greedy_decoder = GreedyCTCDecoder(vocab)

# run model forward with feature --> output

# run greedy decoder
decoded = greedy_decoder.forward()

# possible output if your training work well
# decoded -> ['\\phi', 'Right', '(', 'Right', '0', 'Right', '(', 'Right', 'n', 'Right', ')', 'Right', ')']

# task 5: add calculation of metrics
from torchmetrics.functional.text.helper import _LevenshteinEditDistance


def edit_distance(pred_seq, label_seq):
    # Your code here
    return distance


# Test your implementation
assert edit_distance(['\\phi',
                      'Right',
                      '(',
                      'Right',
                      '0',
                      'Right',
                      '(',
                      'Right',
                      'n',
                      'Right',
                      ')',
                      'Right',
                      ')'],
                     ['\\phi',
                      'Right',
                      '(',
                      'Right',
                      '\\phi',
                      'Right',
                      '(',
                      'Right',
                      'n',
                      'Right',
                      ')',
                      'Right',
                      ')']) == 1

import pytorch_lightning as pl
import torch
from torchmetrics.functional.text.helper import _LevenshteinEditDistance as _LE_distance


class MathOnlineModel(pl.LightningModule):
    def __init__(self, lr=0.001, input_size=4, output_size=109, hidden_size=128, num_layers=3, decoder=None):
        super().__init__()
        self.model = LSTM_TemporalClassification(input_size, hidden_size, num_layers, output_size)
        self.criterion = nn.CTCLoss(blank=output_size - 1)
        self.lr = lr
        self.decoder = decoder

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y, x_lens, y_lens = batch

        # your code to calculate loss

        # your code to calculate total edit distance
        total_edits = 0
        total_lens = 0

        # loop: calculate edit distance
        total_edits += edit_distance
        total_lens += y_len

    self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
    self.log('train_wer', total_edits / total_lens, prog_bar=True, on_step=True, on_epoch=True)
    return loss


def validation_step(self, batch, batch_idx):
    x, y, x_lens, y_lens = batch

    # your code to calculate loss

    # your code to calculate total edit distance
    total_edits = 0
    total_lens = 0

    # loop: calculate edit distance
    total_edits += edit_distance
    total_lens += y_len


self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
self.log('val_wer', total_edits / total_lens, prog_bar=True, on_step=False, on_epoch=True)
return loss


def test_step(self, batch, batch_idx):
    x, y, x_lens, y_lens = batch
    # your code to calculate loss

    # your code to calculate total edit distance
    total_edits = 0
    total_lens = 0

    # loop: calculate edit distance
    total_edits += edit_distance
    total_lens += y_len


self.log('test_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
self.log('test_wer', total_edits / total_lens, prog_bar=True, on_step=False, on_epoch=True)
return loss


def configure_optimizers(self):
    return torch.optim.Adam(self.model.parameters(), lr=self.lr)


from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer

trainer = Trainer(
    callbacks=[
        LearningRateMonitor(logging_interval='step'),
        ModelCheckpoint(filename='{epoch}-{val_wer:.4f}', save_top_k=5, monitor='val_wer', mode='min'),
    ],
    logger=TensorBoardLogger('lightning_logs'),
    check_val_every_n_epoch=1,
    fast_dev_run=False,
    default_root_dir='checkpoint',
    deterministic=False,
    max_epochs=10,
    log_every_n_steps=50,
    devices=1,
)

vocab = Vocab('crohme_vocab.txt')
greedy_decoder = GreedyCTCDecoder(vocab)

# model = MathOnlineModel()
model = MathOnlineModel.load_from_checkpoint('path/to/your/checkpoint.ckpt', decoder=greedy_decoder)

dm = CROHMEDatamodule(root_dir='/kaggle/input/crohme2019/crohme2019',
                      train_annotation='/kaggle/input/crohme2019/crohme2019_train.txt',
                      validation_annotation='/kaggle/input/crohme2019/crohme2019_valid.txt',
                      test_annotation='/kaggle/input/crohme2019/crohme2019_test.txt',
                      vocab_file='crohme_vocab.txt',
                      batch_size=32,
                      num_workers=4
                      )

trainer.fit(model, dm)

# run test on validation dataset
# load your best val_wer model

from pytorch_lightning import Trainer

trainer = Trainer(
    devices=1,
)

# Load the model from a checkpoint
model = MathOnlineModel.load_from_checkpoint(
    'path/to/your_checkpoint.ckpt',
    decoder=GreedyCTCDecoder(dataset.vocab)
)

# Initialize the data module
dm = CROHMEDatamodule(
    root_dir='/kaggle/input/crohme2019/crohme2019',
    train_annotation='/kaggle/input/crohme2019/crohme2019_train.txt',
    validation_annotation='/kaggle/input/crohme2019/crohme2019_valid.txt',
    test_annotation='/kaggle/input/crohme2019/crohme2019_valid.txt',
    vocab_file='crohme_vocab.txt',
    batch_size=32,
    num_workers=4
)

# Test the model
trainer.test(model, datamodule=dm)