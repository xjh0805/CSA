import torch
import torch.utils.data as data
import numpy as np
import nltk
import pickle
import os
from PIL import Image


class PrecompDataset(data.Dataset):
    """
    Load precomputed captions, image features, and co-occurrence matrices
    """

    def __init__(self, data_path, data_split, vocab):
        self.vocab = vocab
        loc = data_path + '/'

        # Load captions
        self.captions = []
        with open(loc + '%s_caps.txt' % data_split, 'rb') as f:
            for line in f:
                self.captions.append(line.strip())

        # Load image features
        self.images = np.load(loc + '%s_ims.npy' % data_split)

        # Load co-occurrence matrices from .pkl files
        with open(loc + '%s_co_matrix_img.pkl' % data_split, 'rb') as f:
            self.co_matrix_img = pickle.load(f)

        with open(loc + '%s_co_matrix_txt.pkl' % data_split, 'rb') as f:
            self.co_matrix_txt = pickle.load(f)

        self.length = len(self.captions)

        # Adjust for dataset redundancy
        if self.images.shape[0] != self.length:
            self.im_div = 5
        else:
            self.im_div = 1

        # Reduce validation set size
        if data_split == 'dev':
            self.length = 5000

    def __getitem__(self, index):
        # Handle image redundancy
        img_id = int(index / self.im_div)
        image = torch.Tensor(self.images[img_id])
        caption = self.captions[index]
        vocab = self.vocab

        # Convert caption (string) to word ids
        tokens = nltk.tokenize.word_tokenize(str(caption).lower())
        caption = [vocab('<start>')] + [vocab(token) for token in tokens] + [vocab('<end>')]
        target = torch.Tensor(caption)

        # Get corresponding co-occurrence matrix rows
        co_matrix_img_row = torch.Tensor(self.co_matrix_img[img_id])
        co_matrix_txt_row = torch.Tensor(self.co_matrix_txt[index])

        return image, target, index, img_id, co_matrix_img_row, co_matrix_txt_row

    def __len__(self):
        return self.length


def collate_fn(data):
    """Build mini-batch tensors from a list of (image, caption, co_matrix_img, co_matrix_txt) tuples."""
    # Sort a data list by caption length
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions, ids, img_ids, co_matrix_img, co_matrix_txt = zip(*data)

    # Merge images into a single tensor
    images = torch.stack(images, 0)

    # Merge captions
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]

    # Convert co-occurrence matrices to tensors
    co_matrix_img = torch.stack(co_matrix_img, 0)
    co_matrix_txt = torch.stack(co_matrix_txt, 0)

    return images, targets, lengths, ids, co_matrix_img, co_matrix_txt


def get_precomp_loader(data_path, data_split, vocab, opt, batch_size=100,
                       shuffle=True, num_workers=2):
    """Returns torch.utils.data.DataLoader for custom dataset."""
    dset = PrecompDataset(data_path, data_split, vocab)

    data_loader = torch.utils.data.DataLoader(dataset=dset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              pin_memory=True,
                                              collate_fn=collate_fn)
    return data_loader


def get_loaders(data_name, vocab, batch_size, workers, opt):
    dpath = os.path.join(opt.data_path, data_name)
    train_loader = get_precomp_loader(dpath, 'train', vocab, opt,
                                      batch_size, True, workers)
    val_loader = get_precomp_loader(dpath, 'dev', vocab, opt,
                                    batch_size, False, workers)
    return train_loader, val_loader


def get_test_loader(split_name, data_name, vocab, batch_size,
                    workers, opt):
    dpath = os.path.join(opt.data_path, data_name)
    test_loader = get_precomp_loader(dpath, split_name, vocab, opt,
                                     batch_size, False, workers)
    return test_loader
