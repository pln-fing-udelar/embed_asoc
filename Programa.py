import os
import codecs
import pickle
import argparse
import random
import torch as t
import numpy as np
import pandas as pd

import sklearn.metrics.pairwise
import io
import math
import matplotlib

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt

from torch import LongTensor as LT
from torch import FloatTensor as FT
import torch.nn as nn

from tqdm import tqdm
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from model import Word2Vec, SGNS
from scipy import stats

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data/', help="data directory path")
    parser.add_argument('--vocab', type=str, default='./data/corpus.txt', help="corpus path for building vocab")
    parser.add_argument('--corpus', type=str, default='./data/corpus.txt', help="corpus path")
    parser.add_argument('--unk', type=str, default='<UNK>', help="UNK token")
    parser.add_argument('--window', type=int, default=5, help="window size")
    parser.add_argument('--max_vocab', type=int, default=20000, help="maximum number of vocab")

    parser.add_argument('--name', type=str, default='sgns', help="model name")
    parser.add_argument('--save_dir', type=str, default='./pts/', help="model directory path")
    parser.add_argument('--e_dim', type=int, default=300, help="embedding dimension")
    parser.add_argument('--n_negs', type=int, default=20, help="number of negative samples")
    parser.add_argument('--epoch', type=int, default=100, help="number of epochs")
    parser.add_argument('--mb', type=int, default=4096, help="mini-batch size")
    parser.add_argument('--ss_t', type=float, default=1e-5, help="subsample threshold")
    parser.add_argument('--weights', action='store_true', help="use weights for negative sampling")
    parser.add_argument('--cuda', action='store_true', help="use CUDA")
    parser.add_argument('--conti', action='store_true', help="continue learning")
    
    parser.add_argument('--result_dir', type=str, default='./result/', help="result directory path")
    parser.add_argument('--model', type=str, default='tsne', choices=['pca', 'tsne'], help="model for visualization")
    parser.add_argument('--top_k', type=int, default=1000, help="scatter top-k words")

    return parser.parse_args()

class Bundler(nn.Module):

    def forward(self, data):
        raise NotImplementedError

    def forward_i(self, data):
        raise NotImplementedError

    def forward_o(self, data):
        raise NotImplementedError


class Word2Vec(Bundler):

    def __init__(self, vocab_size=20000, embedding_size=300, padding_idx=0):
        super(Word2Vec, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.ivectors = nn.Embedding(self.vocab_size, self.embedding_size, padding_idx=padding_idx)
        self.ovectors = nn.Embedding(self.vocab_size, self.embedding_size, padding_idx=padding_idx)
        self.ivectors.weight = nn.Parameter(t.cat([t.zeros(1, self.embedding_size), FT(self.vocab_size - 1, self.embedding_size).uniform_(-0.5 / self.embedding_size, 0.5 / self.embedding_size)]))
        self.ovectors.weight = nn.Parameter(t.cat([t.zeros(1, self.embedding_size), FT(self.vocab_size - 1, self.embedding_size).uniform_(-0.5 / self.embedding_size, 0.5 / self.embedding_size)]))
        self.ivectors.weight.requires_grad = True
        self.ovectors.weight.requires_grad = True

    def forward(self, data):
        return self.forward_i(data)

    def forward_i(self, data):
        v = LT(data)
        v = v.cuda() if self.ivectors.weight.is_cuda else v
        return self.ivectors(v)

    def forward_o(self, data):
        v = LT(data)
        v = v.cuda() if self.ovectors.weight.is_cuda else v
        return self.ovectors(v)


class SGNS(nn.Module):

    def __init__(self, embedding, vocab_size=20000, n_negs=20, weights=None):
        super(SGNS, self).__init__()
        self.embedding = embedding
        self.vocab_size = vocab_size
        self.n_negs = n_negs
        self.weights = None
        if weights is not None:
            wf = np.power(weights, 0.75)
            wf = wf / wf.sum()
            self.weights = FT(wf)

    def forward(self, iword, owords):
        batch_size = iword.size()[0]
        context_size = owords.size()[1]
        if self.weights is not None:
            nwords = t.multinomial(self.weights, batch_size * context_size * self.n_negs, replacement=True).view(batch_size, -1)
        else:
            nwords = FT(batch_size, context_size * self.n_negs).uniform_(0, self.vocab_size - 1).long()
        ivectors = self.embedding.forward_i(iword).unsqueeze(2)
        ovectors = self.embedding.forward_o(owords)
        nvectors = self.embedding.forward_o(nwords).neg()
        oloss = t.bmm(ovectors, ivectors).squeeze().sigmoid().log().mean(1)
        nloss = t.bmm(nvectors, ivectors).squeeze().sigmoid().log().view(-1, context_size, self.n_negs).sum(2).mean(1)
        return -(oloss + nloss).mean()


class Preprocess(object):

    def __init__(self, window=5, unk='<UNK>', data_dir='./data/'):
        self.window = window
        self.unk = unk
        self.data_dir = data_dir

    def skipgram(self, sentence, i):
        iword = sentence[i]
        left = sentence[max(i - self.window, 0): i]
        right = sentence[i + 1: i + 1 + self.window]
        return iword, [self.unk for _ in range(self.window - len(left))] + left + right + [self.unk for _ in range(self.window - len(right))]

    def build(self, filepath, max_vocab=20000):
        print("building vocab...")
        step = 0
        self.wc = {self.unk: 1}
        with codecs.open(filepath, 'r', encoding='utf-8') as file:
            for line in file:
                step += 1
                if not step % 1000:
                    print("working on {}kth line".format(step // 1000), end='\r')
                line = line.strip()
                if not line:
                    continue
                sent = line.split()
                for word in sent:
                    self.wc[word] = self.wc.get(word, 0) + 1
        print("")
        self.idx2word = [self.unk] + sorted(self.wc, key=self.wc.get, reverse=True)[:max_vocab - 1]
        self.word2idx = {self.idx2word[idx]: idx for idx, _ in enumerate(self.idx2word)}
        self.vocab = set([word for word in self.word2idx])
        pickle.dump(self.wc, open(os.path.join(self.data_dir, 'wc.dat'), 'wb'))
        pickle.dump(self.vocab, open(os.path.join(self.data_dir, 'vocab.dat'), 'wb'))
        pickle.dump(self.idx2word, open(os.path.join(self.data_dir, 'idx2word.dat'), 'wb'))
        pickle.dump(self.word2idx, open(os.path.join(self.data_dir, 'word2idx.dat'), 'wb'))
        print("build done")

    def convert(self, filepath):
        print("converting corpus...")
        step = 0
        data = []
        with codecs.open(filepath, 'r', encoding='utf-8') as file:
            for line in file:
                step += 1
                if not step % 1000:
                    print("working on {}kth line".format(step // 1000), end='\r')
                line = line.strip()
                if not line:
                    continue
                sent = []
                for word in line.split():
                    if word in self.vocab:
                        sent.append(word)
                    else:
                        sent.append(self.unk)
                for i in range(len(sent)):
                    iword, owords = self.skipgram(sent, i)
                    data.append((self.word2idx[iword], [self.word2idx[oword] for oword in owords]))
        print("")
        pickle.dump(data, open(os.path.join(self.data_dir, 'train.dat'), 'wb'))
        print("conversion done")


class PermutedSubsampledCorpus(Dataset):

    def __init__(self, datapath, ws=None):
        data = pickle.load(open(datapath, 'rb'))
        if ws is not None:
            self.data = []
            for iword, owords in data:
                if random.random() > ws[iword]:
                    self.data.append((iword, owords))
        else:
            self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        iword, owords = self.data[idx]
        return iword, np.array(owords)


def train(args,epoca,continuacion):
    idx2word = pickle.load(open(os.path.join(args.data_dir, 'idx2word.dat'), 'rb'))
    wc = pickle.load(open(os.path.join(args.data_dir, 'wc.dat'), 'rb'))
    wf = np.array([wc[word] for word in idx2word])
    wf = wf / wf.sum()
    ws = 1 - np.sqrt(args.ss_t / wf)
    ws = np.clip(ws, 0, 1)
    vocab_size = len(idx2word)
    weights = wf if args.weights else None
    if not os.path.isdir(args.save_dir):
        os.mkdir(args.save_dir)
    model = Word2Vec(vocab_size=vocab_size, embedding_size=args.e_dim)
    modelpath = os.path.join(args.save_dir, '{}.pt'.format(args.name))
    sgns = SGNS(embedding=model, vocab_size=vocab_size, n_negs=args.n_negs, weights=weights)
    if os.path.isfile(modelpath) and continuacion:
        sgns.load_state_dict(t.load(modelpath))
    if args.cuda:
        sgns = sgns.cuda()
    optim = Adam(sgns.parameters())
    optimpath = os.path.join(args.save_dir, '{}.optim.pt'.format(args.name))
    if os.path.isfile(optimpath) and continuacion:
        optim.load_state_dict(t.load(optimpath))
    for epoch in range(1, epoca + 1):
        dataset = PermutedSubsampledCorpus(os.path.join(args.data_dir, 'train.dat'))
        dataloader = DataLoader(dataset, batch_size=args.mb, shuffle=True)
        total_batches = int(np.ceil(len(dataset) / args.mb))
        pbar = tqdm(dataloader)
        pbar.set_description("[Epoch {}]".format(e))
        for iword, owords in pbar:
            loss = sgns(iword, owords)
            optim.zero_grad()
            loss.backward()
            optim.step()
            pbar.set_postfix(loss=loss.item())
    idx2vec = model.ivectors.weight.data.cpu().numpy()
    pickle.dump(idx2vec, open(os.path.join(args.data_dir, 'idx2vec.dat'), 'wb'))
    t.save(sgns.state_dict(), os.path.join(args.save_dir, '{}.pt'.format(args.name)))
    t.save(optim.state_dict(), os.path.join(args.save_dir, '{}.optim.pt'.format(args.name)))


def evaluate(args):
  wc = pickle.load(open('/clusteruy/home/fgomez/wc.dat', 'rb'))
  words = sorted(wc, key=wc.get, reverse=True)[:3104210]
  word2idx = pickle.load(open('/clusteruy/home/fgomez/word2idx.dat', 'rb'))
  idx2vec = pickle.load(open('/clusteruy/home/fgomez/idx2vec.dat', 'rb'))
  X = [idx2vec[word2idx[word]] for word in words]
  filePathSimlex = "/clusteruy/home/fgomez/simlex.csv"
  ManzanasSimLex = pd.read_csv(filePathSimlex, encoding='UTF-8')
  print("Generando Archivo de Similaridad...")
  lindiceSimlex1 = []
  ManzanasSimLexReal = []
  for n in range(0,1888):
    indice = -1
    try:
      indice = np.array(sklearn.metrics.pairwise.cosine_similarity(np.array(X[words.index(ManzanasSimLex['wordA'][n])].reshape(1,-1)),
      np.array(X[words.index(ManzanasSimLex['wordB'][n])].reshape(1,-1))),dtype=float)[0][0]
    except Exception:
      pass
    if indice==-1:
      pass
    else:
      lindiceSimlex1.append(indice)
      ManzanasSimLexReal.append(ManzanasSimLex['rating'][n])
  print("Fin del Proceso")
  rho, pval= stats.spearmanr(ManzanasSimLexReal, lindiceSimlex1)
  print(rho)
  simlex_bar.insert(epocas_glb,rho)
  r = rho
  num = 3320
  stderr = 1.0 / math.sqrt(num - 3)
  delta = 1.99 * stderr
  lower = math.tanh(math.atanh(r) - delta)
  upper = math.tanh(math.atanh(r) + delta)
  print(lower)
  print(upper)
  yer1.insert(epocas_glb,rho-lower)
  filePathAbstract = "/clusteruy/home/fgomez/similarityList.abstract.es.csv"
  ManzanasAbstract = pd.read_csv(filePathAbstract, encoding='UTF-8')
  print("Generando Archivo de Similaridad Abstract")
  lindiceAbslex1 = []
  ManzanasAbstrcReal = []
  #out_v = io.open('Similaridad.csv', 'w', encoding='utf-8')
  for n in range(0,3321):
    indice = -1
    try:
      indice = np.array(sklearn.metrics.pairwise.cosine_similarity(np.array(X[words.index(ManzanasAbstract['wordA'][n])].reshape(1,-1)),
      np.array(X[words.index(ManzanasAbstract['wordB'][n])].reshape(1,-1))),dtype=float)[0][0]
    except Exception:
      pass
    if indice==-1:
      pass
    else:
      lindiceAbslex1.append(indice)
      ManzanasAbstrcReal.append(ManzanasAbstract['rating'][n])  
  rho, pval= stats.spearmanr(ManzanasAbstrcReal, lindiceAbslex1)
  abstract_bar.insert(epocas_glb,rho)
  r = rho
  num = 3320
  stderr = 1.0 / math.sqrt(num - 3)
  delta = 1.99 * stderr
  lower = math.tanh(math.atanh(r) - delta)
  upper = math.tanh(math.atanh(r) + delta)
  print(rho)
  print(lower)
  print(upper)
  yer2.insert(epocas_glb,rho-lower)
  filePathConcrete = "/clusteruy/home/fgomez/similarityList.concrete.es.csv"
  ManzanasConcrete = pd.read_csv(filePathConcrete, encoding='UTF-8')
  print("Generando Archivo de Similaridad Concrete")
  lindiceConcrete1 = []
  ManzanasConcreteReal = []
  for n in range(0,3321):
    indice = -1
    try:
      indice = np.array(sklearn.metrics.pairwise.cosine_similarity(np.array(X[words.index(ManzanasConcrete['wordA'][n])].reshape(1,-1)),
      np.array(X[words.index(ManzanasConcrete['wordB'][n])].reshape(1,-1))),dtype=float)[0][0]
    except Exception:
	    pass
    if indice==-1:
 	    pass
    else:
 	    lindiceConcrete1.append(indice)
 	    ManzanasConcreteReal.append(ManzanasConcrete['rating'][n])
  rho, pval= stats.spearmanr(ManzanasConcreteReal, lindiceConcrete1)
  concrete_bar.insert(epocas_glb,rho)
  r = rho
  num = 3320
  stderr = 1.0 / math.sqrt(num - 3)
  delta = 1.99 * stderr
  lower = math.tanh(math.atanh(r) - delta)
  upper = math.tanh(math.atanh(r) + delta)
  print(rho)
  print(lower)
  print(upper)
  yer3.insert(epocas_glb,rho-lower)
  test = plt.figure()
  barWidth = 0.3
  r1 = np.arange(len(simlex_bar))
  r2 = [x + barWidth for x in r1]
  r3 = [x + barWidth for x in r2]
  # Create blue bars
  plt.bar(r1, simlex_bar, width = barWidth, color = 'blue', edgecolor = 'black', yerr=yer1, capsize=7, label='Simlex')
  # Create cyan bars
  plt.bar(r2, abstract_bar, width = barWidth, color = 'cyan', edgecolor = 'black', yerr=yer2, capsize=7, label='Abstract')
  # Create cyan bars
  plt.bar(r3, concrete_bar, width = barWidth, color = 'red', edgecolor = 'black', yerr=yer3, capsize=7, label='Concrete')
  plt.xticks([r + barWidth for r in range(len(simlex_bar))], eje_x_epocas)
  plt.ylabel('height')
  plt.legend()
  plt.figure(figsize=(50,50))
  test.savefig(os.path.join(args.data_dir, "resultados") + '.png')

def plot(args):
    wc = pickle.load(open(os.path.join(args.data_dir, 'wc.dat'), 'rb'))
    words = sorted(wc, key=wc.get, reverse=True)[:args.top_k]
    if args.model == 'pca':
        model = PCA(n_components=2)
    elif args.model == 'tsne':
        model = TSNE(n_components=2, perplexity=30, init='pca', method='exact', n_iter=5000)
    word2idx = pickle.load(open(os.path.join(args.data_dir, 'word2idx.dat'), 'rb'))
    idx2vec = pickle.load(open(os.path.join(args.data_dir, 'idx2vec.dat'), 'rb'))
    X = [idx2vec[word2idx[word]] for word in words]
    X = model.fit_transform(X)
    plt.figure(figsize=(18, 18))
    for i in range(len(X)):
        plt.text(X[i, 0], X[i, 1], words[i], bbox=dict(facecolor='blue', alpha=0.1))
    plt.xlim((np.min(X[:, 0]), np.max(X[:, 0])))
    plt.ylim((np.min(X[:, 1]), np.max(X[:, 1])))
    if not os.path.isdir(args.result_dir):
        os.mkdir(args.result_dir)
    plt.savefig(os.path.join(args.result_dir, args.model) + '.png')
    
    
if __name__ == '__main__':
    args = parse_args()
    preprocess = Preprocess(window=args.window, unk=args.unk, data_dir=args.data_dir)
    preprocess.build(args.vocab, max_vocab=args.max_vocab)
    preprocess.convert(args.corpus)
    epocas_glb=args.epoch
    simlex_bar = []
    abstract_bar = []
    concrete_bar = []
    yer1 = []
    yer2 = []
    yer3 = []
    eje_x_epocas = []
    if args.conti:
      continuacion=True
    else:
      continuacion=False
    if epocas_glb>1:
      for e in range(0,epocas_glb):
        epoca=1
        if e>1:
          continuacion=True
        train(parse_args(),epoca,continuacion)
        eje_x_epocas.insert(e,"Epoca {}".format(e))
        evaluate(parse_args())
    matplotlib.rc('font', family='AppleGothic')
    plot(parse_args())
