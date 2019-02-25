import torch
import logging
import codecs
import numpy as np
import gzip
logger = logging.getLogger(__name__)


class Embeddings(torch.nn.Module):
    def __init__(self, n_d, word2id, embs=None, fix_emb=True, oov='<oov>', pad='<pad>', normalize=True,
                 input_field_name=None):
        super(Embeddings, self).__init__()
        if embs is not None:
            embwords, embvecs = embs
            # for word in embwords:
            #  assert word not in word2id, "Duplicate words in pre-trained embeddings"
            #  word2id[word] = len(word2id)

            logger.info("{} word embeddings to be initialized.".format(len(word2id)))
            if n_d != len(embvecs[0]):
                logger.warning("[WARNING] n_d ({}) != word vector size ({}). Use {} for embeddings.".format(
                    n_d, len(embvecs[0]), len(embvecs[0])))
                n_d = len(embvecs[0])

        self.input_field_name = input_field_name
        self.word2id = word2id
        self.id2word = {i: word for word, i in word2id.items()}
        self.n_V, self.n_d = len(word2id), n_d
        self.oovid = word2id[oov]
        self.padid = word2id[pad]
        self.embedding = torch.nn.Embedding(self.n_V, n_d, padding_idx=self.padid)
        scale = np.sqrt(3.0 / n_d)
        self.embedding.weight.data.uniform_(-scale, scale)

        if embs is not None:
            emb_words, emb_vecs = embs
            weight = self.embedding.weight
            for emb_word, emb_vec in zip(emb_words, emb_vecs):
                if emb_word not in word2id:
                    continue
                i = word2id[emb_word]
                weight.data[i].copy_(torch.from_numpy(emb_vec))
            logger.info("embedding shape: {}".format(weight.size()))

        if normalize:
            weight = self.embedding.weight
            norms = weight.data.norm(2, 1)
            if norms.dim() == 1:
                norms = norms.unsqueeze(1)
            weight.data.div_(norms.expand_as(weight.data))

        if fix_emb:
            self.embedding.weight.requires_grad = False

    def forward(self, input_):
        return self.embedding(input_)

    def get_output_dim(self):
        return self.n_d


def load_embedding_npz(path: str):
    data = np.load(path)
    return [str(w) for w in data['words']], data['vals']


def load_embedding_txt(path: str, has_header: bool):
    words, vals = [], []
    if path.endswith('.gz'):
        fin = gzip.open(path, 'rb')
    else:
        fin = codecs.open(path, 'r', encoding='utf-8', errors='ignore')
    if has_header:
        fin.readline()

    dim = None
    cnt = 0
    for line in fin:
        cnt += 1
        line = line.strip()
        if line:
            parts = line.split()
            if dim is None:
                dim = len(parts[1:])
            elif dim != len(parts[1:]):
                logger.info('unequal number of fields in line {}: {}, expected {}'.format(cnt, len(parts[1:]), dim))
                continue
            words.append(parts[0])
            vals += [float(x) for x in parts[1:]]  # equal to append
    return words, np.asarray(vals).reshape(len(words), -1)  # reshape


def load_embedding(path: str, has_header: bool):
    if path.endswith(".npz"):
        return load_embedding_npz(path)
    else:
        return load_embedding_txt(path, has_header)
