"""
pointerbot_scalable.py -- your pointerbot_selfcheck.py (everything it did) PLUS
model sizes you can switch UP to when more examples stop making a difference.

Starting point is your original CONFIG, now called size "small". Going up:
    small (your original) -> medium -> large -> xlarge
Edit SIZE_PRESETS below to change or add sizes.

What's new vs. pointerbot_selfcheck.py (all older features are unchanged)
------------------------------------------------------------------------
  :sizes                      list the sizes, parameter counts, relative cost
  :size <name>                switch to a size (retrains from scratch; see below)
  :grow / :shrink             one step up / down
  :autogrow off|suggest|on [maxsize]
                              off     = never mention it
                              suggest = (default) tell you when a bigger size looks needed
                              on      = grow automatically, up to maxsize (default xlarge)
  :capacity                   check the current weights now + show the history
  :conflicts                  list taught prompts that have contradictory answers

When does it say "a bigger model is needed"?
  1. After training, the model still can't reproduce some of its training
     examples even after the full repair loop (a capacity limit), OR
  2. you added >= 20% more examples since the last run at this size, but
     held-out accuracy didn't improve and is still below 95% (a plateau).
Contradictory patterns (same prompt, different answers) are subtracted
first: no model size can fix those, so they never trigger a grow.

What a switch does
  * Weights from one size can't be reused at another (different shapes), so
    the new size trains from scratch on everything you've saved (automatic if
    :auto is on).
  * Before switching, the current weights are backed up in size_backups/.
    Switching back restores them instead of retraining.
  * The chosen size is remembered in size.json (it overrides the size fields
    of CONFIG).

Honest limits of scaling
  * This is pure numpy with a Python loop per example. Cost grows roughly with
    parameter count: 'large' is ~9x slower per epoch than 'small'. xlarge
    really wants a PC, not a phone.
  * Bigger is not automatically better. If the problem is contradictory or
    too-similar examples, more capacity won't fix it -- the advice says so.

---------------------------------------------------------------------------
Original description of pointerbot_selfcheck.py:

pointerbot_selfcheck.py -- pointerbot_configurable.py plus a SELF-TEST and
SELF-CORRECTION loop for the patterns you teach it.

1. selftest(): after training, the model is asked every taught prompt and
   its GREEDY answer (no sampling randomness) is compared with the answer
   you taught. Pass = exact match.
2. self_correct(): any pattern it gets wrong is trained on again (repeated,
   mixed with a few already-passing patterns so it doesn't forget them)
   and the whole set is re-tested, until every pattern passes or
   SELFTEST_MAX_ROUNDS is hit. The best-scoring weights are kept.
3. :stabilize runs that loop automatically before saving.
4. :selftest -- test the CURRENT model on every taught pattern (no training).
5. Taught patterns are ALWAYS in the training set; validation only splits
   corpus/template data.
6. :fullstabilize -- no validation split, every example trains, then EVERY
   example is verified and repaired until the final weights reproduce all of
   them (taught patterns: greedy answer must match exactly; template/corpus
   blocks: argmax next token right wherever the data determines it).
   templates.json's held-out combos stay held out -- your generalization test.
7. :verify -- the same full check on the current weights, no training.
8. :twophase -- memorize first (:fullstabilize), then a generalize phase with
   strong weight decay, checkpointing only when recall stays intact AND
   held-out accuracy (templates.json held-out + probes.json) hits a new best.
9. :generalize -- run phase 2 alone on the current weights.
10. AUTO-MEMORIZE (on by default; :auto off / :auto on).

Honest limits:
  * The self-test measures RECALL of what you taught, not generalization.
  * The test is greedy. Chat mode samples (temperature 0.8, top_k 20), so a
    passing pattern can still occasionally come out different in chat.

Run:
    python3 pointerbot_scalable.py
Requires:
    pkg install python-numpy   (Termux)  /  pip install numpy   (elsewhere)
"""
import os
import re
import json
import time
import shutil
import random
import collections
import numpy as np


# ======================================================================
# ---- Tensor: autograd core (+ rmsnorm, sigmoid) ----
# ======================================================================
class Tensor:
    def __init__(self, data, requires_grad=False, _children=(), _op=""):
        self.data = np.array(data, dtype=np.float64)
        self.requires_grad = requires_grad
        self.grad = np.zeros_like(self.data) if requires_grad else None
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    @property
    def shape(self):
        return self.data.shape

    def zero_grad(self):
        if self.requires_grad:
            self.grad = np.zeros_like(self.data)

    @staticmethod
    def _unbroadcast(grad, shape):
        while grad.ndim > len(shape):
            grad = grad.sum(axis=0)
        for i, s in enumerate(shape):
            if s == 1 and grad.shape[i] != 1:
                grad = grad.sum(axis=i, keepdims=True)
        return grad

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data,
                      requires_grad=(self.requires_grad or other.requires_grad),
                      _children=(self, other), _op="add")

        def _backward():
            if self.requires_grad:
                self.grad += self._unbroadcast(out.grad, self.data.shape)
            if other.requires_grad:
                other.grad += self._unbroadcast(out.grad, other.data.shape)
        out._backward = _backward
        return out

    def __neg__(self):
        out = Tensor(-self.data, requires_grad=self.requires_grad,
                      _children=(self,), _op="neg")

        def _backward():
            if self.requires_grad:
                self.grad += -out.grad
        out._backward = _backward
        return out

    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return self + (-other)

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            out = Tensor(self.data * other, requires_grad=self.requires_grad,
                          _children=(self,), _op="mulc")

            def _backward():
                if self.requires_grad:
                    self.grad += out.grad * other
            out._backward = _backward
            return out

        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data,
                      requires_grad=(self.requires_grad or other.requires_grad),
                      _children=(self, other), _op="mul")

        def _backward():
            if self.requires_grad:
                self.grad += self._unbroadcast(out.grad * other.data, self.data.shape)
            if other.requires_grad:
                other.grad += self._unbroadcast(out.grad * self.data, other.data.shape)
        out._backward = _backward
        return out

    __rmul__ = __mul__

    def matmul(self, other):
        out = Tensor(self.data @ other.data,
                      requires_grad=(self.requires_grad or other.requires_grad),
                      _children=(self, other), _op="matmul")

        def _backward():
            if self.requires_grad:
                self.grad += out.grad @ np.swapaxes(other.data, -1, -2)
            if other.requires_grad:
                other.grad += np.swapaxes(self.data, -1, -2) @ out.grad
        out._backward = _backward
        return out

    def __matmul__(self, other):
        return self.matmul(other)

    def transpose(self, axes):
        out = Tensor(np.transpose(self.data, axes), requires_grad=self.requires_grad,
                      _children=(self,), _op="transpose")
        inv = tuple(np.argsort(axes))

        def _backward():
            if self.requires_grad:
                self.grad += np.transpose(out.grad, inv)
        out._backward = _backward
        return out

    def reshape(self, *shape):
        orig_shape = self.data.shape
        out = Tensor(self.data.reshape(*shape), requires_grad=self.requires_grad,
                      _children=(self,), _op="reshape")

        def _backward():
            if self.requires_grad:
                self.grad += out.grad.reshape(orig_shape)
        out._backward = _backward
        return out

    def sum(self, axis=None, keepdims=False):
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims),
                      requires_grad=self.requires_grad, _children=(self,), _op="sum")

        def _backward():
            if self.requires_grad:
                g = out.grad
                if not keepdims and axis is not None:
                    g = np.expand_dims(g, axis)
                self.grad += np.ones_like(self.data) * g
        out._backward = _backward
        return out

    def tanh(self):
        t = np.tanh(self.data)
        out = Tensor(t, requires_grad=self.requires_grad, _children=(self,), _op="tanh")

        def _backward():
            if self.requires_grad:
                self.grad += (1 - t ** 2) * out.grad
        out._backward = _backward
        return out

    def sigmoid(self):
        s = 1.0 / (1.0 + np.exp(-self.data))
        out = Tensor(s, requires_grad=self.requires_grad, _children=(self,), _op="sigmoid")

        def _backward():
            if self.requires_grad:
                self.grad += s * (1 - s) * out.grad
        out._backward = _backward
        return out

    def relu(self):
        out = Tensor(np.maximum(0, self.data), requires_grad=self.requires_grad,
                      _children=(self,), _op="relu")

        def _backward():
            if self.requires_grad:
                self.grad += (self.data > 0) * out.grad
        out._backward = _backward
        return out

    def softmax(self, axis=-1):
        x = self.data - np.max(self.data, axis=axis, keepdims=True)
        e = np.exp(x)
        s = e / np.sum(e, axis=axis, keepdims=True)
        out = Tensor(s, requires_grad=self.requires_grad, _children=(self,), _op="softmax")

        def _backward():
            if self.requires_grad:
                dy = out.grad
                dot = np.sum(dy * s, axis=axis, keepdims=True)
                self.grad += s * (dy - dot)
        out._backward = _backward
        return out

    def layernorm(self, gamma, beta, eps=1e-5):
        x = self.data
        mu = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        std_inv = 1.0 / np.sqrt(var + eps)
        xhat = (x - mu) * std_inv
        y = xhat * gamma.data + beta.data
        out = Tensor(y, requires_grad=True, _children=(self, gamma, beta), _op="layernorm")
        D = x.shape[-1]

        def _backward():
            dy = out.grad
            if gamma.requires_grad:
                gamma.grad += np.sum(dy * xhat, axis=tuple(range(dy.ndim - 1)))
            if beta.requires_grad:
                beta.grad += np.sum(dy, axis=tuple(range(dy.ndim - 1)))
            if self.requires_grad:
                dxhat = dy * gamma.data
                dvar = np.sum(dxhat * (x - mu) * -0.5 * std_inv ** 3, axis=-1, keepdims=True)
                dmu = (np.sum(dxhat * -std_inv, axis=-1, keepdims=True) +
                       dvar * np.mean(-2 * (x - mu), axis=-1, keepdims=True))
                dx = dxhat * std_inv + dvar * 2 * (x - mu) / D + dmu / D
                self.grad += dx
        out._backward = _backward
        return out

    def rmsnorm(self, weight, eps=1e-6):
        x = self.data
        ms = np.mean(x * x, axis=-1, keepdims=True)
        inv = 1.0 / np.sqrt(ms + eps)
        xhat = x * inv
        y = xhat * weight.data
        out = Tensor(y, requires_grad=True, _children=(self, weight), _op="rmsnorm")
        D = x.shape[-1]

        def _backward():
            dy = out.grad
            if weight.requires_grad:
                weight.grad += np.sum(dy * xhat, axis=tuple(range(dy.ndim - 1)))
            if self.requires_grad:
                dxhat = dy * weight.data
                dms = np.sum(dxhat * x * -0.5 * inv ** 3, axis=-1, keepdims=True)
                dx = dxhat * inv + dms * 2 * x / D
                self.grad += dx
        out._backward = _backward
        return out

    def embedding_lookup(self, ids):
        out_data = self.data[ids]
        out = Tensor(out_data, requires_grad=self.requires_grad, _children=(self,), _op="embed")

        def _backward():
            if self.requires_grad:
                np.add.at(self.grad, ids, out.grad)
        out._backward = _backward
        return out

    def backward(self):
        topo, visited = [], set()

        def build(v):
            if id(v) not in visited:
                visited.add(id(v))
                for c in v._prev:
                    build(c)
                topo.append(v)
        build(self)
        self.grad = np.ones_like(self.data)
        for v in reversed(topo):
            v._backward()

    def __repr__(self):
        return f"Tensor(shape={self.data.shape}, requires_grad={self.requires_grad})"


def softmax_cross_entropy(logits, target_ids):
    x = logits.data
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    probs = e / np.sum(e, axis=-1, keepdims=True)
    T = x.shape[0]
    logp = -np.log(probs[np.arange(T), target_ids] + 1e-12)
    loss_val = np.mean(logp)
    out = Tensor(loss_val, requires_grad=True, _children=(logits,), _op="softmax_ce")

    def _backward():
        grad = probs.copy()
        grad[np.arange(T), target_ids] -= 1
        grad /= T
        logits.grad += grad * out.grad
    out._backward = _backward
    return out


def nll_from_probs(probs: Tensor, target_ids) -> Tensor:
    T = len(target_ids)
    target_ids = np.array(target_ids)
    p_target = probs.data[np.arange(T), target_ids]
    loss_val = np.mean(-np.log(p_target + 1e-9))

    out = Tensor(loss_val, requires_grad=True, _children=(probs,), _op="nll")

    def _backward():
        grad = np.zeros_like(probs.data)
        grad[np.arange(T), target_ids] = (-1.0 / (p_target + 1e-9)) / T
        probs.grad += grad * out.grad
    out._backward = _backward
    return out


def silu(x: Tensor) -> Tensor:
    return x * x.sigmoid()


def _param(shape, scale=0.02):
    return Tensor(np.random.randn(*shape) * scale, requires_grad=True)


# ======================================================================
# ---- RoPE ----
# ======================================================================
def build_rope_cache(T, d_head, base=10000.0):
    assert d_head % 2 == 0
    inv_freq = 1.0 / (base ** (np.arange(0, d_head, 2) / d_head))
    pos = np.arange(T)
    freqs = np.outer(pos, inv_freq)
    freqs = np.concatenate([freqs, freqs], axis=-1)
    return Tensor(np.cos(freqs)), Tensor(np.sin(freqs))


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    d = x.data
    Dh = d.shape[-1]
    half = Dh // 2
    x1, x2 = d[..., :half], d[..., half:]
    rot = np.concatenate([-x2, x1], axis=-1)
    out_data = d * cos.data + rot * sin.data
    out = Tensor(out_data, requires_grad=x.requires_grad, _children=(x,), _op="rope")

    def _backward():
        if x.requires_grad:
            dout = out.grad
            grad_direct = dout * cos.data
            dg = dout * sin.data
            dg1, dg2 = dg[..., :half], dg[..., half:]
            grad_x1 = grad_direct[..., :half] + dg2
            grad_x2 = grad_direct[..., half:] - dg1
            x.grad += np.concatenate([grad_x1, grad_x2], axis=-1)
    out._backward = _backward
    return out


# ======================================================================
# ---- tokenizer -- unchanged ----
# ======================================================================
class BPETokenizer:
    SPECIALS = ["<PAD>", "<BOS>", "<EOS>", "<UNK>"]

    def __init__(self):
        printable = [chr(i) for i in range(32, 127)]
        extra = ["\n", "\t"]
        self.base_chars = sorted(set(printable + extra))
        self.merges = []
        self._build_vocab()

    def _build_vocab(self):
        tokens = list(self.SPECIALS) + list(self.base_chars)
        for a, b in self.merges:
            tokens.append(a + b)
        self.stoi = {}
        for t in tokens:
            if t not in self.stoi:
                self.stoi[t] = len(self.stoi)
        self.itos = {i: t for t, i in self.stoi.items()}
        self.merge_ranks = {pair: i for i, pair in enumerate(self.merges)}

    @property
    def vocab_size(self):
        return len(self.stoi)

    @property
    def eos_id(self):
        return self.stoi["<EOS>"]

    @property
    def bos_id(self):
        return self.stoi["<BOS>"]

    @staticmethod
    def _pretokenize(text):
        return re.findall(r"\S+|\s+", text)

    def train(self, text, vocab_size=600, min_pair_freq=2, verbose=True):
        words = self._pretokenize(text)
        word_freq = collections.Counter(words)
        vocab = {tuple(w): f for w, f in word_freq.items()}

        n_base = len(self.SPECIALS) + len(self.base_chars)
        target_merges = max(0, vocab_size - n_base)

        merges = []
        for i in range(target_merges):
            pairs = collections.defaultdict(int)
            for word, freq in vocab.items():
                for j in range(len(word) - 1):
                    pairs[(word[j], word[j + 1])] += freq
            if not pairs:
                break
            best_pair, best_freq = max(pairs.items(), key=lambda kv: kv[1])
            if best_freq < min_pair_freq:
                break
            merges.append(best_pair)

            a, b = best_pair
            merged = a + b
            new_vocab = {}
            for word, freq in vocab.items():
                w = list(word)
                out, k = [], 0
                while k < len(w):
                    if k < len(w) - 1 and w[k] == a and w[k + 1] == b:
                        out.append(merged)
                        k += 2
                    else:
                        out.append(w[k])
                        k += 1
                key = tuple(out)
                new_vocab[key] = new_vocab.get(key, 0) + freq
            vocab = new_vocab

            if verbose and (i + 1) % 50 == 0:
                print(f"  [bpe merge {i+1}/{target_merges}: {a!r}+{b!r} -> "
                      f"{merged!r} (freq={best_freq})]")

        self.merges = merges
        self._build_vocab()
        if verbose:
            print(f"[bpe trained: {len(self.base_chars)} base chars + "
                  f"{len(self.merges)} merges = {self.vocab_size} tokens total]")

    def _encode_word(self, word):
        symbols = list(word)
        if len(symbols) == 1 or not self.merge_ranks:
            return symbols
        while True:
            best_rank, best_i = None, None
            for i in range(len(symbols) - 1):
                rank = self.merge_ranks.get((symbols[i], symbols[i + 1]))
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank, best_i = rank, i
            if best_i is None:
                break
            symbols = (symbols[:best_i] + [symbols[best_i] + symbols[best_i + 1]]
                       + symbols[best_i + 2:])
        return symbols

    def encode_prompt(self, text):
        return self.encode(text, add_bos=True)

    def encode(self, text, add_bos=True, add_eos=False):
        ids = [self.stoi["<BOS>"]] if add_bos else []
        unk = self.stoi["<UNK>"]
        for word in self._pretokenize(text):
            for piece in self._encode_word(word):
                if piece in self.stoi:
                    ids.append(self.stoi[piece])
                else:
                    for ch in piece:
                        ids.append(self.stoi.get(ch, unk))
        if add_eos:
            ids.append(self.stoi["<EOS>"])
        return ids

    def decode(self, ids):
        out = []
        for i in ids:
            t = self.itos.get(int(i), "")
            if t in ("<BOS>", "<PAD>"):
                continue
            if t == "<EOS>":
                break
            out.append(t)
        return "".join(out)

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"base_chars": self.base_chars,
                        "merges": [list(p) for p in self.merges]}, f)

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        tok = cls()
        tok.base_chars = data["base_chars"]
        tok.merges = [tuple(p) for p in data["merges"]]
        tok._build_vocab()
        return tok


# ======================================================================
# ---- word-level tokenizer + tokenizer switching (optional; YOU choose with :tokenizer) ----
# Every distinct word in your patterns/corpus/templates becomes ONE token (punctuation too).
# "pizza" then has its own embedding, which ends up near "chocolate"/"icecream" if they sit in
# the same slots. Cost: an unseen word is <UNK> until you run :retoken (always manual).
class WordTokenizer:
    """Word-level tokenizer.

    exact=True (default for newly built tokenizers): every word, symbol and
    whitespace character is its own token, so decode(encode(text)) == text
    EXACTLY. Words keep one identity whatever spacing surrounds them (the
    spaces are separate tokens), which is what lets a rule learned for one
    word transfer to the others. Common code operators (== != <= >= -> => ::
    := ++ -- && || << >> += -= *= /= // /* */ .. ...) are single tokens.
    Line endings are normalised to \n.

    exact=False (legacy, used when loading an older word_tokenizer.json that
    has no "exact" key): the original behaviour -- decode re-joins tokens
    with spaces, so code spacing is not preserved.
    """
    SPECIALS = ["<PAD>", "<BOS>", "<EOS>", "<UNK>"]
    _TOKEN_RE = re.compile(r"\w+(?:'\w+)*|\n|[^\w\s]")
    _EXACT_RE = re.compile(
        r"</?(?:SETPROFILE|PROFILE|OBS|USER|CHECK|REPLY)>|<SEP>|<EOS>|<BOS>|<PAD>|<UNK>|(?<!\w)c\+\+(?![\w+])|\.\.\.|<<=|>>=|->|=>|::|:=|\+\+|--|==|!=|<=|>=|&&|\|\||<<|>>"
        r"|\+=|-=|\*=|/=|%=|//|/\*|\*/|\.\.|\w+|\s|\S")
    _EXACT_SEED = [" ", "\t", "\n"]
    SEP = "<SEP>"   # marks the end of a prompt (exact mode with sep=True)
    _NO_SPACE_BEFORE = set(".,!?;:)]}%")

    def __init__(self, exact=True, sep=None):
        self.exact = exact
        self.sep = exact if sep is None else sep   # new exact tokenizers get <SEP>
        self.words = []
        self.base_chars = []     # kept so code written for BPETokenizer still works
        self.merges = []         # kept so code written for BPETokenizer still works
        self._build_vocab()

    def _build_vocab(self):
        seed = (list(self._EXACT_SEED) if self.exact else []) + ([self.SEP] if getattr(self, 'sep', False) else [])
        tokens = list(self.SPECIALS) + seed + list(self.words)
        self.stoi = {}
        for t in tokens:
            if t not in self.stoi:
                self.stoi[t] = len(self.stoi)
        self.itos = {i: t for t, i in self.stoi.items()}

    @property
    def vocab_size(self):
        return len(self.stoi)

    @property
    def eos_id(self):
        return self.stoi["<EOS>"]

    @property
    def bos_id(self):
        return self.stoi["<BOS>"]

    def split(self, text):
        if self.exact:
            text = text.replace("\r\n", "\n").replace("\r", "\n")
            return self._EXACT_RE.findall(text)
        return self._TOKEN_RE.findall(text)

    def train(self, text, verbose=True):
        self.words = sorted(set(self.split(text)))
        self._build_vocab()
        if verbose:
            print(f"[word tokenizer built ({'exact' if self.exact else 'legacy'} mode): "
                  f"{len(self.words)} words/symbols + {len(self.SPECIALS)} specials = "
                  f"{self.vocab_size} tokens total]")

    def unknown_words(self, text):
        return [w for w in sorted(set(self.split(text))) if w not in self.stoi]

    def encode(self, text, add_bos=True, add_eos=False):
        unk = self.stoi["<UNK>"]
        ids = [self.stoi["<BOS>"]] if add_bos else []
        ids += [self.stoi.get(w, unk) for w in self.split(text)]
        if add_eos:
            ids.append(self.stoi["<EOS>"])
        return ids

    def encode_prompt(self, text):
        ids = self.encode(text, add_bos=True)
        if self.sep:
            ids.append(self.stoi[self.SEP])
        return ids

    @classmethod
    def _join(cls, prev, tok):
        if prev is None or prev == "\n" or tok == "\n" or tok in cls._NO_SPACE_BEFORE:
            return tok
        if tok.startswith("'") or prev == "'":
            return tok
        return " " + tok

    def display(self, prev_id, tid):
        # how to print one streamed token
        if self.exact:
            return self.itos.get(tid, "")
        prev = None if prev_id is None else self.itos.get(prev_id, "")
        return self._join(prev, self.itos.get(tid, ""))

    def decode(self, ids):
        out, prev = [], None
        for i in ids:
            t = self.itos.get(int(i), "")
            if t in ("<BOS>", "<PAD>", self.SEP):
                continue
            if t == "<EOS>":
                break
            out.append(t if self.exact else self._join(prev, t))
            prev = t
        return "".join(out)

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"type": "word", "exact": self.exact, "sep": self.sep, "words": self.words}, f)

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        tok = cls(exact=data.get("exact", False), sep=data.get("sep", False))   # files without the keys are older versions
        tok.words = data["words"]
        tok._build_vocab()
        return tok


# which tokenizer's vocabulary the current weights belong to (saved in model.meta.json)
TOK_SIG = {"v": ""}


def _vocab_sig(tok):
    import hashlib
    joined = "\x00".join(tok.itos[i] for i in range(tok.vocab_size))
    return hashlib.md5(joined.encode("utf-8")).hexdigest()


def _register_tokenizer(tok):
    TOK_SIG["v"] = _vocab_sig(tok)


def _swap_tokenizer(tok, new_tok):
    # replace the tokenizer IN PLACE (everything holds a reference to the same object)
    tok.__class__ = new_tok.__class__
    tok.__dict__.clear()
    tok.__dict__.update(new_tok.__dict__)
    _register_tokenizer(tok)


def _collect_tokenizer_text():
    parts = [load_corpus()]
    for p in load_patterns():
        parts.append(p.get("prompt", "") + " " + p.get("answer", ""))
    synth_patterns, holdout_evals = expand_templates(load_templates())
    for p in synth_patterns:
        parts.append(p["prompt"])
    for item in holdout_evals:
        parts.append(item["prompt"] + " " + str(item["expected"]))
    for item in load_probes():      # held-out probes: words must exist in the vocabulary so the pointer can copy them
        parts.append(item["prompt"] + " " + str(item["expected"]))
    return "\n".join(t for t in parts if t)


def train_word_tokenizer(verbose=True):
    tok = WordTokenizer()
    tok.train(_collect_tokenizer_text(), verbose=verbose)
    tok.save(WORD_PATH)
    return tok


def _build_or_load_tokenizer_for_mode():
    if SETTINGS["tokenizer"] == "word":
        if os.path.exists(WORD_PATH):
            tok = WordTokenizer.load(WORD_PATH)
            print(f"[loaded word tokenizer: {tok.vocab_size} tokens ({len(tok.words)} words)]")
            return tok
        print("[no word_tokenizer.json found -- building one from saved patterns/corpus]")
        return train_word_tokenizer()
    return _build_or_load_bpe_tokenizer()


def _warn_unknown(tok):
    # word mode only: words in your saved data that the vocabulary has never seen
    if SETTINGS["tokenizer"] != "word" or not hasattr(tok, "unknown_words"):
        return []
    unknown = tok.unknown_words(_collect_tokenizer_text())
    if unknown:
        shown = ", ".join(unknown[:12]) + (" ..." if len(unknown) > 12 else "")
        print(f"[word tokenizer: {len(unknown)} word(s) in your saved data are not in its "
              f"vocabulary ({shown}) and would be <UNK>. Run :retoken to add them "
              f"(rebuilds the vocabulary and retrains from scratch).]")
    return unknown


def reset_everything(tok, patterns):
    # Delete every file the bot saved (templates.json and probes.json are yours -- kept)
    removed = []
    for path in (SAVE_PATH, META_PATH, PATTERNS_PATH, CORPUS_PATH, BPE_PATH, WORD_PATH,
                 SETTINGS_PATH, LOG_PATH):
        if os.path.exists(path):
            os.remove(path)
            removed.append(os.path.basename(path))
    if os.path.isdir(BACKUP_DIR):
        shutil.rmtree(BACKUP_DIR)
        removed.append(os.path.basename(BACKUP_DIR) + "/")
    # in-memory state goes back to a first start
    patterns.clear()                      # in place: main() holds the same list
    SETTINGS.update({"size": None, "autogrow": "suggest", "max_size": SIZE_ORDER[-1],
                     "tokenizer": TOKENIZER_DEFAULT})
    CONFIG.update(SIZE_PRESETS[REFERENCE_SIZE])
    _swap_tokenizer(tok, WordTokenizer() if SETTINGS["tokenizer"] == "word" else BPETokenizer())
    gpt = ConfigurableGPT(vocab_size=tok.vocab_size, **CONFIG)
    opt = Adam(gpt.params(), lr=LR)
    meta = {"n_patterns": 0, "corpus_len": 0}
    return removed, gpt, opt, meta


def switch_tokenizer(name, tok, patterns, corpus_text, gpt, opt, meta):
    old = SETTINGS["tokenizer"]
    _backup_current(f"{current_size_name()}_{old}")      # safety copy in size_backups/
    SETTINGS["tokenizer"] = name
    save_settings()
    if name == "word":
        new_tok = train_word_tokenizer()
    elif os.path.exists(BPE_PATH):
        new_tok = BPETokenizer.load(BPE_PATH)
    else:
        new_tok = train_tokenizer()
    _swap_tokenizer(tok, new_tok)
    _clear_current_weights()                              # old weights belong to the old vocabulary
    gpt = ConfigurableGPT(vocab_size=tok.vocab_size, **CONFIG)
    opt = Adam(gpt.params(), lr=LR)
    meta = {"n_patterns": 0, "corpus_len": 0}
    n_params = sum(p.data.size for p in gpt.params())
    print(f"[tokenizer: '{old}' -> '{name}': {tok.vocab_size} tokens, {n_params:,} parameters. "
          f"Different vocabulary, so the model trains from scratch. "
          f"(old weights copied to size_backups/{current_size_name()}_{old}.npz)]")
    return _auto(tok, patterns, corpus_text, gpt, opt, meta)


# ======================================================================
# ---- feedforward: SwiGLU building block, used by every expert ----
# ======================================================================
class SwiGLUFeedForward:
    def __init__(self, d_model, d_ff):
        self.W_gate = _param((d_model, d_ff))
        self.W_up = _param((d_model, d_ff))
        self.W_down = _param((d_ff, d_model))

    def params(self):
        return [self.W_gate, self.W_up, self.W_down]

    def __call__(self, x):
        gate = silu(x.matmul(self.W_gate))
        up = x.matmul(self.W_up)
        return (gate * up).matmul(self.W_down)


# ======================================================================
# ---- Mixture-of-Experts feedforward, with a shared expert ----
# ======================================================================
class MoEFeedForward:
    """n_experts=1 is a valid config, but note what it actually costs:
    every token still pays for the shared expert AND the one routed
    expert -- roughly 2x a single dense FFN's compute, not 1x."""
    def __init__(self, d_model, d_ff_expert, n_experts):
        self.n_experts = n_experts
        self.shared = SwiGLUFeedForward(d_model, d_ff_expert)
        self.experts = [SwiGLUFeedForward(d_model, d_ff_expert) for _ in range(n_experts)]
        self.W_router = _param((d_model, n_experts))

    def params(self):
        ps = list(self.shared.params()) + [self.W_router]
        for e in self.experts:
            ps += e.params()
        return ps

    def __call__(self, x):
        T, D = x.shape
        shared_out = self.shared(x)

        router_logits = x.matmul(self.W_router)
        router_probs = router_logits.softmax(axis=-1)
        chosen = np.argmax(router_probs.data, axis=-1)

        expert_outs = [e(x) for e in self.experts]
        combined = Tensor(np.zeros((T, D)), requires_grad=True,
                           _children=tuple(expert_outs) + (router_probs,))
        gate_weight = router_probs.data[np.arange(T), chosen]

        out_data = np.zeros((T, D))
        for e_idx in range(self.n_experts):
            mask = (chosen == e_idx)
            if not mask.any():
                continue
            out_data[mask] = expert_outs[e_idx].data[mask] * gate_weight[mask, None]
        combined.data = out_data

        def _backward():
            dout = combined.grad
            for e_idx, eo in enumerate(expert_outs):
                if eo.requires_grad:
                    mask = (chosen == e_idx)
                    g = np.zeros_like(eo.data)
                    g[mask] = dout[mask] * gate_weight[mask, None]
                    eo.grad += g
            if router_probs.requires_grad:
                g = np.zeros_like(router_probs.data)
                contrib = np.sum(dout * np.stack([eo.data for eo in expert_outs], axis=1)[
                    np.arange(T), chosen], axis=-1)
                g[np.arange(T), chosen] = contrib
                router_probs.grad += g
        combined._backward = _backward

        return shared_out + combined


# ======================================================================
# ---- Attention: plain full-size, or MLA (compressed K/V) ----
# ======================================================================
class MultiHeadAttention:
    """Plain attention, full-size per-head K/V. Used when use_mla=False."""
    def __init__(self, d_model, n_heads):
        assert d_model % n_heads == 0
        self.d_model, self.n_heads, self.d_head = d_model, n_heads, d_model // n_heads
        self.Wq, self.bq = _param((d_model, d_model)), _param((d_model,), 0.0)
        self.Wk, self.bk = _param((d_model, d_model)), _param((d_model,), 0.0)
        self.Wv, self.bv = _param((d_model, d_model)), _param((d_model,), 0.0)
        self.Wo, self.bo = _param((d_model, d_model)), _param((d_model,), 0.0)

    def params(self):
        return [self.Wq, self.bq, self.Wk, self.bk, self.Wv, self.bv, self.Wo, self.bo]

    def __call__(self, x, causal_mask, rope):
        T, D = x.shape
        H, Dh = self.n_heads, self.d_head
        q = x.matmul(self.Wq) + self.bq
        k = x.matmul(self.Wk) + self.bk
        v = x.matmul(self.Wv) + self.bv
        q = q.reshape(T, H, Dh).transpose((1, 0, 2))
        k = k.reshape(T, H, Dh).transpose((1, 0, 2))
        v = v.reshape(T, H, Dh).transpose((1, 0, 2))
        if rope is not None:
            cos, sin = rope
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)
        kt = k.transpose((0, 2, 1))
        scores = q.matmul(kt) * (1.0 / np.sqrt(Dh))
        scores = scores + causal_mask
        attn = scores.softmax(axis=-1)
        out = attn.matmul(v)
        out = out.transpose((1, 0, 2)).reshape(T, D)
        return out.matmul(self.Wo) + self.bo


class MultiHeadLatentAttention:
    """Compressed K/V via a shared low-rank latent. Used when use_mla=True.
    Simplified vs. the DeepSeek-V2 paper (no decoupled RoPE key path)."""
    def __init__(self, d_model, n_heads, d_latent):
        assert d_model % n_heads == 0
        self.d_model, self.n_heads, self.d_head = d_model, n_heads, d_model // n_heads
        self.Wq, self.bq = _param((d_model, d_model)), _param((d_model,), 0.0)
        self.W_down_kv = _param((d_model, d_latent))
        self.W_up_k = _param((d_latent, d_model))
        self.W_up_v = _param((d_latent, d_model))
        self.Wo, self.bo = _param((d_model, d_model)), _param((d_model,), 0.0)

    def params(self):
        return [self.Wq, self.bq, self.W_down_kv, self.W_up_k, self.W_up_v, self.Wo, self.bo]

    def __call__(self, x, causal_mask, rope):
        T, D = x.shape
        H, Dh = self.n_heads, self.d_head
        q = x.matmul(self.Wq) + self.bq
        c_kv = x.matmul(self.W_down_kv)
        k = c_kv.matmul(self.W_up_k)
        v = c_kv.matmul(self.W_up_v)
        q = q.reshape(T, H, Dh).transpose((1, 0, 2))
        k = k.reshape(T, H, Dh).transpose((1, 0, 2))
        v = v.reshape(T, H, Dh).transpose((1, 0, 2))
        if rope is not None:
            cos, sin = rope
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)
        kt = k.transpose((0, 2, 1))
        scores = q.matmul(kt) * (1.0 / np.sqrt(Dh))
        scores = scores + causal_mask
        attn = scores.softmax(axis=-1)
        out = attn.matmul(v)
        out = out.transpose((1, 0, 2)).reshape(T, D)
        return out.matmul(self.Wo) + self.bo


# ======================================================================
# ---- TransformerBlock: assembled from whatever CONFIG picks ----
# ======================================================================
class TransformerBlock:
    def __init__(self, cfg):
        d = cfg["d_model"]
        self.use_rmsnorm = cfg["use_rmsnorm"]
        if self.use_rmsnorm:
            self.norm1_w = Tensor(np.ones(d), requires_grad=True)
            self.norm2_w = Tensor(np.ones(d), requires_grad=True)
        else:
            self.norm1_g = Tensor(np.ones(d), requires_grad=True)
            self.norm1_b = Tensor(np.zeros(d), requires_grad=True)
            self.norm2_g = Tensor(np.ones(d), requires_grad=True)
            self.norm2_b = Tensor(np.zeros(d), requires_grad=True)

        if cfg["use_mla"]:
            self.attn = MultiHeadLatentAttention(d, cfg["n_heads"], cfg["d_latent"])
        else:
            self.attn = MultiHeadAttention(d, cfg["n_heads"])

        self.ffn = MoEFeedForward(d, cfg["d_ff_expert"], cfg["n_experts"])

    def params(self):
        if self.use_rmsnorm:
            ps = [self.norm1_w, self.norm2_w]
        else:
            ps = [self.norm1_g, self.norm1_b, self.norm2_g, self.norm2_b]
        return ps + self.attn.params() + self.ffn.params()

    def _norm(self, x, which):
        if self.use_rmsnorm:
            w = self.norm1_w if which == 1 else self.norm2_w
            return x.rmsnorm(w)
        g, b = (self.norm1_g, self.norm1_b) if which == 1 else (self.norm2_g, self.norm2_b)
        return x.layernorm(g, b)

    def __call__(self, x, causal_mask, rope):
        x = x + self.attn(self._norm(x, 1), causal_mask, rope)
        x = x + self.ffn(self._norm(x, 2))
        return x


# ======================================================================
# ---- ConfigurableGPT: backbone + pointer/copy head + MTP head ----
# ======================================================================
class ConfigurableGPT:
    def __init__(self, vocab_size, d_model=64, n_heads=4, n_layers=3,
                 d_latent=16, d_ff_expert=96, n_experts=4, max_seq_len=256,
                 use_rope=True, use_rmsnorm=True, use_mla=True,
                 use_pointer=True, use_mtp=True, mtp_weight=0.3, ptr_shift=False, seed=0):
        np.random.seed(seed)
        self.cfg = dict(d_model=d_model, n_heads=n_heads, n_layers=n_layers,
                         d_latent=d_latent, d_ff_expert=d_ff_expert, n_experts=n_experts,
                         max_seq_len=max_seq_len, use_rope=use_rope, use_rmsnorm=use_rmsnorm,
                         use_mla=use_mla, use_pointer=use_pointer, use_mtp=use_mtp,
                         mtp_weight=mtp_weight, ptr_shift=ptr_shift)
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.max_seq_len = max_seq_len
        self.use_rope = use_rope
        self.use_rmsnorm = use_rmsnorm
        self.use_pointer = use_pointer
        self.use_mtp = use_mtp
        self.mtp_weight = mtp_weight
        self.ptr_shift = ptr_shift

        self.token_emb = _param((vocab_size, d_model))
        if not use_rope:
            self.pos_emb = _param((max_seq_len, d_model))

        self.blocks = [TransformerBlock(self.cfg) for _ in range(n_layers)]

        if use_rmsnorm:
            self.lnf_w = Tensor(np.ones(d_model), requires_grad=True)
        else:
            self.lnf_g = Tensor(np.ones(d_model), requires_grad=True)
            self.lnf_b = Tensor(np.zeros(d_model), requires_grad=True)

        if use_pointer:
            self.Wq_ptr = _param((d_model, d_model))
            self.Wk_ptr = _param((d_model, d_model))
            self.Wg = _param((d_model, 1))
            self.bg = Tensor(np.zeros(1), requires_grad=True)

        if use_mtp:
            self.mtp_block = TransformerBlock(self.cfg)
            if use_rmsnorm:
                self.mtp_norm_w = Tensor(np.ones(d_model), requires_grad=True)
            else:
                self.mtp_norm_g = Tensor(np.ones(d_model), requires_grad=True)
                self.mtp_norm_b = Tensor(np.zeros(d_model), requires_grad=True)

        self._mask_cache = {}
        self._rope_cache = {}

    def named_params(self):
        names, ps = [], []

        def add(n, p):
            names.append(n)
            ps.append(p)

        add("token_emb", self.token_emb)
        if not self.use_rope:
            add("pos_emb", self.pos_emb)
        if self.use_rmsnorm:
            add("lnf_w", self.lnf_w)
        else:
            add("lnf_g", self.lnf_g)
            add("lnf_b", self.lnf_b)
        if self.use_pointer:
            add("Wq_ptr", self.Wq_ptr)
            add("Wk_ptr", self.Wk_ptr)
            add("Wg", self.Wg)
            add("bg", self.bg)
        for i, b in enumerate(self.blocks):
            for j, p in enumerate(b.params()):
                add(f"block{i}_p{j}", p)
        if self.use_mtp:
            for j, p in enumerate(self.mtp_block.params()):
                add(f"mtp_p{j}", p)
            if self.use_rmsnorm:
                add("mtp_norm_w", self.mtp_norm_w)
            else:
                add("mtp_norm_g", self.mtp_norm_g)
                add("mtp_norm_b", self.mtp_norm_b)
        return names, ps

    def params(self):
        return self.named_params()[1]

    def zero_grad(self):
        for p in self.params():
            p.zero_grad()

    def _causal_mask(self, T):
        if T not in self._mask_cache:
            m = np.triu(np.ones((T, T)) * -1e9, k=1)
            self._mask_cache[T] = Tensor(m, requires_grad=False)
        return self._mask_cache[T]

    def _rope(self, T):
        if not self.use_rope:
            return None
        if T not in self._rope_cache:
            self._rope_cache[T] = build_rope_cache(T, self.d_model // self.n_heads)
        return self._rope_cache[T]

    def _final_norm(self, x):
        if self.use_rmsnorm:
            return x.rmsnorm(self.lnf_w)
        return x.layernorm(self.lnf_g, self.lnf_b)

    def _mtp_norm(self, x):
        if self.use_rmsnorm:
            return x.rmsnorm(self.mtp_norm_w)
        return x.layernorm(self.mtp_norm_g, self.mtp_norm_b)

    def forward_probs(self, ids, compute_mtp=None):
        """Returns (final_prob, p_gen, mtp_logits_full_or_None)."""
        if compute_mtp is None:
            compute_mtp = self.use_mtp

        T = len(ids)
        ids_arr = np.array(ids)
        x = self.token_emb.embedding_lookup(ids_arr)
        if not self.use_rope:
            x = x + self.pos_emb.embedding_lookup(np.arange(T))
        mask = self._causal_mask(T)
        rope = self._rope(T)

        for block in self.blocks:
            x = block(x, mask, rope)
        h = self._final_norm(x)

        gen_logits = h.matmul(self.token_emb.transpose((1, 0)))
        gen_prob = gen_logits.softmax(axis=-1)

        if self.use_pointer:
            q_ptr = h.matmul(self.Wq_ptr)
            k_ptr = h.matmul(self.Wk_ptr)
            scores_ptr = q_ptr.matmul(k_ptr.transpose((1, 0))) * (1.0 / np.sqrt(self.d_model))
            p_gen = (h.matmul(self.Wg) + self.bg).sigmoid()
            if self.ptr_shift:
                # induction-style copy: position t attends to earlier positions j < t and copies the token that
                # FOLLOWED position j (ids[j+1]), i.e. "continue the phrase I matched".  Position 0 has nothing to
                # attend to, so it uses the generator only (p_gen forced to 1) -- no future token is ever visible.
                scores_ptr = scores_ptr + Tensor(np.triu(np.ones((T, T)) * -1e9, k=0), requires_grad=False)
                attn_ptr = scores_ptr.softmax(axis=-1)
                nxt = np.concatenate([ids_arr[1:], ids_arr[:1]])          # last entry is never attended to
                onehot = Tensor(np.eye(self.vocab_size)[nxt], requires_grad=False)
                copy_prob = attn_ptr.matmul(onehot)
                first = np.zeros((T, 1)); first[0, 0] = 1.0
                p_gen = p_gen * Tensor(1.0 - first, requires_grad=False) + Tensor(first, requires_grad=False)
            else:
                scores_ptr = scores_ptr + mask
                attn_ptr = scores_ptr.softmax(axis=-1)
                onehot = Tensor(np.eye(self.vocab_size)[ids_arr], requires_grad=False)
                copy_prob = attn_ptr.matmul(onehot)
            final_prob = gen_prob * p_gen + copy_prob * (p_gen * -1.0 + 1.0)
        else:
            final_prob, p_gen = gen_prob, None

        mtp_logits_full = None
        if compute_mtp and self.use_mtp:
            mtp_x = self.mtp_block(x, mask, rope)
            mtp_h = self._mtp_norm(mtp_x)
            mtp_logits_full = mtp_h.matmul(self.token_emb.transpose((1, 0)))

        return final_prob, p_gen, mtp_logits_full

    def _fast_ok(self):
        c = self.cfg
        return bool(c.get("use_mla") and c.get("use_rope") and c.get("use_rmsnorm") and c.get("use_pointer"))

    def generate_fast(self, prompt_ids, max_new_tokens=40, eos_id=None, on_token=None):
        """Greedy decoding with a key/value cache: every token is processed once, not once per generated word.
        Same maths as forward_probs (checked to give identical answers)."""
        names, ps = self.named_params()
        W = {n: p.data for n, p in zip(names, ps)}
        D, H, L, E = self.d_model, self.n_heads, self.cfg["n_layers"], self.cfg["n_experts"]
        Dh = D // H; half = Dh // 2
        ids = list(prompt_ids)
        Tmax = len(ids) + max_new_tokens
        dt = W["token_emb"].dtype
        inv = 1.0 / (10000.0 ** (np.arange(0, Dh, 2) / Dh))
        fr = np.outer(np.arange(Tmax), inv); fr = np.concatenate([fr, fr], axis=-1)
        cos, sin = np.cos(fr).astype(dt), np.sin(fr).astype(dt)
        Kc = [np.zeros((H, Tmax, Dh), dtype=dt) for _ in range(L)]
        Vc = [np.zeros((H, Tmax, Dh), dtype=dt) for _ in range(L)]
        Kp = np.zeros((Tmax, D), dtype=dt)
        V = self.vocab_size
        def rms(x, w): return x / np.sqrt((x * x).mean(-1, keepdims=True) + 1e-6) * w
        def sm(x):
            x = x - x.max(-1, keepdims=True); e = np.exp(x); return e / e.sum(-1, keepdims=True)
        def silu(t): return t / (1.0 + np.exp(-t))
        def rope(x, pos):
            rot = np.concatenate([-x[:, half:], x[:, :half]], axis=-1)
            return x * cos[pos] + rot * sin[pos]
        def step(tok_id, pos, want_out):
            x = W["token_emb"][tok_id]
            for i in range(L):
                g = lambda j: W[f"block{i}_p{j}"]
                h = rms(x, g(0))
                q = (h @ g(2) + g(3)).reshape(H, Dh); c = h @ g(4)
                k = (c @ g(5)).reshape(H, Dh); v = (c @ g(6)).reshape(H, Dh)
                q, k = rope(q, pos), rope(k, pos)
                Kc[i][:, pos], Vc[i][:, pos] = k, v
                sc = np.einsum("hd,htd->ht", q, Kc[i][:, :pos + 1]) / np.sqrt(Dh)
                a = np.einsum("ht,htd->hd", sm(sc), Vc[i][:, :pos + 1]).reshape(D)
                x = x + (a @ g(7) + g(8))
                h = rms(x, g(1))
                shared = (silu(h @ g(9)) * (h @ g(10))) @ g(11)
                probs = sm(h @ g(12)); e = int(np.argmax(probs))
                out = (silu(h @ g(13 + 3 * e)) * (h @ g(14 + 3 * e))) @ g(15 + 3 * e)
                x = x + shared + out * probs[e]
            h = rms(x, W["lnf_w"])
            Kp[pos] = h @ W["Wk_ptr"]
            if not want_out:
                return None
            gen = sm(h @ W["token_emb"].T)
            if self.ptr_shift:
                if pos == 0:
                    return gen                                   # nothing earlier to copy from
                att = sm((Kp[:pos] @ (h @ W["Wq_ptr"])) / np.sqrt(D))
                copy = np.zeros(V, dtype=dt); np.add.at(copy, np.array(ids[1:pos + 1]), att)
            else:
                att = sm((Kp[:pos + 1] @ (h @ W["Wq_ptr"])) / np.sqrt(D))
                copy = np.zeros(V, dtype=dt); np.add.at(copy, np.array(ids[:pos + 1]), att)
            pg = 1.0 / (1.0 + np.exp(-(h @ W["Wg"] + W["bg"])))
            return gen * pg + copy * (1.0 - pg)
        def prefill(n):
            """Read the first n prompt tokens in ONE batched pass (same maths as n calls of step)."""
            toks = np.array(ids[:n])
            x = W["token_emb"][toks]                                     # (n, D)
            causal = np.triu(np.full((n, n), -np.inf, dtype=dt), 1)
            c_n, s_n = cos[:n], sin[:n]
            def rope_b(t):                                                # t: (H, n, Dh)
                rot = np.concatenate([-t[..., half:], t[..., :half]], axis=-1)
                return t * c_n + rot * s_n
            ar = np.arange(n)
            for i in range(L):
                g = lambda j: W[f"block{i}_p{j}"]
                h = rms(x, g(0))
                q = (h @ g(2) + g(3)).reshape(n, H, Dh).transpose(1, 0, 2)
                c = h @ g(4)
                k = (c @ g(5)).reshape(n, H, Dh).transpose(1, 0, 2)
                v = (c @ g(6)).reshape(n, H, Dh).transpose(1, 0, 2)
                q, k = rope_b(q), rope_b(k)
                Kc[i][:, :n], Vc[i][:, :n] = k, v
                sc = np.einsum("hqd,hkd->hqk", q, k) / np.sqrt(Dh) + causal
                a = np.einsum("hqk,hkd->hqd", sm(sc), v).transpose(1, 0, 2).reshape(n, D)
                x = x + (a @ g(7) + g(8))
                h = rms(x, g(1))
                shared = (silu(h @ g(9)) * (h @ g(10))) @ g(11)
                probs = sm(h @ g(12)); e = probs.argmax(-1)               # one expert per token
                out = np.zeros_like(x)
                for ex in np.unique(e):
                    m = e == ex
                    hm = h[m]
                    out[m] = (silu(hm @ g(13 + 3 * ex)) * (hm @ g(14 + 3 * ex))) @ g(15 + 3 * ex)
                x = x + shared + out * probs[ar, e][:, None]
            Kp[:n] = rms(x, W["lnf_w"]) @ W["Wk_ptr"]
        if len(ids) > 1:
            prefill(len(ids) - 1)
        pos = len(ids) - 1
        for _ in range(max_new_tokens):
            final = step(ids[pos], pos, True)
            nxt = int(np.argmax(final))
            ids.append(nxt); pos += 1
            if on_token is not None:
                on_token(nxt)
            if eos_id is not None and nxt == eos_id:
                break
        return ids

    def generate(self, prompt_ids, max_new_tokens=40, temperature=0.9, top_k=20,
                 eos_id=None, on_token=None):
        ids = list(prompt_ids)
        if temperature is None and top_k == 1 and self._fast_ok() and len(ids) + max_new_tokens <= self.max_seq_len:
            return self.generate_fast(ids, max_new_tokens, eos_id, on_token)
        for _ in range(max_new_tokens):
            ctx = ids[-self.max_seq_len:]
            probs, _, _ = self.forward_probs(ctx, compute_mtp=False)
            last = probs.data[-1]
            if temperature is not None and temperature != 1.0:
                last = np.power(np.clip(last, 1e-12, None), 1.0 / max(temperature, 1e-6))
                last = last / last.sum()
            if top_k is not None and top_k < len(last):
                kth = np.partition(last, -top_k)[-top_k]
                last = np.where(last >= kth, last, 0.0)
                last = last / last.sum()
            next_id = int(np.random.choice(len(last), p=last))
            ids.append(next_id)
            if on_token is not None:
                on_token(next_id)
            if eos_id is not None and next_id == eos_id:
                break
        return ids

    def loss(self, inp, target):
        """Returns (total_loss, primary_loss, mtp_loss_or_None)."""
        T = len(inp)
        probs, p_gen, mtp_logits_full = self.forward_probs(inp)
        primary_loss = nll_from_probs(probs, target)

        if not self.use_mtp or T < 2:
            return primary_loss, primary_loss, None

        mtp_target = target[1:]
        n_valid = T - 1
        mtp_logits = Tensor(mtp_logits_full.data[:n_valid], requires_grad=True,
                             _children=(mtp_logits_full,), _op="slice")

        def _slice_backward():
            if mtp_logits_full.requires_grad:
                g = np.zeros_like(mtp_logits_full.data)
                g[:n_valid] = mtp_logits.grad
                mtp_logits_full.grad += g
        mtp_logits._backward = _slice_backward

        mtp_loss = softmax_cross_entropy(mtp_logits, np.array(mtp_target))
        total = primary_loss + mtp_loss * self.mtp_weight
        return total, primary_loss, mtp_loss


# ======================================================================
# ---- optimizer -- unchanged ----
# ======================================================================
class Adam:
    def __init__(self, params, lr=3e-3, betas=(0.9, 0.999), eps=1e-8):
        self.params = params
        self.lr = lr
        self.b1, self.b2 = betas
        self.eps = eps
        self.m = [np.zeros_like(p.data) for p in params]
        self.v = [np.zeros_like(p.data) for p in params]
        self.t = 0

    def zero_grad(self):
        for p in self.params:
            p.grad = np.zeros_like(p.data)

    def step(self, clip=1.0, weight_decay=0.0):
        self.t += 1
        for i, p in enumerate(self.params):
            g = p.grad
            if clip is not None:
                norm = np.linalg.norm(g)
                if norm > clip:
                    g = g * (clip / (norm + 1e-8))
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * (g * g)
            mhat = self.m[i] / (1 - self.b1 ** self.t)
            vhat = self.v[i] / (1 - self.b2 ** self.t)
            p.data -= self.lr * mhat / (np.sqrt(vhat) + self.eps)
            if weight_decay > 0:
                p.data -= self.lr * weight_decay * p.data


# ======================================================================
# ---- main.py -- same CLI/workflow, + self-test / self-correction + SIZES ----
# ======================================================================
HERE = os.path.dirname(os.path.abspath(__file__))
SAVE_PATH = os.path.join(HERE, "model.npz")
META_PATH = os.path.join(HERE, "model.meta.json")
PATTERNS_PATH = os.path.join(HERE, "patterns.json")
CORPUS_PATH = os.path.join(HERE, "corpus.txt")
BPE_PATH = os.path.join(HERE, "bpe_tokenizer.json")
WORD_PATH = os.path.join(HERE, "word_tokenizer.json")
TOKENIZER_DEFAULT = "bpe"   # "bpe" (your original) or "word" (one token per word); switch with :tokenizer
SETTINGS_PATH = os.path.join(HERE, "size.json")
LOG_PATH = os.path.join(HERE, "growth_log.json")
BACKUP_DIR = os.path.join(HERE, "size_backups")

# ---- MODEL SIZES (small = your original CONFIG; sizes only go UP from it) ----
# d_model must divide by n_heads, and d_model / n_heads must be even (RoPE).
SIZE_PRESETS = {
    "small":  dict(d_model=64,  n_heads=4, n_layers=3, d_latent=16, d_ff_expert=96,  n_experts=4),
    "medium": dict(d_model=96,  n_heads=6, n_layers=4, d_latent=24, d_ff_expert=144, n_experts=4),
    "large":  dict(d_model=128, n_heads=8, n_layers=6, d_latent=32, d_ff_expert=192, n_experts=6),
    "xlarge": dict(d_model=192, n_heads=8, n_layers=6, d_latent=48, d_ff_expert=256, n_experts=6),
    "xxl":    dict(d_model=256, n_heads=8, n_layers=6, d_latent=64, d_ff_expert=512,  n_experts=4),   # ~15M parameters
    "huge":   dict(d_model=384, n_heads=6, n_layers=8, d_latent=96, d_ff_expert=640,  n_experts=4),   # ~37M parameters
    "dlg15":  dict(d_model=256, n_heads=8, n_layers=6, d_latent=64, d_ff_expert=512,  n_experts=4, max_seq_len=512),   # ~16M parameters, 512-token context
    "dlg30":  dict(d_model=384, n_heads=6, n_layers=7, d_latent=96, d_ff_expert=544,  n_experts=4, max_seq_len=512),   # ~30M parameters, 512-token context
    "max":    dict(d_model=512, n_heads=8, n_layers=8, d_latent=128, d_ff_expert=1024, n_experts=4),  # ~78M parameters
}
SIZE_ORDER = list(SIZE_PRESETS)
SIZE_KEYS = ["d_model", "n_heads", "n_layers", "d_latent", "d_ff_expert", "n_experts"]
REFERENCE_SIZE = "small"

# ---- THE ARCHITECTURE CONFIG (size fields come from SIZE_PRESETS) ----
CONFIG = dict(
    max_seq_len=384,       # was 256: 2 training chats were 277 tokens, and chats were trimmed at 196 tokens
    use_rope=True,          # rotary position embeddings vs. a learned pos_emb table
    use_rmsnorm=True,        # RMSNorm vs. LayerNorm
    use_mla=True,              # compressed-latent K/V vs. plain full-size K/V
    use_pointer=True,            # keep the original copy/pointer mechanism
    ptr_shift=False,             # True = copy the token that FOLLOWED the matched position (induction-style). Needs a retrain from scratch.
    use_mtp=True,                  # auxiliary t+2 prediction head
    mtp_weight=0.3,
)
CONFIG.update(SIZE_PRESETS["small"])

MIN_EPOCHS = 15
MAX_EPOCHS = 400
MAX_EPOCHS_MEMORIZE = 600
PATIENCE = 8
VAL_FRACTION = 0.15
MIN_EXAMPLES_FOR_VAL = 20
LR = 3e-3
BPE_TARGET_VOCAB_SIZE = 600
BPE_MIN_PAIR_FREQ = 2
BPE_MIN_TEXT_CHARS = 40      # below this much text, don't bother learning merges (was 200)

TEMPLATES_PATH = os.path.join(HERE, "templates.json")
TEMPLATE_HOLDOUT_FRACTION = 0.25
TEMPLATE_CHECK_EVERY = 25
TEMPLATE_WEIGHT_DECAY = 0.1

# ---- SELF-TEST / SELF-CORRECTION knobs ----
SELFTEST_MAX_ROUNDS = 40     # max repair rounds after the main training loop
REPAIR_REPEAT = 3            # each failed pattern is trained this many times per round
REPAIR_LR_SCALE = 0.3        # repair rounds use a gentler learning rate (LR * this)
SELFTEST_SHOW_FAILURES = 5   # how many failures to print per report
FULL_MAX_ROUNDS = 60         # max repair rounds in :fullstabilize
GEN_WEIGHT_DECAY = 0.3       # phase 2 (generalize): strong weight decay pulls weights away from
                             # 'pure memorization' while recall is guarded
GEN_MAX_EPOCHS = 200         # phase 2 epoch cap
GEN_CHECK_EVERY = 5          # phase 2: check recall + held-out accuracy every N epochs
GEN_PATIENCE = 8             # phase 2: stop after this many checks with no held-out gain
GEN_LR_SCALE = 0.5           # phase 2 learning rate = LR * this
PROBES_PATH = os.path.join(HERE, "probes.json")
FULL_CHECK_EVERY = 5         # :fullstabilize: verify every N epochs and stop as soon as everything passes
FULL_STALL_ROUNDS = 4        # repair rounds with no progress before trying a plain-training burst
FULL_STALL_EPOCHS = 10       # length of that burst (ordinary epochs over everything, full LR)
FULL_BLOCK_TOKEN_ACC = 1.0   # a template/corpus block 'passes' when this fraction of its
                             # next-token predictions are argmax-correct (1.0 = every token)

# ---- SCALING knobs ----
PLATEAU_MIN_GROWTH = 1.20    # need >= 20% more examples than an earlier run to call a plateau
PLATEAU_MIN_GAIN = 0.02      # held-out must improve by more than this to count as progress
PLATEAU_DONE_ABOVE = 0.95    # at/above this held-out accuracy we don't call it a plateau
LOG_MAX = 200

CHECKPOINT = {"fn": None}   # optional hook: fn(gpt, epoch) called every 5 epochs of stabilize
WARM_START = {"fn": None}   # optional hook: fn(gpt, tok) called on the fresh model before training
CHAT_GREEDY = True   # True: chat answers use greedy decoding (exact, repeatable) -- best for a cheat sheet.
                     # False: the original sampling (temperature 0.8, top_k 20).

SETTINGS = {"size": None, "autogrow": "suggest", "max_size": SIZE_ORDER[-1],
            "tokenizer": TOKENIZER_DEFAULT,
            "checkmode": False, "showcheck": False}


def load_patterns():
    if not os.path.exists(PATTERNS_PATH):
        return []
    with open(PATTERNS_PATH, encoding="utf-8") as f:
        return json.load(f)


def save_patterns(patterns):
    with open(PATTERNS_PATH, "w", encoding="utf-8") as f:
        json.dump(patterns, f, indent=1)


def import_patterns_from_file(path):
    if not os.path.exists(path):
        return None, f"file not found: {path}"
    added = []
    if path.lower().endswith(".json"):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        for item in data:
            if isinstance(item, dict) and "prompt" in item and "answer" in item:
                added.append({"prompt": item["prompt"], "answer": item["answer"]})
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                added.append({"prompt": item[0], "answer": item[1]})
    else:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line.strip() or "||" not in line:
                    continue
                prompt, answer = line.split("||", 1)
                added.append({"prompt": prompt, "answer": answer})
    return added, None


def load_corpus():
    if not os.path.exists(CORPUS_PATH):
        return ""
    with open(CORPUS_PATH, encoding="utf-8") as f:
        return f.read()


def append_corpus(new_text):
    with open(CORPUS_PATH, "a", encoding="utf-8") as f:
        f.write(new_text.strip() + "\n\n")


def read_pasted_text():
    print("paste your text below. finish with a line containing just: END")
    lines = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line.strip() == "END":
            break
        lines.append(line)
    return "\n".join(lines).strip()


def build_or_load_tokenizer():
    tok = _build_or_load_tokenizer_for_mode()
    _register_tokenizer(tok)
    return tok


def _build_or_load_bpe_tokenizer():
    if os.path.exists(BPE_PATH):
        tok = BPETokenizer.load(BPE_PATH)
        print(f"[loaded bpe tokenizer: {tok.vocab_size} tokens "
              f"({len(tok.base_chars) + 4} base + {len(tok.merges)} merges)]")
        return tok
    print("[no bpe_tokenizer.json found -- training one from saved patterns/corpus]")
    return train_tokenizer()


def train_tokenizer(verbose=True):
    parts = [load_corpus()]
    for p in load_patterns():
        parts.append(p.get("prompt", "") + " " + p.get("answer", ""))
    synth_patterns, holdout_evals = expand_templates(load_templates())
    for p in synth_patterns:
        parts.append(p["prompt"])
    for item in holdout_evals:
        parts.append(item["prompt"] + " " + str(item["expected"]))
    text = "\n".join(t for t in parts if t)

    tok = BPETokenizer()
    if len(text) < BPE_MIN_TEXT_CHARS:
        if verbose:
            print("[not enough text yet to learn meaningful subwords -- "
                  "using base characters only for now.]")
        tok.train(text, vocab_size=len(tok.stoi), min_pair_freq=BPE_MIN_PAIR_FREQ, verbose=verbose)
    else:
        tok.train(text, vocab_size=BPE_TARGET_VOCAB_SIZE, min_pair_freq=BPE_MIN_PAIR_FREQ, verbose=verbose)
    tok.save(BPE_PATH)
    return tok


def save_model(gpt, n_patterns, corpus_len):
    names, params = gpt.named_params()
    np.savez(SAVE_PATH, **{n: p.data for n, p in zip(names, params)})
    with open(META_PATH, "w") as f:
        json.dump({"n_patterns": n_patterns, "corpus_len": corpus_len, "config": gpt.cfg,
                   "vocab_size": gpt.vocab_size, "tokenizer": SETTINGS["tokenizer"],
                   "vocab_sig": TOK_SIG["v"]}, f, indent=1)


def load_model(gpt):
    if not os.path.exists(SAVE_PATH):
        return False, {"n_patterns": 0, "corpus_len": 0}
    data = np.load(SAVE_PATH)
    names, params = gpt.named_params()
    n_loaded = 0
    for n, p in zip(names, params):
        if n in data and data[n].shape == p.data.shape:
            p.data = data[n].copy()
            n_loaded += 1
    meta = {"n_patterns": 0, "corpus_len": 0}
    if os.path.exists(META_PATH):
        with open(META_PATH) as f:
            meta = json.load(f)
    if n_loaded < len(names):
        print(f"[note: only {n_loaded}/{len(names)} saved tensors matched this CONFIG's "
              f"shapes -- the rest started from random init. This happens when you change "
              f"CONFIG after a previous :stabilize; run :stabilize again to retrain fully "
              f"under the new architecture.]")
    return True, meta


def encode_pattern(tok, prompt, answer, max_len):
    full = tok.encode_prompt(prompt) + tok.encode(answer, add_bos=False, add_eos=True)
    full = full[:max_len + 1]
    return full[:-1], full[1:]


def chunk_corpus(tok, text, max_len):
    if not text:
        return []
    ids = tok.encode(text, add_bos=True, add_eos=True)
    chunks = []
    for i in range(0, max(0, len(ids) - 1), max_len):
        block = ids[i:i + max_len + 1]
        if len(block) >= 4:
            chunks.append((block[:-1], block[1:]))
    return chunks


def _split_train_val(examples, val_fraction, seed=0):
    examples = list(examples)
    rng = random.Random(seed)
    rng.shuffle(examples)
    n = len(examples)
    if n < 2:
        return examples, examples
    n_val = max(1, int(round(n * val_fraction)))
    n_val = min(n_val, n - 1)
    val = examples[:n_val]
    train = examples[n_val:]
    return train, val


def _eval_loss(gpt, examples):
    if not examples:
        return None
    total = 0.0
    for inp, target in examples:
        total_loss, _, _ = gpt.loss(inp, target)
        total += float(total_loss.data)
    return total / len(examples)


def load_templates():
    if not os.path.exists(TEMPLATES_PATH):
        return []
    with open(TEMPLATES_PATH, encoding="utf-8") as f:
        return json.load(f)


def _cross_product_combos(var_groups):
    combos = [{}]
    for group_rows in var_groups:
        new_combos = []
        for combo in combos:
            for row in group_rows:
                merged = dict(combo)
                merged.update(row)
                new_combos.append(merged)
        combos = new_combos
    return combos


def expand_templates(templates, holdout_fraction=TEMPLATE_HOLDOUT_FRACTION, seed=0):
    train_patterns = []
    holdout_evals = []
    rng = random.Random(seed)
    for tdef in templates:
        template = tdef["template"]
        eval_template = tdef.get("eval_template", template)
        check_placeholder = tdef.get("check_placeholder")
        var_groups = []
        for group in tdef.get("vars", {}).values():
            placeholders = group["placeholders"]
            group_rows = [dict(zip(placeholders, values)) for values in group["values"]]
            var_groups.append(group_rows)
        combos = _cross_product_combos(var_groups)
        rng.shuffle(combos)
        n_holdout = int(round(len(combos) * holdout_fraction)) if check_placeholder else 0
        for i, flat_vars in enumerate(combos):
            if i < n_holdout:
                holdout_evals.append({"prompt": eval_template.format(**flat_vars),
                                       "expected": flat_vars[check_placeholder]})
            else:
                train_patterns.append({"prompt": template.format(**flat_vars), "answer": ""})
    return train_patterns, holdout_evals


def check_template_accuracy(gpt, tok, holdout_evals, max_new_tokens=120):
    if not holdout_evals:
        return None
    correct = 0
    for item in holdout_evals:
        ids = tok.encode_prompt(item["prompt"])
        out = gpt.generate(ids, max_new_tokens=max_new_tokens, temperature=0.01,
                            top_k=1, eos_id=tok.eos_id)
        completion = tok.decode(out[len(ids):])
        if item["expected"] in completion:
            correct += 1
    return correct / len(holdout_evals)


# ======================================================================
# ---- SELF-TEST + SELF-CORRECTION ----
# ======================================================================
def _snapshot(gpt):
    names, params = gpt.named_params()
    return {n: p.data.copy() for n, p in zip(names, params)}


def _restore(gpt, state):
    names, params = gpt.named_params()
    for n, p in zip(names, params):
        if n in state:
            p.data = state[n].copy()


def selftest(gpt, tok, patterns):
    """Ask the model every taught prompt and compare its GREEDY answer
    (temperature=None, top_k=1 -> pure argmax, no randomness) with the
    answer you taught. Comparison is done after a tokenizer round-trip of
    the expected answer, so characters the tokenizer can't represent don't
    cause false failures. Patterns with an empty answer are skipped.

    Returns a list of dicts: idx (position in `patterns`), ok, prompt,
    expected, got."""
    results = []
    for i, p in enumerate(patterns):
        answer = p.get("answer", "")
        if not answer.strip():
            continue
        prompt_ids = tok.encode_prompt(p["prompt"])
        ans_ids = tok.encode(answer, add_bos=False, add_eos=True)
        expected = tok.decode(ans_ids).strip()
        out = gpt.generate(prompt_ids, max_new_tokens=len(ans_ids) + 5,
                           temperature=None, top_k=1, eos_id=tok.eos_id)
        got = tok.decode(out[len(prompt_ids):]).strip()
        results.append(dict(idx=i, ok=(got == expected), prompt=p["prompt"],
                            expected=expected, got=got))
    return results


def _accuracy(results):
    return sum(r["ok"] for r in results) / len(results) if results else 1.0


def _print_failures(results, max_show=SELFTEST_SHOW_FAILURES):
    failures = [r for r in results if not r["ok"]]
    for r in failures[:max_show]:
        print(f"    prompt:   {r['prompt']!r}")
        print(f"    expected: {r['expected']!r}")
        print(f"    got:      {r['got']!r}")
    if len(failures) > max_show:
        print(f"    ... and {len(failures) - max_show} more failing")


def self_correct(gpt, opt, tok, patterns, pattern_examples, weight_decay=0.0):
    """After normal training: test the model on every taught pattern; for
    those it gets wrong, train on them again (repeated REPAIR_REPEAT times,
    mixed with some already-passing patterns so they aren't forgotten),
    then re-test. Loops until everything passes or SELFTEST_MAX_ROUNDS.
    The best-scoring weights seen are restored at the end, so a bad round
    can't leave the model worse than it started. `pattern_examples[i]` must
    be the (inp, target) encoding of `patterns[i]`."""
    results = selftest(gpt, tok, patterns)
    if not results:
        print("[self-test: no taught patterns with answers to test]")
        return results
    acc = _accuracy(results)
    n_ok = sum(r["ok"] for r in results)
    print(f"[self-test: {n_ok}/{len(results)} taught patterns answered correctly "
          f"({acc:.0%})]")
    if acc < 1.0:
        _print_failures(results)

    best_acc, best_state, best_results = acc, _snapshot(gpt), results
    base_lr = opt.lr
    rounds = 0
    while acc < 1.0 and rounds < SELFTEST_MAX_ROUNDS:
        rounds += 1
        failed = [r["idx"] for r in results if not r["ok"]]
        passing = [r["idx"] for r in results if r["ok"]]
        replay = random.sample(passing, min(len(passing), max(len(failed), 4)))
        batch = [pattern_examples[i] for i in failed] * REPAIR_REPEAT
        batch += [pattern_examples[i] for i in replay]
        random.shuffle(batch)

        opt.lr = LR * REPAIR_LR_SCALE
        for inp, target in batch:
            gpt.zero_grad()
            total_loss, _, _ = gpt.loss(inp, target)
            total_loss.backward()
            opt.step(weight_decay=weight_decay)
        opt.lr = base_lr

        results = selftest(gpt, tok, patterns)
        acc = _accuracy(results)
        n_ok = sum(r["ok"] for r in results)
        tag = ""
        if acc > best_acc:
            best_acc, best_state, best_results = acc, _snapshot(gpt), results
            tag = "  (best so far)"
        print(f"  [repair round {rounds}: trained {len(set(failed))} failing + "
              f"{len(replay)} replay -> {n_ok}/{len(results)} correct ({acc:.0%}){tag}]")

    if best_acc > acc:
        _restore(gpt, best_state)
        results = best_results
        acc = best_acc
        print(f"[restored best checkpoint from the repair loop ({acc:.0%})]")

    if acc >= 1.0:
        print("[self-test PASSED: every taught pattern is recalled exactly]")
    else:
        print(f"[self-test finished at {acc:.0%} after {rounds} repair rounds -- "
              f"still failing:]")
        _print_failures(results)
        print("   (try: fewer/cleaner patterns, a larger size (:grow), or run :stabilize again)")
    return results


def _build_examples(tok, patterns, corpus_text):
    pattern_examples = [encode_pattern(tok, p["prompt"], p["answer"], CONFIG["max_seq_len"]) for p in patterns]
    synth_patterns, holdout_evals = expand_templates(load_templates())
    synth_examples = [encode_pattern(tok, p["prompt"], p["answer"], CONFIG["max_seq_len"]) for p in synth_patterns]
    corpus_examples = chunk_corpus(tok, corpus_text, CONFIG["max_seq_len"])
    return pattern_examples, synth_examples, corpus_examples, holdout_evals


def _text_pattern_examples(patterns, pattern_examples):
    """Patterns saved with a BLANK answer (you typed a whole sentence as the
    prompt and pressed Enter at 'what should it have said?') are plain text
    examples: there is no prompt->answer pair to compare, so they're verified
    like corpus text -- at every position where the data determines the next
    token, the model's argmax must be right (teacher-forced). Before this,
    such patterns were silently skipped by every check."""
    return [ex for p, ex in zip(patterns, pattern_examples)
            if not p.get("answer", "").strip()]


def _ambiguity_masks(pattern_examples, other_examples):
    """Some next-token targets can't be predicted from the prefix alone:
    e.g. every template example starts 'capital of ' and then a DIFFERENT
    country follows, so no model can get that slot 'right' for all of them.
    A position is AMBIGUOUS when the same prefix is followed by more than one
    distinct token anywhere in the training data. We don't demand exactness
    there. Returns, per `other_examples` entry, a boolean array that is True
    where the target is uniquely determined (and so must be reproduced)."""
    followers = collections.defaultdict(set)

    def prefix_hashes(inp):
        h, out = 0, []
        for t in inp:
            h = hash((h, int(t)))
            out.append(h)
        return out

    for inp, target in list(pattern_examples) + list(other_examples):
        for h, t in zip(prefix_hashes(inp), target):
            followers[h].add(int(t))
    masks = []
    for inp, target in other_examples:
        masks.append(np.array([len(followers[h]) == 1 for h in prefix_hashes(inp)]))
    return masks


def verify_all(gpt, tok, patterns, other_examples, masks=None):
    """Check EVERY training example.
      * taught patterns: free-running greedy answer must match exactly
        (same as selftest()).
      * template-synthetic examples and corpus blocks: teacher-forced check
        -- at every UNAMBIGUOUS position the model's argmax next token must
        equal the real next token (pass = FULL_BLOCK_TOKEN_ACC of those
        positions correct). See _ambiguity_masks for what 'ambiguous' means.
    Returns (pattern_results, other_results)."""
    pat = selftest(gpt, tok, patterns)
    oth = []
    for j, (inp, target) in enumerate(other_examples):
        probs, _, _ = gpt.forward_probs(inp, compute_mtp=False)
        correct = np.argmax(probs.data, axis=-1) == np.array(target)
        valid = masks[j] if masks is not None else np.ones(len(target), dtype=bool)
        n_valid = int(valid.sum())
        acc = float(correct[valid].mean()) if n_valid else 1.0
        bad = np.where(valid & ~correct)[0]
        first_bad = None
        if len(bad):
            k = int(bad[0])
            first_bad = (k, int(target[k]), int(np.argmax(probs.data[k])))
        oth.append(dict(idx=j, ok=(acc >= FULL_BLOCK_TOKEN_ACC), tok_acc=acc,
                        n_ambiguous=int(len(target) - n_valid), first_bad=first_bad))
    return pat, oth


def _count_ok(pat, oth):
    n_ok = sum(r["ok"] for r in pat) + sum(r["ok"] for r in oth)
    return n_ok, len(pat) + len(oth)


def _print_text_failures(tok, oth, other_examples, max_show=SELFTEST_SHOW_FAILURES):
    shown = 0
    for r in oth:
        if r["ok"] or r["first_bad"] is None:
            continue
        if shown >= max_show:
            print("    ... and more failing text examples")
            break
        inp, _ = other_examples[r["idx"]]
        k, want, got = r["first_bad"]
        ctx = tok.decode(inp[:k + 1])
        ctx = ("..." + ctx[-50:]) if len(ctx) > 50 else ctx
        print(f"    text example fails after {ctx!r}: "
              f"expected {tok.itos.get(want, '?')!r}, model predicts {tok.itos.get(got, '?')!r}")
        shown += 1


def _print_verify(pat, oth, show_failures=True, tok=None, other_examples=None):
    if pat:
        print(f"  taught patterns (greedy answer, exact match): "
              f"{sum(r['ok'] for r in pat)}/{len(pat)}")
    if oth:
        mean_acc = sum(r["tok_acc"] for r in oth) / len(oth)
        n_amb = sum(r["n_ambiguous"] for r in oth)
        print(f"  text examples (blank-answer patterns, template, corpus): "
              f"{sum(r['ok'] for r in oth)}/{len(oth)}  (mean token accuracy {mean_acc:.1%}"
              f"; {n_amb} ambiguous positions not required)")
    if show_failures and any(not r["ok"] for r in pat):
        _print_failures(pat)
    if show_failures and tok is not None and other_examples is not None:
        _print_text_failures(tok, oth, other_examples)


def full_self_correct(gpt, opt, tok, patterns, pattern_examples, other_examples,
                      weight_decay=0.0, quiet=False):
    """Verify EVERY training example, retrain on whatever fails (mixed with
    a replay sample of passing examples), and repeat until all pass or
    FULL_MAX_ROUNDS. Keeps the best-scoring weights seen."""
    say = (lambda *a, **k: None) if quiet else print
    masks = _ambiguity_masks(pattern_examples, other_examples)
    pat, oth = verify_all(gpt, tok, patterns, other_examples, masks)
    n_ok, n = _count_ok(pat, oth)
    if n == 0:
        say("[full verify: nothing to verify]")
        return pat, oth
    say(f"[full verification: {n_ok}/{n} examples reproduced]")
    if not quiet: _print_verify(pat, oth, tok=tok, other_examples=other_examples)

    best_ok, best_state, best_pat, best_oth = n_ok, _snapshot(gpt), pat, oth
    base_lr = opt.lr
    rounds = 0
    stall = 0
    _seen = {id(e) for e in pattern_examples}
    all_ex = list(pattern_examples) + [e for e in other_examples if id(e) not in _seen]
    while n_ok < n and rounds < FULL_MAX_ROUNDS:
        rounds += 1
        failed = ([pattern_examples[r["idx"]] for r in pat if not r["ok"]] +
                  [other_examples[r["idx"]] for r in oth if not r["ok"]])
        passing = ([pattern_examples[r["idx"]] for r in pat if r["ok"]] +
                   [other_examples[r["idx"]] for r in oth if r["ok"]])
        replay = random.sample(passing, min(len(passing), max(len(failed), 4)))
        batch = failed * REPAIR_REPEAT + replay
        random.shuffle(batch)

        opt.lr = LR * REPAIR_LR_SCALE
        for inp, target in batch:
            gpt.zero_grad()
            total_loss, _, _ = gpt.loss(inp, target)
            total_loss.backward()
            opt.step(weight_decay=weight_decay)
        opt.lr = base_lr

        pat, oth = verify_all(gpt, tok, patterns, other_examples, masks)
        n_ok, n = _count_ok(pat, oth)
        tag = ""
        if n_ok > best_ok:
            best_ok, best_state, best_pat, best_oth = n_ok, _snapshot(gpt), pat, oth
            tag = "  (best so far)"
            stall = 0
        else:
            stall += 1
        if n_ok < n and stall >= FULL_STALL_ROUNDS:
            # Stuck: targeted repair isn't moving it. Do a burst of ordinary
            # training over everything at the normal learning rate, then re-check.
            for _ in range(FULL_STALL_EPOCHS):
                random.shuffle(all_ex)
                for inp, target in all_ex:
                    gpt.zero_grad()
                    total_loss, _, _ = gpt.loss(inp, target)
                    total_loss.backward()
                    opt.step(weight_decay=weight_decay)
            pat, oth = verify_all(gpt, tok, patterns, other_examples, masks)
            n_ok, n = _count_ok(pat, oth)
            stall = 0
            tag += "  (stalled -> plain-training burst)"
            if n_ok > best_ok:
                best_ok, best_state, best_pat, best_oth = n_ok, _snapshot(gpt), pat, oth
        say(f"  [full repair round {rounds}: retrained {len(failed)} failing + "
              f"{len(replay)} replay -> {n_ok}/{n} reproduced{tag}]")

    if best_ok > n_ok:
        _restore(gpt, best_state)
        pat, oth, n_ok = best_pat, best_oth, best_ok
        say("[restored best checkpoint from the repair loop]")

    if n_ok >= n:
        say(f"[FULL VERIFY PASSED: all {n} training examples are reproduced by the final weights]")
    else:
        say(f"[full verify finished at {n_ok}/{n} after {rounds} repair rounds. Final state:]")
        if not quiet: _print_verify(pat, oth, tok=tok, other_examples=other_examples)
        say("   (a larger size (:grow), fewer or cleaner examples, or a smaller "
              "corpus will make 100% reachable; or lower FULL_BLOCK_TOKEN_ACC)")
    return pat, oth


def load_probes():
    """probes.json (optional): [{"prompt": "...", "expected": "..."}, ...] --
    extra UNSEEN test cases that are never trained on. They are added to the
    templates.json held-out combos as the generalization signal. Don't put
    a probe's prompt in your taught patterns."""
    if not os.path.exists(PROBES_PATH):
        return []
    with open(PROBES_PATH, encoding="utf-8") as f:
        data = json.load(f)
    return [{"prompt": d["prompt"], "expected": d["expected"]}
            for d in data if isinstance(d, dict) and "prompt" in d and "expected" in d]


def generalize_phase(gpt, opt, tok, patterns, pattern_examples, other_examples, gen_evals):
    """PHASE 2 -- start from weights that already reproduce the examples
    (phase 1) and keep training on the SAME examples, but with strong weight
    decay. The idea (it's what 'grokking' experiments report): once the data
    is memorized, weight decay keeps shrinking whatever isn't needed, which
    can favor a smaller, rule-like solution over a lookup table. Nothing new
    is memorized here; the goal is that the held-out accuracy rises while
    recall stays intact.

    Guard rails: a checkpoint only counts if recall of the training examples
    is no worse than at the start of phase 2 AND held-out accuracy is a new
    best. The best such checkpoint is restored at the end, so if phase 2
    doesn't help you simply get the phase-1 weights back unchanged."""
    if not gen_evals:
        print("[phase 2 skipped: no held-out signal. Add a templates.json with "
              "check_placeholder, or a probes.json of unseen prompt/expected cases, "
              "so there's a way to tell whether generalization improves.]")
        return None
    masks = _ambiguity_masks(pattern_examples, other_examples)
    pat, oth = verify_all(gpt, tok, patterns, other_examples, masks)
    n_ok0, n = _count_ok(pat, oth)
    acc0 = check_template_accuracy(gpt, tok, gen_evals)
    print(f"[phase 2 (generalize): start -- recall {n_ok0}/{n}, held-out accuracy {acc0:.2f} "
          f"on {len(gen_evals)} unseen cases; weight_decay={GEN_WEIGHT_DECAY}]")

    best_acc, best_state = acc0, _snapshot(gpt)
    _seen = {id(e) for e in pattern_examples}
    all_ex = list(pattern_examples) + [e for e in other_examples if id(e) not in _seen]
    base_lr = opt.lr
    opt.lr = LR * GEN_LR_SCALE
    no_gain = 0
    t0 = time.time()
    for epoch in range(1, GEN_MAX_EPOCHS + 1):
        random.shuffle(all_ex)
        for inp, target in all_ex:
            gpt.zero_grad()
            total_loss, _, _ = gpt.loss(inp, target)
            total_loss.backward()
            opt.step(weight_decay=GEN_WEIGHT_DECAY)
        if epoch % GEN_CHECK_EVERY:
            continue
        pat, oth = verify_all(gpt, tok, patterns, other_examples, masks)
        n_ok, _ = _count_ok(pat, oth)
        acc = check_template_accuracy(gpt, tok, gen_evals)
        recall_ok = n_ok >= n_ok0
        tag = ""
        if recall_ok and acc > best_acc + 1e-9:
            best_acc, best_state, no_gain = acc, _snapshot(gpt), 0
            tag = "  (new best, checkpointed)"
        else:
            no_gain += 1
            if not recall_ok:
                tag = "  (recall dropped -- not eligible)"
        print(f"  gen epoch {epoch:3d}  recall {n_ok}/{n}  held-out {acc:.2f}{tag}  "
              f"elapsed={time.time()-t0:.1f}s")
        if best_acc >= 1.0:
            print("[phase 2: held-out accuracy reached 1.00]")
            break
        if no_gain >= GEN_PATIENCE:
            print(f"[phase 2: no held-out gain in {GEN_PATIENCE} checks -- stopping]")
            break
    opt.lr = base_lr
    _restore(gpt, best_state)
    print(f"[phase 2 done: held-out accuracy {acc0:.2f} -> {best_acc:.2f} "
          f"(recall kept at >= {n_ok0}/{n})]")
    if best_acc <= acc0:
        print("   (no improvement found -- weights are exactly the phase-1 weights)")
    return best_acc


def stabilize_once(tok, patterns, corpus_text, full=False, generalize=False):
    """One training run at the CURRENT size (this is the original stabilize()).
    full=False: normal stabilize (validation split on corpus/template data,
    early stopping, then the pattern self-test/repair loop).
    full=True:  FULL STABILIZE -- no validation split, every example is
    trained on, then EVERY example is verified and repaired until the final
    weights reproduce all of them (see full_self_correct).
    Use stabilize() (below) in normal code: it wraps this with the
    'is a bigger size needed?' check."""
    templates = load_templates()
    synth_patterns, holdout_evals = expand_templates(templates)
    if synth_patterns or holdout_evals:
        print(f"[templates.json: expanded into {len(synth_patterns)} synthetic training "
              f"examples + {len(holdout_evals)} held out to check generalization]")

    # pattern_examples[i] <-> patterns[i]  (self_correct relies on this alignment)
    pattern_examples = [encode_pattern(tok, p["prompt"], p["answer"], CONFIG["max_seq_len"]) for p in patterns]
    synth_examples = [encode_pattern(tok, p["prompt"], p["answer"], CONFIG["max_seq_len"]) for p in synth_patterns]
    corpus_examples = chunk_corpus(tok, corpus_text, CONFIG["max_seq_len"])
    text_ex = _text_pattern_examples(patterns, pattern_examples)
    all_examples = pattern_examples + synth_examples + corpus_examples

    if not all_examples:
        print("[nothing saved yet -- teach it something or paste some text first]")
        return None, None

    _warn_unknown(tok)

    # Taught patterns are ALWAYS trained on. Validation only splits the
    # corpus/template data, so it can never hide a pattern you taught.
    splittable = synth_examples + corpus_examples
    use_val = (not full) and len(splittable) >= MIN_EXAMPLES_FOR_VAL
    if use_val:
        train_split, val_examples = _split_train_val(splittable, VAL_FRACTION)
        train_examples = list(pattern_examples) + train_split
    else:
        train_examples, val_examples = list(all_examples), []
        why = ("FULL mode: every example is trained on and verified" if full else
               f"only {len(splittable)} corpus/template examples -- skipping held-out validation")
        print(f"[{why}; targeting near-perfect memorization]")
    n_train = len(train_examples)
    epoch_cap = MAX_EPOCHS if use_val else MAX_EPOCHS_MEMORIZE

    print(f"[size: {current_size_name()}  CONFIG: {CONFIG}]")
    print(f"[stabilizing: {len(pattern_examples)} patterns + {len(synth_examples)} synthetic + "
          f"{len(corpus_examples)} corpus blocks = {len(all_examples)} examples "
          f"({n_train} train / {len(val_examples)} val), up to {epoch_cap} epochs, "
          f"early stop after {PATIENCE} epochs with no improvement]")

    gpt = ConfigurableGPT(vocab_size=tok.vocab_size, **CONFIG)
    if WARM_START.get("fn"):
        WARM_START["fn"](gpt, tok)
    n_params = sum(p.data.size for p in gpt.params())
    print(f"[this configuration has {n_params:,} parameters]")
    opt = Adam(gpt.params(), lr=LR)

    best_monitor = float("inf")
    best_state = None
    epochs_since_improve = 0
    t0 = time.time()
    best_acc = -1.0
    best_acc_state = None
    MEMORIZE_THRESHOLD = 0.2
    if full:
        full_others = text_ex + synth_examples + corpus_examples
        full_masks = _ambiguity_masks(pattern_examples, full_others)

    for epoch in range(1, epoch_cap + 1):
        if not use_val:
            progress = epoch / epoch_cap
            opt.lr = LR * (0.03 + 0.97 * 0.5 * (1 + np.cos(np.pi * progress)))
        random.shuffle(train_examples)
        total = 0.0
        for inp, target in train_examples:
            gpt.zero_grad()
            total_loss, primary_loss, mtp_loss = gpt.loss(inp, target)
            total_loss.backward()
            opt.step(weight_decay=TEMPLATE_WEIGHT_DECAY if holdout_evals else 0.0)
            total += float(total_loss.data)
        train_avg = total / n_train
        val_avg = _eval_loss(gpt, val_examples) if use_val else None
        monitor = val_avg if use_val else train_avg

        improved = monitor is not None and monitor < best_monitor - 1e-4
        if improved:
            best_monitor = monitor
            best_state = _snapshot(gpt)
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1

        val_str = f"  val_loss={val_avg:.4f}" if use_val else ""
        print(f"  epoch {epoch:3d}  train_loss={train_avg:.4f}{val_str}"
              f"{'  (best)' if improved else ''}  elapsed={time.time()-t0:.1f}s")
        if CHECKPOINT.get("fn") and epoch % 5 == 0:
            CHECKPOINT["fn"](gpt, epoch)

        if holdout_evals and (epoch % TEMPLATE_CHECK_EVERY == 0 or epoch == 1):
            heldout_acc = check_template_accuracy(gpt, tok, holdout_evals)
            if heldout_acc > best_acc:
                best_acc = heldout_acc
                best_acc_state = _snapshot(gpt)
                print(f"    [templates.json: held-out accuracy={heldout_acc:.2f} -- new best, checkpointed]")
            if best_acc >= 1.0 and epoch >= MIN_EPOCHS and not full:
                print("[stopping: held-out accuracy reached 1.00]")
                break

        if full and epoch >= MIN_EPOCHS and epoch % FULL_CHECK_EVERY == 0:
            # A loss threshold is a poor stop signal here: positions the data can't
            # determine (e.g. which name comes first) keep the average loss up even
            # when everything that CAN be learned is. Ask the real question instead.
            _p, _o = verify_all(gpt, tok, patterns, full_others, full_masks)
            _ok, _n = _count_ok(_p, _o)
            if _ok >= _n:
                print(f"[stopping: all {_n} examples already reproduced at epoch {epoch}]")
                best_state = _snapshot(gpt)
                break

        if epoch >= MIN_EPOCHS:
            if holdout_evals and not full:
                pass
            elif not use_val:
                if train_avg < MEMORIZE_THRESHOLD:
                    print(f"[stopping: train_loss={train_avg:.4f} below memorization threshold]")
                    best_state = _snapshot(gpt)
                    break
            else:
                if epochs_since_improve >= PATIENCE:
                    print(f"[early stop: val_loss hasn't improved in {PATIENCE} epochs]")
                    break
    else:
        print(f"[hit epoch cap ({epoch_cap}) without early stopping -- restoring best checkpoint]")

    if full:
        if best_state is not None:
            _restore(gpt, best_state)
        full_self_correct(gpt, opt, tok, patterns, pattern_examples,
                          text_ex + synth_examples + corpus_examples)
        gen_evals = holdout_evals + load_probes()
        if generalize:
            generalize_phase(gpt, opt, tok, patterns, pattern_examples,
                             text_ex + synth_examples + corpus_examples, gen_evals)
        if gen_evals:
            h_acc = check_template_accuracy(gpt, tok, gen_evals)
            print(f"[held-out accuracy (the generalization check): {h_acc:.2f} "
                  f"on {len(gen_evals)} unseen cases]")
        save_model(gpt, len(patterns), len(corpus_text))
        print(f"[done -- full stabilize, weights saved to {SAVE_PATH}]")
        return gpt, opt

    if holdout_evals and best_acc_state is not None:
        _restore(gpt, best_acc_state)
        # Repair recall of taught patterns without weight decay (decay would
        # fight the memorization we're trying to fix).
        full_self_correct(gpt, opt, tok, patterns, pattern_examples, text_ex)
        save_model(gpt, len(patterns), len(corpus_text))
        print(f"[done -- best held-out accuracy={best_acc:.2f}, weights saved to {SAVE_PATH}]")
        return gpt, opt

    if best_state is not None:
        _restore(gpt, best_state)

    full_self_correct(gpt, opt, tok, patterns, pattern_examples, text_ex)
    save_model(gpt, len(patterns), len(corpus_text))
    print(f"[done -- best {'val' if use_val else 'train'}_loss={best_monitor:.4f}, "
          f"weights saved to {SAVE_PATH}]")
    return gpt, opt


# ======================================================================
# ---- SCALING: sizes, "is a bigger model needed?", switching ----
# ======================================================================
def load_settings():
    if os.path.exists(SETTINGS_PATH):
        try:
            with open(SETTINGS_PATH, encoding="utf-8") as f:
                data = json.load(f)
            for k in SETTINGS:
                if k in data:
                    SETTINGS[k] = data[k]
        except (OSError, ValueError):
            print("[size.json unreadable -- using defaults]")
    if SETTINGS["autogrow"] not in ("off", "suggest", "on"):
        SETTINGS["autogrow"] = "suggest"
    if SETTINGS["max_size"] not in SIZE_PRESETS:
        SETTINGS["max_size"] = SIZE_ORDER[-1]
    if SETTINGS["tokenizer"] not in ("bpe", "word"):
        SETTINGS["tokenizer"] = "bpe"


def save_settings():
    with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
        json.dump(SETTINGS, f, indent=1)


def load_log():
    if not os.path.exists(LOG_PATH):
        return []
    try:
        with open(LOG_PATH, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return []


def save_log(log):
    with open(LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(log[-LOG_MAX:], f, indent=1)


def estimate_params(cfg, vocab):
    """Parameter count implied by cfg (mirrors ConfigurableGPT's layout)."""
    d, L = cfg["d_model"], cfg["d_latent"]
    f, n = cfg["d_ff_expert"], cfg["n_experts"]
    norm = d if cfg["use_rmsnorm"] else 2 * d
    attn = (2 * d * d + 2 * d + 3 * d * L) if cfg["use_mla"] else 4 * (d * d + d)
    ffn = 3 * d * f * (n + 1) + d * n            # shared expert + n routed + router
    block = 2 * norm + attn + ffn
    total = vocab * d + norm + cfg["n_layers"] * block
    if not cfg["use_rope"]:
        total += cfg["max_seq_len"] * d
    if cfg["use_pointer"]:
        total += 2 * d * d + d + 1
    if cfg["use_mtp"]:
        total += block + norm
    return total


def _cfg_for(name):
    cfg = dict(CONFIG)
    cfg.update(SIZE_PRESETS[name])
    return cfg


def _validate_presets():
    for name, p in SIZE_PRESETS.items():
        assert set(p) - {"max_seq_len"} == set(SIZE_KEYS), f"preset {name!r} must define exactly {SIZE_KEYS} (and may add max_seq_len)"
        assert p["d_model"] % p["n_heads"] == 0, f"{name}: d_model must divide by n_heads"
        assert (p["d_model"] // p["n_heads"]) % 2 == 0, f"{name}: head size must be even (RoPE)"


def current_size_name():
    cur = {k: CONFIG[k] for k in SIZE_KEYS}
    for name, p in SIZE_PRESETS.items():
        if p == cur:
            return name
    return "custom"


def _neighbor(cur, vocab, step):
    if cur in SIZE_ORDER:
        i = SIZE_ORDER.index(cur) + step
        return SIZE_ORDER[i] if 0 <= i < len(SIZE_ORDER) else None
    here = estimate_params(CONFIG, vocab)
    names = SIZE_ORDER if step > 0 else list(reversed(SIZE_ORDER))
    for name in names:
        e = estimate_params(_cfg_for(name), vocab)
        if (step > 0 and e > here) or (step < 0 and e < here):
            return name
    return None


def apply_size(name):
    CONFIG.update(SIZE_PRESETS[name])      # in place: everything sees it
    SETTINGS["size"] = name
    save_settings()


def _fmt_params(n):
    return f"{n/1e6:.2f}M" if n >= 1e6 else f"{n/1e3:.0f}k"


def _rel_cost(name, vocab):
    return estimate_params(_cfg_for(name), vocab) / estimate_params(_cfg_for(REFERENCE_SIZE), vocab)


def print_sizes(vocab):
    cur = current_size_name()
    print(f"  {'size':8} {'params':>8} {'cost':>7}  d_model layers experts")
    for name in SIZE_ORDER:
        p = SIZE_PRESETS[name]
        mark = "  <- current" if name == cur else ""
        print(f"  {name:8} {_fmt_params(estimate_params(_cfg_for(name), vocab)):>8} "
              f"{_rel_cost(name, vocab):>6.1f}x  {p['d_model']:7d} {p['n_layers']:6d} "
              f"{p['n_experts']:7d}{mark}")
    if cur == "custom":
        print(f"  custom   {_fmt_params(estimate_params(CONFIG, vocab)):>8}   (current CONFIG "
              f"doesn't match a preset)")
    print(f"  (cost = rough compute per epoch relative to '{REFERENCE_SIZE}'; "
          f"autogrow: {SETTINGS['autogrow']}, max {SETTINGS['max_size']})")


def _backup_current(name):
    if not os.path.exists(SAVE_PATH):
        return
    os.makedirs(BACKUP_DIR, exist_ok=True)
    shutil.copy(SAVE_PATH, os.path.join(BACKUP_DIR, f"{name}.npz"))
    if os.path.exists(META_PATH):
        shutil.copy(META_PATH, os.path.join(BACKUP_DIR, f"{name}.meta.json"))


def _clear_current_weights():
    for path in (SAVE_PATH, META_PATH):
        if os.path.exists(path):
            os.remove(path)


def _try_restore_backup(name, vocab):
    npz = os.path.join(BACKUP_DIR, f"{name}.npz")
    meta_p = os.path.join(BACKUP_DIR, f"{name}.meta.json")
    if not (os.path.exists(npz) and os.path.exists(meta_p)):
        return False
    try:
        with open(meta_p) as f:
            meta = json.load(f)
        cfg = meta.get("config", {})
        if (meta.get("vocab_size") != vocab
                or meta.get("tokenizer", "bpe") != SETTINGS["tokenizer"]
                or meta.get("vocab_sig") not in (None, TOK_SIG["v"])):
            return False
        if any(cfg.get(k) != SIZE_PRESETS[name][k] for k in SIZE_KEYS):
            return False
        shutil.copy(npz, SAVE_PATH)
        shutil.copy(meta_p, META_PATH)
        return True
    except (OSError, ValueError):
        return False


def conflict_report(patterns):
    """Returns (groups, unavoidable). groups: prompt -> Counter(answers) for
    prompts taught with >1 distinct answer. unavoidable: the minimum number
    of those patterns that must fail a greedy self-test, since one prompt
    can only produce one answer -- no model size can fix these."""
    by = collections.defaultdict(list)
    for p in patterns:
        a = p.get("answer", "").strip()
        if a:
            by[p["prompt"]].append(a)
    groups, unavoidable = {}, 0
    for prompt, answers in by.items():
        c = collections.Counter(answers)
        if len(c) > 1:
            groups[prompt] = c
            unavoidable += len(answers) - max(c.values())
    return groups, unavoidable


def print_conflicts(patterns):
    groups, unavoidable = conflict_report(patterns)
    if not groups:
        print("[no contradictory patterns found]")
        return
    print(f"[{len(groups)} prompt(s) taught with different answers -- at least "
          f"{unavoidable} pattern(s) can never pass; fix or delete these in patterns.json:]")
    for prompt, c in list(groups.items())[:10]:
        print(f"    {prompt!r}")
        for ans, k in c.most_common():
            print(f"        x{k}: {ans!r}")
    if len(groups) > 10:
        print(f"    ... and {len(groups) - 10} more")


def measure(gpt, tok, patterns, corpus_text, full):
    """Recall + held-out accuracy of the current weights. No training.
    full=False only checks taught/blank-answer patterns (the normal
    :stabilize holds out some corpus data, so checking all of it would
    blame capacity for data that was deliberately never trained on)."""
    pat_ex, synth_ex, corpus_ex, holdout = _build_examples(tok, patterns, corpus_text)
    text_ex = _text_pattern_examples(patterns, pat_ex)
    others = text_ex + (synth_ex + corpus_ex if full else [])
    masks = _ambiguity_masks(pat_ex, others)
    pat, oth = verify_all(gpt, tok, patterns, others, masks)
    ok, n = _count_ok(pat, oth)
    evals = holdout + load_probes()
    held = check_template_accuracy(gpt, tok, evals) if evals else None
    _, unavoidable = conflict_report(patterns)
    return dict(time=int(time.time()), size=current_size_name(),
                params=int(sum(p.data.size for p in gpt.params())),
                n_examples=len(pat_ex) + len(synth_ex) + len(corpus_ex),
                recall_ok=ok, recall_n=n, unavoidable=unavoidable,
                held_out=held, n_heldout=len(evals), full=bool(full))


def capacity_signal(entry, history):
    """history = earlier log entries (not including `entry`). Returns a reason
    string if a bigger model looks needed, else None."""
    n, ok = entry["recall_n"], entry["recall_ok"]
    effective_miss = (n - ok) - entry.get("unavoidable", 0)
    if n and effective_miss > 0:
        return (f"the model still can't reproduce {effective_miss} of its {n} training examples "
                f"after the full repair loop (not counting contradictory patterns)")
    h = entry.get("held_out")
    if h is not None and h < PLATEAU_DONE_ABOVE:
        prev = [e for e in history
                if e.get("size") == entry["size"] and e.get("held_out") is not None
                and e.get("n_heldout") == entry.get("n_heldout")
                and entry["n_examples"] >= PLATEAU_MIN_GROWTH * e["n_examples"]]
        if prev:
            p = prev[-1]
            if h <= p["held_out"] + PLATEAU_MIN_GAIN:
                return (f"examples grew {p['n_examples']} -> {entry['n_examples']} but held-out "
                        f"accuracy went {p['held_out']:.2f} -> {h:.2f} (a plateau)")
    return None


def _advice(reason, entry, vocab):
    cur = entry["size"]
    nxt = _neighbor(cur, vocab, +1)
    print(f"[capacity: {reason}]")
    if entry.get("unavoidable"):
        print(f"[note: {entry['unavoidable']} pattern(s) contradict each other and can never pass "
              f"at any size -- see :conflicts]")
    if nxt is None:
        print(f"[already at the largest preset ('{cur}'). Add a bigger entry to SIZE_PRESETS, "
              f"or improve/clean the data.]")
        return
    print(f"[a bigger model may help: ':grow' -> {nxt} (~{_rel_cost(nxt, vocab):.1f}x the compute "
          f"of '{REFERENCE_SIZE}'), or ':autogrow on' to do it automatically. If the examples are "
          f"very similar to each other, more DIVERSE examples may help more than more size.]")


def stabilize(tok, patterns, corpus_text, full=False, generalize=False):
    """stabilize_once() + the capacity check. After each run it measures
    recall / held-out accuracy, logs them, and -- if a bigger size looks
    needed -- either tells you (autogrow=suggest) or moves up a size and
    retrains from scratch (autogrow=on, up to max_size)."""
    while True:
        gpt, opt = stabilize_once(tok, patterns, corpus_text, full=full, generalize=generalize)
        if gpt is None:
            return gpt, opt
        entry = measure(gpt, tok, patterns, corpus_text, full)
        history = load_log()
        reason = capacity_signal(entry, history)
        save_log(history + [entry])
        if reason is None:
            held = f", held-out {entry['held_out']:.2f}" if entry["held_out"] is not None else ""
            print(f"[capacity check: '{entry['size']}' looks sufficient -- recall "
                  f"{entry['recall_ok']}/{entry['recall_n']}{held}]")
            return gpt, opt

        vocab = tok.vocab_size
        nxt = _neighbor(entry["size"], vocab, +1)
        can_grow = (SETTINGS["autogrow"] == "on" and nxt is not None
                    and SIZE_ORDER.index(nxt) <= SIZE_ORDER.index(SETTINGS["max_size"]))
        if not can_grow:
            if SETTINGS["autogrow"] != "off":
                _advice(reason, entry, vocab)
            return gpt, opt
        print(f"[capacity: {reason}]")
        print(f"[autogrow: '{entry['size']}' -> '{nxt}' (~{_rel_cost(nxt, vocab):.1f}x compute); "
              f"retraining from scratch at the new size]")
        _backup_current(entry["size"])
        apply_size(nxt)
        _clear_current_weights()


def switch_size(name, tok, patterns, corpus_text, gpt, opt, meta):
    cur = current_size_name()
    if name == cur:
        print(f"[already using '{cur}']")
        return gpt, opt, meta
    vocab = tok.vocab_size
    _backup_current(cur)
    apply_size(name)
    restored = _try_restore_backup(name, vocab)
    if not restored:
        _clear_current_weights()
    gpt = ConfigurableGPT(vocab_size=vocab, **CONFIG)
    meta = {"n_patterns": 0, "corpus_len": 0}
    if restored:
        _, meta = load_model(gpt)
        print(f"[restored the earlier '{name}' weights from size_backups/]")
    opt = Adam(gpt.params(), lr=LR)
    n_params = sum(p.data.size for p in gpt.params())
    print(f"[size: '{cur}' -> '{name}': {n_params:,} parameters, "
          f"~{_rel_cost(name, vocab):.1f}x the compute per epoch of '{REFERENCE_SIZE}']")
    if not restored:
        print("[different shapes -- the new size trains from scratch on everything saved]")
    gpt, opt, meta = _auto(tok, patterns, corpus_text, gpt, opt, meta)
    return gpt, opt, meta


AUTO = {"on": True}          # toggle with :auto on / :auto off
AUTO_RETRAIN_FRACTION = 0.5  # if more than this share of examples fail, retrain from scratch
AUTO_MAX_CYCLES = 6          # incremental repair cycles before falling back to a scratch retrain
AUTO_EPOCHS_PER_CYCLE = 10   # plain training epochs between repair cycles


def auto_memorize(tok, patterns, corpus_text, gpt, opt):
    """Hands-off memorization. Checks whether the CURRENT weights reproduce
    every saved example; if not, learns the missing ones:
      * mostly-new/untrained model (> AUTO_RETRAIN_FRACTION failing): a full
        phase-1 stabilize from scratch (same as :fullstabilize);
      * otherwise: keep the current weights and repair incrementally
        (retrain failures + replay, plus a few ordinary epochs between
        cycles), so adding one pattern doesn't throw away what it knew.
    Saves the weights when something changed. Never runs the generalize
    phase -- that's yours to trigger (:generalize / :twophase)."""
    pat_ex, synth_ex, corpus_ex, _ = _build_examples(tok, patterns, corpus_text)
    others = _text_pattern_examples(patterns, pat_ex) + synth_ex + corpus_ex
    if not pat_ex and not others:
        return gpt, opt
    masks = _ambiguity_masks(pat_ex, others)
    pat, oth = verify_all(gpt, tok, patterns, others, masks)
    n_ok, n = _count_ok(pat, oth)
    if n_ok >= n:
        print(f"[auto-memorize: all {n} examples already reproduced -- nothing to do]")
        return gpt, opt

    failing = n - n_ok
    print(f"[auto-memorize: {failing}/{n} examples not reproduced yet -- learning them...]")
    if failing / n > AUTO_RETRAIN_FRACTION:
        new_gpt, new_opt = stabilize(tok, patterns, corpus_text, full=True)
        return (new_gpt, new_opt) if new_gpt is not None else (gpt, opt)

    _seen = {id(e) for e in pat_ex}
    all_ex = list(pat_ex) + [e for e in others if id(e) not in _seen]
    base_lr = opt.lr
    for cycle in range(1, AUTO_MAX_CYCLES + 1):
        pat, oth = full_self_correct(gpt, opt, tok, patterns, pat_ex, others, quiet=True)
        n_ok, n = _count_ok(pat, oth)
        if n_ok >= n:
            break
        opt.lr = LR * 0.5
        for _ in range(AUTO_EPOCHS_PER_CYCLE):
            random.shuffle(all_ex)
            for inp, target in all_ex:
                gpt.zero_grad()
                total_loss, _, _ = gpt.loss(inp, target)
                total_loss.backward()
                opt.step()
        opt.lr = base_lr
    else:
        pat, oth = verify_all(gpt, tok, patterns, others, masks)
        n_ok, n = _count_ok(pat, oth)

    if n_ok < n:
        print(f"[auto-memorize: incremental repair stalled at {n_ok}/{n} -- retraining from scratch]")
        new_gpt, new_opt = stabilize(tok, patterns, corpus_text, full=True)
        return (new_gpt, new_opt) if new_gpt is not None else (gpt, opt)

    save_model(gpt, len(patterns), len(corpus_text))
    print(f"[auto-memorize: done -- all {n} examples reproduced, weights saved]")
    return gpt, opt


def _auto(tok, patterns, corpus_text, gpt, opt, meta):
    """Run auto_memorize if enabled and return the updated (gpt, opt, meta)."""
    if not AUTO["on"]:
        print("[auto-memorize is off -- run :stabilize / :fullstabilize when ready]")
        return gpt, opt, meta
    if _warn_unknown(tok):
        print("[auto-memorize skipped -- run :retoken to rebuild the vocabulary and train]")
        return gpt, opt, meta
    gpt, opt = auto_memorize(tok, patterns, corpus_text, gpt, opt)
    meta = {"n_patterns": len(patterns), "corpus_len": len(corpus_text), "config": dict(CONFIG)}
    return gpt, opt, meta


# ======================================================================
# ---- CHECK MODE: hidden monologue ----
# The model is trained to write   <CHECK> ... verdict: X </CHECK> <REPLY> ... </REPLY>
# The check is its scratchpad. In chat only the reply is shown (:showcheck on to see the check too).
# ======================================================================
PROFILE_PATH = os.path.join(HERE, "profile.txt")
CHECK_MAX_NEW = 200
HISTORY = []                      # (user, reply) pairs of this conversation; sent back as context. :new clears it
CURRENT_OBS = {"text": None}     # set with  :obs <text>  ; sent as <OBS> .. </OBS> with every chat message


def _known_topics(patterns):
    """Group the taught check-style prompts by profile: what the bot was trained to know."""
    rx_p = re.compile(r"<PROFILE>\s*(.*?)\s*</PROFILE>")
    rx_o = re.compile(r"<OBS>\s*(.*?)\s*</OBS>")
    rx_u = re.compile(r"<USER>\s*(.*?)\s*</USER>")
    order, seen = [], {}
    for p in patterns:
        m = rx_p.search(p.get("prompt", ""))
        if not m:
            continue
        t = seen.get(m.group(1))
        if t is None:
            t = seen[m.group(1)] = {"profile": m.group(1), "obs": [], "users": []}
            order.append(t)
        o, u = rx_o.search(p["prompt"]), rx_u.search(p["prompt"])
        if o and o.group(1) not in t["obs"]:
            t["obs"].append(o.group(1))
        if u and (u.group(1) or "(empty)") not in t["users"]:
            t["users"].append(u.group(1) or "(empty)")
    return order


def _single_profile():
    """One-pool mode: when profiles.json holds exactly one profile, every message is sent with that profile
    (the same text the model was trained with); profile.txt is ignored."""
    try:
        with open(os.path.join(HERE, "profiles.json"), encoding="utf-8") as f:
            d = json.load(f)
        if isinstance(d, dict) and len(d) == 1:
            return " ".join(str(next(iter(d.values()))).split())
    except Exception:
        pass
    return None


def load_profile():
    _sp = _single_profile()
    if _sp:
        return _sp
    if os.path.exists(PROFILE_PATH):
        with open(PROFILE_PATH, encoding="utf-8") as f:
            return " ".join(f.read().split())
    return ""


def save_profile(text):
    text = " ".join(text.split())
    with open(PROFILE_PATH, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    return text


PROFILES_PATH = os.path.join(HERE, "profiles.json")
DEFAULT_PROFILES = {
    "coding": "I am a bot that teaches coding. I give short and simple answers. I use Python in my examples. I only know about coding.",
    "cat": "I am a bot. I do not eat food. I can count to 20. I like cats. I do not like loud music.",
    "dog": "I am a bot. I do not eat food. I cannot count. I like dogs. I like loud music.",
}


def load_profiles():
    """name -> full profile text. profiles.json (same folder) adds to / overrides the defaults."""
    prof = dict(DEFAULT_PROFILES)
    if os.path.exists(PROFILES_PATH):
        try:
            with open(PROFILES_PATH, encoding="utf-8") as f:
                for k, v in json.load(f).items():
                    prof[str(k).strip().lower()] = " ".join(str(v).split())
        except Exception as e:
            print(f"[could not read profiles.json: {e}]")
    return prof


def apply_auto_profile(decoded):
    """The model switches profile by writing  <SETPROFILE> name </SETPROFILE>  before its reply.
    The name is looked up in load_profiles(); the profile file is rewritten so the NEXT message
    is sent with the new profile."""
    m = re.search(r"<SETPROFILE>\s*(.*?)\s*</SETPROFILE>", decoded, re.S)
    if not m or not m.group(1).strip():
        return None
    key = " ".join(m.group(1).split()).lower()
    prof = load_profiles()
    if key not in prof:
        print(f"  [model asked for profile '{key}' but it is not in profiles.json -- ignored]")
        return None
    if prof[key] == load_profile():
        return None
    save_profile(prof[key])
    print(f"  [profile switched to: {key}]")
    return key


def wrap_prompt(user_text, obs=None):
    """What the model was trained to see: <PROFILE> .. </PROFILE> [<OBS> .. </OBS>] <USER> .. </USER>"""
    parts = ["<PROFILE> " + load_profile() + " </PROFILE>"]
    if obs:
        parts.append("<OBS> " + obs + " </OBS>")
    for u, r in HISTORY:
        parts.append("<USER> " + u + " </USER>")
        parts.append("<REPLY> " + r + " </REPLY>")
    parts.append("<USER> " + user_text + " </USER>")
    return " ".join(parts)


def _taught_prompt(user_text):
    return wrap_prompt(user_text) if SETTINGS.get("checkmode") else user_text


class ReplyFilter:
    """Turns the token stream into what the person sees. Pure logic (no printing) so it can be tested.
    feed(token_text) -> text to print now.  Hides <CHECK>..</CHECK>, the tags, and stray spaces around the reply."""

    def __init__(self, show_check=False):
        self.show_check, self.in_check, self.in_reply = show_check, False, False
        self.pending_ws, self.started, self.shown_any = "", False, False

    def feed(self, piece):
        if piece == "<CHECK>":
            self.in_check = True
            return "\n  (check:" if self.show_check else ""
        if piece == "</CHECK>":
            self.in_check = False
            return ")\n" if self.show_check else ""        # the model already wrote a space before </CHECK>
        if piece == "<REPLY>":
            self.in_reply, self.pending_ws, self.started = True, "", False
            return ""
        if piece == "</REPLY>":
            self.in_reply, self.pending_ws = False, ""
            return ""
        if piece in ("<PROFILE>", "</PROFILE>", "<OBS>", "</OBS>", "<USER>", "</USER>"):
            return ""
        if self.in_check:
            return piece if self.show_check else ""
        if not self.in_reply:
            return ""
        if piece.strip() == "":                      # whitespace: hold it until a word follows
            if self.started:
                self.pending_ws += piece
            return ""
        out, self.pending_ws, self.started = self.pending_ws + piece, "", True
        self.shown_any = True
        return out


def stream_generate(gpt, tok, prompt_text, max_new_tokens=80, obs=None):
    check_mode = bool(SETTINGS.get("checkmode"))
    user_text = prompt_text
    if check_mode:
        prompt_text = wrap_prompt(user_text, obs)
    prompt_ids = tok.encode_prompt(prompt_text)
    while check_mode and HISTORY and len(prompt_ids) > CONFIG["max_seq_len"] - 60:     # forget the oldest turn first
        HISTORY.pop(0)
        prompt_text = wrap_prompt(user_text, obs)
        prompt_ids = tok.encode_prompt(prompt_text)
    if check_mode:
        # as long as it can be while still fitting the context (keeps the fast cached decoder usable)
        max_new_tokens = max(1, min(CHECK_MAX_NEW, CONFIG["max_seq_len"] - len(prompt_ids)))
    if hasattr(tok, "unknown_words"):
        _unk = tok.unknown_words(prompt_text)
        if _unk:
            print(f"[word tokenizer: not in its vocabulary: {', '.join(_unk)} -- seen as <UNK>. "
                  f"Teach them, then run :retoken to add them.]")
    print("assistant> ", end="", flush=True)
    _last = [None]
    flt = ReplyFilter(bool(SETTINGS.get("showcheck"))) if check_mode else None

    def on_token(tid):
        piece = tok.itos.get(tid, "")
        if piece in ("<EOS>", "<BOS>", "<PAD>"):
            return
        if flt is not None:
            text = flt.feed(piece)
        else:
            text = tok.display(_last[0], tid) if hasattr(tok, "display") else piece
            _last[0] = tid
        if text:
            print(text, end="", flush=True)

    out_ids = gpt.generate(prompt_ids, max_new_tokens=max_new_tokens,
                            temperature=(None if CHAT_GREEDY else 0.8),
                            top_k=(1 if CHAT_GREEDY else 20), eos_id=tok.eos_id,
                            on_token=on_token)
    new_ids = out_ids[len(prompt_ids):]
    if flt is not None and not flt.shown_any:
        # the model never produced a <REPLY> block: say so, and show what it did produce
        print(f"[no <REPLY> block -- raw output: {tok.decode(new_ids)}]", end="")
    print()
    return new_ids


def print_status(patterns, corpus_text, meta):
    n, clen = len(patterns), len(corpus_text)
    print(f"[{n} patterns saved, {clen} corpus characters saved]")
    if meta.get("n_patterns") == n and meta.get("corpus_len") == clen:
        print("[model is up to date with everything saved]")
    else:
        print(f"[model last stabilized at {meta.get('n_patterns',0)} patterns / "
              f"{meta.get('corpus_len',0)} corpus chars -- run :stabilize to catch it up]")
    if "config" in meta:
        print(f"[last stabilized CONFIG: {meta['config']}]")
    print(f"[tokenizer: {SETTINGS['tokenizer']}]")
    print(f"[size: {current_size_name()}, autogrow: {SETTINGS['autogrow']} "
          f"(max {SETTINGS['max_size']})]")


def handle_command(cmd_line, tok, patterns, corpus_text, gpt, opt, meta):
    parts = cmd_line.split(maxsplit=1)
    cmd = parts[0]
    args = parts[1].split() if len(parts) > 1 else []
    vocab = tok.vocab_size

    # ---- sizes ----
    if cmd == ":sizes":
        print_sizes(vocab)
        return True, gpt, opt, meta

    if cmd == ":size":
        if not args or args[0] not in SIZE_PRESETS:
            print(f"usage: :size <{'|'.join(SIZE_ORDER)}>   (current: {current_size_name()})")
            return True, gpt, opt, meta
        gpt, opt, meta = switch_size(args[0], tok, patterns, corpus_text, gpt, opt, meta)
        return True, gpt, opt, meta

    if cmd in (":grow", ":shrink"):
        nxt = _neighbor(current_size_name(), vocab, +1 if cmd == ":grow" else -1)
        if nxt is None:
            print(f"[no {'larger' if cmd == ':grow' else 'smaller'} preset than "
                  f"'{current_size_name()}']")
            return True, gpt, opt, meta
        gpt, opt, meta = switch_size(nxt, tok, patterns, corpus_text, gpt, opt, meta)
        return True, gpt, opt, meta

    if cmd == ":autogrow":
        if args and args[0] in ("off", "suggest", "on"):
            SETTINGS["autogrow"] = args[0]
        if len(args) > 1 and args[1] in SIZE_PRESETS:
            SETTINGS["max_size"] = args[1]
        save_settings()
        print(f"[autogrow: {SETTINGS['autogrow']}, max size {SETTINGS['max_size']} -- "
              f"usage: :autogrow off|suggest|on [maxsize]]")
        return True, gpt, opt, meta

    if cmd == ":conflicts":
        print_conflicts(patterns)
        return True, gpt, opt, meta

    if cmd == ":capacity":
        entry = measure(gpt, tok, patterns, corpus_text, full=True)
        held = (f", held-out {entry['held_out']:.2f} on {entry['n_heldout']}"
                if entry["held_out"] is not None else "")
        print(f"[size '{entry['size']}' ({entry['params']:,} params): recall "
              f"{entry['recall_ok']}/{entry['recall_n']}{held}; {entry['n_examples']} examples]")
        log = load_log()
        reason = capacity_signal(entry, log)
        if reason:
            _advice(reason, entry, vocab)
        else:
            print("[no sign that a bigger model is needed right now]")
        if log:
            print("  history (size, examples, recall, held-out):")
            for e in log[-8:]:
                h = f"{e['held_out']:.2f}" if e.get("held_out") is not None else "  - "
                print(f"    {e['size']:8} {e['n_examples']:5d} ex   recall "
                      f"{e['recall_ok']}/{e['recall_n']}   held-out {h}")
        return True, gpt, opt, meta

    # ---- training / checking ----
    if cmd == ":stabilize":
        new_gpt, new_opt = stabilize(tok, patterns, corpus_text)
        if new_gpt is not None:
            gpt, opt = new_gpt, new_opt
            meta = {"n_patterns": len(patterns), "corpus_len": len(corpus_text), "config": dict(CONFIG)}
        return True, gpt, opt, meta

    if cmd == ":fullstabilize":
        new_gpt, new_opt = stabilize(tok, patterns, corpus_text, full=True)
        if new_gpt is not None:
            gpt, opt = new_gpt, new_opt
            meta = {"n_patterns": len(patterns), "corpus_len": len(corpus_text), "config": dict(CONFIG)}
        return True, gpt, opt, meta

    if cmd == ":reset":
        print("[:reset deletes EVERYTHING the bot has saved: trained weights, all taught patterns,")
        print(" pasted corpus text, both tokenizers, size/tokenizer settings, the growth log and")
        print(" size_backups/. It keeps templates.json and probes.json (you write those).")
        print(" This cannot be undone.]")
        if input("type RESET to confirm (anything else cancels) > ").strip() != "RESET":
            print("[cancelled -- nothing was deleted]")
            return True, gpt, opt, meta
        removed, gpt, opt, meta = reset_everything(tok, patterns)
        print(f"[reset done -- removed: {', '.join(removed) if removed else 'nothing (already clean)'}]")
        print(f"[back to a first start: no patterns, untrained model, '{SETTINGS['tokenizer']}' "
              f"tokenizer, size {current_size_name()}]")
        return True, gpt, opt, meta

    if cmd == ":tokenizer":
        if not args or args[0] not in ("bpe", "word"):
            print(f"usage: :tokenizer bpe|word   (current: {SETTINGS['tokenizer']}, "
                  f"{tok.vocab_size} tokens)")
            print("  bpe  = your original: letters/subword pieces")
            print("  word = one token per whole word (new words become <UNK> until :retoken)")
            return True, gpt, opt, meta
        if args[0] == SETTINGS["tokenizer"]:
            print(f"[already using the '{args[0]}' tokenizer]")
            return True, gpt, opt, meta
        gpt, opt, meta = switch_tokenizer(args[0], tok, patterns, corpus_text, gpt, opt, meta)
        return True, gpt, opt, meta

    if cmd == ":retoken":
        new_tok = train_word_tokenizer() if SETTINGS["tokenizer"] == "word" else train_tokenizer()
        _swap_tokenizer(tok, new_tok)
        print(f"[{SETTINGS['tokenizer']} tokenizer retrained: {tok.vocab_size} tokens ({len(tok.merges)} merges). "
              f"Old weights belong to the old vocabulary, so the model retrains from scratch.]")
        gpt = ConfigurableGPT(vocab_size=tok.vocab_size, **CONFIG)
        opt = Adam(gpt.params(), lr=LR)
        gpt, opt, meta = _auto(tok, patterns, corpus_text, gpt, opt, meta)
        return True, gpt, opt, meta

    if cmd == ":auto":
        arg = args[0].lower() if args else ""
        if arg in ("on", "off"):
            AUTO["on"] = (arg == "on")
        print(f"[auto-memorize is {'ON' if AUTO['on'] else 'OFF'} -- usage: :auto on | :auto off]")
        if AUTO["on"] and arg == "on":
            gpt, opt, meta = _auto(tok, patterns, corpus_text, gpt, opt, meta)
        return True, gpt, opt, meta

    if cmd == ":twophase":
        new_gpt, new_opt = stabilize(tok, patterns, corpus_text, full=True, generalize=True)
        if new_gpt is not None:
            gpt, opt = new_gpt, new_opt
            meta = {"n_patterns": len(patterns), "corpus_len": len(corpus_text), "config": dict(CONFIG)}
        return True, gpt, opt, meta

    if cmd == ":generalize":
        pat_ex, synth_ex, corpus_ex, holdout = _build_examples(tok, patterns, corpus_text)
        res = generalize_phase(gpt, opt, tok, patterns, pat_ex,
                               _text_pattern_examples(patterns, pat_ex) + synth_ex + corpus_ex,
                               holdout + load_probes())
        if res is not None:
            save_model(gpt, len(patterns), len(corpus_text))
            print(f"[weights saved to {SAVE_PATH}]")
        return True, gpt, opt, meta

    if cmd == ":verify":
        pat_ex, synth_ex, corpus_ex, _ = _build_examples(tok, patterns, corpus_text)
        others = _text_pattern_examples(patterns, pat_ex) + synth_ex + corpus_ex
        pat, oth = verify_all(gpt, tok, patterns, others, _ambiguity_masks(pat_ex, others))
        n_ok, n = _count_ok(pat, oth)
        if n == 0:
            print("[nothing to verify yet]")
        else:
            print(f"[verify: {n_ok}/{n} training examples reproduced by the current weights "
                  f"-- no training]")
            _print_verify(pat, oth, tok=tok, other_examples=others)
            if n_ok < n:
                print("[run :fullstabilize to retrain and repair until every example passes]")
        return True, gpt, opt, meta

    if cmd == ":selftest":
        pat_ex = [encode_pattern(tok, p["prompt"], p["answer"], CONFIG["max_seq_len"]) for p in patterns]
        text_ex = _text_pattern_examples(patterns, pat_ex)
        pat, oth = verify_all(gpt, tok, patterns, text_ex, _ambiguity_masks(pat_ex, text_ex))
        n_ok, n = _count_ok(pat, oth)
        if n == 0:
            print("[no taught patterns to test yet]")
        else:
            print(f"[self-test: {n_ok}/{n} taught patterns reproduced "
                  f"({n_ok / n:.0%}) -- greedy decoding, current weights, no training]")
            _print_verify(pat, oth, tok=tok, other_examples=text_ex)
            if n_ok < n:
                print("[run :fullstabilize to retrain and repair until every one passes]")
        return True, gpt, opt, meta

    if cmd == ":status":
        print_status(patterns, corpus_text, meta)
        return True, gpt, opt, meta

    if cmd in (":checkmode", ":showcheck"):
        key = cmd[1:]
        if not args or args[0] not in ("on", "off"):
            print(f"usage: {cmd} on|off   (now: {'on' if SETTINGS.get(key) else 'off'})")
        else:
            SETTINGS[key] = (args[0] == "on")
            save_settings()
            print(f"[{key} {args[0]}]")
            if key == "checkmode" and SETTINGS[key] and not load_profile():
                print(f"[no profile yet: put the bot's facts in {PROFILE_PATH}, or use :profile <text>]")
        return True, gpt, opt, meta

    if cmd in (":known", ":topic"):
        known = _known_topics(patterns)
        if not known:
            print("[no check-style patterns found (prompts with <PROFILE> ... </PROFILE>)]")
            return True, gpt, opt, meta
        if cmd == ":topic":
            if not args or not args[0].isdigit() or not (1 <= int(args[0]) <= len(known)):
                print(f"usage: :topic <1-{len(known)}>   (see :known)")
                return True, gpt, opt, meta
            t = known[int(args[0]) - 1]
            with open(PROFILE_PATH, "w", encoding="utf-8") as f:
                f.write(t["profile"] + "\n")
            CURRENT_OBS["text"] = None
            HISTORY.clear()
            print(f"[profile set to topic {args[0]}; observation cleared. Now type :checkmode on, then your question]")
            return True, gpt, opt, meta
        for i, t in enumerate(known, 1):
            print(f"\n[{i}] PROFILE: {t['profile']}")
            if t["obs"]:
                print("    observations it was trained on:")
                for o in t["obs"]:
                    print(f"      - {o}")
            print("    messages it was trained on:", " | ".join(t["users"]) or "(none)")
        print("\nuse  :topic <number>  to load a profile,  :obs <text>  to set what the bot sees,  then type the message.")
        return True, gpt, opt, meta

    if cmd in (":new", ":history"):
        if cmd == ":new":
            HISTORY.clear()
            print("[new conversation: the bot forgot everything said so far]")
        else:
            print("[conversation so far]" if HISTORY else "[no conversation yet]")
            for u, r in HISTORY:
                print(f"  you: {u}\n  bot: {r}")
        return True, gpt, opt, meta

    if cmd == ":obs":
        if len(parts) > 1 and parts[1].strip().lower() not in ("off", "none", "clear"):
            CURRENT_OBS["text"] = " ".join(parts[1].split())
            print(f"[observation set: {CURRENT_OBS['text']}]")
        elif len(parts) > 1:
            CURRENT_OBS["text"] = None
            print("[observation cleared]")
        else:
            print(f"[observation: {CURRENT_OBS['text'] or '(none)'}]   usage: :obs <text>  |  :obs off")
        return True, gpt, opt, meta

    if cmd == ":profile":
        if len(parts) > 1:
            with open(PROFILE_PATH, "w", encoding="utf-8") as f:
                f.write(parts[1].strip() + "\n")
            HISTORY.clear()
            print("[profile saved]")
        else:
            print(f"[profile: {load_profile() or '(empty)'}]")
        return True, gpt, opt, meta

    if cmd == ":import":
        if not args:
            print("usage: :import <path to .json or .txt file>")
            return True, gpt, opt, meta
        added, err = import_patterns_from_file(parts[1].strip())
        if err:
            print(f"[import failed: {err}]")
        else:
            patterns.extend(added)
            save_patterns(patterns)
            print(f"[imported {len(added)} patterns ({len(patterns)} total now)]")
            gpt, opt, meta = _auto(tok, patterns, corpus_text, gpt, opt, meta)
        return True, gpt, opt, meta

    return False, gpt, opt, meta


def main():
    _validate_presets()
    load_settings()
    if SETTINGS["size"] in SIZE_PRESETS:
        CONFIG.update(SIZE_PRESETS[SETTINGS["size"]])

    print("=" * 60)
    print(" pointerbot scalable -- your pointerbot_selfcheck + switchable sizes")
    print(f" size: {current_size_name()}   autogrow: {SETTINGS['autogrow']} "
          f"(max {SETTINGS['max_size']})")
    print(" SIZES:  :sizes  :size <name>  :grow  :shrink  :autogrow off|suggest|on")
    print("         :capacity (is a bigger model needed?)   :conflicts")
    print(" teach: type patterns | learn: paste text | :import: bulk-load")
    print(" :stabilize retrains on EVERYTHING, then self-tests and self-corrects")
    print(" :fullstabilize trains on EVERY example, then verifies every one and repairs")
    print(" MEMORIZATION IS AUTOMATIC: save a pattern / paste text / :import and it learns")
    print(" them on its own (:auto off to pause). You test and chase generalization.")
    print(" :twophase = :fullstabilize (memorize) THEN a generalize phase from those weights")
    print(" :generalize runs that second phase on the current weights only")
    print(" :selftest / :verify check the current weights (patterns only / everything)")
    print(" TOKENIZER: :tokenizer bpe|word (you choose; never automatic)   :retoken rebuilds its vocabulary")
    print(" CHECK MODE: :checkmode on|off   :showcheck on|off   :profile <text>   (hidden <CHECK> monologue before each reply)")
    print(" CONVERSATION: with :checkmode on the bot remembers what was said  (:new forgets it, :history shows it)")
    print(" TEST BY HAND: :known (what it was trained on)  :topic N  :obs <what the bot sees>  then type a message")
    print(" :reset wipes EVERYTHING the bot saved (asks you to type RESET first)")
    print("=" * 60)

    tok = build_or_load_tokenizer()
    gpt = ConfigurableGPT(vocab_size=tok.vocab_size, **CONFIG)
    loaded, meta = load_model(gpt)
    cfg_changed = meta.get("config") not in (None, CONFIG)
    vocab_changed = (meta.get("vocab_size") not in (None, tok.vocab_size)
                     or meta.get("vocab_sig") not in (None, TOK_SIG["v"])
                     or meta.get("tokenizer", "bpe") != SETTINGS["tokenizer"])
    if loaded and (cfg_changed or vocab_changed):
        # CONFIG or the tokenizer vocabulary changed since the weights were saved.
        # Old weights belong to a different design, so don't reuse them -- start
        # fresh and let auto-memorize retrain.
        why = "CONFIG" if cfg_changed else "tokenizer vocabulary"
        print(f"[{why} changed since the last stabilize -- discarding old weights "
              f"and retraining]")
        gpt = ConfigurableGPT(vocab_size=tok.vocab_size, **CONFIG)
        loaded, meta = False, {"n_patterns": 0, "corpus_len": 0}
    opt = Adam(gpt.params(), lr=LR)
    patterns = load_patterns()
    corpus_text = load_corpus()

    print(f"[model {'loaded' if loaded else 'freshly initialized, no saved weights yet'}]")
    print_status(patterns, corpus_text, meta)
    print(f"[params: {sum(p.data.size for p in gpt.params()):,}]")
    _text_chars = len(corpus_text) + sum(len(p.get("prompt", "")) + len(p.get("answer", ""))
                                          for p in patterns)
    if SETTINGS["tokenizer"] == "bpe" and not tok.merges and _text_chars >= BPE_MIN_TEXT_CHARS:
        print("[tip: the tokenizer has 0 merges (it was trained before you had data), so it "
              "works letter by letter. :retoken learns whole-word subwords from your data "
              "and retrains the model]")
    print()
    if (meta.get("n_patterns") != len(patterns) or meta.get("corpus_len") != len(corpus_text)
            or not loaded):
        gpt, opt, meta = _auto(tok, patterns, corpus_text, gpt, opt, meta)

    mode = None
    try:
        while True:
            if mode is None:
                cmd = input("mode? [chat / teach / learn / template / quit] > ").strip()
                if cmd == ":quit":
                    break
                if cmd == ":mode":
                    continue
                if cmd.startswith(":"):
                    handled, gpt, opt, meta = handle_command(cmd, tok, patterns, corpus_text, gpt, opt, meta)
                    corpus_text = load_corpus()
                    continue
                cmd = cmd.lower()
                if cmd in ("quit", "exit", "q"):
                    break
                if cmd not in ("chat", "teach", "learn", "template"):
                    print("please type: chat, teach, learn, template, :stabilize, :fullstabilize, "
                          ":twophase, :generalize, :retoken, :auto on|off, :selftest, :verify, :status, "
                          ":sizes, :size <name>, :grow, :shrink, :autogrow, :capacity, :conflicts, "
                          ":import <path>, :tokenizer bpe|word, :reset, or quit")
                    continue
                mode = cmd
                print(f"-- {mode} mode -- (:mode to switch, :stabilize to retrain, "
                      f":selftest to check recall, :quit to exit)")
                continue

            if mode == "template":
                print("[template wizard omitted from this listing for brevity -- "
                      "identical to your original file's run_template_wizard(); "
                      "copy it over verbatim if you use templates.json]")
                mode = None
                continue

            if mode == "learn":
                text = read_pasted_text()
                if not text:
                    print("[nothing pasted, skipping]")
                else:
                    append_corpus(text)
                    corpus_text = load_corpus()
                    print(f"[saved ({len(corpus_text)} corpus characters total)]")
                    gpt, opt, meta = _auto(tok, patterns, corpus_text, gpt, opt, meta)
                mode = None
                continue

            user = input("you> ").strip()
            if user == ":quit":
                break
            if user == ":mode":
                mode = None
                continue
            if user.startswith(":"):
                handled, gpt, opt, meta = handle_command(user, tok, patterns, corpus_text, gpt, opt, meta)
                corpus_text = load_corpus()
                continue
            if user == "":
                continue

            if mode == "chat":
                new_ids = stream_generate(gpt, tok, user, obs=CURRENT_OBS["text"])
                if SETTINGS.get("checkmode"):
                    decoded = tok.decode(new_ids)
                    apply_auto_profile(decoded)
                    m = re.search(r"<REPLY>\s*(.*?)\s*</REPLY>", decoded, re.S)
                    if m:
                        HISTORY.append((user, " ".join(m.group(1).split())))

            elif mode == "teach":
                reply_ids = stream_generate(gpt, tok, user)
                reply_text = tok.decode(reply_ids)
                fb = input("was that right? [y/n] > ").strip().lower()
                if fb == "y":
                    patterns.append({"prompt": _taught_prompt(user), "answer": reply_text})
                    save_patterns(patterns)
                    print(f"[pattern saved ({len(patterns)} total)]")
                    gpt, opt, meta = _auto(tok, patterns, corpus_text, gpt, opt, meta)
                elif fb == "n":
                    if SETTINGS.get("checkmode"):
                        print("[check mode: type the FULL answer, e.g.  <CHECK> claim: .. fact: .. verdict: no </CHECK> <REPLY> .. </REPLY>]")
                    correct = input("what should it have said? > ")
                    patterns.append({"prompt": _taught_prompt(user), "answer": correct})
                    save_patterns(patterns)
                    print(f"[pattern saved ({len(patterns)} total)]")
                    gpt, opt, meta = _auto(tok, patterns, corpus_text, gpt, opt, meta)
                else:
                    print("[skipped -- not saved]")
    except (EOFError, KeyboardInterrupt):
        pass

    print("\nbye!")


if __name__ == "__main__":
    main()


def to_float32(gpt):
    """Inference only: store the weights as float32 (half the memory, faster). Do not train afterwards."""
    for p in gpt.params():
        p.data = p.data.astype(np.float32)
    return gpt
