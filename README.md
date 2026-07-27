# Deep Learning from Scratch

### From an autograd engine to language models

Building the core pieces of modern deep learning from nothing — no
`torch.nn`, no `.backward()` I did not write myself — one stage at a time.
Each part is a small, self-contained project with its own notebook and README.

Reimplemented while working through Andrej Karpathy's *Neural Networks: Zero to
Hero*, in my own code and words.

## Roadmap

| Stage | Topic | Status |
|-------|-------|--------|
| [micrograd](micrograd/) | A scalar autograd engine + a tiny neural-net library | ✅ done |
| makemore — bigram | Character-level language model: counts and a 1-layer net | 🔜 next |
| makemore — MLP | An embedding + hidden-layer language model | ⬜ planned |
| makemore — activations / BatchNorm | Making deeper nets trainable | ⬜ planned |
| nanoGPT | A small Transformer / GPT | ⬜ planned |

## The thread

The point of the series is that each stage reuses the last. The autograd engine
built in **micrograd** is the same idea that trains every model that follows —
so by the end the language models are standing on an engine I wrote myself, not
on a framework. That continuity is what turns a set of exercises into one build.

## Repository layout

```
deep-learning-from-scratch/
└── micrograd/          scalar autograd engine + MLP  (README + notebook)
                        more stages land here as the series continues
```

Each stage folder contains a runnable notebook, a short README explaining the
idea and what I took away from it, and any scripts used to produce its figures.

## Getting started

Every notebook has an **Open in Colab** badge in its folder README, so you can
run it in the browser without cloning. To run locally:

```bash
git clone https://github.com/midhun9901/deep-learning-from-scratch.git
```

Dependencies are per-stage and minimal (micrograd's engine needs only the Python
standard library). See each folder's README.
