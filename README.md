# Beam with Controlled Patience

## Introduction
We introduce the patience factor for the commonly-used implementation of beam decoding.

## Machine Translation
Download and uncompress the pretrained multilingual BART models from the [fairseq repository](https://github.com/pytorch/fairseq/tree/main/examples/multilingual#mbart50-models):
```bash
wget https://dl.fbaipublicfiles.com/fairseq/models/mbart50/mbart50.ft.1n.tar.gz
wget https://dl.fbaipublicfiles.com/fairseq/models/mbart50/mbart50.ft.n1.tar.gz
tar xvzf mbart50.ft.1n.tar.gz
tar xvzf mbart50.ft.n1.tar.gz
```
Here are some example commands.
English to German (EN-DE) with FCFS `p=2`.
```bash
python generate_mbart_1n.py  --lang de --out-file mt/en-de/output/newstest2021.en-de.mbart.p2.de --in-file mt/en-de/src/newstest2021.en-de.src.en --patience-factor 2 --model-dir <model_dir>
```
Japanese to English (JA-EN) with FCFS `p=1` (the original algorithm).
```bash
python generate_mbart_n1.py  --lang ja --out-file mt/ja-en/output/newstest2021.ja-en.mbart.p1.en --in-file mt/ja-en/src/newstest2021.ja-en.src.ja --patience-factor 1 --model-dir <model_dir>
```
English to Chinese (EN-ZH) with vanilla decoding.
```bash
python generate_mbart_1n.py  --lang zh --out-file mt/en-zh/output/newstest2021.en-zh.mbart.vanilla.zh --in-file mt/en-zh/src/newstest2021.en-zh.src.en --vanilla --model-dir <model_dir>
```
Polish to English (PL-EN) with greedy decoding.
```bash
python generate_mbart_n1.py  --lang pl --out-file mt/pl-en/output/newstest2020.pl-en.mbart.greedy.en --in-file mt/pl-en/src/newstest2020.pl-en.src.pl --beam 1 --model-dir <model_dir>
```

## Summarization
Download and uncompress the pretrained, finetuned BART models:
```bash
wget https://dl.fbaipublicfiles.com/fairseq/models/bart.large.cnn.tar.gz 
wget https://dl.fbaipublicfiles.com/fairseq/models/bart.large.xsum.tar.gz
tar xvzf bart.large.cnn.tar.gz
tar xvzf bart.large.xsum.tar.gz
```
Here are some example commands.

## Evaluate

## Citation
```
@misc{kasai2022BeamPatience,
    title   = {Beam Decoding with Controlled Patience},
    author  = {},
    year    = {2022},
    url     = {}, 
}
```
<p align="center">
<a href="https://www.cs.washington.edu/research/nlp">
<img src="https://github.com/jungokasai/THumB/blob/master/figs/uwnlp_logo.png" height="100" alt="UWNLP Logo">
</a>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
<a href="https://allenai.org/">
<img src="https://github.com/jungokasai/THumB/blob/master/figs/ai2_logo.png" height="100" alt="AI2 Logo" style="padding-right:160">
</a>
</p>
