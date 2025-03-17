# GuideDiffuSeq - Understanding the Quality-Diversity Trade-off in Diffusion Language Models

### Access the paper [here](https://arxiv.org/pdf/2503.10683).

This is the codebase for GuideDiffuSeq, which proposes several
methods for controlling quality and diversity in
token-level diffusion language models, such as the effect
of classifier-free guidance (CFG).

The code is based on the `diffusers` library.

## Setup
```
git clone https://github.com/zzbuzzard/GuideDiffuSeq
cd GuideDiffuSeq
pip install -r requirements.txt
```
The codebase uses `wandb`, which you may disable with the environment variable `WANDB_MODE=offline`.

## Usage
Create an empty directory `models/[my_model_name]`, and add `config.json` and `train_config.json` files
which specify the parameters required in `config.json`.

Training is then run via
```
python train.py -m models/[my_model_name]
```
Following training, evaluation can be run with `eval.py`. See `python eval.py --help` 
for usage information.

We include the QQP dataset in this repo (datasets/QQP) as it is of a reasonably small size.

## Citation
```
@article{buzzard2025guidediffuseq,
      title={Understanding the Quality-Diversity Trade-off in Diffusion Language Models}, 
      author={Zak Buzzard},
      year={2025},
      url={https://arxiv.org/abs/2503.10683}, 
}
```
