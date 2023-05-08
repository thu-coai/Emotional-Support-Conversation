# Running Scripts for *ESC*

Siyang Liu*, **Chujie Zheng***, Orianna Demasi, Sahand Sabour, Yu Li, Zhou Yu, Yong Jiang and Minlie Huang. **Towards Emotional Support Dialog Systems**. *In ACL 2021*. [[paper]](https://arxiv.org/abs/2106.01144) [[repo]](https://github.com/thu-coai/Emotional-Support-Conversation)

```bib
@inproceedings{liu-etal-2021-towards,
  title={Towards Emotional Support Dialog Systems},
  author={Liu, Siyang  and 
    Zheng, Chujie  and 
    Demasi, Orianna  and 
    Sabour, Sahand  and 
    Li, Yu  and 
    Yu, Zhou  and 
    Jiang, Yong  and 
    Huang, Minlie},
  booktitle={Proceedings of the 59th annual meeting of the Association for Computational Linguistics},
  year={2021}
}
```

## Preparing Enviroment

```bash
conda env create -f env.yml -n cuda
conda activate cuda
```

## Downloading Model

You should first download the [BlenderBot-small](https://huggingface.co/facebook/blenderbot_small-90M) model and replace the fake `pytorch_model.bin` file in `Blenderbot_small-90M` with the true one.

If you would like to evaluate generated results with Embedding-based similarity, you can download my prepared embedding files from [HuggingFace](https://huggingface.co/datasets/chujiezheng/glove_embedding).

## About Postfix

- `_vanilla` denotes the variant directly fine-tuned on ESConv without using strategies
- `_strat` denotes the one that additionally uses the strategy information and supervision

## Preprocessing Training Data

First, enter `_reformat` and run `python process.py`.

Then, run `bash RUN/prepare_vanilla.sh` to preprocess the training data.

## Training Your Model

Run `bash RUN/train_vanilla.sh` to train your model.

## Inference with Your Model

Every time of model training will create a new folder in `DATA/{inputter_name}.{config_name}`, which is named after the time when the training starts. You should select a checkpoint (it may be based on the PPL of validation), and replace the checkpoint path in `RUN/infer_vanilla.sh --load_checkpoint` with the path of your selected checkpoint.

Then, run `bash RUN/infer_vanilla.sh` to do the inference.

**Note**: When you run `infer_strat.sh`, you can change `GOLDEN_TRUTH` in  `inputters/PARAMS.py` to control whether use the golden strategy during inference.

## Interacting with Your Model

Similar to inference, after designating the checkpoint in `RUN/interact_vanilla.sh --load_checkpoint`, run `bash RUN/interact_vanilla.sh`.
