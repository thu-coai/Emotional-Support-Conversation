## Inference and Checkpoint Selection

For inference, we selected the checkpoint with the lowest **PPL (Perplexity)**, following the implementation details described in the original paper.

**Note:** 
- You can check the PPL for each epoch in the `eval_log.csv` file.
- Before running the scripts, you must update the `--load_checkpoint` argument in the shell scripts located under the `RUN/` directory to point to your specific checkpoint path.