# LLaMA 

This repository is intended as a minimal, hackable and readable example to load [LLaMA](https://ai.facebook.com/blog/large-language-model-llama-meta-ai/) ([arXiv](https://arxiv.org/abs/2302.13971v1)) models and run inference.
In order to download the checkpoints and tokenizer, fill this [google form](https://forms.gle/jk851eBVbX1m5TAv5)

### Setup
In a conda env with pytorch / cuda available, run
```
pip install -r requirements.txt
```
Then in this repository
```
pip install -e .
```


### Inference
The provided `example.py` can be run on a single or multi-gpu node with `torchrun` and will output completions for two pre-defined prompts. Using `TARGET_FOLDER` as defined in `download.sh`:
```
torchrun --nproc_per_node MP example.py --ckpt_dir $TARGET_FOLDER/model_size --tokenizer_path $TARGET_FOLDER/tokenizer.model
```

Different models require different MP values:

|  Model | MP |
|--------|----|
| 7B     | 1  |
| 13B    | 2  |
| 33B    | 4  |
| 65B    | 8  |

### Gradio Interface
Use the gradio demo to play around with inference in an interactive way. Make sure to change the model paths in app.py.
```
torchrun --nproc_per_node 1 app.py
```
This demo requires ~16GB GPU memory, but could also run on CPU if you adapt a few lines. See this example for the app in use:

https://user-images.githubusercontent.com/32109055/222568019-44f590ae-724f-4b51-848e-5273fdfe16e2.mp4

### Model Card
See [MODEL_CARD.md](MODEL_CARD.md)

### License
See the [LICENSE](LICENSE) file.
