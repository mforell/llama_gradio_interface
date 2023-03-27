import gradio as gr
from example import load
import torch
import os
from fairscale.nn.model_parallel.initialize import initialize_model_parallel
_path = "/data/models/llama/"
local_rank = int(os.environ.get("LOCAL_RANK", -1))
world_size = int(os.environ.get("WORLD_SIZE", -1))

torch.distributed.init_process_group("nccl" if not os.name == "nt" else "gloo")
initialize_model_parallel(world_size)
torch.cuda.set_device(local_rank)
torch.manual_seed(1)

model_selection = input("Please enter the model you want to use: 7B, 13B, 30B: \n ")
modelpath = ""

while not modelpath:
    if model_selection == "7B":
        modelpath = _path + "llama-7B"
    elif model_selection == "13B":
        modelpath =  _path + "llama-13B"
    elif model_selection == "30B":
        modelpath = _path + "llama-30B"
    else:
        print("Invalid model selection.")
        model_selection = input("Please enter the model you want to use: 7B, 13B, 30B ")

if modelpath:
    print("You have chosen " + model_selection + " and the modelpath is " + modelpath + ".")

generator = load(ckpt_dir=modelpath+"/models/" + model_selection , tokenizer_path=modelpath+"/models/tokenizer.model", local_rank=local_rank, world_size=world_size)

def generate_text(text):
    yield from generator.generate_rolling(text, max_gen_len=512)


examples = [
    # For these prompts, the expected answer is the natural continuation of the prompt
    "I believe the meaning of life is",
    "Simply put, the theory of relativity states that ",
    "Building a website can be done in 10 simple steps:\n",
    # Few shot prompts: https://huggingface.co/blog/few-shot-learning-gpt-neo-and-inference-api
    """Tweet: "I hate it when my phone battery dies."
Sentiment: Negative
###
Tweet: "My day has been 👍"
Sentiment: Positive
###
Tweet: "This is the link to the article"
Sentiment: Neutral
###
Tweet: "This new music video was incredibile"
Sentiment:""",
    """Translate English to French:
sea otter => loutre de mer
peppermint => menthe poivrée
plush girafe => girafe peluche
cheese =>""",
    ]

gr.Interface(
    generate_text,
    "textbox",
    "text",
    title=f"LLama {model_selection}",
    description= f"LLama-{model_selection} large language model.",
    examples=examples
).queue().launch()
