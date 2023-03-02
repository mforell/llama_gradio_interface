import gradio as gr
from example import load
import torch
import os
from fairscale.nn.model_parallel.initialize import initialize_model_parallel

local_rank = int(os.environ.get("LOCAL_RANK", -1))
world_size = int(os.environ.get("WORLD_SIZE", -1))

torch.distributed.init_process_group("nccl")
initialize_model_parallel(world_size)
torch.cuda.set_device(local_rank)
torch.manual_seed(1)

generator = load(ckpt_dir="./models/7B", tokenizer_path="./models/tokenizer.model", local_rank=0, world_size=1)

def generate_text(text):
    yield from generator.generate_rolling(text, max_gen_len=1024)

examples = [
    ["The capital of Germany is the city of"],
    ["Here is my sonnet in the style of Shakespeare about an artificial intelligence:"]
]

gr.Interface(
    generate_text,
    "textbox",
    "text",
    title="LLama 7B",
    description="LLama-7B large language model.",
    examples=examples
).queue().launch()