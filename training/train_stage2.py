import json
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import PeftModel
from trl import SFTTrainer, SFTConfig

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
STAGE1_ADAPTER = "results/checkpoint1"
OUTPUT_DIR = "results/checkpoint2"
DATA_PATH = "data/json_stage2_train.json"
MAX_SEQ_LENGTH = 1024

with open(DATA_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

def format_example(example):
    instruction = example.get("instruction", "").strip()
    input_text = example.get("input", "").strip()
    output = example.get("output", "").strip()

    if input_text:
        text = (
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{input_text}\n\n"
            f"### Response:\n{output}"
        )
    else:
        text = (
            f"### Instruction:\n{instruction}\n\n"
            f"### Response:\n{output}"
        )

    return {"text": text}

dataset = Dataset.from_list(data)
dataset = dataset.map(format_example, remove_columns=dataset.column_names)

tokenizer = AutoTokenizer.from_pretrained(STAGE1_ADAPTER, token=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    token=True,
    dtype=torch.float16,
)

model = PeftModel.from_pretrained(
    base_model,
    STAGE1_ADAPTER,
    is_trainable=True,
)

model.config.use_cache = False
model.gradient_checkpointing_enable()

training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    num_train_epochs=1,
    logging_steps=10,
    save_strategy="epoch",
    fp16=False,
    bf16=False,
    max_grad_norm=0.0,
    report_to="none",
    remove_unused_columns=False,
    max_length=MAX_SEQ_LENGTH,
    packing=False,
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=training_args,
    processing_class=tokenizer,
)

trainer.train()
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"Saved Stage 2 checkpoint to {OUTPUT_DIR}")