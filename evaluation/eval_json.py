import json
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
EVAL_DATA = "data/json_stage2_eval.json"
MAX_NEW_TOKENS = 256


def load_eval_data():
    with open(EVAL_DATA, "r") as f:
        return json.load(f)


def build_prompt(example):
    instruction = example["instruction"]
    input_text = example.get("input", "")

    if input_text:
        return f"""### Instruction:
{instruction}

### Input:
{input_text}

### Response:
"""
    else:
        return f"""### Instruction:
{instruction}

### Response:
"""


def load_base_model():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb,
        device_map="auto",
        torch_dtype=torch.float16,
    )

    return model, tokenizer


def load_adapter(adapter_path):
    model, tokenizer = load_base_model()
    model = PeftModel.from_pretrained(model, adapter_path)
    return model, tokenizer


def generate(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False
        )

    text = tokenizer.decode(out[0], skip_special_tokens=True)

    return text.split("### Response:")[-1].strip()


def is_valid_json(text):
    try:
        json.loads(text)
        return True
    except:
        return False


def evaluate(model, tokenizer, dataset):

    total = len(dataset)
    valid_json = 0
    exact_match = 0

    for example in tqdm(dataset):

        prompt = build_prompt(example)
        pred = generate(model, tokenizer, prompt)
        target = example["output"].strip()

        if is_valid_json(pred):
            valid_json += 1

        if pred.strip() == target:
            exact_match += 1

    return {
        "total": total,
        "valid_json": valid_json,
        "valid_json_rate": valid_json / total,
        "exact_match": exact_match,
        "exact_match_rate": exact_match / total
    }


def main():

    dataset = load_eval_data()

    print("\n===== Evaluating BASE MODEL =====")
    base_model, tokenizer = load_base_model()
    base_results = evaluate(base_model, tokenizer, dataset)
    print(base_results)

    print("\n===== Evaluating CHECKPOINT 1 =====")
    model1, tokenizer = load_adapter("results/checkpoint1")
    r1 = evaluate(model1, tokenizer, dataset)
    print(r1)

    print("\n===== Evaluating CHECKPOINT 2 =====")
    model2, tokenizer = load_adapter("results/checkpoint2")
    r2 = evaluate(model2, tokenizer, dataset)
    print(r2)

    results = {
        "checkpoint0_base": base_results,
        "checkpoint1": r1,
        "checkpoint2": r2
    }

    with open("results/json_eval_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()