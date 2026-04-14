import json
import time
from openai import OpenAI

API_KEY = "utsa-jABQlGLaTrae2bqMHyAvPxTvE9KTP0DEWYIXhvtgkDkVcGjp44rN6G56x1aGiyem"
BASE_URL = "http://149.165.173.247:8888/v1"
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

with open("data/json_prompt_seed.json", "r", encoding="utf-8") as f:
    prompts = json.load(f)

results = []

SYSTEM_PROMPT = (
    "You are a data formatting assistant. "
    "Return only valid JSON. Do not include markdown fences, explanations, or extra text."
)

for item in prompts:
    user_prompt = f"Instruction: {item['instruction']}\n\nInput:\n{item['input']}"

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2,
            max_tokens=300
        )

        output_text = response.choices[0].message.content.strip()

        results.append({
            "id": item["id"],
            "task_type": item["task_type"],
            "instruction": item["instruction"],
            "input": item["input"],
            "output": output_text
        })

        print(f"Generated: {item['id']}")
        time.sleep(0.5)

    except Exception as e:
        print(f"Failed on {item['id']}: {e}")

with open("data/json_teacher_raw.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print("Saved raw teacher outputs to data/json_teacher_raw.json")