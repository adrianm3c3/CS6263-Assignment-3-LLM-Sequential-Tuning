import json

def parse_json(text):
    return json.loads(text)

with open("data/json_teacher_raw.json", "r", encoding="utf-8") as f:
    data = json.load(f)

valid = []
invalid = []

for item in data:
    try:
        parsed = parse_json(item["output"])
        item["parsed_output"] = parsed
        valid.append(item)
    except Exception as e:
        item["error"] = str(e)
        invalid.append(item)

with open("data/json_teacher_valid.json", "w", encoding="utf-8") as f:
    json.dump(valid, f, indent=2, ensure_ascii=False)

with open("data/json_teacher_invalid.json", "w", encoding="utf-8") as f:
    json.dump(invalid, f, indent=2, ensure_ascii=False)

print(f"Valid samples: {len(valid)}")
print(f"Invalid samples: {len(invalid)}")