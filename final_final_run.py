import os
from pathlib import Path
import pandas as pd
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from peft import PeftModel
import torch
from tqdm import tqdm

DATASET_ROOT = Path("~/dataset_root/custom_dataset/custom_dataset").expanduser()
TEST_CSV_PATH = DATASET_ROOT / "test_non_labels.csv"
TEST_IMAGE_DIR = DATASET_ROOT / "test"
MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
LORA_PATH = DATASET_ROOT / "qwen2_vl_lora_finetuned_ff" / "final_lora"
OUTPUT_CSV_PATH = DATASET_ROOT / "test_predictions.csv"

df = pd.read_csv(TEST_CSV_PATH)
required_cols = {"id", "file", "question"}
if not required_cols.issubset(df.columns):
    raise ValueError(f"Missing columns in test_non_labels.csv: {required_cols - set(df.columns)}")

df["image_path"] = df["file"].apply(lambda f: str(TEST_IMAGE_DIR / f.strip()))
df = df[df["image_path"].apply(lambda p: os.path.exists(p))]
if len(df) == 0:
    raise RuntimeError("No valid test images found.")

processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)
model = PeftModel.from_pretrained(model, LORA_PATH)
model.eval()

def generate_prediction(image_path, question):
    instruction = (
        f"{question} "
        "Respond in the following format: a single word or letter, followed by a period and a space, "
        "then a short explanation starting with 'There is' or 'The', beginning with a capital letter and ending with a period. "
        "Do not use any prefixes such as 'Answer:' or 'Explanation:'."
    )
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": instruction},
            ],
        }
    ]
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[text],
        images=[image_path],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=1024
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            use_cache=True
        )

    generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    generated_text = processor.decode(generated_ids, skip_special_tokens=True).strip()
    return generated_text

predictions = []
for _, row in tqdm(df.iterrows(), total=len(df), desc="Generating predictions"):
    pred_text = generate_prediction(row["image_path"], row["question"])
    
    if ". " in pred_text:
        parts = pred_text.split(". ", 1)
        answer = parts[0].strip()
        explanation = parts[1].strip()
        if explanation and not explanation[0].isupper():
            explanation = explanation[0].upper() + explanation[1:] if len(explanation) > 1 else explanation.upper()
        if not explanation.endswith('.'):
            explanation = explanation + '.'
    else:
        answer = pred_text.split()[0] if pred_text else ""
        explanation = pred_text

    predictions.append({
        "id": row["id"],
        "file": row["file"],
        "question": row["question"],
        "answer": answer,
        "explanation": explanation
    })

pred_df = pd.DataFrame(predictions)
pred_df[["id", "file", "question", "answer", "explanation"]].to_csv(OUTPUT_CSV_PATH, index=False)
print(f"Predictions saved to: {OUTPUT_CSV_PATH}")