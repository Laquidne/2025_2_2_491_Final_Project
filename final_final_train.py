import os
from pathlib import Path
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoProcessor,
    Qwen2VLForConditionalGeneration,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model
import torch

DATASET_ROOT = Path("~/dataset_root/custom_dataset/custom_dataset").expanduser()
CSV_PATH = DATASET_ROOT / "train_labels.csv"
IMAGE_DIR = DATASET_ROOT / "train"
MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
OUTPUT_DIR = DATASET_ROOT / "qwen2_vl_lora_finetuned_ff"

df = pd.read_csv(CSV_PATH)
required_cols = {"id", "file", "question", "answer", "explanation"}
if not required_cols.issubset(df.columns):
    raise ValueError(f"Missing columns: {required_cols - set(df.columns)}")

df = df.dropna(subset=["file", "question", "answer", "explanation"])
df = df[df["file"].str.strip() != ""]
df = df[df["question"].str.strip() != ""]

df["image_path"] = df["file"].apply(lambda f: str(IMAGE_DIR / f.strip()))
df = df[df["image_path"].apply(os.path.exists)]
if len(df) == 0:
    raise RuntimeError("No valid images found. Check image paths.")

dataset = Dataset.from_pandas(df[["image_path", "question", "answer", "explanation"]])

processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    bias="none",
    target_modules=["q_proj", "v_proj"],
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)
model.enable_input_require_grads()

training_args = TrainingArguments(
    output_dir=str(OUTPUT_DIR),
    num_train_epochs=5,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
    learning_rate=1e-4,
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    logging_steps=10,
    save_strategy="no",
    bf16=True,
    remove_unused_columns=False,
    report_to="none",
    dataloader_num_workers=4,
    dataloader_pin_memory=True,
    optim="adamw_torch",
    max_grad_norm=1.0,
)

def collate_fn(examples):
    texts = []
    images = []
    for ex in examples:
        image_path = str(ex["image_path"])
        question = str(ex["question"]).strip()
        answer = str(ex["answer"]).strip()
        explanation = str(ex["explanation"]).strip()
        user_text = (
            f"{question} "
            "Respond in the following format: a single word or letter, followed by a period and a space, "
            "then a short explanation starting with 'There is' or 'The', beginning with a capital letter and ending with a period. "
            "Do not use any prefixes such as 'Answer:' or 'Explanation:'."
        )
        assistant_text = f"{answer}. {explanation}"
        messages = [
            {"role": "user", "content": [{"type": "image", "image": image_path}, {"type": "text", "text": user_text}]},
            {"role": "assistant", "content": [{"type": "text", "text": assistant_text}]}
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        texts.append(text)
        images.append(image_path)

    inputs = processor(
        text=texts,
        images=images,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=1024,
    )

    labels = inputs["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100

    for i, text in enumerate(texts):
        assistant_start = text.find(assistant_text)
        if assistant_start == -1:
            continue
        user_part = text[:assistant_start]
        user_len = len(processor.tokenizer(user_part, add_special_tokens=False)["input_ids"])
        labels[i, :user_len] = -100

    inputs["labels"] = labels
    return inputs

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=collate_fn,
)

trainer.train()

final_dir = os.path.join(OUTPUT_DIR, "final_lora")
model.save_pretrained(final_dir)
processor.save_pretrained(final_dir)

print(f"Training completed. Model saved to: {final_dir}")