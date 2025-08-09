import os
import torch
import json
import pandas as pd
from datasets import Dataset, DatasetDict, Audio, load_dataset
from peft import get_peft_model, LoraConfig
from transformers import (
    WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor, WhisperForConditionalGeneration,
    Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
)
from jiwer import compute_measures
from accelerate import Accelerator

# Initialize accelerator for multi-GPU support
accelerator = Accelerator()
device = accelerator.device

import os
import json
# Function to load data
def data_generator(file_path):
    def generator():
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)  # Parse JSON into a Python dictionary or list
        for segment in data:
            # if len(segment['truth_non_normalized'].split()) > 100:
            #     print("Skipping segment : " + segment['truth_non_normalized'])
            #     continue 
            yield {"audio": segment["file_path"], "sentence": segment["transcription"]}
            
    return generator

# Load datasets
train_dataset = Dataset.from_generator(data_generator(f""))#the dataset ptah inside" "

# Cast audio column to process as audio
train_dataset = train_dataset.cast_column("audio", Audio(sampling_rate=16000))


train_dev_split = train_dataset.train_test_split(test_size=0.2, seed=42)

dataset = DatasetDict({
    "train": train_dev_split["train"].shuffle(seed=42),
    "test": train_dev_split['test'].shuffle(seed=42)
})
print(dataset)

# Load Whisper feature extractor & tokenizer
feature_extractor = WhisperFeatureExtractor.from_pretrained("")#the model path inside" "
tokenizer = WhisperTokenizer.from_pretrained("", language="Arabic", task="transcribe")#the model path inside" "
tokenizer.model_max_length = 256  # Ensure tokenizer doesn't generate longer sequences
processor = WhisperProcessor.from_pretrained("", language="Arabic", task="transcribe")#the model path inside" "


def prepare_dataset(batch):
    audio = batch["audio"]
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
    batch["labels"] = tokenizer(batch["sentence"]).input_ids
   
    return batch





dataset = dataset.map(prepare_dataset, remove_columns=["audio", "sentence"], num_proc=16)

print("Done preparing")
print(dataset)

# Define training arguments with multi-GPU support
training_args = Seq2SeqTrainingArguments(
    output_dir=" ",##the output path inside" ""
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=1e-5,
    num_train_epochs=8,
    bf16=True,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_eval_batch_size=32,
    predict_with_generate=True,
    report_to="none",
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=False,
    dataloader_num_workers=16,
    dataloader_pin_memory=True,
    ddp_find_unused_parameters=False,  # ✅ Multi-GPU optimization
    generation_max_length=256,  # Adjust based on dataset
)


import multiprocessing

def compute_single(pred, ref):
    """Compute WER for a single prediction-reference pair with restrictions."""
    if not pred.strip() or not ref.strip():  # Ignore empty predictions/references
        return {"incorrect": 0, "total": 0}
    
    measures = compute_measures(ref, pred)
    return {
        "incorrect": measures["substitutions"] + measures["deletions"] + measures["insertions"],
        "total": measures["substitutions"] + measures["deletions"] + measures["hits"]
    } def compute_parallel(predictions, references):
    """Parallelized WER computation using multiprocessing."""
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        results = pool.starmap(compute_single, zip(predictions, references))

    incorrect = sum(r["incorrect"] for r in results)
    total = sum(r["total"] for r in results)

    return incorrect / total if total > 0 else 0.0


def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = tokenizer.pad_token_id
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    return {"wer": 100 * compute_parallel(pred_str, label_str)}


# Load model and apply LoRA tuning
model = WhisperForConditionalGeneration.from_pretrained("")#the model path inside" "
model.gradient_checkpointing_disable()

lora_config = LoraConfig(r=16, lora_alpha=64, lora_dropout=0.1, target_modules=["q_proj", "v_proj"])
model = get_peft_model(model, lora_config)
model.gradient_checkpointing_disable()


class DataCollatorSpeechSeq2SeqWithPadding:
    def __init__(self, processor, decoder_start_token_id):
        self.processor = processor
        self.decoder_start_token_id = decoder_start_token_id

    def __call__(self, features):
        # Extract input features (audio)
        batch = self.processor.feature_extractor.pad(
            [{"input_features": f["input_features"]} for f in features], return_tensors="pt"
        )

        # Extract labels (text)
        labels_batch = self.processor.tokenizer.pad(
            [{"input_ids": f["labels"]} for f in features], return_tensors="pt"
        )

        labels = labels_batch["input_ids"]

        # Ensure attention_mask is correctly aligned with labels
        if "attention_mask" in labels_batch:
            labels = labels.masked_fill(labels_batch["attention_mask"] == 0, -100)

        # Prevent slicing issues
        if labels.shape[1] > 1 and (labels[:, 0] == self.decoder_start_token_id).all():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor, model.config.decoder_start_token_id)

# Trainer setup
trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)

with torch.no_grad():
    print("Start Eval before training")
    metrics = trainer.evaluate(eval_dataset=dataset['test'])
    print("before training test eval: " + str(metrics))

# Train the model
trainer.train()


best_checkpoint = trainer.state.best_model_checkpoint
if best_checkpoint:
    print("Best model checkpoint based on WER:", best_checkpoint)
else:
    print("No best checkpoint found. Check training logs.")

with torch.no_grad():
    print("Start eval after training")
    metrics = trainer.evaluate(eval_dataset=dataset['test'])
    print("after training test eval: " + str(metrics))