import os
import torch
import json
from pathlib import Path
from datasets import Dataset, DatasetDict, Audio
from peft import get_peft_model, LoraConfig
from transformers import (
    WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor, WhisperForConditionalGeneration,
    Seq2SeqTrainer, Seq2SeqTrainingArguments
)
from jiwer import compute_measures
from accelerate import Accelerator
import multiprocessing
from peft import PeftModel

# Initialize accelerator for multi-GPU support
accelerator = Accelerator()
device = accelerator.device

# ======================================================
# Data Generation: Build dataset on the fly from folders
# ======================================================
from whisper_normalizer.basic import BasicTextNormalizer
normalizer = BasicTextNormalizer()

def clean_transcription(text):
    """Normalize Arabic transcription using Whisper's normalizer and replace 'ة' with 'ه'."""
    norm_text = normalizer(text)
    norm_text = norm_text.replace("ة", "ه")
    return norm_text.strip()

def data_generator_from_folder(base_folder):
    """
    Iterate over all JSON files in the base folder.
    Each JSON file contains a video ID and segments.
    For each segment, construct the full audio file path.
    Only yield examples if the corresponding audio file exists.
    """
    base_path = Path(base_folder)
    for json_file in base_path.glob("*.json"):
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        video_id = data.get("vid")
        segments = data.get("segments", [])
        audio_folder = base_path / video_id
        for seg in segments:
            file_name = seg.get("file_name")
            transcription = seg.get("transcription")
            if file_name and transcription:
                full_audio_path = audio_folder / file_name
                if full_audio_path.exists():
                    yield {"audio": str(full_audio_path),
                           "transcription": clean_transcription(transcription)}
                else:
                    print(f"Warning: File not found: {full_audio_path.name}")

# Set your base folder for the dataset
base_folder = ""#Dataset folder inside " "
dataset = Dataset.from_generator(lambda: data_generator_from_folder(base_folder))
# Cast the "audio" column so that the file paths are converted to actual audio data (16kHz)
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

# Split into train and test sets
train_dev_split = dataset.train_test_split(test_size=0.2, seed=42)
dataset = DatasetDict({
    "train": train_dev_split["train"].shuffle(seed=42).select(range(9000)),
    "test": train_dev_split["test"].shuffle(seed=42).select(range(1800))
})
print("Dataset loaded:")
print({split: len(dataset[split]) for split in dataset})
print({split: list(dataset[split].features.keys()) for split in dataset})


# ======================================================
# Load Whisper Components
# ======================================================
# Use the same feature extractor, tokenizer, and processor as before.
model_path = ""  # base model path for tokenizer/processor
feature_extractor = WhisperFeatureExtractor.from_pretrained(model_path)
tokenizer = WhisperTokenizer.from_pretrained(model_path, language="Arabic", task="transcribe")
tokenizer.model_max_length = 256
processor = WhisperProcessor.from_pretrained(model_path, language="Arabic", task="transcribe")

# ======================================================
# Prepare Dataset: Batched mapping with custom normalization
# ======================================================
def normalize_arabic(text):
    norm_text = normalizer(text)
    norm_text = norm_text.replace("ة", "ه")
    return norm_text.strip()

def alternative_prepare_dataset(batch):
    input_features = []
    # Process each audio sample in the batch
    for audio in batch["audio"]:
        # Since the "audio" column is now cast, each sample should be a dict.
        if isinstance(audio, dict):
            feats = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features
            # Avoid ambiguity by explicitly checking that feats is not None and has length > 0.
            input_features.append(feats[0] if (feats is not None and len(feats) > 0) else None)
        else:
            print("Warning: audio is not a dict, value:", audio)
            input_features.append(None)
    batch["input_features"] = input_features

    # Normalize and tokenize the transcriptions
    normalized_texts = [normalize_arabic(t) for t in batch["transcription"]]
    batch["labels"] = [
        tokenizer(text, truncation=True, max_length=256).input_ids
        for text in normalized_texts
    ]

    return batch

dataset = dataset.map(
    alternative_prepare_dataset,
    remove_columns=["audio", "transcription"],
    batched=True,
    num_proc=16
)
print("Mapping completed:")
print({split: len(dataset[split]) for split in dataset})
print({split: list(dataset[split].features.keys()) for split in dataset})


# ======================================================
# Define Accelerator-friendly compute_metrics functions
# ======================================================
def compute_single(pred, ref):
    if not pred.strip() or not ref.strip():
        return {"incorrect": 0, "total": 0}
    measures = compute_measures(ref, pred)
    return {
        "incorrect": measures["substitutions"] + measures["deletions"] + measures["insertions"],
        "total": measures["substitutions"] + measures["deletions"] + measures["hits"]
    }

def compute_parallel(predictions, references):
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

# ======================================================
# Model Loading and LoRA Tuning from Checkpoint
# ======================================================
# Instead of loading from the base model path, load from the LoRA-finetuned checkpoint.
checkpoint_path = ""#here should be the path of the best chekpoint that you want to continue fine-tuning on
#model = WhisperForConditionalGeneration.from_pretrained(checkpoint_path)
#model = PeftModel.from_pretrained(model_path,checkpoint_path)
base_model = WhisperForConditionalGeneration.from_pretrained(model_path)
model = PeftModel.from_pretrained(base_model, checkpoint_path)
model = model.merge_and_unload()

# Wrap gradient checkpoint disabling in a try/except block in case the method or its internals (_require_grads_hook) are unavailable.
try:
    model.gradient_checkpointing_disable()
except AttributeError as e:
    print("Warning: gradient_checkpointing_disable() failed:", e)

# Set up the LoRA configuration and wrap the model. (Keep same parameters.)
lora_config = LoraConfig(r=16, lora_alpha=64, lora_dropout=0.1, target_modules=["q_proj", "v_proj"])
model = get_peft_model(model, lora_config)

try:
    model.gradient_checkpointing_disable()
except AttributeError as e:
    print("Warning: gradient_checkpointing_disable() after PEFT wrapping failed:", e)

# ======================================================
# Data Collator Definition
# ======================================================
class DataCollatorSpeechSeq2SeqWithPadding:
    def __init__(self, processor, decoder_start_token_id):
        self.processor = processor
        self.decoder_start_token_id = decoder_start_token_id

    def __call__(self, features):
        batch = self.processor.feature_extractor.pad(
            [{"input_features": f["input_features"]} for f in features], return_tensors="pt"
        )
        labels_batch = self.processor.tokenizer.pad(
            [{"input_ids": f["labels"]} for f in features], return_tensors="pt"
        )
        labels = labels_batch["input_ids"]
        if "attention_mask" in labels_batch:
            labels = labels.masked_fill(labels_batch["attention_mask"] == 0, -100)
        if labels.shape[1] > 1 and (labels[:, 0] == self.decoder_start_token_id).all():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor, model.config.decoder_start_token_id)

# ======================================================
# Trainer Setup with Accelerator
# ======================================================
training_args = Seq2SeqTrainingArguments(
    output_dir=checkpoint_path,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=5e-6,
    num_train_epochs=6,
    fp16=True,
    eval_strategy="epoch",      # Evaluate at each epoch
    save_strategy="epoch",      # Save checkpoint at the end of each epoch
    # Do not limit the number of checkpoints so that every checkpoint is saved:
    save_total_limit=None,      # or remove this parameter entirely if you prefer
    per_device_eval_batch_size=32,
    predict_with_generate=True,
    report_to="none",
    load_best_model_at_end=False,  # Disable loading only the best model so every epoch is saved
    metric_for_best_model="wer",     # Still compute the metric for reference (not for checkpoint filtering)
    greater_is_better=False,
    push_to_hub=False,
    dataloader_num_workers=16,
    dataloader_pin_memory=True,
    ddp_find_unused_parameters=False,
    generation_max_length=256,
    lr_scheduler_type="linear",
    warmup_ratio=0.05
    #resume_from_checkpoint=(checkpoint_path,model_path)
)

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
    metrics = trainer.evaluate(eval_dataset=dataset["test"])
    print("Before training test eval:", metrics)

# ======================================================
# Training
# ======================================================
# Resume training from the provided checkpoint.
trainer.train()

# Checkpoint saving: Every epoch checkpoint is now saved,
# regardless of whether its WER is worse than that of the resumed checkpoint.
# After training, this prints information about the last checkpoint.
best_checkpoint = trainer.state.best_model_checkpoint
if best_checkpoint:
    print("Best model checkpoint based on WER:", best_checkpoint)
else:
    print("No best checkpoint found. Check training logs.")

with torch.no_grad():
    print("Start Eval after training")
    metrics = trainer.evaluate(eval_dataset=dataset["test"])
    print("After training test eval:", metrics)

import torch.distributed as dist
if dist.is_initialized():
    dist.destroy_process_group()