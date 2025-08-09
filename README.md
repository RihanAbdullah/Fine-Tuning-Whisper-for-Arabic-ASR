[README-whisper.md](https://github.com/user-attachments/files/21696774/README-whisper.md)
# Fine-Tuning Whisper for Arabic ASR
This repo contains code only; datasets and checkpoints are not included.
This repository contains code for **two-stage fine-tuning** of OpenAI's **Whisper** model for Arabic Automatic Speech Recognition (ASR), including Modern Standard Arabic and various dialects.

## 📌 Overview
The fine-tuning process was conducted in two stages:
1. **First Stage of Fine-Tuning** – Fine-tuned Whisper on the Common Voice Arabic dataset.
2. **Second Stage of Fine-Tuning (Resumed on Different Dataset)** – Took the best checkpoint from Stage 1 and continued fine-tuning on the MASC dataset.

Both stages were evaluated on **Flures**, an unseen dataset, to measure generalization ability.

---

## 📂 Repository Structure
```
.
├── First_Stage_Fine-tuning_Whisper.py   # Fine-tuning on Common Voice (Stage 1)
├── Second_Stage_Fine-tuning_Whisper.py  # Continued fine-tuning on MASC (Stage 2)
├── requirements.txt                     # Dependencies list
└── README.md                            # Project documentation
```

---

## ⚙️ Features
- **Two-stage fine-tuning** pipeline for improved ASR performance.
- **Dataset loading** from structured JSON and audio folders.
- **Arabic text normalization** with Whisper’s normalizer and custom refinements.
- **LoRA integration** for efficient training.
- **Multi-GPU training** with Hugging Face Accelerate.
- **Evaluation metrics**:
  - **WER** (Word Error Rate)
  - **CER** (Character Error Rate)

---

## 📊 Evaluation Metrics
- **Word Error Rate (WER)**: Measures word-level transcription accuracy.
- **Character Error Rate (CER)**: Measures character-level accuracy — useful for morphologically rich languages.

---

## 🚀 Usage

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare dataset
The dataset should follow this structure:
```
dataset/
│
├── file1.json
├── file2.json
└── audio_folder/
    ├── audio1.wav
    ├── audio2.wav
    └── ...
```
Each JSON file should include:
```json
{
    "vid": "video_id",
    "segments": [
        {"file_name": "audio1.wav", "transcription": "Arabic text..."},
        {"file_name": "audio2.wav", "transcription": "Arabic text..."}
    ]
}
```

### 3. Run fine-tuning
```bash
python First_Stage_Fine-tuning_Whisper.py
```

### 4. Resume from Stage 1 checkpoint (Stage 2 fine-tuning)
```bash
python Second_Stage_Fine-tuning_Whisper.py
```

---

## 📈 Results on Flures (Unseen Dataset)

| Stage                                     | WER (%) | CER  |
|-------------------------------------------|---------|------|
| First Stage Fine-Tuning (Common Voice)    | 10.0    | 0.036|
| Second Stage Fine-Tuning (Resumed on MASC)| 9.7     | 0.033|

*Flures dataset was not seen during training and used solely for evaluation.*

---

## 🛠 Requirements
- Python 3.9+
- PyTorch (GPU recommended)
- Hugging Face Transformers
- Datasets
- Accelerate
- JiWER

---

## 📄 License
 Thi project is intended license under the MIT License.
