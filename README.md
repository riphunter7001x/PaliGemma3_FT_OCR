# **Fine-Tuning Vision-Language Model: PaliGemma 3**

## **Overview**
This repository contains the fine-tuning process of **PaliGemma 3**, a Vision-Language model, using **Hugging Face's SFTTrainer**. The fine-tuned model has been successfully trained and pushed to **Hugging Face Model Hub**.

## **Model Details**
- **Base Model:** PaliGemma 3
- **Fine-Tuning Framework:** Hugging Face **SFTTrainer**
- **Task:** OCR invoices with  Vision-Language Models
- **Dataset:** Custom OCR dataset for invoice text extraction
- **Training Hardware:** **NVIDIA L40S GPU**



## **BLEU Score Evaluation**
After fine-tuning, the model was evaluated using **BLEU (Bilingual Evaluation Understudy)** metric, achieving the following results:

```json
{
  "bleu": 0.9701,
  "precisions": [0.9881, 0.9761, 0.9641, 0.9526],
  "brevity_penalty": 1.0,
  "length_ratio": 1.0,
  "translation_length": 3280,
  "reference_length": 3280
}
```

### **Interpretation of BLEU Score**
- **BLEU Score:** **97.01%** (Excellent OCR accuracy)
- **High Precision:**
  - Unigram: **98.81%**
  - Bigram: **97.61%**
  - Trigram: **96.41%**
  - 4-gram: **95.26%**
- **No Brevity Penalty:** OCR output matches the reference length exactly.

## **Translation Error Rate (TER) Evaluation**
In addition to BLEU, the model was evaluated using **Translation Error Rate (TER)**, a metric that measures the number of edits required to convert generated text into the reference text. Unlike BLEU, which focuses on word overlap, TER accounts for insertions, deletions, substitutions, and shifts.

```json
{
  "score": 6.3312,
  "num_edits": 39,
  "ref_length": 616.0
}
```

### **Interpretation of TER Score**
- **TER Score:** **6.33%** (Low error rate, indicating high accuracy)
- **Number of Edits:** **39** modifications required to align the OCR output with the reference
- **Reference Length:** **616 words**

A **lower TER score** indicates better translation quality. Given the **low TER score**, the model's performance in OCR-based text generation is highly accurate with minimal errors.

## **Model Deployment**
The fine-tuned model has been **pushed to Hugging Face** for public access.

👉 **[Access the Model on Hugging Face](https://huggingface.co/your-model-link)**

## **How to Use the Model**
To use the fine-tuned **PaliGemma 3** model for OCR tasks:

```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model_name = "your-huggingface-model-name"
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

text = "Upload your image or document here"
inputs = tokenizer(text, return_tensors="pt")
outputs = model.generate(**inputs)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## **Future Improvements**
- Enhance robustness on diverse invoice formats
- Optimize inference speed for real-time applications
- Experiment with additional Vision-Language tasks

---
🚀 **Fine-Tuned PaliGemma 3 for OCR - High Accuracy & Deployed on Hugging Face!**

