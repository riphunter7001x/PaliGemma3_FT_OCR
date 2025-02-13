# **Fine-Tuning Vision-Language Model: PaliGemma 3**

## **Overview**
This repository contains the fine-tuning process of **PaliGemma 3**, a Vision-Language model, using **Hugging Face's SFTTrainer**. The fine-tuned model has been successfully trained and pushed to **Hugging Face Model Hub**.

## **Model Details**
- **Base Model:** PaliGemma 3
- **Fine-Tuning Framework:** Hugging Face **SFTTrainer**
- **Task:** Vision-Language Understanding
- **Dataset:** Custom OCR dataset for invoice text extraction
- **Training Hardware:** **NVIDIA L40S GPU**

## **Hardware Specifications**
| GPU Model          | Temperature | Power Usage | Memory Usage |
|--------------------|-------------|------------|--------------|
| NVIDIA L40S       | 27°C        | 31W / 350W | 0MiB / 46068MiB |

## **Training Details**
- **Total Epochs:** 50
- **Final Training Metrics:**
  ```json
  {
    "global_step": 900,
    "training_loss": 0.0464,
    "metrics": {
      "train_runtime": 2694.7514,
      "train_samples_per_second": 2.783,
      "train_steps_per_second": 0.334,
      "total_flos": 1.4488688790846672e+17,
      "train_loss": 0.0464,
      "epoch": 47.3733
    }
  }
  ```

## **BLEU Score Evaluation**
After fine-tuning, the model was evaluated using **BLEU (Bilingual Evaluation Understudy)** metric, achieving the following results:

```json
{
  "bleu": 0.9712,
  "precisions": [0.9881, 0.9768, 0.9657, 0.9548],
  "brevity_penalty": 1.0,
  "length_ratio": 1.0003,
  "translation_length": 3281,
  "reference_length": 3280
}
```

### **Interpretation of BLEU Score**
- **BLEU Score:** **97.12%** (Excellent OCR accuracy)
- **High Precision:**
  - Unigram: **98.81%**
  - Bigram: **97.68%**
  - Trigram: **96.57%**
  - 4-gram: **95.48%**
- **No Brevity Penalty:** OCR output matches the reference length closely.

## **Translation Error Rate (TER) Evaluation**
In addition to BLEU, the model was evaluated using **Translation Error Rate (TER)**, a metric that measures the number of edits required to convert generated text into the reference text. Unlike BLEU, which focuses on word overlap, TER accounts for insertions, deletions, substitutions, and shifts.

```json
{
  "score": 6.0065,
  "num_edits": 37,
  "ref_length": 616.0
}
```

### **Interpretation of TER Score**
- **TER Score:** **6.01%** (Low error rate, indicating high accuracy)
- **Number of Edits:** **37** modifications required to align the OCR output with the reference
- **Reference Length:** **616 words**

A **lower TER score** indicates better translation quality. Given the **low TER score**, the model's performance in OCR-based text generation is highly accurate with minimal errors.

## **Demo Input Image & Extracted Data**
Below is a sample OCR result obtained from a demo input image:

**Image Path:** `path/to/demo_image.jpg`

```json
{
    "route": "V183-RZ-924",
    "pallet_number": "14",
    "delivery_date": "5/3/2024",
    "load": "4",
    "dock": "D20",
    "shipment_id": "P24812736099",
    "destination": "726 Meghan Brooks, Amyberg, IA 67863",
    "asn_number": "2211190904",
    "salesman": "RYAN GREEN",
    "products": [
        {
            "description": "293847 - ROLL OF METAL WIRE",
            "cases": "16",
            "sales_units": "8",
            "layers": "4"
        },
        {
            "description": "958273 - CASE OF SPRAY MOPS",
            "cases": "16",
            "sales_units": "8",
            "layers": "3"
        },
        {
            "description": "298693 - CASE OF MULTI-SURFACE SPRAY",
            "cases": "2",
            "sales_units": "4",
            "layers": "2"
        }
    ],
    "total_cases": "34",
    "total_units": "20",
    "total_layers": "9",
    "printed_date": "12/05/2024 10:14",
    "page_number": "91"
}
```

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

