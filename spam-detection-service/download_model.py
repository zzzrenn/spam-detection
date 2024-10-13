from transformers import AutoModelForSequenceClassification, AutoTokenizer
import os

# download the model weights
MODEL = "mrm8488/bert-tiny-finetuned-sms-spam-detection"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)
# save the model weights
save_dir = "weights/bert-tiny"
os.makedirs(save_dir, exist_ok=True)
tokenizer.save_pretrained(save_dir)
model.save_pretrained(save_dir)