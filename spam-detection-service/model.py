# model.py
from transformers import AutoModelForSequenceClassification, AutoTokenizer

class Model:
  """A model class to lead the model and tokenizer"""

  def __init__(self) -> None:
    pass
  
  def load_model():
    model = AutoModelForSequenceClassification.from_pretrained("./weights/bert-tiny/")
    return model

  def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("./weights/bert-tiny/")
    return tokenizer