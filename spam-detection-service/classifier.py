# classifier.py
from torch.nn.functional import softmax
from model import Model
import numpy as np

class Classifier:
    def __init__(self):
        self.model = Model.load_model()
        self.tokenizer = Model.load_tokenizer()

    def get_classification_label_and_score(self, text: str):
        result = {}
        labels = ["Non-spam", "Spam"]
        encoded_input = self.tokenizer(text, return_tensors='pt')
        output = self.model(**encoded_input)
        scores = output[0][0]
        scores = softmax(scores, dim=0).detach().numpy()
        idx = np.argmax(scores)
        result["label"] = str(labels[idx])
        result["score"] = np.round(float(scores[idx]), 4)
        return result
    
if __name__ == "__main__":
    classifier = Classifier()
    text = "As a valued customer, I am pleased to advise you that following recent review of your Mob No. you are awarded with a £1500 Bonus Prize, call 09066364589"
    result = classifier.get_sentiment_label_and_score(text)
    print(result)
