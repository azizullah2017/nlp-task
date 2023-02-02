import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np


class IntentClassifier:
    def __init__(self,classes,model,tokenizer,label_encoder):
        self.classes = classes
        self.classifier = model
        self.tokenizer = tokenizer
        self.label_encoder = label_encoder

    def get_intent(self,text):
        self.text = [text]
        self.test_keras = self.tokenizer.texts_to_sequences(self.text)
        self.test_keras_sequence = pad_sequences(self.test_keras, maxlen=16, padding='post')
        self.pred = self.classifier.predict(self.test_keras_sequence)
        return self.label_encoder.inverse_transform(np.argmax(self.pred,1))[0]

# IntentClassifier
# from tensorflow.keras.models import load_model

# model = load_model('models/intents.h5')

# with open('utils/classes.pkl','rb') as file:
#   classes = pickle.load(file)

# with open('utils/tokenizer.pkl','rb') as file:
#   tokenizer = pickle.load(file)

# with open('utils/label_encoder.pkl','rb') as file:
#   label_encoder = pickle.load(file)
     
# nlu = IntentClassifier(classes,model,tokenizer,label_encoder)
     
# print("========================")
# print(nlu.get_intent("is it cold in India right now"))
# print("========================")
# print(nlu.get_intent("how much tax is on my salary"))

