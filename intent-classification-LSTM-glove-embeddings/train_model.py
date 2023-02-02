import numpy as np
import json
     

# Loading json data
with open('data_full.json') as file:
  data = json.loads(file.read())

# Loading out-of-scope intent data
val_oos = np.array(data['oos_val'])
train_oos = np.array(data['oos_train'])
test_oos = np.array(data['oos_test'])

# Loading other intents data
val_others = np.array(data['val'])
train_others = np.array(data['train'])
test_others = np.array(data['test'])

# Merging out-of-scope and other intent data
val = np.concatenate([val_oos,val_others])
train = np.concatenate([train_oos,train_others])
test = np.concatenate([test_oos,test_others])
     

data = np.concatenate([train,test,val])
data = data.T

text = data[0]
labels = data[1]

# Splitting the dataset

from sklearn.model_selection import train_test_split
     

train_txt,test_txt,train_label,test_labels = train_test_split(text,labels,test_size = 0.3)
     
# Preprocessing split dataset

# from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.text    import Tokenizer

# from keras.preprocessing.text import Tokenizer

from tensorflow.keras.preprocessing.sequence import pad_sequences

ls=[]
for c in train_txt:
    ls.append(len(c.split()))
maxLen=int(np.percentile(ls, 98))
     

embeddings_index={}
with open('glove.6B.100d.txt', encoding='utf8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
     

all_embs = np.stack(embeddings_index.values())
emb_mean,emb_std = all_embs.mean(), all_embs.std()
# emb_mean,emb_std
max_num_words = 40000
embedding_dim=len(embeddings_index['the'])
classes = np.unique(labels)

tokenizer = Tokenizer(num_words=max_num_words)
tokenizer.fit_on_texts(train_txt)

train_sequences = tokenizer.texts_to_sequences(train_txt)
train_sequences = pad_sequences(train_sequences, maxlen=maxLen, padding='post')
test_sequences = tokenizer.texts_to_sequences(test_txt)
test_sequences = pad_sequences(test_sequences, maxlen=maxLen, padding='post')
word_index = tokenizer.word_index
     

num_words = min(max_num_words, len(word_index) )+1
embedding_matrix = np.random.normal(emb_mean, emb_std, (num_words, embedding_dim))
for word, i in word_index.items():
    if i >= max_num_words:
        break
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
     

from sklearn.preprocessing import OneHotEncoder,LabelEncoder

label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(classes)

onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoder.fit(integer_encoded)

train_label_encoded = label_encoder.transform(train_label)
train_label_encoded = train_label_encoded.reshape(len(train_label_encoded), 1)
train_label = onehot_encoder.transform(train_label_encoded)
     

test_labels_encoded = label_encoder.transform(test_labels)
test_labels_encoded = test_labels_encoded.reshape(len(test_labels_encoded), 1)
test_labels = onehot_encoder.transform(test_labels_encoded)


# Model preparation

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout, LSTM, Activation, Bidirectional,Embedding
     

model = Sequential()

model.add(Embedding(num_words, 100, trainable=False,input_length=train_sequences.shape[1], weights=[embedding_matrix]))
model.add(Bidirectional(LSTM(256, return_sequences=True, recurrent_dropout=0.1, dropout=0.1), 'concat'))
model.add(Dropout(0.3))
model.add(LSTM(256, return_sequences=False, recurrent_dropout=0.1, dropout=0.1))
model.add(Dropout(0.3))
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(classes.shape[0], activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
history = model.fit(train_sequences, train_label, epochs = 30,
          batch_size = 64, shuffle=True,
          validation_data=[test_sequences, test_labels])

# one epoch = one forward pass and one backward pass of all the training examples
# batch size = the number of training examples in one forward/backward pass. The higher the batch size, the more memory space you'll need

import matplotlib.pyplot as plt
# %matplotlib inline
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

import pickle
import json
     

model.save('models/intents.h5')

with open('utils/classes.pkl','wb') as file:
   pickle.dump(classes,file)

with open('utils/tokenizer.pkl','wb') as file:
   pickle.dump(tokenizer,file)

with open('utils/label_encoder.pkl','wb') as file:
   pickle.dump(label_encoder,file)

# class IntentClassifier:
#     def __init__(self,classes,model,tokenizer,label_encoder):
#         self.classes = classes
#         self.classifier = model
#         self.tokenizer = tokenizer
#         self.label_encoder = label_encoder

#     def get_intent(self,text):
#         self.text = [text]
#         self.test_keras = self.tokenizer.texts_to_sequences(self.text)
#         self.test_keras_sequence = pad_sequences(self.test_keras, maxlen=16, padding='post')
#         self.pred = self.classifier.predict(self.test_keras_sequence)
#         return label_encoder.inverse_transform(np.argmax(self.pred,1))[0]

# import pickle

# from tensorflow.keras.models import load_model

# model = load_model('models/intents.h5')

# with open('utils/classes.pkl','rb') as file:
#   classes = pickle.load(file)

# with open('utils/tokenizer.pkl','rb') as file:
#   tokenizer = pickle.load(file)

# with open('utils/label_encoder.pkl','rb') as file:
#   label_encoder = pickle.load(file)
     

# nlu = IntentClassifier(classes,model,tokenizer,label_encoder)
     

# print(nlu.get_intent("is it cold in India right now"))


# https://github.com/hetpandya/intent-classifier/blob/main/intent_classification.ipynb