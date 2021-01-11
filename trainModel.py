from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import pandas as pd
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import LSTM, Bidirectional, Dense, Embedding, Dropout

wordTotal = 10000
maxLen = 60

# load training data
train = pd.read_csv('train.txt', header=None, sep=';', names=['Input', 'Sentiment'], encoding='utf-8')
validation = pd.read_csv('val.txt', header=None, sep=';', names=['Input', 'Sentiment'], encoding='utf-8')


# tokenize data
X = train['Input']
tokenizer = Tokenizer(wordTotal, lower=True, oov_token='OOV')
tokenizer.fit_on_texts(X)

X_train = tokenizer.texts_to_sequences(X)
X_train = pad_sequences(X_train, maxlen=maxLen, padding='post')
X_val = validation['Input']
X_val = tokenizer.texts_to_sequences(X_val)
X_val = pad_sequences(X_val, maxlen=maxLen, padding='post')

train['Sentiment'] = train.Sentiment.replace({'joy': 0, 'anger': 1, 'love': 2, 'sadness': 3, 'fear': 4, 'surprise': 5})
Y_train = train['Sentiment'].values
Y_train = to_categorical(Y_train)

Y_val = validation.Sentiment.replace({'joy': 0, 'anger': 1, 'love': 2, 'sadness': 3, 'fear': 4, 'surprise': 5})
Y_val = to_categorical(Y_val)


# construct model
model = Sequential()
model.add(Embedding(wordTotal, 128, input_length=maxLen))
model.add(Dropout(0.8))
model.add(Bidirectional(LSTM(100, return_sequences=True)))
model.add(Bidirectional(LSTM(150)))
model.add(Dense(6, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
hist = model.fit(X_train, Y_train, epochs=15, validation_data=(X_val, Y_val))

# save final Model
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("model.h5")
print("Saved model to disk")
