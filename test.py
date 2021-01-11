from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import model_from_json
import pandas as pd


wordTotal = 10000


train = pd.read_csv('train.txt', header=None, sep=';', names=['Input', 'Sentiment'], encoding='utf-8')
X = train['Input']
tokenizer = Tokenizer(wordTotal, lower=True, oov_token='OOV')
tokenizer.fit_on_texts(X)

# load weights for  model

json_file = open('model.json', 'r')
json = json_file.read()
json_file.close()
model = model_from_json(json)
model.load_weights("model.h5")


def predictEmotion(sentence):

    str = [sentence]
    seq = tokenizer.texts_to_sequences(str)
    padded = pad_sequences(seq, maxlen=60, padding='post')
    prediction = model.predict_classes(padded)

    dictionary = {'joy': 0, 'anger': 1, 'love': 2, 'sadness': 3, 'fear': 4, 'surprise': 5}
    for key, val in dictionary.items():
        if (val == prediction):
            prediction = key

    print("The emotion is " + prediction)


# change sentence variable to predict emotion
sentence = "machine learning is fun"

predictEmotion(sentence)
