from typing import List, Union
import tensorflow as tf
import re
from fastapi import FastAPI, Response, status
from pydantic import BaseModel
import uvicorn
import numpy as np
import traceback
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk
from nltk.corpus import stopwords
import pandas as pd

nltk.download('stopwords')
stopwords_uses = stopwords.words('english')

maxlen = 10  # Set the desired max sequence length
pad_type = 'post'
trunc_type = 'post'

df_vegan_test = pd.read_csv('endpoint/df_vegan_test.csv')

def preprocess_title(sentence):
    # Remove trailing spaces
    sentence = re.sub(r'\s$', '', sentence)

    # Remove special characters
    sentence = re.sub(r'[^a-zA-Z0-9 ]', '', sentence).lower()

    # Remove stopwords
    sentence = ' '.join([word for word in sentence.split() if word not in stopwords_uses])
    return sentence
tokenizer = Tokenizer(oov_token='<OOV>')

input_title = df_vegan_test['title']  # Add your input titles here
processed_titles = [preprocess_title(title) for title in input_title]
tokenizer.fit_on_texts(processed_titles)
input_seq = tokenizer.texts_to_sequences(processed_titles)

word_index = tokenizer.word_index
# print(word_index)

input_padded = pad_sequences(input_seq, padding=pad_type, truncating=trunc_type, maxlen=maxlen)

def preprocess_texts_input(sentences):
    processed_sentences = [preprocess_title(sentence) for sentence in sentences]
    tokenizer.fit_on_texts(processed_sentences)
    input_seq = tokenizer.texts_to_sequences(processed_sentences)
    input_seq = pad_sequences(input_seq, padding=pad_type, truncating=trunc_type, maxlen=maxlen)
    return input_seq

# sentences = ['This is a sample sentence.', 'Another sample sentence.']
# processed_sentences = preprocess_texts_input(sentences)
# print("Processed Sentences:")
# print(processed_sentences)


class ModelInput(BaseModel):
    sentence: str

app = FastAPI()

model = tf.keras.models.load_model('endpoint/Trained E16, BiLSTM1, MaxP F32 KS5.h5')
# model.summary()

@app.get("/")
async def index():
    return "Hello from the recommendation system endpoint"

tokenizer = Tokenizer()

@app.post("/recommend")
async def recommendation(request: ModelInput):
    try:
        sentences = [request.sentence]
        sentences = preprocess_texts_input(sentences)
        # print(sentences)
        prediction = model.predict(sentences)

        threshold = 0.5

        recommendation = []
        for index, pred in enumerate(prediction[0]):
            if pred > threshold:
                recommendation.append(str(df_vegan_test['title'].iloc[index]))
        top_recommendations = recommendation
        json_response = {"topRecommendationRecipes": top_recommendations}
        return json_response
    except Exception as e:
        print(traceback.format_exc())
        return Response(content=str(e), status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
