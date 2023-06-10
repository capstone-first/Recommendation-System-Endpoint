from typing import List
import tensorflow as tf
import re
from fastapi import FastAPI, Response, status
from pydantic import BaseModel
import uvicorn
import numpy as np
import traceback
from typing import Union
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stopwords_uses = stopwords.words('english')

maxlen = 100  # Set the desired max sequence length
pad_type = 'post'
trunc_type = 'post'

def preprocess_title(sentence):
    # Remove trailing spaces
    sentence = re.sub(r'\s$', '', sentence)

    # Remove special characters
    sentence = re.sub(r'[^a-zA-Z0-9 ]', '', sentence).lower()

    # Remove stopwords
    sentence = ' '.join([word for word in sentence.split() if word not in stopwords_uses])
    return sentence

tokenizer = Tokenizer(oov_token='<OOV>')
word_index = {}

def preprocess_texts_input(sentences):
    processed_sentences = []
    for sentence in sentences:
        sentence = preprocess_title(sentence)
        processed_sentences.append(sentence)
    tokenizer.fit_on_texts(processed_sentences)
    input_seq = tokenizer.texts_to_sequences(processed_sentences)
    input_seq = pad_sequences(input_seq, padding=pad_type, truncating=trunc_type, maxlen=maxlen)
    return input_seq

class ModelInput(BaseModel):
    sentence: Union[str, List[str]]

app = FastAPI()

model = tf.keras.models.load_model('endpoint/test_model.h5')

@app.get("/")
async def index():
    return "Hello from the recommendation system endpoint"

tokenizer = Tokenizer()

@app.post("/recommend")
async def recommendation(request: ModelInput):
    try:
        sentences = request.sentence
        sentences = preprocess_texts_input(sentences)
        prediction = model.predict(sentences)

        # get the output with the highest probability
        top_indices = np.argsort(prediction[0])[-5:][::-1]
        print("Top Indices:", top_indices)

        word_index_reverse = {index: word for word, index in tokenizer.word_index.items()}
        print("Word Index Reverse:", word_index_reverse)

        top_recipes = [word_index_reverse[index] for index in top_indices if index in word_index_reverse]

        json_response = {"top recommendation recipes": top_recipes}
        return json_response
    except Exception as e:
        print(traceback.format_exc())
        return Response(content=str(e), status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)

