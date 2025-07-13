from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle
import heapq
import os

model = load_model(r"D:\College\dl\next word predictor\next_word_model.h5")

with open(r"D:\College\dl\next word predictor\tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

total_words = len(tokenizer.word_index) + 1
max_seq_len = model.input_shape[1] + 1  

app = Flask(__name__, template_folder='templates')

def predict_next_word(seed_text):
    seq = tokenizer.texts_to_sequences([seed_text])[0]
    seq = pad_sequences([seq], maxlen=max_seq_len - 1, padding='pre')
    preds = model.predict(seq, verbose=0)[0]
    next_idx = np.argmax(preds)
    return tokenizer.index_word.get(next_idx, '')

def beam_search_predict(seed_text, beam_width=3, next_words=5):
    sequences = [(seed_text, 0.0)]
    for _ in range(next_words):
        all_candidates = []
        for sent, score in sequences:
            token_seq = tokenizer.texts_to_sequences([sent])[0]
            token_seq = pad_sequences([token_seq], maxlen=max_seq_len - 1, padding='pre')
            preds = model.predict(token_seq, verbose=0)[0]
            top_idxs = np.argsort(preds)[-beam_width:]
            for idx in top_idxs:
                word = tokenizer.index_word.get(idx, '')
                new_sent = sent + ' ' + word
                new_score = score - np.log(preds[idx] + 1e-10)
                all_candidates.append((new_sent, new_score))
        sequences = heapq.nsmallest(beam_width, all_candidates, key=lambda x: x[1])
    return [s for s, _ in sequences]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    seed = data.get('seed')
    beam_width = int(data.get('beam_width', 3))
    next_words = int(data.get('next_words', 5))

    if not seed:
        return jsonify({'error': 'Seed text is required.'}), 400

    next_word = predict_next_word(seed)
    beam_output = beam_search_predict(seed, beam_width, next_words)

    return jsonify({
        'seed': seed,
        'next_word': next_word,
        'beam_search': beam_output
    })

if __name__ == '__main__':
    app.run(debug=True)
