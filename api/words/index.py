from http.server import BaseHTTPRequestHandler
from urllib import parse
import json
import gensim
import urllib.request
import os

# URL to download the FastText vectors
url = 'https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.vec.gz'
model_path = './cc.en.300.vec.gz'

# Download the model if it doesn't exist
if not os.path.exists(model_path):
    urllib.request.urlretrieve(url, model_path)

# Load the model
model = gensim.models.KeyedVectors.load_word2vec_format(model_path)

class handler(BaseHTTPRequestHandler):

    def _set_headers(self):
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')  # Allow all origins
        self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')

    def do_OPTIONS(self):
        self.send_response(200)
        self._set_headers()
        self.end_headers()

    def do_GET(self):
        s = self.path
        dic = dict(parse.parse_qsl(parse.urlsplit(s).query))

        word = dic["word"]
        answer = dic['answer']

        # Get vectors from Gensim FastText
        word_vector = model[word]
        answer_vector = model[answer]

        # Calculate cosine similarity
        sim = self.cosine_similarity(answer_vector, word_vector)
        score = sim * 100
        result = json.dumps({"score": score})

        self.send_response(200)
        self._set_headers()
        self.end_headers()
        self.wfile.write(result.encode())
        return

    def cosine_similarity(self, vec1, vec2):
        dot_product = vec1.dot(vec2)
        norm_vec1 = (vec1 ** 2).sum() ** 0.5
        norm_vec2 = (vec2 ** 2).sum() ** 0.5
        return dot_product / (norm_vec1 * norm_vec2)