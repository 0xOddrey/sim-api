from http.server import BaseHTTPRequestHandler
from urllib import parse
import json
import fasttext
import fasttext.util

# Download and load FastText English vectors
fasttext.util.download_model('en', if_exists='ignore')
ft = fasttext.load_model('cc.en.300.bin')

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

        # Get vectors from FastText
        word_vector = ft.get_word_vector(word)
        answer_vector = ft.get_word_vector(answer)

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