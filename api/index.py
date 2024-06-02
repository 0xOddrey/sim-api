from http.server import BaseHTTPRequestHandler
from urllib import parse
import spacy
import json
import string

# Load the spaCy model
nlp = spacy.load("en_core_web_md")

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
        word = dic.get("word", "").strip().lower()
        answer = dic.get("answer", "").strip().lower()

        # Handle cases where input is missing
        if not word or not answer:
            result = json.dumps({"error": "Both 'word' and 'answer' parameters are required."})
            self.send_response(400)
            self._set_headers()
            self.end_headers()
            self.wfile.write(result.encode())
            return

        # Remove punctuation and split into tokens
        word_tokens = [token.lemma_ for token in nlp(word) if token.is_alpha]
        answer_tokens = [token.lemma_ for token in nlp(answer) if token.is_alpha]

        # Check for high overlap using lemmatized forms
        overlap = len(set(word_tokens).intersection(answer_tokens))
        total = max(len(word_tokens), len(answer_tokens))

        if total > 0 and overlap / total > 0.8:
            result = json.dumps({"score": 0})
        else:
            word_doc = nlp(" ".join(word_tokens))
            answer_doc = nlp(" ".join(answer_tokens))
            sim = word_doc.similarity(answer_doc)
            score = sim * 100
            result = json.dumps({"score": score})

        self.send_response(200)
        self._set_headers()
        self.end_headers()
        self.wfile.write(result.encode())
        return