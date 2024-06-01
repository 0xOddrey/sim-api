from http.server import BaseHTTPRequestHandler
from urllib import parse
import spacy
import json


nlp = spacy.load("en-core-web-trf", disable=["tagger", "attribute_ruler", "lemmatizer"])



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
		print(word)
		sim = nlp(answer).similarity(nlp(word))
		score =  sim * 100
		result = json.dumps({"score": score})
		self.send_response(200)
		self._set_headers()
		self.end_headers()
		self.wfile.write(result.encode())
		return