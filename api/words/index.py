from http.server import BaseHTTPRequestHandler
from urllib import parse
import spacy
import json



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
		word = dic["word"]
		answer = dic['answer']
		#lower case the words
		word = word.lower()
		answer = answer.lower()
		#check to ensure letters in each word don't overlap by 90% 
		word_letters = word.split()
		answer_letters = answer.split()
		#how to check word like lion and lions are too similar 
		overlap = 0
		total = 0
		for letter in word_letters:
			if letter in answer_letters:
				overlap += 1
			total += 1
		if overlap/total > 0.9:
			result = json.dumps({"score": 0})
			self.send_response(200)
			self._set_headers()
			self.end_headers()
			self.wfile.write(result.encode())
			return
		sim = nlp(word).similarity(nlp(answer))
		score =  sim * 100
		result = json.dumps({"score": score})
		self.send_response(200)
		self._set_headers()
		self.end_headers()
		self.wfile.write(result.encode())
		return