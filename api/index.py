from http.server import BaseHTTPRequestHandler
from urllib import parse
import json


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
		result = json.dumps({"score": 5})
		self.send_response(200)
		self._set_headers()
		self.end_headers()
		self.wfile.write(result.encode())
		return