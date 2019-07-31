'''
Forked from https://github.com/3Top/word2vec-api/blob/master/word2vec-api.py
'''
from flask import Flask, request, jsonify
from flask_restful import Resource, Api, reqparse
import numpy as np
import sys
import argparse
from gensim.models import KeyedVectors
import logging
import json
import time

app = Flask(__name__)
api = Api(app)


def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.sqrt(np.dot(vec1, vec1) * np.dot(vec2, vec2)))


def filter_words(words):
    if words is None:
        return
    return [word for word in words if word in model.vocab]

class Vector(Resource):
    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument('word', type=str, required=True, action='append')
        args = parser.parse_args()
        word = args.get('word')[0]
        if word in model.vocab:
            return json.dumps(model[word].tolist())
        else:
            return "out of vocabulary"



class Similarity(Resource):
    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument('w2', type=str, required=True, action='append')
        parser.add_argument('w1', type=str, required=True, action='append')
        args = parser.parse_args()
        w1 = args.get('w1')[0]
        w2 = args.get('w2')[0]
        logging.debug(f"Requesting similarity between {w1} and {w2}")
        if w1 in model.vocab and w2 in model.vocab:
            return json.dumps(model.similarity(w1, w2).tolist()) 
        else:
            logging.debug("Words are not in the vocabulary..")
            return "out of vocabulary"



class HelloWorld(Resource):
    def get(self):
        return {'hello': 'world'}



if __name__ == '__main__':
    global model
    p = argparse.ArgumentParser()
    p.add_argument('--model', help='Path to the trained model')
    p.add_argument('--port', help='Server port, default: 5000', default=5000)
    p.add_argument('--host', help='Server host, default: 0.0.0.0', default='0.0.0.0')
    p.add_argument('--path', help='Server path, default:', default='')
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO)
    start_time = time.time()
    logging.info(f"Loading Word2Vec model...")
    model = KeyedVectors.load_word2vec_format(args.model)
    logging.info(f"Done in --- {time.time() - start_time} seconds ---")
    logging.info("Starting Server...")
    api.add_resource(HelloWorld, args.path+'/')
    api.add_resource(Similarity, args.path+'/similarity')
    api.add_resource(Vector, args.path+'/vector')
    app.run(host=args.host, port=args.port)
