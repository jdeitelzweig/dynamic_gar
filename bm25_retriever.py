import json
import numpy as np
from pyserini.search import SimpleSearcher

class Query:
	def __init__(self, qid, question, answers):
		self.id = qid
		self.question = question
		self.answers = answers

	def answer_overlap(self, doc):
		contains_answer = False
		for answer in self.answers:
			if answer in doc:
				contains_answer = True
		return contains_answer

	def __str__(self):
		return str(self.id)

	def __repr__(self):
		return str(self)

	def __hash__(self):
		return hash(self.id)


def top_k(searcher, ranked_queries, k):
	acc = []
	for q, docids in ranked_queries.items():
		answer_possible = 0
		for docid in docids[:k]:
			doc = searcher.doc(docid)
			if q.answer_overlap(doc.raw()):
				answer_possible = 1
		acc.append(answer_possible)
	return np.mean(acc)


def main():
	# Initialize searcher from prebuilt wikipedia index
	searcher = SimpleSearcher('indexes/enwiki-prebuilt')
	searcher.set_rm3()

	# Get natural questions queries
	queries = []
	with open("/n/fs/nlp-jacksond/datasets/nq-open/dev_preprocessed.json") as f:
		nq_data = json.load(f)
		for query in nq_data["data"]:
			queries.append(Query(query["id"], query["question"], query["answers"]))

	# Find top documents for each query
	ranked_queries = {}
	for q in queries:
		hits = searcher.search(q.question, 1000)
		docids = [hit.docid.strip() for hit in hits]
		ranked_queries[q] = docids

	# Print top k accuracy
	print(f"k=20: {top_k(searcher, ranked_queries, 20)}")
	print(f"k=100: {top_k(searcher, ranked_queries, 100)}")
	print(f"k=1000: {top_k(searcher, ranked_queries, 1000)}")


if __name__ == "__main__":
	main()
