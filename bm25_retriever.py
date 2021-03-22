import numpy as np
from pyserini.search import SimpleSearcher

class Query:
	def __init__(self, qid, query, answers):
		self.id = qid
		self.query = query
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
	query_strings = ["what does hp mean in war and order", "who wrote the first declaration of human rights"]
	qids = ["test_4", "test_5"]
	answers = ["hit points or health points", "Cyrus"]
	for i, q in enumerate(query_strings):
		queries.append(Query(qids[i], query_strings[i], [answers[i]]))

	# Find top documents for each query
	ranked_queries = {}
	for q in queries:
		hits = searcher.search(q.query, 1000)
		docids = [hit.docid.strip() for hit in hits]
		ranked_queries[q] = docids

	# Print top k accuracy
	print(f"k=20: {top_k(searcher, ranked_queries, 20)}")
	print(f"k=100: {top_k(searcher, ranked_queries, 100)}")
	print(f"k=1000: {top_k(searcher, ranked_queries, 1000)}")


if __name__ == "__main__":
	main()
