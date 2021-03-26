import json
import argparse
from pyserini.search import SimpleSearcher

class Query:
	def __init__(self, qid, question, answers):
		self.id = qid
		self.question = question
		self.answers = answers

	def __str__(self):
		return str(self.id)

	def __repr__(self):
		return str(self)

	def __hash__(self):
		return hash(self.id)


def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Get top k accuracy for a dataset on wikipedia')
	parser.add_argument('--input', type=str, required=True,
		help="Path to input dataset")
	parser.add_argument('--index', type=str, required=True,
		help="Path to prebuilt indexes")
	parser.add_argument('--batch-size', type=int, metavar='num', required=False,
		default=1, help="Specify batch size to search the collection concurrently.")
	parser.add_argument('--threads', type=int, metavar='num', required=False,
		default=1, help="Maximum number of threads to use.")
	parser.add_argument('--output', type=str, required=True,
		help="Path to output file")
	args = parser.parse_args()

	# Initialize searcher from prebuilt wikipedia index
	searcher = SimpleSearcher(args.index)

	# Get queries
	queries = {}
	with open(args.input) as f:
		data = json.load(f)
		for query in data["data"]:
			queries[query["id"]] = Query(query["id"], query["question"], query["answers"])

	# Find top documents for each query
	ranked_queries = {}

	if args.batch_size <= 1 and args.threads <= 1:
		for qid, q in queries.items():
			hits = searcher.search(q.question, 1000)
			ranked_queries[q] = hits
	else:
		for qs in batch(list(queries.values()), args.batch_size):
			hits = searcher.batch_search([q.question for q in qs], [q.id for q in qs], 1000, args.threads)
			hits = {queries[q]: v for q, v in hits.items()}
			ranked_queries.update(hits)

	output_dict = {}
	for q, hits in ranked_queries.items():
		output_dict[q.id] = {"question": q.question, "answers": q.answers, "contexts": []}
		for hit in hits:
			docid = hit.docid.strip()
			ctx = json.loads(searcher.doc(docid).raw())['contents']
			out = {'docid': docid, 'score': hit.score, 'text': ctx}
			output_dict[q.id]["contexts"].append()

	with open(args.output, "w+") as f:
		json.dump(output_dict, f, indent=4)