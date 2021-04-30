import json
from tqdm import tqdm
import argparse

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('k', type=int)
	parser.add_argument('file', type=argparse.FileType('r'), nargs='+')
	args = parser.parse_args()

	context_dicts = []

	for f in args.file:
		context_dicts.append(json.load(f))

	out = {}

	# Loop through questions (each document should have same)
	for qid in tqdm(context_dicts[0]):
		fused_contexts = []
		context_nums = [0] * len(context_dicts)
		# Loop through documents
		i = 0
		# Make sure no duplicates are added
		added_docids = set()
		# Keep track of which documents are exhausted
		finished_documents = set()

		while len(fused_contexts) < args.k:
			if len(finished_documents) == len(context_dicts):
				break

			# Get context from document i
			if context_nums[i] >= len(context_dicts[i][qid]["contexts"]):
				finished_documents.add(i)
				i = (i + 1) % len(context_dicts)
				continue
			context = context_dicts[i][qid]["contexts"][context_nums[i]]
			# Keep searching until a new context is found
			while context["docid"] in added_docids:
				context_nums[i] += 1
				if context_nums[i] >= len(context_dicts[i][qid]["contexts"]):
					context = None
					break
				context = context_dicts[i][qid]["contexts"][context_nums[i]]
			if context is not None:
				# Found context not in list
				context_nums[i] += 1
				added_docids.add(context["docid"])
				# Add document to fused contexts
				fused_contexts.append(context)
			# Go to next document
			i = (i + 1) % len(context_dicts)

		out[qid] = context_dicts[0][qid]
		out[qid]["contexts"] = fused_contexts

	json.dump(out, open("out.json", "w+"), indent=4)



if __name__ == "__main__":
	main()
