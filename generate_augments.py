import json
import argparse
from tqdm import trange
from transformers import BartTokenizer, BartForConditionalGeneration


def main():
	parser = argparse.ArgumentParser(description='Generate augments for queries')
	parser.add_argument('--input', type=str, required=True,
		help="Path to input dataset")
	parser.add_argument('--model', type=str, required=True,
		help="Path to pretrained model")
	parser.add_argument('--output', type=str, required=True,
		help="Path to output file")
	args = parser.parse_args()

	model = BartForConditionalGeneration.from_pretrained(args.model)
	tokenizer = BartTokenizer.from_pretrained(args.model)

	queries = []
	with open(args.input) as f:
		data = json.load(f)
		for query in data["data"]:
			queries.append(query["question"])

	batch_size = 32
	out = []
	for idx in trange(0, len(queries), batch_size):
		batch = queries[idx:min(idx+batch_size, len(queries))]
		inputs = tokenizer(batch, max_length=1024, padding="max_length", truncation=True, return_tensors='pt')
		summary_ids = model.generate(inputs['input_ids'], num_beams=4, max_length=50, early_stopping=True)
		out.extend([tokenizer.decode(g, clean_up_tokenization_spaces=False) for g in summary_ids])
	
	#assert len(out) == len(queries)
	print(len(out), len(queries))	
	for query, augment in zip(data["data"], out):
		augment = augment.replace("</s>", " ")
		augment = augment.replace("<s>", "")
		query["question"] = query["question"] + " " + augment

	with open(args.output, "w+") as f:
		json.dump(data, f)

if __name__ == "__main__":
	main()
