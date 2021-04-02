import json
import argparse
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

	inputs = tokenizer(["when did bat out of hell get released?", "who sings does he love me with reba?", 
		"what is the name of wonder womans mother?"], max_length=1024, return_tensors='pt')
	summary_ids = model.generate(inputs['input_ids'], num_beams=4, max_length=5, early_stopping=True)
	out = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids]
	print(out)

if __name__ == "__main__":
	main()
