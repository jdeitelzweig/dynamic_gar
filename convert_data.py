import json

datapath = "/n/fs/nlp-jacksond/datasets/natural-questions/dev.json"
augment_type = "sentence"
remove_multi = False

data = json.load(open(datapath, "r"))
data = data["data"]
questions = {}

for article in data:
    for paragraph in article["paragraphs"]:
        for question in paragraph["qas"]:
            if question["question"] in questions:
                augments = questions[question["question"]]
            else:
                augments = []

            if augment_type == "answer":
                for answer in question["answers"]:
                    augments.append(answer["text"])
            elif augment_type == "title":
                augments.append(article["title"])
            elif augment_type == "sentence":
                for answer in question["answers"]:
                    answer_loc = answer["answer_start"]
                    start = paragraph["context"][:answer_loc].split(".")[-1]
                    end = paragraph["context"][answer_loc:].split(".")[0]
                    sentence = start + end + "."
                    augments.append(sentence)
            else:
                raise ValueError("Not a valid augmentation")
            questions[question["question"]] = augments

with open('output.json', 'w+') as outfile:
    for question, augments in questions.items():
        if not augments:
            continue
        if remove_multi:
            json.dump({'question': question, 'augment': augments[0]}, outfile)
        else:
            json.dump({'question': question, 'augment': "</s>".join(augments)}, outfile)
        outfile.write('\n')

