model = MT5ForConditionalGeneration.from_pretrained("models/paraphrasing_pt")
tokenizer = MT5Tokenizer.from_pretrained("models/paraphrasing_pt")

text = tokenizer("eu gosto muito de ver você pelas manhãs", return_tensors="pt")

res = model.generate(inputs=text['input_ids'])

print(tokenizer.decode(res[0]))
