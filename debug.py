text = tokenizer("eu gosto muito de ver você pelas manhãs", return_tensors="pt")

res = model.generate(inputs=text['input_ids'])

tokenizer.decode(res[0])
