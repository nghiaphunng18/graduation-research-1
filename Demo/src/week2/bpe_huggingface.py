from transformers import GPT2Tokenizer, GPT2Model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2Model.from_pretrained('gpt2')
text = "I am learning NLP"
encoded_input = tokenizer(text, return_tensors='pt')
print(encoded_input)
