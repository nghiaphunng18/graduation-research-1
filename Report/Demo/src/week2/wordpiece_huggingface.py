from transformers import BertTokenizer, BertModel
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
text = "I am leaning NLP"
encoded_input = tokenizer(text, return_tensors='pt')
print(encoded_input)