import spacy

# load the language model
nlp = spacy.load('en_core_web_sm')

text = "This is an example sentence. However, it contains stop words!"
doc = nlp(text)

filtered_text = [token.text for token in doc if not token.is_stop]

print("original text: ", text)
print("filtered_text: ", " ".join(filtered_text))