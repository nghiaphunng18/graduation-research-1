import spacy

# load the english language model
nlp = spacy.load('en_core_web_sm')

text = '''Good muffins cost $3.88\\nin New York. Please buuy me two of them.\n\nThanks.'''

# process the text
doc = nlp(text)

for sent in doc.sents:
    print(sent.text)