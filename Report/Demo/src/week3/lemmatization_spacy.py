import spacy
nlp = spacy.load('en_core_web_sm')

# create a doc object
doc = nlp(u'the bats saw the cats with best stripes hanging upside down by their feet')
print("doc: ", doc)

# create list of tokens from given string
tokens = []
for token in doc:
    tokens.append(token)
print("tokens: ", tokens)

lemmatized_sentence = " ".join([token.lemma_ for token in doc])
print("lemmatized_sentence: ", lemmatized_sentence)