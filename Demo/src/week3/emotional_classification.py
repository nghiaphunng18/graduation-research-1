import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.read_csv('../../datasets/week3/IMDBDataset.csv')
print(df.columns)
# Index(['review', 'sentiment'], dtype='object')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Tokenize the text
    tokens = nltk.word_tokenize(text)
    # Lemmatize and remove stop words
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    # Re-join tokens into a string
    return ' '.join(lemmatized_tokens)

# Apply preprocessing to each review
df['processed_review'] = df['review'].apply(preprocess_text)
print(df.head())

# Assuming df['processed_review'] and df['sentiment'] are your data
X = df['processed_review']
y = df['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)  # Convert sentiment to binary

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Tokenize text
tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)
word_index = tokenizer.word_index

# Convert text to sequences and pad them to ensure uniform length
sequences_train = tokenizer.texts_to_sequences(X_train)
sequences_test = tokenizer.texts_to_sequences(X_test)

padded_train = pad_sequences(sequences_train, maxlen=200, truncating='post', padding='post')
padded_test = pad_sequences(sequences_test, maxlen=200, truncating='post', padding='post')

# Define the model
model = Sequential([
    Embedding(input_dim=5000, output_dim=300, input_length=200),
    Bidirectional(LSTM(64)),
    Dense(24, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(padded_train, y_train, epochs=10, validation_data=(padded_test, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(padded_test, y_test)
print(f'Loss: {loss}, Accuracy: {accuracy}')

# test
def predict_sentiment(text):
    process_text = preprocess_text(text)
    sequence = tokenizer.texts_to_sequences([process_text])
    padded_sequence = pad_sequences(sequence, maxlen=200, truncating='post', padding='post')
    prediction = model.predict(padded_sequence)[0][0]

    sentiment = "Positive" if prediction >= 0.5 else "Negative"
    print(f"Review: {text} -> {sentiment}")

new_review = "I love this movie! The acting was great and the story was amazing."
predict_sentiment(new_review)

new_review_2 = "This movie was terrible. I wasted two hours of my life."
predict_sentiment(new_review_2)


