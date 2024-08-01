import requests
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Attention, Concatenate

# URL of the dataset on GitHub
url = 'https://raw.githubusercontent.com/bhkenealey/CS4375_Final_Project/main/dataset/dialogs.txt'

# Fetch the data from GitHub
response = requests.get(url)
data = response.text

# Split lines into input-output pairs
lines = data.split('\n')
pairs = [line.strip().split('\t') for line in lines if line.count('\t')]
print(pairs)

# Separate inputs and targets
inputs, targets = zip(*pairs)

# Define start and end tokens
start_token = '<start>'
end_token = '<end>'

# Add start and end tokens
input_texts = [start_token + ' ' + text + ' ' + end_token for text in inputs]
target_texts = [start_token + ' ' + text + ' ' + end_token for text in targets]

# Tokenize the sentences
tokenizer = Tokenizer(filters='')
tokenizer.fit_on_texts(input_texts + target_texts)
input_sequences = tokenizer.texts_to_sequences(input_texts)
target_sequences = tokenizer.texts_to_sequences(target_texts)

# Pad sequences
max_length = max(max(len(seq) for seq in input_sequences), max(len(seq) for seq in target_sequences) + 1)
input_sequences = pad_sequences(input_sequences, maxlen=max_length, padding='post')
target_sequences = pad_sequences(target_sequences, maxlen=max_length, padding='post')

# Create target data for the decoder
target_sequences_input = target_sequences[:, :-1]
target_sequences_output = target_sequences[:, 1:]

# Vocabulary size
vocab_size = len(tokenizer.word_index) + 1

# Define model inputs
encoder_inputs = Input(shape=(max_length,), name='encoder_inputs')
decoder_inputs = Input(shape=(max_length-1,), name='decoder_inputs')

# Encoder
embedding_dim = 256
encoder_embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim, name='encoder_embedding')(encoder_inputs)
encoder_lstm = LSTM(embedding_dim, return_sequences=True, return_state=True, name='encoder_lstm')
encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)
encoder_states = [state_h, state_c]

# Decoder with Attention
decoder_embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim, name='decoder_embedding')(decoder_inputs)
decoder_lstm = LSTM(embedding_dim, return_sequences=True, return_state=True, name='decoder_lstm')
decoder_lstm_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)

# Attention layer
attention = Attention(name='attention_layer')
attention_output = attention([decoder_lstm_outputs, encoder_outputs])
decoder_concat_input = Concatenate(axis=-1)([decoder_lstm_outputs, attention_output])

# Dense layer
decoder_dense = Dense(vocab_size, activation='softmax', name='decoder_dense')
decoder_outputs = decoder_dense(decoder_concat_input)

# Define the model
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
batch_size = 32
epochs = 100

model.fit(
    [input_sequences, target_sequences_input],
    np.expand_dims(target_sequences_output, -1),
    batch_size=batch_size,
    epochs=epochs,
    validation_split=0.2
)

# Encoder model
encoder_model = Model(encoder_inputs, [encoder_outputs, state_h, state_c])

# Decoder setup
decoder_state_input_h = Input(shape=(embedding_dim,))
decoder_state_input_c = Input(shape=(embedding_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

decoder_hidden_state_input = Input(shape=(max_length, embedding_dim))
decoder_lstm_outputs, state_h, state_c = decoder_lstm(
    decoder_embedding, initial_state=decoder_states_inputs
)
attention_output = attention([decoder_lstm_outputs, decoder_hidden_state_input])
decoder_concat_input = Concatenate(axis=-1)([decoder_lstm_outputs, attention_output])
decoder_outputs = decoder_dense(decoder_concat_input)

decoder_model = Model(
    [decoder_inputs] + [decoder_hidden_state_input, decoder_state_input_h, decoder_state_input_c],
    [decoder_outputs] + [state_h, state_c]
)

# Function to generate responses with attention
def decode_sequence(input_seq):
    encoder_outputs, state_h, state_c = encoder_model.predict(input_seq)
    states_value = [state_h, state_c]

    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = tokenizer.word_index[start_token]

    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + [encoder_outputs] + states_value)

        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_word = tokenizer.index_word[sampled_token_index]
        decoded_sentence += ' ' + sampled_word

        if sampled_word == end_token or len(decoded_sentence) > max_length:
            stop_condition = True

        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index

        states_value = [h, c]

    return decoded_sentence.strip(end_token)

# Function to respond to input
def respond_to_input(input_text):
    input_sequence = tokenizer.texts_to_sequences([start_token + ' ' + input_text + ' ' + end_token])
    input_sequence = pad_sequences(input_sequence, maxlen=max_length, padding='post')
    response = decode_sequence(input_sequence)
    return response

# Main loop to interact with the bot
def chat_with_bot():
    print("Start chatting with the bot (type 'exit' to stop):")
    while True:
        input_text = input("You: ")
        if input_text.lower() == 'exit':
            print("Ending chat. Goodbye!")
            break
        response = respond_to_input(input_text)
        print("Bot:", response)

# Run the chat loop
chat_with_bot()
