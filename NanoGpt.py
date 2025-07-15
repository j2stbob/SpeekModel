import tensorflow as tf
import numpy as np


with open("Datas/451f.txt", encoding='utf-8') as f:
    data = f.readlines()
data = [line.strip() for line in data if line.strip()]
data = ' '.join(data)

alph = set(data)
encode = {char: idx for idx, char in enumerate(alph)}
decode = {idx: char for char, idx in encode.items()}


encode_text = [encode[char] for char in data]


batch_data = []
y = []

for i in range(0, len(encode_text) - 16):
    batch_data.append(encode_text[i:i + 16])
    y.append(encode_text[i + 16])

x = np.array(batch_data)
y = np.array(y).reshape(-1, 1)
x = np.reshape(x, (x.shape[0], x.shape[1], 1))


model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(128, input_shape=(16, 1)),
    tf.keras.layers.Dense(len(encode), activation='softmax')
])


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])



model.fit(x, y, epochs=10)

def generate(input_text, symbols):
    encode_input = [encode[char] for char in input_text]

    for i in range(symbols):
        if len(encode_input) < 16:
            padded_input = [0] * (16 - len(encode_input)) + encode_input
        else:
            padded_input = encode_input[-16:]

        x_input = np.array(padded_input)
        x_input = np.reshape(x_input, (1, 16, 1))

        y_pred = model.predict(x_input)
        next_char_index = np.argmax(y_pred[0])



        encode_input.append(next_char_index)
    return ''.join([decode[idx] for idx in encode_input])


sample_text = "Монтэг двин"
gen = generate(sample_text, 16)
print(gen)