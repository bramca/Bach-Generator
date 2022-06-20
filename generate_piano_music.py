import os
import random
import re
from music21 import converter, instrument, note, chord
from music21.midi import MidiException
from music21 import *
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking, Embedding
import numpy as np
from sklearn.model_selection import train_test_split
# music21 doc: https://web.mit.edu/music21/doc/moduleReference/moduleStream.html?highlight=stream%20part#music21.stream.Stream.notes
# RNN: https://towardsdatascience.com/recurrent-neural-networks-by-example-in-python-ffd204f99470
# Piano songs:
# 1  ./bach\aof\can1.mid
# 2  ./bach\aof\can2.mid
# 3  ./bach\aof\can3.mid
# 4  ./bach\aof\can4.mid
# 5  ./bach\aof\cnt1.mid
# 6  ./bach\aof\cnt3.mid
# 7  ./bach\aof\tri1.mid
# 8  ./bach\aof\unfin.mid
# 9  ./bach\cantatas\jesu2.mid
# 10 ./bach\fugues\fuguecm.mid
# 11 ./bach\fugues\fuguegm.mid
# 12 ./bach\gold\aria.mid
# 13 ./bach\gold\goldberg.mid
# 14 ./bach\partitas\cou1.mid
# 15 ./bach\partitas\cou2.mid
# 16 ./bach\partitas\gig1.mid
# 17 ./bach\partitas\pre1.mid
# 18 ./bach\partitas\ron2.mid
# 19 ./bach\wtcbki\fugue9.mid
# 20 ./bach\wtcbkii\fugue9.mid

piano_files = open("piano_songs.txt", "r")
musicstrings = []
pianoparts = []
it = 0
for file in piano_files.read().split("\n"):
    if file.endswith(".mid"):
        try:
            midi_file = converter.parse(file)
            parts = instrument.partitionByInstrument(midi_file)
            # print("_____________ parts info ________________")
            # print(parts)
            if parts is not None:
                for part in parts:
                    if part.getInstrument().instrumentName == "Piano":
                        # part.show('text')
                        # print("__________ file info ____________")
                        # print(file)
                        print("_____________ part info _____________")
                        # print(part)
                        # sp = midi.realtime.StreamPlayer(part)
                        # sp.play()
                        g = ""
                        offset  = -1
                        for el in part:
                            #  or isinstance(el, note.Rest) ?
                            if isinstance(el, note.Note):
                                if offset == el.offset:
                                    g += '%s' % (el.name)
                                else:
                                    g += ' (%.2f)%s' % (el.offset, el.name)
                                offset = el.offset
                        musicstrings.append(g)
                        print(g)
        except MidiException as e:
            print(e)
    it += 1
    if it == 10:
        break
piano_files.close()

# print(musicstrings)

tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='', lower=False, split=' ')

tokenizer.fit_on_texts(musicstrings)

seq = tokenizer.texts_to_sequences(musicstrings)

print("___________________ seq info __________________")
print(len(seq))
print(len(seq[0]))
print(seq)

idx_word = tokenizer.index_word

print("_________ idx_word info ___________")
print(' '.join(idx_word[w] for w in seq[0]))

features = []
labels = []

training_length = 50

# Iterate through the sequences of tokens
for s in seq:

    # Create multiple training examples from each sequence
    for i in range(training_length, len(s)):

        # Extract the features and label
        extract = s[i - training_length:i + 1]

        # Set the features and label
        features.append(extract[:-1])
        labels.append(extract[-1])

features = np.array(features)

print("_______________ features info ______________________")
print(features.shape)
print(features)

# number of words in vocabulary
num_words = len(idx_word) + 1
print("number of words in voc: %d" % (num_words))

label_array = np.zeros((len(features), num_words), dtype=np.int8)

for example_index, word_index in enumerate(labels):
    label_array[example_index, word_index] = 1

print("__________________ label array info _________________________")
print(label_array)
print(label_array.shape)

X_train, X_test, y_train, y_test = train_test_split(features, label_array, shuffle=False)


print("_______ X train info _________")
print(X_train.shape)
print(X_train)

print("___________ X test info _________________")
print(X_test.shape)
print(X_test)

# print(idx_word[np.argmax(label_array[12])])
if os.path.isfile("model.h5"):
    print("loading model")
    model = tf.keras.models.load_model("model.h5")
else:
    print("training model")
    model = Sequential()
    # Embedding layer
    model.add(
        Embedding(input_length=training_length,
                  input_dim=num_words,
                  output_dim=100))
    # Masking layer for pre-trained embeddings
    model.add(Masking(mask_value=0.0))
    # Recurrent layer
    model.add(LSTM(64, return_sequences=False, 
                   dropout=0.1, recurrent_dropout=0.1))
    # Fully connected layer
    model.add(Dense(64, activation='relu'))
    # Dropout for regularization
    model.add(Dropout(0.5))
    # Output layer
    model.add(Dense(num_words, activation='softmax'))
    # Compile the model
    model.compile(
        optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, batch_size=20, epochs=30)
    print("Evaluate on test data")
    results = model.evaluate(X_test, y_test, batch_size=10)
    print("test loss, test acc:", results)
    print("model saved in model.h5")
    model.save("model.h5")

print("________________ model info _________________")
print(model.summary())

print("________________ generating notes ________________")
num_gen_notes = 100
start = 3
print("generating %d notes with seed:" % (num_gen_notes))
next_seq_to_predict = seq[start][0:training_length]
features_to_notes = [idx_word[f] for f in next_seq_to_predict]
print(' '.join(features_to_notes))
offset_pat = re.compile('\((.+)\)')
note_pat = re.compile('[A-Z][#|-]?')
p = stream.Part()
p.insert(instrument.Piano())
pred_to_stream = stream.Stream()
for fn in features_to_notes:
    fn_offset = float(offset_pat.match(fn).group(1))
    for nn in note_pat.findall(fn):
        pn = note.Note(nn)
        p.insert(fn_offset, pn)
last_offset = fn_offset
original_offset = -1
for i in range(0, num_gen_notes):
    # print(next_seq_to_predict)
    predictions = model.predict(np.reshape(next_seq_to_predict, (1, len(next_seq_to_predict))))
    final_predictions = (predictions == predictions.max(axis=1)[:,None]).astype(int)
    pred_to_notes = [idx_word[np.argmax(pred)] for pred in final_predictions]
    # print(pred_to_notes)
    # pred_to_notes.sort(key=lambda x: float(offset_pat.match(x).group(1)))
    # print(len(pred_to_notes))
    n = pred_to_notes[0]
    n_offset = float(offset_pat.match(n).group(1))
    # print(n_offset)
    # print(note_pat.findall(n))
    if n_offset - last_offset > 1:
        if original_offset > 0 and original_offset == n_offset:
            n_offset = last_offset
        else:
            print("----- offset adjusted ----")
            print("original one: %.2f" % (n_offset))
            original_offset = n_offset
            n_offset = last_offset + random.choice([0.25, 0.5, 1])
            print("new one: %.2f" % (n_offset))
    last_offset = n_offset
    for sn in note_pat.findall(n):
        pn = note.Note(sn)
        p.insert(n_offset, pn)
    next_seq_to_predict = np.delete(next_seq_to_predict, 0)
    next_seq_to_predict = np.append(next_seq_to_predict, tokenizer.texts_to_sequences([n])[0])
pred_to_stream.insert(0, p)
pred_to_stream.show('text')
sp = midi.realtime.StreamPlayer(pred_to_stream)
sp.play()
# mf = midi.translate.streamToMidiFile(pred_to_stream)
# mf.open('generated.midi', 'wb')
# mf.write()
# mf.close()

print("______________________ original notes _______________________")
original = seq[start][0:(training_length + num_gen_notes)]
original_to_notes = [idx_word[f] for f in original]
print(' '.join(features_to_notes))
original_to_stream = stream.Stream()
original_p = stream.Part()
original_p.insert(instrument.Piano())
for fn in original_to_notes:
    fn_offset = float(offset_pat.match(fn).group(1))
    for nn in note_pat.findall(fn):
        pn = note.Note(nn)
        original_p.insert(fn_offset, pn)
original_to_stream.insert(0, original_p)
original_to_stream.show('text')
sp = midi.realtime.StreamPlayer(original_to_stream)
sp.play()
# mf = midi.translate.streamToMidiFile(original_to_stream)
# mf.open('generated.midi', 'wb')
# mf.write()
# mf.close()
