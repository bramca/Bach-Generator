# Bach Generator
This repo contains a `python` script that can parse midi (.mid) files and train a machine learning model on those files to generate new music.
It will only train on the piano parts from the given midi files.

# Requirements
`python 3.x`:
```
music21
tensorflow 1.14.0
numpy
sklearn
```

# How to run
Just run the command `python generate_piano_music.py`.
If the file `model.h5` exists it will load that into the model else it will train a new model.
If you want to add new music files you can just put them into the repo and provide the path in the [piano_songs.txt](./piano_songs.txt) file.

By default the generated music is not saved, it will just be played.
To save the generated music you need to add the following snippet to the [generate_piano_music.py](./generate_piano_music.py) file.
``` python
mf = midi.translate.streamToMidiFile(pred_to_stream)
mf.open('generated.midi', 'wb')
mf.write()
mf.close()
```
