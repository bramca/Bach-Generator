import os
from music21 import converter, instrument, note, chord
from music21.midi import MidiException
from music21 import *

instr = instrument.Piano
piano_file = open("piano_songs.txt", "w")
for root, dirs, files in os.walk("./bach"):
    for file in files:
        if file.endswith(".mid"):
            f = os.path.join(root, file)
            # print(os.path.join(root, file))
            try:
                midi_file = converter.parse(f)
                parts = instrument.partitionByInstrument(midi_file)
                if parts is not None:
                    for part in parts:
                        if isinstance(part.getInstrument(), instr):
                            print("instrument:")
                            print(part.getInstrument().instrumentName)
                            print(os.path.join(root, file))
                            piano_file.write("%s\n" % os.path.join(root, file))
            except MidiException as e:
                print(e)

piano_file.close()
