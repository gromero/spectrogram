Usage
=====

Generate a 1x49x10x1 audio spectrogram for audio recognition:

```sh
$ python3 ./sg.py ./e5e54cee_nohash_1_no.wav --output input_no.npz
```

Then use the `.npz` input in TVM:

```sh
$ tvmc run ./kws_ref_model.tar -i ./input_no.npz --print-top 12
2022-09-05 22:11:27.934 INFO load_module /tmp/tmp1inyr871/mod.so
[[   3   11    2   10    9    8    7    6    5    4    1    0]
 [  98 -106 -121 -128 -128 -128 -128 -128 -128 -128 -128 -128]]
```

where in `kws_ref_model.tflite` model (keyword spotting the following label indeces apply:

`['Down', 'Go', 'Left', 'No', 'Off', 'On', 'Right', 'Stop', 'Up', 'Yes', 'Silence', 'Unknown']`

Hence 3 => "No" utterance, so "No" scores above `98`, i.e. is `> 0`, so recognized.
