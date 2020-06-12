This code is used to train the audio to expression network.
-> train_audio2expressionsAttentionTMP.sh
You need to provide the training data that fits your face model.
The face model is defined in "BaselModel", you need to provide the average model, a mask for the mouth and the basis vectors.

As training data you need to provide data that fits the "data/multi_face_audio_eq_tmp_cached_dataset.py" data loader.