This code runs the inference given a source audio and a target video that has been tracked.
The process is started using "transfer.sh" (where you can also specify the target sequences -> TARGET_ACTOR_LIST).
In the "transfer.py" you can specify the source sequences (search for "source_actors").
Make sure that you specify the dimensions of your blendshape model in "BaselModel/basel_model.py" (-> N_EXPRESSIONS).

Note that you have to extract the deepspeech features in a preprocessing step.
In the datasets folder you will find an example how the data should look like.
The deepspeech features are provided as npy files, while for the target sequence you also have to provide the expressions (visually tracked blendshape coefficients).
If you have a different data format you need to adapt the data loader (data/face_dataset.py).

Once you have the prepared data you can run the script.
It will optimize for the mapping from the audio-expression space to your blendshape model space.
The mapping is stored in the "mappings" folder (note: that it caches the mappings there and reuses it for the next run. If you change something, you need to delete this cache).
The final output is stored in the "datasets/TRANSFERS" folder as a list of estimated expressions using the source audio features.

Given these expressions you need to generate new uv maps for the target video using the rigid pose of the target video (only replacing the expressions).
These can then be used in the deferred neural rendering framework.