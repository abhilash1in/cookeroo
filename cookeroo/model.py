import os
import librosa
import numpy as np
import pandas as pd
from .utils import _get_subdirectory_names, _get_file_paths, _is_existing_dir, _clear_dir
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
from keras.models import model_from_json
from datetime import datetime
from pathlib import Path
import pickle



class CookerooModel():
    def __init__(self, data_base_path, model_base_path, extension):
        super().__init__()
        # data_base_path = '/Users/abhilash1in/Documents/Projects/Cookeroo/data/'
        # extension = 'wav'
        self._data_base_path = data_base_path
        self._extension = extension
        self._raw_data_path = os.path.join(self._data_base_path, 'raw')
        self._sliced_data_path = os.path.join(self._data_base_path, 'sliced')
        self._model_base_path = os.path.join(model_base_path, 'models')
        self._model_checkpoint_path = os.path.join(self._model_base_path, 'checkpoints')
        Path(self._model_checkpoint_path).mkdir(parents=True, exist_ok=True)

    def _get_librosa_audio_objects(self, file_paths):
        librosa_audio_objects = []
        for file_path in file_paths:
            librosa_audio, librosa_sample_rate = librosa.load(file_path)
            librosa_audio_objects.append((librosa_audio, librosa_sample_rate))
        return librosa_audio_objects

    def _get_training_data(self, librosa_audio_objects, category):
        features = []
        for librosa_audio_tuple in librosa_audio_objects:
            audio = librosa_audio_tuple[0]
            sample_rate = librosa_audio_tuple[1]
            data = self._extract_features(audio, sample_rate)
            features.append([data, category])
        return features

    def _extract_features(self, audio, sample_rate):
        try:
            mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
            mfccsscaled = np.mean(mfccs.T, axis=0)
        except Exception:
            print("Error encountered while parsing librosa audio")
            return None
        return mfccsscaled

    def _build_model(self, num_labels, filter_size=2):
        # num_labels = yy.shape[1]
        # Construct model
        self.model = Sequential()

        self.model.add(Dense(256, input_shape=(40,)))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.5))

        self.model.add(Dense(256))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.5))

        self.model.add(Dense(num_labels))
        self.model.add(Activation('softmax'))

        # Compile the model
        self.model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

        return self.model

    def _evaluate_model(self, x, y, verbose=0):
        score = self.model.evaluate(x, y, verbose=0)
        accuracy = 100 * score[1]
        return "%.4f%%" % accuracy

    def _train(self, x_train, y_train, x_test, y_test, checkpoint_filepath, num_epochs=100, num_batch_size=32):
        checkpointer = ModelCheckpoint(
            filepath=os.path.join(checkpoint_filepath, 'weights.best.basic_mlp.hdf5'), verbose=1,
            save_best_only=True)
        start = datetime.now()
        print(y_train)
        self.model.fit(x_train, y_train, batch_size=num_batch_size, epochs=num_epochs,
                       validation_data=(x_test, y_test), callbacks=[checkpointer], verbose=1)
        duration = datetime.now() - start
        print("Training completed in time: ", duration)

    def train(self):
        _clear_dir(self._model_base_path)
        if not _is_existing_dir(self._sliced_data_path):
            raise ValueError(('Could not find \'sliced\' directory at \'{path}\'. '
                              'Place your training audio data in subdirectories under \'sliced\' '
                              'directory and try again.').format(path=self._sliced_data_path))

        # sliced > subdirectory names => category names
        categories = _get_subdirectory_names(self._sliced_data_path)

        if len(categories) == 0:
            raise ValueError(('Could not find subdirectories (for categories) under \'sliced\' directory ({path}). '
                              'Place your training audio data in subdirectories (one subdirectory for each category) '
                              'under \'sliced\' directory and try again.').format(path=self._sliced_data_path))
        features = []
        for category in categories:
            # path to the individual category sliced directory
            category_data_sliced_directory_path = os.path.join(self._sliced_data_path, category)
            # get list of file paths with given extension
            category_data_sliced_file_paths = _get_file_paths(category_data_sliced_directory_path, self._extension)
            # list of librosa audio objects (one for each file)
            librosa_audio_objects_for_category = self._get_librosa_audio_objects(category_data_sliced_file_paths)
            # extract features from librosa audio object and apply label
            features.extend(self._get_training_data(librosa_audio_objects_for_category, category))
        featuresdf = pd.DataFrame(features, columns=['feature', 'class_label'])
        print('Finished feature extraction from ', len(featuresdf), ' files')

        # Convert features and corresponding classification labels into numpy arrays
        X = np.array(featuresdf.feature.tolist())
        y = np.array(featuresdf.class_label.tolist())

        # Encode the classification labels
        self._le = LabelEncoder()
        yy = to_categorical(self._le.fit_transform(y))
        label_encoder_output = open(os.path.join(self._model_base_path, 'classes.pkl'), 'wb')
        pickle.dump(self._le, label_encoder_output)
        x_train, x_test, y_train, y_test = train_test_split(X, yy, test_size=0.2, random_state=42)
        self._build_model(num_labels=yy.shape[1])
        # Calculate pre-training accuracy
        pre_training_accuracy = self._evaluate_model(x_test, y_test, verbose=0)
        print('Pre-training accuracy = {}'.format(pre_training_accuracy))

        self._train(x_train, y_train, x_test, y_test, checkpoint_filepath=self._model_checkpoint_path,
                    num_batch_size=32, num_epochs=100)

        training_accuracy = self._evaluate_model(x_train, y_train)
        testing_accuracy = self._evaluate_model(x_test, y_test)
        model_json = self.model.to_json()
        json_file_path = os.path.join(self._model_base_path, 'model.json')
        with open(json_file_path, 'w') as json_file:
            json_file.write(model_json)
        self.model.save_weights(os.path.join(self._model_base_path, 'model.h5'))
        print("Saving model")
        print('Training accuracy = {}'.format(training_accuracy))
        print('Testing accuracy = {}'.format(testing_accuracy))

    def predict(self, input_path):
        try:
            json_file = open(os.path.join(self._model_base_path, 'model.json'), 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            loaded_model = model_from_json(loaded_model_json)
            loaded_model.load_weights(os.path.join(self._model_base_path, 'model.h5'), 'r')
            label_encoder_file = open(os.path.join(self._model_base_path, 'classes.pkl'), 'rb')
            label_encoder = pickle.load(label_encoder_file)
            label_encoder_file.close()
            print("Model loaded from disk")
            librosa_object_tuple = self._get_librosa_audio_objects([input_path])[0]
            audio = librosa_object_tuple[0]
            sample_rate = librosa_object_tuple[1]
            data = self._extract_features(audio, sample_rate)
            if(data.ndim == 1):
                data = np.array([data])
            prediction_vector = loaded_model.predict_classes(data)
            predicted_class = label_encoder.inverse_transform(prediction_vector)
            print("The predicted class is:", predicted_class[0], '\n')
            predicted_proba_vector = self.model.predict_proba(data)
            predicted_proba = predicted_proba_vector[0]
            for i in range(len(predicted_proba)):
                category = label_encoder.inverse_transform(np.array([i]))
                print(category[0], "\t\t : ", format(predicted_proba[i], '.32f'))

        except Exception as e:
            print(str(e))
