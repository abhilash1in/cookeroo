# -*- coding: utf-8 -*-
import os
from pydub import AudioSegment
from pathlib import Path
import librosa
import numpy as np

# pipenv install pydub
# brew install ffmpeg


class DataPrep():
    def __init__(self, data_base_path, extension):
        super().__init__()
        # data_base_path = '/Users/abhilash1in/Documents/Projects/Cookeroo/data/'
        # extension = 'wav'
        self._data_base_path = data_base_path
        self._extension = extension

        self._raw_data_path = os.path.join(self._data_base_path, 'raw')
        self._sliced_data_path = os.path.join(self._data_base_path, 'sliced')
        self._sliced_audio_segments = {}

    def _get_file_paths(self, directory_path, extension):
        file_paths = []
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            if file_path.endswith(extension) and os.path.isfile(file_path):
                file_paths.append(file_path)
        return file_paths

    def _get_audio_segments_wav(self, wav_file_paths):
        audio_segments = []
        for file_path in wav_file_paths:
            if file_path.endswith(file_path):
                audio_segment = AudioSegment.from_wav(file_path)
                audio_segments.append(audio_segment)
        return audio_segments

    def _get_sliced_audio_segments(self, audio_segments, slice_interval, allow_gaps=True):
        res_sliced_audio_segments = []
        for audio_segment in audio_segments:
            # print(audio_segment_tuple[0] + ' ' + str(len(get_sliced_audio_segments(audio_segment_tuple, slice_interval))))
            sliced_audio_segments = self._slice_audio_segment(audio_segment, slice_interval, allow_gaps)
            res_sliced_audio_segments.extend(sliced_audio_segments)
        return res_sliced_audio_segments

    def _slice_audio_segment(self, audio_segment, slice_interval, allow_gaps=True):
        sliced_audio_segments = []
        start = 0
        while start < len(audio_segment):
            end = start + slice_interval
            sliced_audio_segment = audio_segment[start:end]
            if len(sliced_audio_segment) >= len(audio_segment) and not allow_gaps:
                break
            elif len(sliced_audio_segment) < slice_interval and not allow_gaps:
                break
            else:
                sliced_audio_segments.append(sliced_audio_segment)
            start = end
        return sliced_audio_segments

    def _export_audio_segments(self, audio_segments, directory_path, extension):
        for index, sliced_audio_segment in enumerate(audio_segments):
            sliced_audio_segment.export(
                os.path.join(directory_path, '{filename}.{extension}'.format(filename=index, extension=extension)),
                format=extension)

    def _clear_dir(self, directory_path):
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                os.remove(os.path.join(root, file))

    def _extract_features(self, file_path):
        try:
            audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
            mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
            mfccsscaled = np.mean(mfccs.T, axis=0)

        except Exception:
            print("Error encountered while parsing file: ", file_path)
            return None

        return mfccsscaled

    def _is_existing_dir(self, directory_path):
        if os.path.exists(directory_path) and os.path.isdir(directory_path):
            return True
        return False

    def export(self):
        for category, audio_segments in self._sliced_audio_segments.items():
            # path to the individual category sliced directory
            category_data_sliced_directory_path = os.path.join(self._sliced_data_path, category)
            # created the sliced directory if it does not exist
            Path(category_data_sliced_directory_path).mkdir(parents=True, exist_ok=True)
            # clear old data in sliced directory
            self._clear_dir(category_data_sliced_directory_path)
            # save sliced AudioSegments to sliced directory
            self._export_audio_segments(audio_segments, category_data_sliced_directory_path, self._extension)

    def slice(self, slice_interval):
        # slice_interval = 3 * 1000  # 3000 miliseconds = 3 seconds

        # raw > subdirectory names = category names
        if not self._is_existing_dir(self._raw_data_path):
            raise ValueError(('Could not find \'raw\' directory at \'{raw_data_path}\'. '
                              'Place your audio data in subdirectories under \'raw\' directory '
                              'and try again.').format(raw_data_path=self._raw_data_path))

        # raw > subdirectory names = category names
        categories = next(os.walk(self._raw_data_path))[1]
        # exclude hidden files
        categories = list(filter(lambda folder_name: not str(folder_name).startswith('.'), categories))

        if len(categories) == 0:
            raise ValueError(('Could not find subdirectories (for categories) under \'raw\' directory ({raw_data_path}). '
                              'Place your audio data in subdirectories (one subdirectory for each category) '
                              'under \'raw\' directory and try again.').format(raw_data_path=self._raw_data_path))

        for category in categories:
            # path to the individual category raw directory
            category_data_raw_directory_path = os.path.join(self._raw_data_path, category)

            # get list of file paths with given extension
            category_data_raw_file_paths = self._get_file_paths(category_data_raw_directory_path, self._extension)
            # list of AudioSegments (one for each file)
            audio_segments = self._get_audio_segments_wav(category_data_raw_file_paths)
            # list of all sliced AudioSegments combined
            self._sliced_audio_segments[category] = self._get_sliced_audio_segments(audio_segments, slice_interval)

        return self._sliced_audio_segments
