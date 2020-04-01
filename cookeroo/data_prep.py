# -*- coding: utf-8 -*-
import os
from pydub import AudioSegment
from pathlib import Path
import librosa
import numpy as np

# pipenv install pydub
# brew install ffmpeg


class DataPrep():
    def __init__(self):
        super().__init__()

    def _get_file_paths(self, folder_path, extension):
        file_paths = []
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
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
        res_sliced_audo_segments = []
        for audio_segment in audio_segments:
            # print(audio_segment_tuple[0] + ' ' + str(len(get_sliced_audio_segments(audio_segment_tuple, slice_interval))))
            sliced_audio_segments = self._slice_audio_segment(audio_segment, slice_interval, allow_gaps)
            res_sliced_audo_segments.extend(sliced_audio_segments)
        return res_sliced_audo_segments

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

    def _export_audio_segments(self, audio_segments, folder_path, extension):
        for index, sliced_audo_segment in enumerate(audio_segments):
            sliced_audo_segment.export(
                os.path.join(folder_path, '{filename}.{extension}'.format(filename=index, extension=extension)),
                format=extension)

    def _clear_dir(self, folder_path):
        for root, dirs, files in os.walk(folder_path):
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

    def slice(self, data_base_path, extension, slice_interval):
        # data_base_path = '/Users/abhilash1in/Documents/Projects/Cookeroo/data/'
        # extension = 'wav'
        # slice_interval = 3 * 1000  # 3000 miliseconds = 3 seconds

        raw_data_path = os.path.join(data_base_path, 'raw')
        sliced_data_path = os.path.join(data_base_path, 'sliced')

        # raw > subdirectory names = category names
        categories = next(os.walk(raw_data_path))[1]

        for category in categories:
            # path to the individual category raw folder
            category_data_raw_folder_path = os.path.join(raw_data_path, category)

            # path to the individual category sliced folder
            category_data_sliced_folder_path = os.path.join(sliced_data_path, category)
            # created the sliced folder if it does not exist
            Path(category_data_sliced_folder_path).mkdir(parents=True, exist_ok=True)
            # clear old data in sliced folder
            self._clear_dir(category_data_sliced_folder_path)

            # get list of file paths with given extension
            category_data_raw_file_paths = self._get_file_paths(category_data_raw_folder_path, extension)
            # list of AudioSegments (one for each file)
            audio_segments = self._get_audio_segments_wav(category_data_raw_file_paths)
            # list of all sliced AudioSegments combined
            sliced_audo_segments = self._get_sliced_audo_segments(audio_segments, slice_interval)
            # save sliced AudioSegments to sliced folder
            self._export_audio_segments(sliced_audo_segments, category_data_sliced_folder_path, extension)
