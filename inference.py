from Preprocessing import features
import numpy as np
import pandas as pd
import librosa
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import joblib
import warnings
from sklearn.exceptions import InconsistentVersionWarning


def transform_data(wav_path):
    y, sr = librosa.load(wav_path, sr=44100)

    # Preprocessing 과정
    mfcc = features.extract_mfcc(y, sr)
    pitch = features.extract_pitch(y, sr)
    f0_pyworld = features.extract_f0_pyworld(y, sr)
    spectral_flux = features.extract_spectral_flux(y, sr)
    spectral_entropy = features.extract_spectral_entropy(y, sr)

    # 추출된 feature 병합한 dataframe을 concated_df로 선언 후, return
    # 피처 병합
    features_dict = {
        'mfcc': mfcc,
        'pitch': pitch,
        'f0_pyworld': f0_pyworld,
        'spectral_flux': spectral_flux,
        'spectral_entropy': spectral_entropy,
    }
    concatenated_df = merge_features(features_dict)

    # pad_or_truncate 적용
    X = pad_or_truncate(concatenated_df.values)
    X = torch.tensor(X, dtype=torch.float32)

    return X


def merge_features(features_dict):
    # 피처들을 하나의 데이터프레임으로 병합
    df_list = []
    for key, df in features_dict.items():
        # 각 DataFrame의 행 수를 통일
        df_list.append(df)

    # 열 방향으로 병합
    concatenated_df = pd.concat(df_list, axis=1)
    concatenated_df = concatenated_df.fillna(0)
    return concatenated_df


def pad_or_truncate(features, max_length=850):
    length, feature_dim = features.shape
    if length > max_length:
        return features[:max_length]
    elif length < max_length:
        pad_width = max_length - length
        padding = np.zeros((pad_width, feature_dim))
        return np.vstack((features, padding))
    return features


def inference(wav_path, model_path, scaler_path):
    # InconsistentVersionWarning 경고 무시
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

    # 1. WAV 파일 전처리
    # print(f"Processing file: {wav_path}")
    features = transform_data(wav_path).numpy()
    # print(f"Original feature shape: {features.shape}")  # Debugging

    # 2. 차원 변환 (단일 시퀀스 처리)
    # 학습 시 데이터가 (850, feature_dim) 형태였다면 이를 맞추기 위해 아래처럼 처리
    features = features.reshape(1, -1)
    # print(f"Reshaped feature shape: {features.shape}")  # Debugging

    # 3. SVM 모델 및 스케일러 로드
    # print("Loading model and scaler...")
    scaler = joblib.load(scaler_path)
    model = joblib.load(model_path)

    # 4. Feature 정규화
    # print("Scaling features...")
    if features.shape[1] != scaler.n_features_in_:
        raise ValueError(
            f"Mismatch in feature dimensions. Expected {scaler.n_features_in_}, but got {features.shape[1]}"
        )
    features = scaler.transform(features)

    # 5. 추론 수행
    # print("Performing inference...")
    prediction = model.predict(features)

    # print(f"Final Prediction: {prediction[0]}")
    return prediction[0]


if __name__ == "__main__":
    # 본 코드 셀은 예시를 위한 것으로, 서버에서 사용할 때에는 inference 함수만 가져다가 사용하면 됨.
    # 예시 파일 경로
    example_wav_path = "/Users/imdohyeon/Documents/PythonWorkspace/Lieon-ai/Dataset_sample/Val/Audio/data27_augmented_4.wav"
    example_model_path = "/Users/imdohyeon/Documents/PythonWorkspace/Lieon-ai/svm_model.pkl"
    example_scaler_path = "/Users/imdohyeon/Documents/PythonWorkspace/Lieon-ai/scaler.pkl"

    # 추론 실행
    result = inference(example_wav_path, example_model_path, example_scaler_path)
    print(result)