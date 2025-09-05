import os
import re
import numpy as np
import torch
import torchaudio
import gradio as gr
from os import getenv
import json
import time
from pathlib import Path

from zonos.model import Zonos, DEFAULT_BACKBONE_CLS as ZonosBackbone
from zonos.conditioning import make_cond_dict, supported_language_codes
from zonos.utils import DEFAULT_DEVICE as device

# 자동처리(무조건 조건) 설정 매핑
AUTO_SETTING_MAP = {
    "자동 감정": "emotion",
    "자동 vq": "vqscore_8",
    "자동 주파수": "fmax",
    "자동 음높이": "pitch_std",
    "자동 속도": "speaking_rate",
    "자동 dnsmos": "dnsmos_ovrl",
    "자동 노이즈": "speaker_noised",
    "자동 화자": "speaker",
}

# 사용자 설정 디렉토리
USER_DATA_DIR = os.path.join(os.path.expanduser("~"), "ZonosData")

# 설정 파일 경로
CONFIG_FILE = os.path.join(os.path.expanduser("~"), "zonos_config.json")

# 설정 파일에서 경로 불러오기
if os.path.exists(CONFIG_FILE):
    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            config = json.load(f)
            USER_DATA_DIR = config.get("data_dir", USER_DATA_DIR)
    except Exception as e:
        print(f"설정 파일 불러오기 오류: {str(e)}")

CURRENT_MODEL_TYPE = None
CURRENT_MODEL = None

# 여러 화자의 임베딩을 저장할 딕셔너리
SPEAKER_EMBEDDINGS = {}
SPEAKER_AUDIO_PATHS = {}

# 설정값의 기본값을 저장하는 딕셔너리
DEFAULT_SETTINGS = {
    "emotion1": 1.0,  # 행복
    "emotion2": 0.05,  # 슬픔
    "emotion3": 0.05,  # 혐오
    "emotion4": 0.05,  # 두려움
    "emotion5": 0.05,  # 놀람
    "emotion6": 0.05,  # 분노
    "emotion7": 0.1,  # 기타
    "emotion8": 0.2,  # 중립
    "vq_single": 0.78,  # VQ 스코어
    "fmax": 24000,  # 최대 주파수 (Hz)
    "pitch_std": 45.0,  # 음높이 표준편차
    "speaking_rate": 15.0,  # 말하기 속도
    "dnsmos_ovrl": 4.0,  # DNSMOS 전체
    "speaker_noised": False,  # 화자 노이즈 제거
    "cfg_scale": 2.0,  # CFG 스케일
    "seed": 420,  # 시드
}

# 설정 이름 매핑 (화자 설정 문자열 -> 내부 변수명)
SETTING_NAME_MAP = {
    "happy": "emotion1",
    "sad": "emotion2",
    "disgust": "emotion3",
    "fear": "emotion4",
    "surprise": "emotion5",
    "anger": "emotion6",
    "other": "emotion7",
    "neutral": "emotion8",
    "vq": "vq_single",
    "fmax": "fmax",
    "pitch": "pitch_std",
    "rate": "speaking_rate",
    "dnsmos": "dnsmos_ovrl",
    "noised": "speaker_noised",
    "cfg": "cfg_scale",
    "seed": "seed",
}

# 저장된 설정 프리셋을 위한 디렉토리
PRESETS_DIR = os.path.join(USER_DATA_DIR, "presets")
DIALOGUES_DIR = os.path.join(USER_DATA_DIR, "saved_dialogues")
SPEAKER_AUDIO_DIR = os.path.join(USER_DATA_DIR, "speaker_audio")

# 디렉토리가 없으면 생성
os.makedirs(PRESETS_DIR, exist_ok=True)
os.makedirs(DIALOGUES_DIR, exist_ok=True)
os.makedirs(SPEAKER_AUDIO_DIR, exist_ok=True)


def generate_settings_string(
    speaker_name,
    emotion1,
    emotion2,
    emotion3,
    emotion4,
    emotion5,
    emotion6,
    emotion7,
    emotion8,
    vq_score,
    fmax,
    pitch,
    speech_rate,
    dnsmos,
    noised,
    cfg,
    seed,
    use_seed,
):
    """설정값을 포함한 화자 문자열 생성"""
    settings = []

    # 감정 설정 추가
    if emotion1 != DEFAULT_SETTINGS["emotion1"]:
        settings.append(f"happy:{emotion1:.2f}")
    if emotion2 != DEFAULT_SETTINGS["emotion2"]:
        settings.append(f"sad:{emotion2:.2f}")
    if emotion3 != DEFAULT_SETTINGS["emotion3"]:
        settings.append(f"disgust:{emotion3:.2f}")
    if emotion4 != DEFAULT_SETTINGS["emotion4"]:
        settings.append(f"fear:{emotion4:.2f}")
    if emotion5 != DEFAULT_SETTINGS["emotion5"]:
        settings.append(f"surprise:{emotion5:.2f}")
    if emotion6 != DEFAULT_SETTINGS["emotion6"]:
        settings.append(f"anger:{emotion6:.2f}")
    if emotion7 != DEFAULT_SETTINGS["emotion7"]:
        settings.append(f"other:{emotion7:.2f}")
    if emotion8 != DEFAULT_SETTINGS["emotion8"]:
        settings.append(f"neutral:{emotion8:.2f}")

    # 음성 품질 설정 추가
    if vq_score != DEFAULT_SETTINGS["vq_single"]:
        settings.append(f"vq:{vq_score:.2f}")
    if fmax != DEFAULT_SETTINGS["fmax"]:
        settings.append(f"fmax:{int(fmax)}")
    if pitch != DEFAULT_SETTINGS["pitch_std"]:
        settings.append(f"pitch:{pitch:.1f}")
    if speech_rate != DEFAULT_SETTINGS["speaking_rate"]:
        settings.append(f"rate:{speech_rate:.1f}")
    if dnsmos != DEFAULT_SETTINGS["dnsmos_ovrl"]:
        settings.append(f"dnsmos:{dnsmos:.1f}")

    # 노이즈 제거 설정 추가
    if noised != DEFAULT_SETTINGS["speaker_noised"]:
        settings.append(f"noised:{str(noised).lower()}")

    # CFG 설정 추가
    if cfg != DEFAULT_SETTINGS["cfg_scale"]:
        settings.append(f"cfg:{cfg:.1f}")

    # 시드 설정 추가 (체크박스로 활성화된 경우)
    if use_seed:
        settings.append(f"seed:{int(seed)}")

    # 설정이 있는 경우 포맷팅, 없으면 기본 형식 반환
    if settings:
        return f"[{speaker_name}|{'|'.join(settings)}]: "
    else:
        return f"[{speaker_name}]: "


def split_into_sentences(text):
    # Simplified sentence splitting (handles basic cases)
    sentences = re.split(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)(?=\s|[A-Z]|$)", text)
    return [s.strip() for s in sentences if s.strip()]


def count_words(text):
    return len(text.split())


def split_into_chunks(text, word_limit=50):
    sentences = split_into_sentences(text)
    chunks = []
    current_chunk = []
    current_word_count = 0

    for sentence in sentences:
        sentence_word_count = count_words(sentence)

        if sentence_word_count > word_limit:
            # Handle very long sentences
            if current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_word_count = 0

            # Split long sentence into smaller parts
            long_sentence_parts = re.split(r"(?<=[,.])\s+", sentence)
            for part in long_sentence_parts:
                part_word_count = count_words(part)
                if current_word_count + part_word_count <= word_limit:
                    current_chunk.append(part)
                    current_word_count += part_word_count
                else:
                    if current_chunk:
                        chunks.append(" ".join(current_chunk))
                    current_chunk = [part]
                    current_word_count = part_word_count
            if current_chunk:  # Add any remaining part
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_word_count = 0

        elif current_word_count + sentence_word_count <= word_limit:
            current_chunk.append(sentence)
            current_word_count += sentence_word_count
        else:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_word_count = sentence_word_count

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def measure_audio_level(audio):
    """오디오의 RMS 레벨을 dB로 반환합니다."""
    # 음성 신호가 너무 작으면(무음일 경우) 기본값 반환
    if np.max(np.abs(audio)) < 1e-6:
        return -100.0  # 매우 작은 레벨의 기본값
        
    # RMS를 dBFS로 변환
    rms = np.sqrt(np.mean(audio**2))
    level_db = 20 * np.log10(rms / 1.0)
    
    return level_db

def balance_first_chunk(audio_segments):
    """첫 청크와 나머지 청크의 음량 차이를 차이를 바탕으로 첫 청크의 음량을 조정합니다."""
    if len(audio_segments) <= 1:
        return audio_segments
        
    # 첫 청크 레벨 측정
    first_chunk_level = measure_audio_level(audio_segments[0])
    
    # 나머지 청크의 평균 레벨 측정
    rest_chunk_levels = [measure_audio_level(chunk) for chunk in audio_segments[1:]]
    valid_levels = [level for level in rest_chunk_levels if level > -90.0]  # 극단적으로 낮은 값은 제외
    
    if not valid_levels:
        return audio_segments  # 유효한 레벨이 없는 경우 원본 반환
    
    avg_rest_level = sum(valid_levels) / len(valid_levels)
    
    # 레벨 차이 계산 (dB)
    level_diff = avg_rest_level - first_chunk_level
    
    # 차이가 크지 않으면 조정하지 않음
    if abs(level_diff) < 3.0:  # 3dB 미만의 차이는 무시
        return audio_segments
    
    print(f"첫 청크 레벨: {first_chunk_level:.1f}dB, 나머지 평균 레벨: {avg_rest_level:.1f}dB, 차이: {level_diff:.1f}dB")
    
    # 첫 청크에 게인 적용
    gain = 10 ** (level_diff / 20.0)
    adjusted_first_chunk = audio_segments[0] * gain
    
    # 클리핑 방지
    if np.max(np.abs(adjusted_first_chunk)) > 0.999:
        adjusted_first_chunk = adjusted_first_chunk / np.max(np.abs(adjusted_first_chunk)) * 0.999
    
    # 조정된 첫 청크와 나머지 청크 합치기
    balanced_segments = [adjusted_first_chunk] + audio_segments[1:]
    
    return balanced_segments

def normalize_all_chunks_to_loudest(audio_segments, first_chunk_boost_db=2.0):
    """모든 청크의 음량을 측정하고 가장 큰 음량으로 정규화합니다.
    first_chunk_boost_db: 첫 번째 청크에 추가로 적용할 부스트(dB)"""
    if len(audio_segments) <= 1:
        return audio_segments
    
    # 모든 청크의 레벨 측정
    chunk_levels = [measure_audio_level(chunk) for chunk in audio_segments]
    valid_levels = [level for level in chunk_levels if level > -90.0]  # 극단적으로 낮은 값은 제외
    
    if not valid_levels:
        return audio_segments  # 유효한 레벨이 없는 경우 원본 반환
    
    # 가장 큰 레벨 찾기
    max_level = max(valid_levels)
    
    print(f"각 청크 레벨: {[f'{level:.1f}dB' for level in chunk_levels]}")
    print(f"가장 큰 레벨: {max_level:.1f}dB")
    
    # 모든 청크를 가장 큰 레벨로 정규화
    normalized_segments = []
    for i, (chunk, level) in enumerate(zip(audio_segments, chunk_levels)):
        if level < -90.0:  # 무음에 가까운 청크는 건너뜀
            normalized_segments.append(chunk)
            continue
        
        # 레벨 차이 계산 (dB)
        level_diff = max_level - level
        
        # 첫 번째 청크의 경우 추가 부스트 적용
        if i == 0:
            level_diff += first_chunk_boost_db
            print(f"첫 청크 추가 부스트 {first_chunk_boost_db}dB 적용")
        
        # 차이가 작으면 조정하지 않음 
        # (첫 번째 청크의 경우 부스트가 적용된 차이 사용)
        if abs(level_diff) < 1.0 and i != 0:  # 1dB 미만의 차이는 무시 (첫 청크 제외)
            normalized_segments.append(chunk)
            continue
        
        # 게인 적용
        gain = 10 ** (level_diff / 20.0)
        adjusted_chunk = chunk * gain
        
        # 클리핑 방지
        if np.max(np.abs(adjusted_chunk)) > 0.999:
            adjusted_chunk = adjusted_chunk / np.max(np.abs(adjusted_chunk)) * 0.999
        
        normalized_segments.append(adjusted_chunk)
        print(f"청크 {i}: {level:.1f}dB → {max_level:.1f}dB (조정: {level_diff:.1f}dB)")
    
    return normalized_segments

def normalize_audio(audio, target_level=-23.0):
    """정규화된 오디오를 반환합니다. target_level은 dBFS(dB relative to full scale)입니다."""
    # 음성 신호가 너무 작으면(무음일 경우) 정규화를 건너뜁니다
    if np.max(np.abs(audio)) < 1e-6:
        return audio
        
    # 현재 음성 레벨 계산 (RMS를 dBFS로 변환)
    rms = np.sqrt(np.mean(audio**2))
    current_level = 20 * np.log10(rms / 1.0)
    
    # 원하는 레벨로 조정하기 위한 게인 계산
    gain = 10**((target_level - current_level) / 20)
    
    # 게인 적용
    normalized_audio = audio * gain
    
    # 클리핑 방지
    if np.max(np.abs(normalized_audio)) > 0.999:
        normalized_audio = normalized_audio / np.max(np.abs(normalized_audio)) * 0.999
        
    return normalized_audio

def apply_volume_adjustment(audio, db_adjustment):
    """오디오 볼륨을 주어진 dB 값만큼 조정합니다."""
    if db_adjustment == 0:
        return audio
        
    # dB to linear gain conversion
    gain = 10 ** (db_adjustment / 20.0)
    
    # 볼륨 조정
    adjusted_audio = audio * gain
    
    # 클리핑 방지
    if np.max(np.abs(adjusted_audio)) > 0.999:
        adjusted_audio = adjusted_audio / np.max(np.abs(adjusted_audio)) * 0.999
        
    return adjusted_audio
def concatenate_audio(audio_segments, silence_duration=0.2):
    """Concatenates audio segments with optional silence between them."""
    silence = np.zeros(int(48000 * silence_duration))  # Assuming 48kHz sample rate
    concatenated = []
    for audio in audio_segments:
        concatenated.append(audio)
        concatenated.append(silence)  # Add silence
    return np.concatenate(concatenated)
def load_model_if_needed(model_choice: str):
    global CURRENT_MODEL_TYPE, CURRENT_MODEL
    if CURRENT_MODEL_TYPE != model_choice:
        if CURRENT_MODEL is not None:
            del CURRENT_MODEL
            torch.cuda.empty_cache()
        print(f"Loading {model_choice} model...")
        CURRENT_MODEL = Zonos.from_pretrained(model_choice, device=device)
        CURRENT_MODEL.requires_grad_(False).eval()
        CURRENT_MODEL_TYPE = model_choice
        print(f"{model_choice} model loaded successfully!")
    return CURRENT_MODEL


def update_ui(model_choice):
    """
    Dynamically show/hide UI elements based on the model's conditioners.
    We do NOT display 'language_id' or 'ctc_loss' even if they exist in the model.
    """
    model = load_model_if_needed(model_choice)
    cond_names = [c.name for c in model.prefix_conditioner.conditioners]
    print("Conditioners in this model:", cond_names)

    text_update = gr.update(visible=("espeak" in cond_names))
    language_update = gr.update(visible=("espeak" in cond_names))
    speaker_management_update = gr.update(visible=("speaker" in cond_names))
    prefix_audio_update = gr.update(visible=True)
    emotion1_update = gr.update(visible=("emotion" in cond_names))
    emotion2_update = gr.update(visible=("emotion" in cond_names))
    emotion3_update = gr.update(visible=("emotion" in cond_names))
    emotion4_update = gr.update(visible=("emotion" in cond_names))
    emotion5_update = gr.update(visible=("emotion" in cond_names))
    emotion6_update = gr.update(visible=("emotion" in cond_names))
    emotion7_update = gr.update(visible=("emotion" in cond_names))
    emotion8_update = gr.update(visible=("emotion" in cond_names))
    vq_single_slider_update = gr.update(visible=("vqscore_8" in cond_names))
    fmax_slider_update = gr.update(visible=("fmax" in cond_names))
    pitch_std_slider_update = gr.update(visible=("pitch_std" in cond_names))
    speaking_rate_slider_update = gr.update(visible=("speaking_rate" in cond_names))
    dnsmos_slider_update = gr.update(visible=("dnsmos_ovrl" in cond_names))
    speaker_noised_checkbox_update = gr.update(visible=("speaker_noised" in cond_names))
    unconditional_keys_update = gr.update(
        choices=[name for name in cond_names if name not in ("espeak", "language_id")]
    )

    return (
        text_update,
        language_update,
        speaker_management_update,
        prefix_audio_update,
        emotion1_update,
        emotion2_update,
        emotion3_update,
        emotion4_update,
        emotion5_update,
        emotion6_update,
        emotion7_update,
        emotion8_update,
        vq_single_slider_update,
        fmax_slider_update,
        pitch_std_slider_update,
        speaking_rate_slider_update,
        dnsmos_slider_update,
        speaker_noised_checkbox_update,
        unconditional_keys_update,
    )


# 새로운 화자를 추가하는 함수 - 수정: 양쪽 드롭다운 모두 업데이트
def add_speaker(model_choice, speaker_name, speaker_audio, speaker_list):
    global SPEAKER_EMBEDDINGS, SPEAKER_AUDIO_PATHS

    if not speaker_name.strip():
        return gr.update(), gr.update(), "화자 이름을 입력해주세요."

    if speaker_name in SPEAKER_EMBEDDINGS:
        return gr.update(), gr.update(), f"'{speaker_name}' 화자가 이미 존재합니다."

    if speaker_audio is None:
        return gr.update(), gr.update(), "화자 오디오를 업로드해주세요."

    selected_model = load_model_if_needed(model_choice)

    try:
        # 오디오 파일을 지정된 폴더로 복사
        safe_name = re.sub(r'[\\/*?:"<>|]', "_", speaker_name)
        audio_ext = os.path.splitext(speaker_audio)[1]
        new_audio_path = os.path.join(SPEAKER_AUDIO_DIR, f"{safe_name}{audio_ext}")
        
        # 원본 파일 복사
        import shutil
        shutil.copy2(speaker_audio, new_audio_path)
        
        # 화자 임베딩 생성
        wav, sr = torchaudio.load(new_audio_path)
        embedding = selected_model.make_speaker_embedding(wav, sr)
        embedding = embedding.to(device, dtype=torch.bfloat16)

        # 화자 정보 저장
        SPEAKER_EMBEDDINGS[speaker_name] = embedding
        SPEAKER_AUDIO_PATHS[speaker_name] = new_audio_path

        speaker_list = list(SPEAKER_EMBEDDINGS.keys())

        # 양쪽 드롭다운 모두 업데이트
        dropdown_update = gr.update(choices=speaker_list, value=speaker_name)
        inline_dropdown_update = gr.update(choices=speaker_list, value=speaker_name)

        return (
            dropdown_update,
            inline_dropdown_update,
            f"'{speaker_name}' 화자가 추가되었습니다. (저장 위치: {new_audio_path})",
        )
    except Exception as e:
        return gr.update(), gr.update(), f"화자 추가 중 오류 발생: {str(e)}"


# 화자를 삭제하는 함수 - 수정: 양쪽 드롭다운 모두 업데이트
def remove_speaker(speaker_name, speaker_list):
    global SPEAKER_EMBEDDINGS, SPEAKER_AUDIO_PATHS

    if speaker_name not in SPEAKER_EMBEDDINGS:
        return gr.update(), gr.update(), f"'{speaker_name}' 화자를 찾을 수 없습니다."

    del SPEAKER_EMBEDDINGS[speaker_name]
    del SPEAKER_AUDIO_PATHS[speaker_name]

    speaker_list = list(SPEAKER_EMBEDDINGS.keys())

    if speaker_list:
        dropdown_update = gr.update(choices=speaker_list, value=speaker_list[0])
        inline_dropdown_update = gr.update(choices=speaker_list, value=speaker_list[0])
        return (
            dropdown_update,
            inline_dropdown_update,
            f"'{speaker_name}' 화자가 삭제되었습니다.",
        )
    else:
        dropdown_update = gr.update(choices=[], value=None)
        inline_dropdown_update = gr.update(choices=[], value=None)
        return (
            dropdown_update,
            inline_dropdown_update,
            f"'{speaker_name}' 화자가 삭제되었습니다.",
        )


# 대화 스크립트를 파싱하는 함수 - 화자별 설정을 포함하도록 수정
def parse_dialogue(dialogue_text):
    """
    대화 스크립트를 파싱하여 화자와 텍스트, 설정으로 분리

    형식: [화자명|설정1:값1|설정2:값2]: 텍스트
    예시: [Alice|happy:0.8|sad:0.2|rate:20.0]: 안녕하세요!
    """
    lines = dialogue_text.strip().split("\n")
    dialogue_parts = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # [화자명|설정1:값1|설정2:값2]: 텍스트 형식 파싱
        match = re.match(r"\[([^\]]+)\]:(.*)", line)
        if match:
            speaker_with_settings = match.group(1).strip()
            text = match.group(2).strip()

            # 설정이 있는 경우 파싱
            parts = speaker_with_settings.split("|")
            speaker = parts[0].strip()

            # 설정값 파싱
            settings = {}
            # 화자별 무조건 설정 추가
            speaker_unconditional_keys = []
            
            for part in parts[1:]:
                # ":"가 있는 설정 처리 (키:값 형태)
                if ":" in part:
                    key, value = part.split(":", 1)
                    key = key.strip().lower()

                    # 설정 이름 매핑 적용
                    if key in SETTING_NAME_MAP:
                        key = SETTING_NAME_MAP[key]

                    # 설정값 변환 (boolean 또는 float)
                    if key == "speaker_noised":
                        value = value.strip().lower() in ("true", "yes", "t", "y", "1")
                    else:
                        try:
                            value = float(value.strip())
                        except ValueError:
                            print(f"경고: 설정값 '{value}' 변환 실패, 기본값 사용")
                            continue

                    settings[key] = value
                # ":"가 없는 설정 처리 (키만 있는 형태)
                else:
                    # 자동 XXX 설정 처리
                    part = part.strip()
                    if part in AUTO_SETTING_MAP:
                        # 무조건 설정 추가
                        speaker_unconditional_keys.append(AUTO_SETTING_MAP[part])
            
            # 무조건 설정 저장
            if speaker_unconditional_keys:
                settings["unconditional_keys"] = speaker_unconditional_keys

            dialogue_parts.append((speaker, text, settings))
        else:
            # 형식이 맞지 않는 경우, 마지막 화자의 텍스트로 추가
            if dialogue_parts:
                last_speaker, last_text, last_settings = dialogue_parts[-1]
                dialogue_parts[-1] = (
                    last_speaker,
                    last_text + " " + line,
                    last_settings,
                )
            else:
                # 첫 번째 줄부터 형식이 맞지 않는 경우
                dialogue_parts.append(("Unknown", line, {}))

    return dialogue_parts


def process_audio(final_audio, normalize_audio_toggle, volume_adjustment):
    """오디오를 처리하고 반환합니다."""
    # 오디오 정규화 적용 (Toggle이 켜져 있을 때)
    if normalize_audio_toggle:
        final_audio = normalize_audio(final_audio)
        
    # 볼륨 조정 적용
    if volume_adjustment != 0:
        final_audio = apply_volume_adjustment(final_audio, volume_adjustment)
        
    return final_audio

def generate_multi_speaker_audio(
    model_choice,
    dialogue_text,
    language,
    prefix_audio,
    e1,
    e2,
    e3,
    e4,
    e5,
    e6,
    e7,
    e8,
    vq_single,
    fmax,
    pitch_std,
    speaking_rate,
    dnsmos_ovrl,
    speaker_noised,
    cfg_scale,
    linear,
    confidence,
    quadratic,
    seed,
    randomize_seed,
    unconditional_keys,
    volume_adjustment,
    normalize_audio_toggle,
    progress=gr.Progress(),
):
    """
    여러 화자의 대화를 생성하는 함수 - 화자별 설정 적용 가능
    """
    global SPEAKER_EMBEDDINGS, DEFAULT_SETTINGS

    try:
        # UI의 설정값을 기본값으로 설정
        DEFAULT_SETTINGS.update(
            {
                "emotion1": float(e1),
                "emotion2": float(e2),
                "emotion3": float(e3),
                "emotion4": float(e4),
                "emotion5": float(e5),
                "emotion6": float(e6),
                "emotion7": float(e7),
                "emotion8": float(e8),
                "vq_single": float(vq_single),
                "fmax": float(fmax),
                "pitch_std": float(pitch_std),
                "speaking_rate": float(speaking_rate),
                "dnsmos_ovrl": float(dnsmos_ovrl),
                "speaker_noised": bool(speaker_noised),
                "cfg_scale": float(cfg_scale),
                "seed": int(seed),
            }
        )

        if not dialogue_text.strip():
            return (None, None), seed, "대화 텍스트를 입력해주세요."

        if not SPEAKER_EMBEDDINGS:
            return (None, None), seed, "적어도 하나의 화자를 추가해주세요."

        # 대화 파싱 - 화자별 설정 포함
        dialogue_parts = parse_dialogue(dialogue_text)

        if not dialogue_parts:
            return (None, None), seed, "유효한 대화를 찾을 수 없습니다."

        # 화자가 존재하는지 검증
        unknown_speakers = []
        for speaker, _, _ in dialogue_parts:
            if speaker not in SPEAKER_EMBEDDINGS and speaker != "Unknown":
                unknown_speakers.append(speaker)

        if unknown_speakers:
            return (
                (None, None),
                seed,
                f"다음 화자를 찾을 수 없습니다: {', '.join(unknown_speakers)}",
            )

        audio_segments = []
        current_seed = seed
        global_randomize_seed = randomize_seed

        # 총 진행 단계 계산
        total_steps = 0
        for _, text, _ in dialogue_parts:
            chunks = split_into_chunks(text)
            for chunk in chunks:
                estimated_generation_duration = 30 * len(chunk) / 400
                total_steps += int(estimated_generation_duration * 86)

        steps_so_far = 0

        # 각 대화 부분마다 음성 생성 - 각 화자별 설정 적용
        for part_idx, (speaker_name, text, speaker_settings) in enumerate(
            dialogue_parts
        ):
            chunks = split_into_chunks(text)

            for i, chunk in enumerate(chunks):
                print(
                    f"Processing speaker '{speaker_name}', chunk {i+1}/{len(chunks)}: {chunk[:50]}..."
                )

                # 화자 임베딩 가져오기
                if speaker_name in SPEAKER_EMBEDDINGS:
                    speaker_embedding = SPEAKER_EMBEDDINGS[speaker_name]
                else:
                    # Unknown 화자인 경우 첫 번째 화자 사용
                    speaker_name = list(SPEAKER_EMBEDDINGS.keys())[0]
                    speaker_embedding = SPEAKER_EMBEDDINGS[speaker_name]

                # 오디오 생성 (첫 번째 부분의 첫 번째 청크에만 오디오 프리픽스 사용)
                selected_model = load_model_if_needed(model_choice)

                # 기본 설정과 화자별 설정 병합 (화자별 설정이 우선)
                # 각 설정 항목에 대해 화자별 설정이 있으면 그것을 사용, 없으면 기본값 사용
                emotion1_val = speaker_settings.get(
                    "emotion1", DEFAULT_SETTINGS["emotion1"]
                )
                emotion2_val = speaker_settings.get(
                    "emotion2", DEFAULT_SETTINGS["emotion2"]
                )
                emotion3_val = speaker_settings.get(
                    "emotion3", DEFAULT_SETTINGS["emotion3"]
                )
                emotion4_val = speaker_settings.get(
                    "emotion4", DEFAULT_SETTINGS["emotion4"]
                )
                emotion5_val = speaker_settings.get(
                    "emotion5", DEFAULT_SETTINGS["emotion5"]
                )
                emotion6_val = speaker_settings.get(
                    "emotion6", DEFAULT_SETTINGS["emotion6"]
                )
                emotion7_val = speaker_settings.get(
                    "emotion7", DEFAULT_SETTINGS["emotion7"]
                )
                emotion8_val = speaker_settings.get(
                    "emotion8", DEFAULT_SETTINGS["emotion8"]
                )
                vq_val = speaker_settings.get(
                    "vq_single", DEFAULT_SETTINGS["vq_single"]
                )
                fmax_val = speaker_settings.get("fmax", DEFAULT_SETTINGS["fmax"])
                pitch_std_val = speaker_settings.get(
                    "pitch_std", DEFAULT_SETTINGS["pitch_std"]
                )
                speaking_rate_val = speaker_settings.get(
                    "speaking_rate", DEFAULT_SETTINGS["speaking_rate"]
                )
                dnsmos_ovrl_val = speaker_settings.get(
                    "dnsmos_ovrl", DEFAULT_SETTINGS["dnsmos_ovrl"]
                )
                speaker_noised_bool = speaker_settings.get(
                    "speaker_noised", DEFAULT_SETTINGS["speaker_noised"]
                )

                # 화자별 CFG와 시드 설정 적용
                cfg_scale_val = float(
                    speaker_settings.get("cfg_scale", DEFAULT_SETTINGS["cfg_scale"])
                )

                # 시드 설정 (화자별 시드가 있으면 사용, 없으면 현재 시드 사용)
                speaker_seed = speaker_settings.get("seed", None)
                # 화자별 시드가 지정되어 있으면 그것을 사용
                if speaker_seed is not None:
                    current_seed = int(speaker_seed)
                    # 화자별 시드가 지정된 경우 무작위화 비활성화
                    local_randomize_seed = False
                else:
                    # 화자별 시드가 없으면 전역 무작위화 설정 따름
                    local_randomize_seed = global_randomize_seed

                linear_val = float(linear)
                confidence_val = float(confidence)
                quadratic_val = float(quadratic)
                max_new_tokens = 86 * 30

                if local_randomize_seed:
                    current_seed = torch.randint(0, 2**32 - 1, (1,)).item()
                torch.manual_seed(current_seed)

                audio_prefix_codes = None
                if part_idx == 0 and i == 0 and prefix_audio is not None:
                    wav_prefix, sr_prefix = torchaudio.load(prefix_audio)
                    wav_prefix = wav_prefix.mean(0, keepdim=True)
                    wav_prefix = selected_model.autoencoder.preprocess(
                        wav_prefix, sr_prefix
                    )
                    wav_prefix = wav_prefix.to(device, dtype=torch.float32)
                    audio_prefix_codes = selected_model.autoencoder.encode(
                        wav_prefix.unsqueeze(0)
                    )

                # 화자별 감정 설정 적용
                emotion_tensor = torch.tensor(
                    [
                        emotion1_val,
                        emotion2_val,
                        emotion3_val,
                        emotion4_val,
                        emotion5_val,
                        emotion6_val,
                        emotion7_val,
                        emotion8_val,
                    ],
                    device=device,
                )

                vq_tensor = torch.tensor([vq_val] * 8, device=device).unsqueeze(0)

                # 화자별 무조건 설정 처리
                # 기본 무조건 설정과 화자별 무조건 설정 합치기
                current_unconditional_keys = list(unconditional_keys)
                
                # 화자별 unconditional_keys가 있는지 확인
                if "unconditional_keys" in speaker_settings:
                    speaker_unconditional = speaker_settings["unconditional_keys"]
                    # 기존 리스트에 없는 항목만 추가
                    for key in speaker_unconditional:
                        if key not in current_unconditional_keys:
                            current_unconditional_keys.append(key)
                    print(f"Speaker '{speaker_name}' has custom unconditional settings: {speaker_unconditional}")
                
                cond_dict = make_cond_dict(
                    text=chunk,
                    language=language,
                    speaker=speaker_embedding,
                    emotion=emotion_tensor,
                    vqscore_8=vq_tensor,
                    fmax=fmax_val,
                    pitch_std=pitch_std_val,
                    speaking_rate=speaking_rate_val,
                    dnsmos_ovrl=dnsmos_ovrl_val,
                    speaker_noised=speaker_noised_bool,
                    device=device,
                    unconditional_keys=current_unconditional_keys,
                )
                conditioning = selected_model.prepare_conditioning(cond_dict)

                estimated_generation_duration = 30 * len(chunk) / 400
                estimated_total_steps = int(estimated_generation_duration * 86)

                def update_progress(
                    _frame: torch.Tensor, step: int, _total_steps: int
                ) -> bool:
                    progress((steps_so_far + step, total_steps))
                    return True

                try:
                    codes = selected_model.generate(
                        prefix_conditioning=conditioning,
                        audio_prefix_codes=audio_prefix_codes,
                        max_new_tokens=max_new_tokens,
                        cfg_scale=cfg_scale_val,
                        batch_size=1,
                        sampling_params=dict(
                            linear=linear_val, conf=confidence_val, quad=quadratic_val
                        ),
                        callback=update_progress,
                        disable_torch_compile=(
                            True if "transformer" in model_choice else False
                        ),
                    )

                    wav_out = selected_model.autoencoder.decode(codes).cpu().detach()
                    if wav_out.dim() == 2 and wav_out.size(0) > 1:
                        wav_out = wav_out[0:1, :]

                    # 생성된 오디오를 세그먼트 리스트에 추가
                    audio_segments.append(wav_out.squeeze().numpy())
                except Exception as e:
                    print(f"오디오 생성 중 오류 발생: {str(e)}")
                    # 오류 발생 시 빈 오디오 세그먼트 추가 (1초 무음)
                    audio_segments.append(np.zeros(48000))

                # 진행 상황 업데이트
                steps_so_far += estimated_total_steps

        # 모든 오디오 세그먼트 합치기
        if audio_segments:
            try:
                # 음량 차이 조정 - 모든 청크를 가장 큰 음량으로 조정, 첫 청크에 +3dB 추가 부스트 적용
                normalized_segments = normalize_all_chunks_to_loudest(audio_segments, first_chunk_boost_db=3.0)
                # 조정된 청크들을 합치기
                final_audio = concatenate_audio(normalized_segments)
                # 빈 오디오인지 확인
                if len(final_audio) == 0 or np.all(final_audio == 0):
                    return (
                        (None, None),
                        current_seed,
                        "생성된 오디오가 없습니다. 다시 시도해주세요.",
                    )
                    
                # 오디오 처리 적용
                processed_audio = process_audio(final_audio, normalize_audio_toggle, volume_adjustment)
                
                return (
                    (48000, processed_audio),
                    current_seed,
                    "대화 음성이 성공적으로 생성되었습니다.",
                )
            except Exception as e:
                print(f"오디오 합치기 중 오류 발생: {str(e)}")
                return (
                    (None, None),
                    current_seed,
                    f"오디오 합치기 중 오류 발생: {str(e)}",
                )
        else:
            return (
                (None, None),
                current_seed,
                "생성된 오디오가 없습니다. 다시 시도해주세요."
                )

    except Exception as e:
        print(f"음성 생성 중 오류 발생: {str(e)}")
        import traceback

        traceback.print_exc()
        return (None, None), seed, f"음성 생성 중 오류 발생: {str(e)}"


def save_speakers(speaker_list):
    """화자 목록을 파일로 저장"""
    global SPEAKER_AUDIO_PATHS

    try:
        speaker_data = {}
        for speaker_name in SPEAKER_AUDIO_PATHS:
            speaker_data[speaker_name] = SPEAKER_AUDIO_PATHS[speaker_name]

        speakers_json_path = os.path.join(USER_DATA_DIR, "saved_speakers.json")
        with open(speakers_json_path, "w", encoding="utf-8") as f:
            json.dump(speaker_data, f, ensure_ascii=False, indent=2)

        return f"화자 목록이 {speakers_json_path} 파일로 저장되었습니다."
    except Exception as e:
        return f"화자 목록 저장 중 오류 발생: {str(e)}"


# 화자 목록 불러오기 함수 수정 - 양쪽 드롭다운 업데이트
def load_speakers(model_choice):
    """저장된 화자 목록을 불러옴"""
    global SPEAKER_EMBEDDINGS, SPEAKER_AUDIO_PATHS

    try:
        speakers_json_path = os.path.join(USER_DATA_DIR, "saved_speakers.json")
        if not os.path.exists(speakers_json_path):
            return gr.update(), gr.update(), "저장된 화자 목록을 찾을 수 없습니다."

        with open(speakers_json_path, "r", encoding="utf-8") as f:
            speaker_data = json.load(f)

        selected_model = load_model_if_needed(model_choice)

        # 기존 목록 초기화
        SPEAKER_EMBEDDINGS = {}
        SPEAKER_AUDIO_PATHS = {}

        # 화자 임베딩 재생성
        for speaker_name, audio_path in speaker_data.items():
            if os.path.exists(audio_path):
                wav, sr = torchaudio.load(audio_path)
                embedding = selected_model.make_speaker_embedding(wav, sr)
                embedding = embedding.to(device, dtype=torch.bfloat16)

                SPEAKER_EMBEDDINGS[speaker_name] = embedding
                SPEAKER_AUDIO_PATHS[speaker_name] = audio_path

        speaker_list = list(SPEAKER_EMBEDDINGS.keys())

        if speaker_list:
            dropdown_update = gr.update(choices=speaker_list, value=speaker_list[0])
            inline_dropdown_update = gr.update(
                choices=speaker_list, value=speaker_list[0]
            )
            return (
                dropdown_update,
                inline_dropdown_update,
                f"{len(speaker_list)}개의 화자를 불러왔습니다. (저장 위치: {USER_DATA_DIR})",
            )
        else:
            dropdown_update = gr.update(choices=[], value=None)
            inline_dropdown_update = gr.update(choices=[], value=None)
            return dropdown_update, inline_dropdown_update, "불러올 화자가 없습니다."
    except Exception as e:
        return gr.update(), gr.update(), f"화자 목록 불러오기 중 오류 발생: {str(e)}"


# 설정 추가 함수 - 간소화된 버전
def append_settings_to_dialogue(dialogue_text, settings_string):
    """대화 텍스트 끝에 설정 문자열 추가"""
    if dialogue_text.strip():
        # 마지막에 개행이 없으면 추가
        if not dialogue_text.endswith("\n"):
            dialogue_text += "\n"
        return dialogue_text + settings_string
    return settings_string


# 현재 설정값을 저장하는 함수
def save_settings_preset(
    preset_name,
    e1,
    e2,
    e3,
    e4,
    e5,
    e6,
    e7,
    e8,
    vq,
    fmax,
    pitch,
    rate,
    dnsmos,
    noised,
    cfg,
    seed,
    use_seed,
):
    """현재 설정값을 프리셋으로 저장"""
    if not preset_name.strip():
        return gr.update(), "프리셋 이름을 입력해주세요."

    preset_data = {
        "emotion1": float(e1),
        "emotion2": float(e2),
        "emotion3": float(e3),
        "emotion4": float(e4),
        "emotion5": float(e5),
        "emotion6": float(e6),
        "emotion7": float(e7),
        "emotion8": float(e8),
        "vq_single": float(vq),
        "fmax": float(fmax),
        "pitch_std": float(pitch),
        "speaking_rate": float(rate),
        "dnsmos_ovrl": float(dnsmos),
        "speaker_noised": bool(noised),
        "cfg_scale": float(cfg),
        "seed": int(seed),
        "use_seed": bool(use_seed),
    }

    # 파일명으로 사용할 수 있게 이름 정리
    safe_name = re.sub(r'[\\/*?:"<>|]', "_", preset_name)
    filename = os.path.join(PRESETS_DIR, f"{safe_name}.json")

    try:
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(preset_data, f, ensure_ascii=False, indent=2)

        # 프리셋 목록 업데이트
        preset_files = get_preset_list()

        return (
            gr.update(choices=preset_files, value=safe_name),
            f"'{preset_name}' 프리셋이 저장되었습니다. (저장 위치: {filename})",
        )
    except Exception as e:
        return gr.update(), f"프리셋 저장 중 오류 발생: {str(e)}"


# 저장된 프리셋 목록 가져오기
def get_preset_list():
    """저장된 프리셋 파일 목록 반환"""
    try:
        preset_files = [
            os.path.splitext(f)[0]
            for f in os.listdir(PRESETS_DIR)
            if f.endswith(".json")
        ]
        return sorted(preset_files)
    except Exception:
        return []


# 저장된 대화 목록 가져오기
def get_dialogue_list():
    """저장된 대화 파일 목록 반환"""
    try:
        dialogue_files = [
            os.path.splitext(f)[0]
            for f in os.listdir(DIALOGUES_DIR)
            if f.endswith(".txt")
        ]
        return sorted(dialogue_files)
    except Exception:
        return []


# 현재 불러온 프리셋 데이터 저장
CURRENT_LOADED_PRESET = None

# 프리셋 불러오기
def load_settings_preset(preset_name):
    """저장된 프리셋에서 설정값 불러오기 (메모리에만 저장, UI에 적용하지 않음)"""
    global CURRENT_LOADED_PRESET
    
    if not preset_name:
        return "프리셋을 선택해주세요."

    try:
        filename = os.path.join(PRESETS_DIR, f"{preset_name}.json")
        with open(filename, "r", encoding="utf-8") as f:
            preset_data = json.load(f)

        # 현재 불러온 프리셋 저장
        CURRENT_LOADED_PRESET = preset_data
        
        return f"'{preset_name}' 프리셋이 불러와졌습니다. '미리설정 적용하기' 버튼을 클릭하여 적용하세요."
    except Exception as e:
        return f"프리셋 불러오기 중 오류 발생: {str(e)}"


# 불러온 프리셋 적용
def apply_settings_preset():
    """불러온 프리셋을 UI에 적용하기"""
    global CURRENT_LOADED_PRESET
    
    if CURRENT_LOADED_PRESET is None:
        return [gr.update() for _ in range(17)] + ["불러온 프리셋이 없습니다. 먼저 프리셋을 불러오세요."]
    
    try:
        # 프리셋 데이터에서 각 설정값 추출
        e1 = CURRENT_LOADED_PRESET.get("emotion1", DEFAULT_SETTINGS["emotion1"])
        e2 = CURRENT_LOADED_PRESET.get("emotion2", DEFAULT_SETTINGS["emotion2"])
        e3 = CURRENT_LOADED_PRESET.get("emotion3", DEFAULT_SETTINGS["emotion3"])
        e4 = CURRENT_LOADED_PRESET.get("emotion4", DEFAULT_SETTINGS["emotion4"])
        e5 = CURRENT_LOADED_PRESET.get("emotion5", DEFAULT_SETTINGS["emotion5"])
        e6 = CURRENT_LOADED_PRESET.get("emotion6", DEFAULT_SETTINGS["emotion6"])
        e7 = CURRENT_LOADED_PRESET.get("emotion7", DEFAULT_SETTINGS["emotion7"])
        e8 = CURRENT_LOADED_PRESET.get("emotion8", DEFAULT_SETTINGS["emotion8"])
        vq = CURRENT_LOADED_PRESET.get("vq_single", DEFAULT_SETTINGS["vq_single"])
        fmax = CURRENT_LOADED_PRESET.get("fmax", DEFAULT_SETTINGS["fmax"])
        pitch = CURRENT_LOADED_PRESET.get("pitch_std", DEFAULT_SETTINGS["pitch_std"])
        rate = CURRENT_LOADED_PRESET.get("speaking_rate", DEFAULT_SETTINGS["speaking_rate"])
        dnsmos = CURRENT_LOADED_PRESET.get("dnsmos_ovrl", DEFAULT_SETTINGS["dnsmos_ovrl"])
        noised = CURRENT_LOADED_PRESET.get("speaker_noised", DEFAULT_SETTINGS["speaker_noised"])
        cfg = CURRENT_LOADED_PRESET.get("cfg_scale", DEFAULT_SETTINGS["cfg_scale"])
        seed = CURRENT_LOADED_PRESET.get("seed", DEFAULT_SETTINGS["seed"])
        use_seed = CURRENT_LOADED_PRESET.get("use_seed", False)

        return [
            gr.update(value=e1),
            gr.update(value=e2),
            gr.update(value=e3),
            gr.update(value=e4),
            gr.update(value=e5),
            gr.update(value=e6),
            gr.update(value=e7),
            gr.update(value=e8),
            gr.update(value=vq),
            gr.update(value=fmax),
            gr.update(value=pitch),
            gr.update(value=rate),
            gr.update(value=dnsmos),
            gr.update(value=noised),
            gr.update(value=cfg),
            gr.update(value=seed),
            gr.update(value=use_seed),
            "불러온 프리셋이 성공적으로 적용되었습니다.",
        ]
    except Exception as e:
        return [gr.update() for _ in range(17)] + [
            f"프리셋 적용 중 오류 발생: {str(e)}"
        ]


# 프리셋 삭제
def delete_settings_preset(preset_name):
    """저장된 프리셋 삭제"""
    if not preset_name:
        return gr.update(), "삭제할 프리셋을 선택해주세요."

    try:
        filename = os.path.join(PRESETS_DIR, f"{preset_name}.json")
        if os.path.exists(filename):
            os.remove(filename)

            # 프리셋 목록 업데이트
            preset_files = get_preset_list()
            if preset_files:
                return (
                    gr.update(choices=preset_files, value=preset_files[0]),
                    f"'{preset_name}' 프리셋이 삭제되었습니다.",
                )
            else:
                return (
                    gr.update(choices=[], value=None),
                    f"'{preset_name}' 프리셋이 삭제되었습니다.",
                )
        else:
            return gr.update(), f"'{preset_name}' 프리셋을 찾을 수 없습니다."
    except Exception as e:
        return gr.update(), f"프리셋 삭제 중 오류 발생: {str(e)}"


# 대화 저장 함수
def save_dialogue(dialogue_text, dialogue_name):
    """대화 텍스트를 파일로 저장"""
    if not dialogue_text.strip():
        return gr.update(), "저장할 대화 텍스트가 없습니다."

    if not dialogue_name.strip():
        # 이름이 지정되지 않은 경우 날짜와 시간을 이용해 자동 생성
        dialogue_name = f"대화_{time.strftime('%Y%m%d_%H%M%S')}"

    # 파일명으로 사용할 수 있게 이름 정리
    safe_name = re.sub(r'[\\/*?:"<>|]', "_", dialogue_name)
    filename = os.path.join(DIALOGUES_DIR, f"{safe_name}.txt")

    try:
        with open(filename, "w", encoding="utf-8") as f:
            f.write(dialogue_text)

        # 대화 목록 업데이트
        dialogue_files = get_dialogue_list()

        return (
            gr.update(choices=dialogue_files, value=safe_name),
            f"'{dialogue_name}' 대화가 저장되었습니다. (저장 위치: {filename})",
        )
    except Exception as e:
        return gr.update(), f"대화 저장 중 오류 발생: {str(e)}"


# 대화 불러오기
def load_dialogue(dialogue_name):
    """저장된 대화 텍스트 불러오기"""
    if not dialogue_name:
        return gr.update(), "불러올 대화를 선택해주세요."

    try:
        filename = os.path.join(DIALOGUES_DIR, f"{dialogue_name}.txt")
        with open(filename, "r", encoding="utf-8") as f:
            dialogue_text = f.read()

        return (
            gr.update(value=dialogue_text),
            f"'{dialogue_name}' 대화가 불러와졌습니다.",
        )
    except Exception as e:
        return gr.update(), f"대화 불러오기 중 오류 발생: {str(e)}"


# 대화 삭제
def delete_dialogue(dialogue_name):
    """저장된 대화 삭제"""
    if not dialogue_name:
        return gr.update(), "삭제할 대화를 선택해주세요."

    try:
        filename = os.path.join(DIALOGUES_DIR, f"{dialogue_name}.txt")
        if os.path.exists(filename):
            os.remove(filename)

            # 대화 목록 업데이트
            dialogue_files = get_dialogue_list()
            if dialogue_files:
                return (
                    gr.update(choices=dialogue_files, value=dialogue_files[0]),
                    f"'{dialogue_name}' 대화가 삭제되었습니다.",
                )
            else:
                return (
                    gr.update(choices=[], value=None),
                    f"'{dialogue_name}' 대화가 삭제되었습니다.",
                )
        else:
            return gr.update(), f"'{dialogue_name}' 대화를 찾을 수 없습니다."
    except Exception as e:
        return gr.update(), f"대화 삭제 중 오류 발생: {str(e)}"


def build_interface():
    supported_models = []
    if "transformer" in ZonosBackbone.supported_architectures:
        supported_models.append("Zyphra/Zonos-v0.1-transformer")

    if "hybrid" in ZonosBackbone.supported_architectures:
        supported_models.append("Zyphra/Zonos-v0.1-hybrid")
    else:
        print(
            "| The current ZonosBackbone does not support the hybrid architecture, meaning only the transformer model will be available in the model selector.\n"
            "| This probably means the mamba-ssm library has not been installed."
        )

    with gr.Blocks() as demo:
        gr.Markdown("# 멀티 화자 대화 음성 생성")

        with gr.Row():
            with gr.Column():
                model_choice = gr.Dropdown(
                    choices=supported_models,
                    value=supported_models[0],
                    label="Zonos 모델 타입",
                    info="사용할 모델 변형을 선택하세요.",
                )
                
                # 데이터 저장 경로 설정
                data_dir_input = gr.Textbox(
                    value=USER_DATA_DIR,
                    label="데이터 저장 경로",
                    info="화자 오디오, 프리셋, 대화 등이 저장될 경로를 설정하세요."
                )
                
                def update_data_directory(new_path):
                    global USER_DATA_DIR, PRESETS_DIR, DIALOGUES_DIR, SPEAKER_AUDIO_DIR
                    if not new_path or not new_path.strip():
                        return "경로가 비어있습니다. 기본 경로를 사용합니다."
                    
                    try:
                        # 경로 설정 업데이트
                        USER_DATA_DIR = os.path.abspath(new_path)
                        PRESETS_DIR = os.path.join(USER_DATA_DIR, "presets")
                        DIALOGUES_DIR = os.path.join(USER_DATA_DIR, "saved_dialogues")
                        SPEAKER_AUDIO_DIR = os.path.join(USER_DATA_DIR, "speaker_audio")
                        
                        # 디렉토리 생성
                        os.makedirs(PRESETS_DIR, exist_ok=True)
                        os.makedirs(DIALOGUES_DIR, exist_ok=True)
                        os.makedirs(SPEAKER_AUDIO_DIR, exist_ok=True)
                        
                        # 설정 파일에 경로 저장
                        config = {"data_dir": USER_DATA_DIR}
                        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
                            json.dump(config, f, ensure_ascii=False, indent=2)
                        
                        return f"데이터 저장 경로가 '{USER_DATA_DIR}'(으)로 설정되었습니다."
                    except Exception as e:
                        return f"경로 설정 중 오류 발생: {str(e)}"
                
                update_dir_button = gr.Button("저장 경로 설정")

        with gr.Row():
            with gr.Column(scale=3):
                gr.Markdown("## 대화 입력")
                gr.Markdown(
                    """각 대화는 다음 형식으로 입력하세요:
                
**기본 형식:** `[화자명]: 텍스트`

**화자별 설정 형식:** `[화자명|설정1:값1|설정2:값2]: 텍스트`

**화자별 자동처리 설정:** `[화자명|자동 감정|설정1:값1]: 텍스트`

**예시:**
```
[Alice]: 안녕하세요, 만나서 반갑습니다.
[Bob|happy:0.5|sad:0.3|rate:20.0|cfg:2.5]: 네, 저도 만나서 반갑습니다.
[Alice|자동 감정|pitch:60.0|seed:42]: 오늘 날씨가 정말 좋네요!
```

**지원 설정:**
- happy, sad, disgust, fear, surprise, anger, other, neutral: 감정 (0.0~1.0)
- vq: VQ 스코어 (0.5~0.8)
- fmax: 최대 주파수 (0~24000)
- pitch: 음높이 표준편차 (0.0~300.0)
- rate: 말하기 속도 (5.0~30.0)
- dnsmos: DNSMOS 전체 (1.0~5.0)
- noised: 화자 노이즈 제거 (true/false)
- cfg: CFG 스케일 (1.0~5.0)
- seed: 시드 (정수)

**자동처리 설정 (무조건 설정):**
- 자동 감정: 기본 감정 설정 무시하고 자동 생성
- 자동 vq: VQ 스코어 자동 생성
- 자동 주파수: 최대 주파수 자동 생성
- 자동 음높이: 음높이 표준편차 자동 생성
- 자동 속도: 말하기 속도 자동 생성
- 자동 dnsmos: DNSMOS 자동 생성
- 자동 노이즈: 노이즈 제거 자동 생성
- 자동 화자: 화자 설정 자동 생성
                """
                )

                # 대화 텍스트
                dialogue_text = gr.Textbox(
                    label="대화 스크립트",
                    value="[Alice]: 안녕하세요, 만나서 반갑습니다.\n[Bob|happy:0.5|sad:0.3|rate:20.0|cfg:2.5]: 네, 저도 만나서 반갑습니다.\n[Alice|자동 감정|pitch:60.0]: 오늘 날씨가 정말 좋네요!",
                    lines=10,
                )

                # 설정 추가 버튼을 대화 입력 옆으로 이동
                with gr.Row():
                    speaker_dropdown_inline = gr.Dropdown(
                        label="화자",
                        choices=[],
                        info="설정을 추가할 화자 선택",
                    )
                    insert_settings_button = gr.Button(
                        "선택한 화자와 설정 추가", variant="primary"
                    )

                # 대화 저장/불러오기 섹션
                with gr.Row():
                    with gr.Column(scale=3):
                        dialogue_name_input = gr.Textbox(
                            label="대화 이름", placeholder="저장할 대화 이름 입력"
                        )
                    with gr.Column(scale=1):
                        save_dialogue_button = gr.Button("대화 저장")

                with gr.Row():
                    with gr.Column(scale=3):
                        dialogue_dropdown = gr.Dropdown(
                            label="저장된 대화",
                            choices=get_dialogue_list(),
                            info="불러올 대화 선택",
                        )
                    with gr.Column(scale=1):
                        load_dialogue_button = gr.Button("대화 불러오기")
                        delete_dialogue_button = gr.Button("대화 삭제", variant="stop")

                language = gr.Dropdown(
                    choices=supported_language_codes,
                    value=(
                        "ko"
                        if "ko" in supported_language_codes
                        else supported_language_codes[0]
                    ),
                    label="언어 코드",
                    info="언어 코드를 선택하세요.",
                    allow_custom_value=True,
                )

            with gr.Column(scale=2):
                gr.Markdown("## 화자 관리")
                with gr.Row():
                    speaker_name_input = gr.Textbox(
                        label="화자 이름", placeholder="예: Alice"
                    )
                    speaker_audio_input = gr.Audio(
                        label="화자 오디오 (참조용)", type="filepath"
                    )

                with gr.Row():
                    add_speaker_button = gr.Button("화자 추가")
                    remove_speaker_button = gr.Button("화자 삭제")
                    save_speakers_button = gr.Button("화자 목록 저장")
                    load_speakers_button = gr.Button("화자 목록 불러오기")

                speaker_dropdown = gr.Dropdown(
                    label="화자 목록", choices=[], info="현재 등록된 화자 목록"
                )

                speaker_message = gr.Textbox(label="상태 메시지", interactive=False)
                
                # 데이터 저장 경로 설정 버튼 이벤트 연결
                update_dir_button.click(fn=update_data_directory, inputs=[data_dir_input], outputs=[speaker_message])

                prefix_audio = gr.Audio(
                    value="assets/silence_100ms.wav",
                    label="선택적 프리픽스 오디오 (이 오디오에 이어서 생성)",
                    type="filepath",
                )

        with gr.Row():
            with gr.Column():
                gr.Markdown(
                    "## 기본 조건부 매개변수 (화자별 설정으로 재정의될 수 있음)"
                )
                dnsmos_slider = gr.Slider(
                    1.0, 5.0, value=4.0, step=0.1, label="DNSMOS 전체"
                )
                fmax_slider = gr.Slider(
                    0, 24000, value=24000, step=1, label="최대 주파수 (Hz)"
                )
                vq_single_slider = gr.Slider(0.5, 0.8, 0.78, 0.01, label="VQ 스코어")
                pitch_std_slider = gr.Slider(
                    0.0, 300.0, value=45.0, step=1, label="음높이 표준편차"
                )
                speaking_rate_slider = gr.Slider(
                    5.0, 30.0, value=15.0, step=0.5, label="말하기 속도"
                )
                speaker_noised_checkbox = gr.Checkbox(
                    label="화자 노이즈 제거?", value=False
                )

            with gr.Column():
                gr.Markdown("## 생성 매개변수")
                cfg_scale_slider = gr.Slider(1.0, 5.0, 2.0, 0.1, label="CFG 스케일")
                seed_number = gr.Number(label="시드", value=420, precision=0)
                randomize_seed_toggle = gr.Checkbox(
                    label="시드 무작위화 (생성 전)", value=True
                )
                use_seed_in_settings = gr.Checkbox(
                    label="설정에 시드 포함", value=False
                )

                # 설정 프리셋 저장/불러오기 섹션
                gr.Markdown("## 설정 프리셋")
                with gr.Row():
                    with gr.Column(scale=3):
                        preset_name_input = gr.Textbox(
                            label="프리셋 이름", placeholder="저장할 프리셋 이름 입력"
                        )
                    with gr.Column(scale=1):
                        save_preset_button = gr.Button("현재 설정 저장")

                with gr.Row():
                    with gr.Column(scale=3):
                        preset_dropdown = gr.Dropdown(
                            label="저장된 프리셋",
                            choices=get_preset_list(),
                            info="불러올 프리셋 선택",
                        )
                    with gr.Column(scale=1):
                        load_preset_button = gr.Button("프리셋 불러오기")
                        apply_preset_button = gr.Button("미리설정 적용하기", variant="primary")
                        delete_preset_button = gr.Button("프리셋 삭제", variant="stop")

        with gr.Accordion("기본 감정 제어", open=False):
            gr.Markdown(
                "### 감정 슬라이더 (화자별 설정으로 재정의될 수 있음)\n"
                "주의: 이 슬라이더들은 직관적으로 작동하지 않을 수 있으며 원하는 효과를 얻기 위해 시행착오가 필요할 수 있습니다.\n"
                "특정 설정은 모델을 불안정하게 만들 수 있습니다. 감정을 무조건 설정으로 하면 도움이 될 수 있습니다."
            )
            with gr.Row():
                emotion1 = gr.Slider(0.0, 1.0, 1.0, 0.05, label="행복")
                emotion2 = gr.Slider(0.0, 1.0, 0.05, 0.05, label="슬픔")
                emotion3 = gr.Slider(0.0, 1.0, 0.05, 0.05, label="혐오")
                emotion4 = gr.Slider(0.0, 1.0, 0.05, 0.05, label="두려움")
            with gr.Row():
                emotion5 = gr.Slider(0.0, 1.0, 0.05, 0.05, label="놀람")
                emotion6 = gr.Slider(0.0, 1.0, 0.05, 0.05, label="분노")
                emotion7 = gr.Slider(0.0, 1.0, 0.1, 0.05, label="기타")
                emotion8 = gr.Slider(0.0, 1.0, 0.2, 0.05, label="중립")

        with gr.Accordion("샘플링", open=False):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### NovelAi의 통합 샘플러")
                    linear_slider = gr.Slider(
                        -2.0,
                        2.0,
                        0.5,
                        0.01,
                        label="선형 (0으로 설정하면 통합 샘플링 비활성화)",
                        info="높은 값은 출력을 덜 무작위로 만듭니다.",
                    )
                    confidence_slider = gr.Slider(
                        -2.0,
                        2.0,
                        0.40,
                        0.01,
                        label="신뢰도",
                        info="낮은 값은 무작위 출력을 더 무작위로 만듭니다.",
                    )
                    quadratic_slider = gr.Slider(
                        -2.0,
                        2.0,
                        0.00,
                        0.01,
                        label="이차",
                        info="높은 값은 낮은 확률을 더 낮게 만듭니다.",
                    )

        with gr.Accordion("고급 매개변수", open=False):
            gr.Markdown(
                "### 무조건 토글\n"
                "체크박스를 선택하면 모델이 해당 조건 값을 무시하고 무조건으로 만듭니다.\n"
                '실제로는 주어진 조건 기능이 제약되지 않고 "자동으로 채워진다"는 의미입니다.'
            )
            with gr.Row():
                unconditional_keys = gr.CheckboxGroup(
                    [
                        "speaker",
                        "emotion",
                        "vqscore_8",
                        "fmax",
                        "pitch_std",
                        "speaking_rate",
                        "dnsmos_ovrl",
                        "speaker_noised",
                    ],
                    value=["emotion"],
                    label="무조건 키",
                )

        with gr.Column():
            gr.Markdown("## 오디오 처리 설정")
            with gr.Row():
                normalize_audio_toggle = gr.Checkbox(
                    label="오디오 정규화 적용", 
                    value=True,
                    info="전체 음성에 정규화를 적용하여 일관된 음량을 유지합니다."
                )
                volume_adjustment = gr.Slider(
                    minimum=-20, 
                    maximum=20, 
                    value=0, 
                    step=0.5, 
                    label="볼륨 조정 (dB)",
                    info="오디오 볼륨을 조정합니다. 양수 값은 볼륨을 키우고, 음수 값은 볼륨을 줄입니다."
                )
            generate_button = gr.Button("오디오 생성", variant="primary")
            output_audio = gr.Audio(label="생성된 오디오", type="numpy", autoplay=True)
            output_message = gr.Textbox(label="출력 메시지", interactive=False)

        # 화자 관리 이벤트 연결
        add_speaker_button.click(
            fn=add_speaker,
            inputs=[
                model_choice,
                speaker_name_input,
                speaker_audio_input,
                speaker_dropdown,
            ],
            outputs=[speaker_dropdown, speaker_message],
        )

        # 화자 관리 이벤트에서 반환된 목록을 인라인 드롭다운에도 적용
        def update_speaker_dropdown_inline(dropdown_update, message):
            global SPEAKER_EMBEDDINGS
            speaker_list = list(SPEAKER_EMBEDDINGS.keys())
            if speaker_list:
                return gr.update(choices=speaker_list, value=speaker_list[0]), message
            else:
                return gr.update(choices=[], value=None), message

        add_speaker_button.click(
            fn=lambda: (
                gr.update(
                    choices=list(SPEAKER_EMBEDDINGS.keys()),
                    value=(
                        list(SPEAKER_EMBEDDINGS.keys())[0]
                        if SPEAKER_EMBEDDINGS
                        else None
                    ),
                ),
                "화자 목록이 업데이트되었습니다.",
            ),
            inputs=[],
            outputs=[speaker_dropdown_inline, speaker_message],
        )

        remove_speaker_button.click(
            fn=remove_speaker,
            inputs=[speaker_dropdown, speaker_dropdown],
            outputs=[speaker_dropdown, speaker_message],
        )

        # 화자가 삭제되면 인라인 드롭다운도 업데이트
        remove_speaker_button.click(
            fn=lambda: (
                gr.update(
                    choices=list(SPEAKER_EMBEDDINGS.keys()),
                    value=(
                        list(SPEAKER_EMBEDDINGS.keys())[0]
                        if SPEAKER_EMBEDDINGS
                        else None
                    ),
                ),
                "화자 목록이 업데이트되었습니다.",
            ),
            inputs=[],
            outputs=[speaker_dropdown_inline, speaker_message],
        )

        save_speakers_button.click(
            fn=save_speakers, inputs=[speaker_dropdown], outputs=[speaker_message]
        )

        load_speakers_button.click(
            fn=load_speakers,
            inputs=[model_choice],
            outputs=[speaker_dropdown, speaker_dropdown_inline, speaker_message],
        )

        # 화자가 로드되면 인라인 드롭다운도 업데이트
        load_speakers_button.click(
            fn=lambda: (
                gr.update(
                    choices=list(SPEAKER_EMBEDDINGS.keys()),
                    value=(
                        list(SPEAKER_EMBEDDINGS.keys())[0]
                        if SPEAKER_EMBEDDINGS
                        else None
                    ),
                ),
                "화자 목록이 업데이트되었습니다.",
            ),
            inputs=[],
            outputs=[speaker_dropdown_inline, speaker_message],
        )

        # 설정 추가 이벤트 연결
        def handle_append_settings(
            speaker,
            text,
            e1,
            e2,
            e3,
            e4,
            e5,
            e6,
            e7,
            e8,
            vq,
            fmax,
            pitch,
            rate,
            dnsmos,
            noised,
            cfg,
            seed,
            use_seed,
        ):
            if not speaker:
                return text, "화자를 선택해주세요."

            settings_string = generate_settings_string(
                speaker,
                e1,
                e2,
                e3,
                e4,
                e5,
                e6,
                e7,
                e8,
                vq,
                fmax,
                pitch,
                rate,
                dnsmos,
                noised,
                cfg,
                seed,
                use_seed,
            )

            new_text = append_settings_to_dialogue(text, settings_string)
            return new_text, f"'{speaker}' 화자의 설정이 추가되었습니다."

        insert_settings_button.click(
            fn=handle_append_settings,
            inputs=[
                speaker_dropdown_inline,  # 인라인 화자 드롭다운 사용
                dialogue_text,
                emotion1,
                emotion2,
                emotion3,
                emotion4,
                emotion5,
                emotion6,
                emotion7,
                emotion8,
                vq_single_slider,
                fmax_slider,
                pitch_std_slider,
                speaking_rate_slider,
                dnsmos_slider,
                speaker_noised_checkbox,
                cfg_scale_slider,
                seed_number,
                use_seed_in_settings,
            ],
            outputs=[dialogue_text, speaker_message],
        )

        # 설정 프리셋 저장 이벤트 연결
        save_preset_button.click(
            fn=save_settings_preset,
            inputs=[
                preset_name_input,
                emotion1,
                emotion2,
                emotion3,
                emotion4,
                emotion5,
                emotion6,
                emotion7,
                emotion8,
                vq_single_slider,
                fmax_slider,
                pitch_std_slider,
                speaking_rate_slider,
                dnsmos_slider,
                speaker_noised_checkbox,
                cfg_scale_slider,
                seed_number,
                use_seed_in_settings,
            ],
            outputs=[preset_dropdown, speaker_message],
        )

        # 설정 프리셋 불러오기 이벤트 연결
        load_preset_button.click(
            fn=load_settings_preset,
            inputs=[preset_dropdown],
            outputs=[speaker_message],
        )
        
        # 불러온 프리셋 적용 이벤트 연결
        apply_preset_button.click(
            fn=apply_settings_preset,
            inputs=[],
            outputs=[
                emotion1,
                emotion2,
                emotion3,
                emotion4,
                emotion5,
                emotion6,
                emotion7,
                emotion8,
                vq_single_slider,
                fmax_slider,
                pitch_std_slider,
                speaking_rate_slider,
                dnsmos_slider,
                speaker_noised_checkbox,
                cfg_scale_slider,
                seed_number,
                use_seed_in_settings,
                speaker_message,
            ],
        )

        # 설정 프리셋 삭제 이벤트 연결
        delete_preset_button.click(
            fn=delete_settings_preset,
            inputs=[preset_dropdown],
            outputs=[preset_dropdown, speaker_message],
        )

        # 대화 저장 이벤트 연결
        save_dialogue_button.click(
            fn=save_dialogue,
            inputs=[dialogue_text, dialogue_name_input],
            outputs=[dialogue_dropdown, speaker_message],
        )

        # 대화 불러오기 이벤트 연결
        load_dialogue_button.click(
            fn=load_dialogue,
            inputs=[dialogue_dropdown],
            outputs=[dialogue_text, speaker_message],
        )

        # 대화 삭제 이벤트 연결
        delete_dialogue_button.click(
            fn=delete_dialogue,
            inputs=[dialogue_dropdown],
            outputs=[dialogue_dropdown, speaker_message],
        )

        # 오디오 생성 이벤트 연결
        generate_button.click(
            fn=generate_multi_speaker_audio,
            inputs=[
                model_choice,
                dialogue_text,
                language,
                prefix_audio,
                emotion1,
                emotion2,
                emotion3,
                emotion4,
                emotion5,
                emotion6,
                emotion7,
                emotion8,
                vq_single_slider,
                fmax_slider,
                pitch_std_slider,
                speaking_rate_slider,
                dnsmos_slider,
                speaker_noised_checkbox,
                cfg_scale_slider,
                linear_slider,
                confidence_slider,
                quadratic_slider,
                seed_number,
                randomize_seed_toggle,
                unconditional_keys,
                volume_adjustment,
                normalize_audio_toggle,
            ],
            outputs=[output_audio, seed_number, output_message],
        )

        # 모델 변경 시 UI 업데이트
        model_choice.change(
            fn=update_ui,
            inputs=[model_choice],
            outputs=[
                dialogue_text,
                language,
                speaker_dropdown,
                prefix_audio,
                emotion1,
                emotion2,
                emotion3,
                emotion4,
                emotion5,
                emotion6,
                emotion7,
                emotion8,
                vq_single_slider,
                fmax_slider,
                pitch_std_slider,
                speaking_rate_slider,
                dnsmos_slider,
                speaker_noised_checkbox,
                unconditional_keys,
            ],
        )
        
        # 시작 시 안내 메시지
        demo.load(lambda: f"Zonos TTS 시스템이 시작되었습니다. 데이터 저장 경로: {USER_DATA_DIR}", None, speaker_message)

    return demo


if __name__ == "__main__":
    demo = build_interface()
    share = getenv("GRADIO_SHARE", "False").lower() in ("true", "1", "t")
    host = getenv("GRADIO_HOST", "0.0.0.0")
    demo.launch(server_name=host, inbrowser=True, share=share)
