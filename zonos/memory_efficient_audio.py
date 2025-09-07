"""
메모리 효율적 오디오 처리 시스템

이 모듈은 임시파일 기반으로 메모리 사용량을 최소화하면서
고품질 오디오 생성 및 후처리를 수행하는 시스템을 제공합니다.
"""

import os
import gc
import json
import uuid
import logging
import psutil
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Generator, Callable, Union

import numpy as np
import torch
import soundfile as sf
from scipy import signal
from scipy.stats import zscore

logger = logging.getLogger(__name__)

# ==================== 메모리 모니터링 유틸리티 ====================

def check_memory_usage(threshold_percent: float = 80.0) -> Dict[str, Any]:
    """메모리 사용량 체크 및 경고"""
    
    memory = psutil.virtual_memory()
    gpu_memory = {}
    
    # GPU 메모리 체크 (가능한 경우)
    if torch.cuda.is_available():
        gpu_memory = {
            'allocated_mb': torch.cuda.memory_allocated() / (1024**2),
            'reserved_mb': torch.cuda.memory_reserved() / (1024**2),
            'max_allocated_mb': torch.cuda.max_memory_allocated() / (1024**2)
        }
    
    memory_info = {
        'ram_percent': memory.percent,
        'ram_available_gb': memory.available / (1024**3),
        'ram_used_gb': memory.used / (1024**3),
        'gpu_memory': gpu_memory,
        'warning': memory.percent > threshold_percent
    }
    
    if memory_info['warning']:
        logger.warning(f"🚨 높은 메모리 사용량 감지: {memory.percent:.1f}% (임계값: {threshold_percent}%)")
        if gpu_memory:
            logger.warning(f"GPU 메모리: {gpu_memory['allocated_mb']:.1f}MB 사용 중")
    
    return memory_info

def force_memory_cleanup():
    """강제 메모리 정리"""
    logger.info("🧹 강제 메모리 정리 실행 중...")
    
    # Python GC
    collected = gc.collect()
    
    # GPU 메모리 정리
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()  # 모든 CUDA 연산 완료 대기
    
    logger.info(f"메모리 정리 완료: {collected}개 객체 해제")

# ==================== 데이터 클래스 ====================

@dataclass
class SegmentMetadata:
    """세그먼트 메타데이터"""
    sequence_id: int
    speaker_name: str
    text: str
    settings_used: Dict[str, Any]
    generation_timestamp: datetime
    
    def to_dict(self) -> Dict:
        return {
            'sequence_id': self.sequence_id,
            'speaker_name': self.speaker_name,
            'text': self.text,
            'settings_used': self.settings_used,
            'generation_timestamp': self.generation_timestamp.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'SegmentMetadata':
        return cls(
            sequence_id=data['sequence_id'],
            speaker_name=data['speaker_name'], 
            text=data['text'],
            settings_used=data['settings_used'],
            generation_timestamp=datetime.fromisoformat(data['generation_timestamp'])
        )

@dataclass
class SegmentFile:
    """임시파일로 저장된 세그먼트 정보"""
    audio_path: Path
    metadata_path: Path
    sequence_id: int
    speaker_name: str
    duration: float
    file_size: int
    created_at: datetime = field(default_factory=datetime.now)

@dataclass  
class AudioSegment:
    """메모리상의 오디오 세그먼트 (임시 로드용)"""
    audio_data: np.ndarray
    sample_rate: int
    metadata: SegmentMetadata

@dataclass
class VolumeMetrics:
    """볼륨 분석 메트릭"""
    rms_db: float
    peak_db: float
    lufs: Optional[float] = None
    dynamic_range: Optional[float] = None
    silence_ratio: Optional[float] = None

@dataclass
class GlobalVolumeAnalysis:
    """전체 세그먼트 볼륨 분석 결과"""
    segment_volumes: List[Dict]
    global_stats: Dict[str, float]

@dataclass
class CompositionResult:
    """통합 결과"""
    output_path: str
    stats: Dict[str, Any]

@dataclass
class DialogueSegment:
    """파싱된 대화 세그먼트"""
    sequence_id: int
    speaker_name: str
    text: str
    settings: Dict[str, Any]

# ==================== 임시파일 관리 시스템 ====================

class SegmentFileManager:
    """세그먼트 임시파일 관리 시스템"""
    
    def __init__(self, temp_dir: str = "temp_audio_segments"):
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(exist_ok=True)
        self.session_id = str(uuid.uuid4())[:8]
        self.segment_files: List[SegmentFile] = []
        
        logger.info(f"SegmentFileManager initialized with session {self.session_id}")
    
    def save_segment_to_file(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        metadata: SegmentMetadata
    ) -> SegmentFile:
        """세그먼트를 임시파일로 저장하고 메모리 해제"""
        
        file_id = f"{self.session_id}_seg_{metadata.sequence_id:03d}"
        audio_path = self.temp_dir / f"{file_id}.wav"
        metadata_path = self.temp_dir / f"{file_id}_meta.json"
        
        try:
            # 오디오 파일 저장
            sf.write(str(audio_path), audio_data, sample_rate)
            
            # 메타데이터 JSON 저장
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata.to_dict(), f, ensure_ascii=False, indent=2)
            
            segment_file = SegmentFile(
                audio_path=audio_path,
                metadata_path=metadata_path,
                sequence_id=metadata.sequence_id,
                speaker_name=metadata.speaker_name,
                duration=len(audio_data) / sample_rate,
                file_size=audio_path.stat().st_size
            )
            
            self.segment_files.append(segment_file)
            
            # 메모리 해제 힌트
            del audio_data
            gc.collect()
            
            logger.debug(f"Saved segment {metadata.sequence_id} to {audio_path}")
            return segment_file
            
        except Exception as e:
            logger.error(f"Failed to save segment {metadata.sequence_id}: {e}")
            raise
    
    def load_segment_from_file(self, segment_file: SegmentFile) -> AudioSegment:
        """임시파일에서 세그먼트 로드"""
        
        try:
            # 오디오 로드
            audio_data, sample_rate = sf.read(str(segment_file.audio_path))
            
            # 메타데이터 로드
            with open(segment_file.metadata_path, 'r', encoding='utf-8') as f:
                metadata_dict = json.load(f)
            
            metadata = SegmentMetadata.from_dict(metadata_dict)
            
            return AudioSegment(
                audio_data=audio_data,
                sample_rate=sample_rate,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Failed to load segment from {segment_file.audio_path}: {e}")
            raise
    
    def load_segments_batch(
        self, 
        segment_files: List[SegmentFile],
        max_memory_mb: int = 512
    ) -> Generator[List[AudioSegment], None, None]:
        """메모리 제한 내에서 배치 단위로 세그먼트 로드"""
        
        current_batch = []
        current_memory_mb = 0
        
        for seg_file in segment_files:
            # 압축 해제 후 예상 메모리 사용량 (WAV는 대략 2배)
            estimated_mb = seg_file.file_size / (1024 * 1024) * 2
            
            if current_memory_mb + estimated_mb > max_memory_mb and current_batch:
                # 현재 배치 반환
                batch_segments = []
                for sf_item in current_batch:
                    try:
                        segment = self.load_segment_from_file(sf_item)
                        batch_segments.append(segment)
                    except Exception as e:
                        logger.warning(f"Failed to load segment {sf_item.sequence_id}: {e}")
                        continue
                
                yield batch_segments
                
                # 메모리 해제
                for seg in batch_segments:
                    del seg.audio_data
                del batch_segments
                gc.collect()
                
                current_batch = []
                current_memory_mb = 0
            
            current_batch.append(seg_file)
            current_memory_mb += estimated_mb
        
        # 마지막 배치 처리
        if current_batch:
            batch_segments = []
            for sf_item in current_batch:
                try:
                    segment = self.load_segment_from_file(sf_item)
                    batch_segments.append(segment)
                except Exception as e:
                    logger.warning(f"Failed to load segment {sf_item.sequence_id}: {e}")
                    continue
            
            if batch_segments:
                yield batch_segments
    
    def cleanup_temp_files(self):
        """세션 임시파일 정리"""
        
        cleaned_count = 0
        for seg_file in self.segment_files:
            try:
                if seg_file.audio_path.exists():
                    seg_file.audio_path.unlink()
                    cleaned_count += 1
                if seg_file.metadata_path.exists():
                    seg_file.metadata_path.unlink()
                    cleaned_count += 1
            except Exception as e:
                logger.warning(f"임시파일 삭제 실패 {seg_file.audio_path}: {e}")
        
        # 정규화된 파일들도 정리
        try:
            for temp_file in self.temp_dir.glob(f"{self.session_id}_*.wav"):
                temp_file.unlink()
                cleaned_count += 1
            for temp_file in self.temp_dir.glob(f"{self.session_id}_*.json"):
                temp_file.unlink()
                cleaned_count += 1
        except Exception as e:
            logger.warning(f"추가 임시파일 정리 실패: {e}")
        
        self.segment_files.clear()
        logger.info(f"Cleaned up {cleaned_count} temporary files for session {self.session_id}")
    
    def get_total_duration(self) -> float:
        """전체 세그먼트 길이 계산"""
        return sum(seg_file.duration for seg_file in self.segment_files)
    
    def get_speaker_segments(self, speaker_name: str) -> List[SegmentFile]:
        """특정 화자의 세그먼트 파일들만 반환"""
        return [
            seg_file for seg_file in self.segment_files 
            if seg_file.speaker_name == speaker_name
        ]
    
    def get_session_info(self) -> Dict[str, Any]:
        """세션 정보 반환"""
        total_size = sum(seg.file_size for seg in self.segment_files)
        speakers = set(seg.speaker_name for seg in self.segment_files)
        
        return {
            'session_id': self.session_id,
            'segment_count': len(self.segment_files),
            'total_duration': self.get_total_duration(),
            'total_file_size': total_size,
            'unique_speakers': list(speakers),
            'temp_dir': str(self.temp_dir)
        }

# ==================== 볼륨 분석 유틸리티 ====================

class VolumeAnalyzer:
    """오디오 볼륨 분석 유틸리티"""
    
    @staticmethod
    def analyze_segment_volume(audio_data: np.ndarray) -> VolumeMetrics:
        """단일 세그먼트의 볼륨 분석"""
        
        if len(audio_data) == 0:
            return VolumeMetrics(rms_db=-100.0, peak_db=-100.0)
        
        # RMS 계산
        rms = np.sqrt(np.mean(audio_data ** 2))
        rms_db = 20 * np.log10(rms + 1e-10)
        
        # Peak 계산
        peak = np.max(np.abs(audio_data))
        peak_db = 20 * np.log10(peak + 1e-10)
        
        # 무음 비율 계산 (간단한 임계값 기반)
        silence_threshold = 0.01
        silent_samples = np.sum(np.abs(audio_data) < silence_threshold)
        silence_ratio = silent_samples / len(audio_data)
        
        # 다이나믹 레인지 (간단한 계산)
        if len(audio_data) > 1000:
            # 상위 1%와 하위 1% 값으로 다이나믹 레인지 추정
            sorted_abs = np.sort(np.abs(audio_data))
            p99 = sorted_abs[int(0.99 * len(sorted_abs))]
            p1 = sorted_abs[int(0.01 * len(sorted_abs))]
            dynamic_range = 20 * np.log10((p99 + 1e-10) / (p1 + 1e-10))
        else:
            dynamic_range = peak_db - rms_db
        
        return VolumeMetrics(
            rms_db=rms_db,
            peak_db=peak_db,
            dynamic_range=dynamic_range,
            silence_ratio=silence_ratio
        )
    
    @staticmethod
    def calculate_global_stats(volume_data: List[Dict]) -> Dict[str, float]:
        """전체 볼륨 통계 계산"""
        
        if not volume_data:
            return {}
        
        rms_values = [v['rms_db'] for v in volume_data if v['rms_db'] > -90]
        peak_values = [v['peak_db'] for v in volume_data if v['peak_db'] > -90]
        
        return {
            'mean_rms_db': float(np.mean(rms_values)) if rms_values else -60.0,
            'std_rms_db': float(np.std(rms_values)) if rms_values else 5.0,
            'mean_peak_db': float(np.mean(peak_values)) if peak_values else -20.0,
            'std_peak_db': float(np.std(peak_values)) if peak_values else 5.0,
            'max_peak_db': float(np.max(peak_values)) if peak_values else -20.0,
            'min_rms_db': float(np.min(rms_values)) if rms_values else -60.0
        }

# ==================== 세그먼트 생성 시스템 ====================

class MemoryEfficientSegmentGenerator:
    """메모리 효율적 세그먼트 생성기"""
    
    def __init__(
        self, 
        model, 
        speaker_embeddings: Dict[str, torch.Tensor],
        file_manager: SegmentFileManager
    ):
        self.model = model
        self.speaker_embeddings = speaker_embeddings
        self.file_manager = file_manager
        
    def generate_segments_to_files(
        self, 
        segments: List[DialogueSegment],
        progress_callback: Optional[Callable] = None
    ) -> List[SegmentFile]:
        """세그먼트들을 생성하여 직접 파일로 저장"""
        
        segment_files = []
        
        for i, dialogue_seg in enumerate(segments):
            try:
                # 🔍 메모리 모니터링 (매 10개 세그먼트마다)
                if i % 10 == 0:
                    memory_info = check_memory_usage(threshold_percent=75.0)
                    if memory_info['warning']:
                        logger.warning(f"세그먼트 {i}: 메모리 사용량 높음, 강제 정리 실행")
                        force_memory_cleanup()
                
                if progress_callback:
                    progress_callback(
                        i / len(segments), 
                        f"세그먼트 {i+1}/{len(segments)} 생성 중... ({dialogue_seg.speaker_name})"
                    )
                
                # 1. 화자 임베딩 가져오기
                if dialogue_seg.speaker_name not in self.speaker_embeddings:
                    logger.warning(f"Unknown speaker: {dialogue_seg.speaker_name}, skipping")
                    continue
                    
                speaker_embedding = self.speaker_embeddings[dialogue_seg.speaker_name]
                
                # 2. 음성 생성 (Zonos 모델 사용)
                audio_data = self._generate_audio_for_segment(
                    dialogue_seg, speaker_embedding
                )
                
                if audio_data is None or len(audio_data) == 0:
                    logger.warning(f"Empty audio generated for segment {i}")
                    continue
                
                # 3. 메타데이터 준비
                metadata = SegmentMetadata(
                    sequence_id=dialogue_seg.sequence_id,
                    speaker_name=dialogue_seg.speaker_name,
                    text=dialogue_seg.text,
                    settings_used=dialogue_seg.settings,
                    generation_timestamp=datetime.now()
                )
                
                # 4. 즉시 파일로 저장 (메모리 해제)
                segment_file = self.file_manager.save_segment_to_file(
                    audio_data=audio_data,
                    sample_rate=48000,  # Zonos 기본 샘플레이트
                    metadata=metadata
                )
                
                segment_files.append(segment_file)
                
                # 5. 명시적 메모리 해제 (audio_data는 이미 파일 저장 시 해제됨)
                # speaker_embedding은 다른 곳에서 관리되므로 여기서 해제하지 않음
                
                # 6. GPU 메모리 정리 (필요시 - 주기적 정리)
                if i % 5 == 0 and torch.cuda.is_available():  # 5개 세그먼트마다 정리
                    torch.cuda.empty_cache()
                    gc.collect()
                
                logger.debug(f"Generated and saved segment {i}: {dialogue_seg.speaker_name}")
                
            except Exception as e:
                logger.error(f"Failed to generate segment {i}: {e}")
                continue
        
        if progress_callback:
            progress_callback(1.0, f"모든 세그먼트 생성 완료! ({len(segment_files)}개)")
        
        # 🎯 최종 메모리 상태 체크
        final_memory = check_memory_usage()
        logger.info(f"생성 완료: {len(segment_files)}/{len(segments)} 세그먼트")
        logger.info(f"최종 메모리 사용량: RAM {final_memory['ram_percent']:.1f}%")
        if final_memory['gpu_memory']:
            logger.info(f"최종 GPU 메모리: {final_memory['gpu_memory']['allocated_mb']:.1f}MB")
        
        return segment_files
    
    def _generate_audio_for_segment(
        self, 
        segment: DialogueSegment, 
        speaker_embedding: torch.Tensor
    ) -> Optional[np.ndarray]:
        """단일 세그먼트의 음성 생성"""
        
        try:
            # Zonos 모델을 사용한 음성 생성 (기존 gradio_interface.py 로직 기반)
            from zonos.conditioning import make_cond_dict
            from zonos.utils import DEFAULT_DEVICE as device
            
            # 감정 벡터 준비 (기본값 또는 설정값 사용)
            emotion_values = [
                segment.settings.get("emotion1", 0.0),
                segment.settings.get("emotion2", 0.0), 
                segment.settings.get("emotion3", 0.0),
                segment.settings.get("emotion4", 0.0),
                segment.settings.get("emotion5", 0.0),
                segment.settings.get("emotion6", 0.0),
                segment.settings.get("emotion7", 0.0),
                segment.settings.get("emotion8", 0.0)
            ]
            emotion_tensor = torch.tensor(emotion_values, device=device)
            
            # VQ 점수 준비
            vq_single = segment.settings.get("vq_single", 0.0)
            vq_tensor = torch.tensor([vq_single] * 8, device=device).unsqueeze(0)
            
            # 조건부 딕셔너리 생성
            cond_dict = make_cond_dict(
                text=segment.text[:500],  # 길이 제한
                language=segment.settings.get("language", "ko"),
                speaker=speaker_embedding,
                emotion=emotion_tensor,
                vqscore_8=vq_tensor,
                fmax=segment.settings.get("fmax", 8000.0),
                pitch_std=segment.settings.get("pitch_std", 1.0),
                speaking_rate=segment.settings.get("speaking_rate", 1.0),
                dnsmos_ovrl=segment.settings.get("dnsmos_ovrl", 3.0),
                speaker_noised=segment.settings.get("speaker_noised", False),
                device=device,
                unconditional_keys=segment.settings.get("unconditional_keys", []),
            )
            
            # 조건부 준비
            conditioning = self.model.prepare_conditioning(cond_dict)
            
            # 음성 생성
            codes = self.model.generate(
                prefix_conditioning=conditioning,
                max_new_tokens=min(86 * 20, 86 * int(len(segment.text) / 10) + 86),  # 텍스트 길이에 따른 동적 조정
                cfg_scale=segment.settings.get("cfg_scale", 3.0),
                batch_size=1,
                sampling_params=dict(
                    linear=float(segment.settings.get("linear", 0.0)),
                    conf=float(segment.settings.get("confidence", 1.0)), 
                    quad=float(segment.settings.get("quadratic", 0.0))
                )
            )
            
            # 오디오 디코딩
            wav_out = self.model.autoencoder.decode(codes).cpu().detach()
            if wav_out.dim() == 2 and wav_out.size(0) > 1:
                wav_out = wav_out[0:1, :]
            
            # numpy 배열로 변환
            audio_data = wav_out.squeeze().numpy()
            
            # 🔥 GPU/CPU Tensor 명시적 해제 (메모리 누수 방지)
            del codes, wav_out
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
            logger.debug(f"Generated audio segment: {audio_data.shape}, text: '{segment.text[:50]}...'")
            return audio_data
            
        except Exception as e:
            logger.error(f"Audio generation failed for segment '{segment.text[:50]}...': {e}")
            import traceback
            traceback.print_exc()
            return None

# ==================== 배치 볼륨 처리 시스템 ====================

class BatchVolumeProcessor:
    """파일 기반 배치 볼륨 처리"""
    
    def __init__(self, file_manager: SegmentFileManager):
        self.file_manager = file_manager
        self.volume_analyzer = VolumeAnalyzer()
    
    def analyze_all_segments(self) -> GlobalVolumeAnalysis:
        """모든 세그먼트 파일의 볼륨 분석"""
        
        volume_data = []
        
        try:
            # 배치 단위로 파일 로드 및 분석
            for batch_segments in self.file_manager.load_segments_batch(max_memory_mb=256):
                for segment in batch_segments:
                    volume_metrics = self.volume_analyzer.analyze_segment_volume(
                        segment.audio_data
                    )
                    
                    volume_data.append({
                        'sequence_id': segment.metadata.sequence_id,
                        'speaker_name': segment.metadata.speaker_name,
                        'rms_db': volume_metrics.rms_db,
                        'peak_db': volume_metrics.peak_db,
                        'dynamic_range': volume_metrics.dynamic_range,
                        'silence_ratio': volume_metrics.silence_ratio
                    })
                
                # 배치 처리 후 메모리 해제
                del batch_segments
                gc.collect()
            
            global_stats = self.volume_analyzer.calculate_global_stats(volume_data)
            
            logger.info(f"Analyzed {len(volume_data)} segments for volume")
            return GlobalVolumeAnalysis(
                segment_volumes=volume_data,
                global_stats=global_stats
            )
            
        except Exception as e:
            logger.error(f"Volume analysis failed: {e}")
            raise
    
    def normalize_all_segments(
        self, 
        volume_analysis: GlobalVolumeAnalysis,
        target_lufs: float = -23.0
    ) -> List[SegmentFile]:
        """모든 세그먼트를 정규화하여 새 파일로 저장"""
        
        normalized_files = []
        
        try:
            # 정규화 파라미터 계산
            normalization_params = self._calculate_normalization_params(
                volume_analysis, target_lufs
            )
            
            # 배치별로 정규화 처리
            for batch_segments in self.file_manager.load_segments_batch(max_memory_mb=256):
                for segment in batch_segments:
                    # 화자별 정규화 적용
                    speaker_params = normalization_params.get(
                        segment.metadata.speaker_name,
                        {'gain_db': 0.0}  # 기본값
                    )
                    
                    normalized_audio = self._apply_normalization(
                        segment.audio_data, speaker_params
                    )
                    
                    # 정규화된 파일 저장
                    normalized_file = self._save_normalized_segment(
                        normalized_audio, segment.sample_rate, segment.metadata
                    )
                    normalized_files.append(normalized_file)
                    
                    # 메모리 해제
                    del normalized_audio
                
                del batch_segments
                gc.collect()
            
            logger.info(f"Normalized {len(normalized_files)} segments")
            return normalized_files
            
        except Exception as e:
            logger.error(f"Normalization failed: {e}")
            raise
    
    def _calculate_normalization_params(
        self, 
        analysis: GlobalVolumeAnalysis, 
        target_lufs: float
    ) -> Dict[str, Dict[str, float]]:
        """화자별 정규화 파라미터 계산"""
        
        params = {}
        
        # 화자별 그룹화
        speaker_volumes = {}
        for vol_data in analysis.segment_volumes:
            speaker = vol_data['speaker_name']
            if speaker not in speaker_volumes:
                speaker_volumes[speaker] = []
            speaker_volumes[speaker].append(vol_data['rms_db'])
        
        # 화자별 평균 RMS 계산 및 정규화 게인 결정
        global_mean_rms = analysis.global_stats.get('mean_rms_db', -20.0)
        target_rms_db = target_lufs + 4.0  # 대략적인 LUFS to RMS 변환
        
        for speaker, rms_values in speaker_volumes.items():
            speaker_mean_rms = np.mean(rms_values)
            gain_db = target_rms_db - speaker_mean_rms
            
            # 극단적인 값 제한
            gain_db = np.clip(gain_db, -20.0, 20.0)
            
            params[speaker] = {
                'gain_db': float(gain_db),
                'original_rms': float(speaker_mean_rms)
            }
        
        logger.debug(f"Calculated normalization params: {params}")
        return params
    
    def _apply_normalization(
        self, 
        audio_data: np.ndarray, 
        params: Dict[str, float]
    ) -> np.ndarray:
        """정규화 적용"""
        
        gain_db = params.get('gain_db', 0.0)
        
        if abs(gain_db) < 0.1:  # 매우 작은 게인은 무시
            return audio_data.copy()
        
        # dB to linear 변환
        gain_linear = 10 ** (gain_db / 20.0)
        
        # 게인 적용
        normalized = audio_data * gain_linear
        
        # 클리핑 방지
        if np.max(np.abs(normalized)) > 0.95:
            peak_reduction = 0.95 / np.max(np.abs(normalized))
            normalized *= peak_reduction
        
        return normalized
    
    def _save_normalized_segment(
        self, 
        audio_data: np.ndarray, 
        sample_rate: int, 
        metadata: SegmentMetadata
    ) -> SegmentFile:
        """정규화된 세그먼트를 새 파일로 저장"""
        
        file_id = f"{self.file_manager.session_id}_norm_{metadata.sequence_id:03d}"
        audio_path = self.file_manager.temp_dir / f"{file_id}.wav"
        metadata_path = self.file_manager.temp_dir / f"{file_id}_meta.json"
        
        try:
            sf.write(str(audio_path), audio_data, sample_rate)
            
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata.to_dict(), f, ensure_ascii=False, indent=2)
            
            return SegmentFile(
                audio_path=audio_path,
                metadata_path=metadata_path,
                sequence_id=metadata.sequence_id,
                speaker_name=metadata.speaker_name,
                duration=len(audio_data) / sample_rate,
                file_size=audio_path.stat().st_size
            )
            
        except Exception as e:
            logger.error(f"Failed to save normalized segment: {e}")
            raise

# ==================== 스트리밍 오디오 통합 시스템 ====================

class StreamingAudioComposer:
    """메모리 효율적 스트리밍 통합"""
    
    def __init__(self, sample_rate: int = 48000):
        self.sample_rate = sample_rate
        self.chunk_size = 1024 * 512  # 512KB 청크
    
    def compose_from_files(
        self,
        normalized_files: List[SegmentFile],
        output_path: str,
        spacing_samples: int = 9600,  # 200ms @ 48kHz
        progress_callback: Optional[Callable] = None
    ) -> CompositionResult:
        """정규화된 파일들을 스트리밍으로 통합"""
        
        total_segments = len(normalized_files)
        composition_stats = {
            'total_duration': 0.0,
            'total_samples': 0,
            'segments_processed': 0,
            'spacing_added_count': 0
        }
        
        try:
            # 시퀀스 ID로 정렬
            sorted_files = sorted(normalized_files, key=lambda x: x.sequence_id)
            
            with sf.SoundFile(
                output_path, 
                'w', 
                samplerate=self.sample_rate, 
                channels=1
            ) as output_file:
                
                for i, seg_file in enumerate(sorted_files):
                    
                    if progress_callback:
                        progress_callback(
                            i / total_segments, 
                            f"통합 중: {i+1}/{total_segments} ({seg_file.speaker_name})"
                        )
                    
                    # 세그먼트 파일을 청크 단위로 스트리밍
                    try:
                        with sf.SoundFile(str(seg_file.audio_path), 'r') as input_file:
                            
                            while True:
                                chunk = input_file.read(self.chunk_size)
                                if len(chunk) == 0:
                                    break
                                
                                output_file.write(chunk)
                                composition_stats['total_samples'] += len(chunk)
                        
                        # 세그먼트 간 간격 추가 (마지막 세그먼트 제외)
                        if i < total_segments - 1:
                            spacing_audio = np.zeros(spacing_samples, dtype=np.float32)
                            output_file.write(spacing_audio)
                            composition_stats['total_samples'] += spacing_samples
                            composition_stats['spacing_added_count'] += 1
                        
                        composition_stats['segments_processed'] += 1
                        
                    except Exception as e:
                        logger.warning(f"Failed to process segment file {seg_file.audio_path}: {e}")
                        continue
            
            composition_stats['total_duration'] = composition_stats['total_samples'] / self.sample_rate
            
            if progress_callback:
                progress_callback(1.0, "통합 완료!")
            
            logger.info(
                f"Composition complete: {composition_stats['segments_processed']} segments, "
                f"{composition_stats['total_duration']:.1f}s total"
            )
            
            return CompositionResult(
                output_path=output_path,
                stats=composition_stats
            )
            
        except Exception as e:
            logger.error(f"Audio composition failed: {e}")
            raise

# ==================== 대화 파싱 유틸리티 ====================

class DialogueParser:
    """대화 텍스트를 세그먼트로 파싱"""
    
    def __init__(self, default_settings: Optional[Dict[str, Any]] = None):
        self.default_settings = default_settings or {}
    
    def parse_dialogue(self, dialogue_text: str) -> List[DialogueSegment]:
        """대화 텍스트를 DialogueSegment 리스트로 변환"""
        
        segments = []
        lines = dialogue_text.strip().split('\n')
        sequence_id = 0
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # 화자와 텍스트 분리 (기존 로직 참조)
            speaker_name, text, settings = self._parse_line(line)
            
            if speaker_name and text:
                segments.append(DialogueSegment(
                    sequence_id=sequence_id,
                    speaker_name=speaker_name,
                    text=text,
                    settings=settings
                ))
                sequence_id += 1
        
        return segments
    
    def _parse_line(self, line: str) -> Tuple[str, str, Dict[str, Any]]:
        """단일 라인에서 화자명, 텍스트, 설정 추출"""
        
        # 기본 형식: "화자명: 텍스트"
        if ':' in line:
            parts = line.split(':', 1)
            speaker_name = parts[0].strip()
            remaining = parts[1].strip()
            
            # 설정 추출 (괄호 안의 설정들)
            settings = self.default_settings.copy()
            text = remaining
            
            # 간단한 설정 파싱 구현 (필요시 확장)
            # 예: "안녕하세요 [happy=0.8, rate=1.2]"
            if '[' in text and ']' in text:
                import re
                pattern = r'\[(.*?)\]'
                matches = re.findall(pattern, text)
                
                for match in matches:
                    setting_pairs = match.split(',')
                    for pair in setting_pairs:
                        if '=' in pair:
                            key, value = pair.split('=', 1)
                            key = key.strip()
                            value = value.strip()
                            
                            # 값 타입 변환 시도
                            try:
                                if '.' in value:
                                    settings[key] = float(value)
                                elif value.lower() in ['true', 'false']:
                                    settings[key] = value.lower() == 'true'
                                else:
                                    settings[key] = int(value)
                            except:
                                settings[key] = value
                
                # 설정 부분 제거
                text = re.sub(r'\[.*?\]', '', text).strip()
            
            return speaker_name, text, settings
        
        # 화자명이 없는 경우
        return "Unknown", line, self.default_settings.copy()

@dataclass
class MemoryEfficientResult:
    """메모리 효율적 처리 결과"""
    final_audio_path: Path
    integrated_audio_path: Path  # 후처리 이전 통합본
    volume_analysis: GlobalVolumeAnalysis
    composition_result: CompositionResult
    file_manager: SegmentFileManager
    session_info: Dict[str, Any]

# ==================== 메인 통합 시스템 ====================

class MemoryEfficientAudioSystem:
    """완전한 메모리 효율적 오디오 처리 시스템"""
    
    def __init__(
        self, 
        model, 
        speaker_embeddings: Dict[str, torch.Tensor],
        temp_dir: str = "temp_audio_segments",
        default_settings: Optional[Dict[str, Any]] = None
    ):
        self.model = model
        self.speaker_embeddings = speaker_embeddings
        
        # 파일 기반 처리 컴포넌트
        self.file_manager = SegmentFileManager(temp_dir)
        self.dialogue_parser = DialogueParser(default_settings)
        self.segment_generator = MemoryEfficientSegmentGenerator(
            model, speaker_embeddings, self.file_manager
        )
        self.volume_processor = BatchVolumeProcessor(self.file_manager)
        self.streaming_composer = StreamingAudioComposer()
        
        logger.info(f"MemoryEfficientAudioSystem initialized with {len(speaker_embeddings)} speakers")
    
    def generate_complete_audio(
        self,
        dialogue_text: str,
        spacing_ms: float = 200.0,
        target_lufs: float = -23.0,
        progress_callback: Optional[Callable] = None
    ) -> MemoryEfficientResult:
        """완전한 메모리 효율적 오디오 생성"""
        
        try:
            # Phase 1: 대화 파싱 (5%)
            if progress_callback:
                progress_callback(0.05, "대화 파싱 중...")
            
            segments = self.dialogue_parser.parse_dialogue(dialogue_text)
            
            if not segments:
                raise ValueError("유효한 대화 세그먼트를 찾을 수 없습니다.")
            
            logger.info(f"Parsed {len(segments)} dialogue segments")
            
            # Phase 2: 세그먼트별 생성 및 임시파일 저장 (5% ~ 60%)
            if progress_callback:
                progress_callback(0.1, "음성 생성 중...")
            
            segment_files = self.segment_generator.generate_segments_to_files(
                segments, 
                lambda p, msg: progress_callback(0.1 + p * 0.5, msg) if progress_callback else None
            )
            
            if not segment_files:
                raise ValueError("음성 생성에 실패했습니다.")
            
            # Phase 3: 전체 볼륨 분석 (60% ~ 70%)
            if progress_callback:
                progress_callback(0.6, "볼륨 분석 중...")
            
            volume_analysis = self.volume_processor.analyze_all_segments()
            
            # Phase 4: 배치별 정규화 (70% ~ 80%)
            if progress_callback:
                progress_callback(0.7, "정규화 중...")
            
            normalized_files = self.volume_processor.normalize_all_segments(
                volume_analysis, target_lufs
            )
            
            # Phase 5: 스트리밍 통합 (80% ~ 95%)
            if progress_callback:
                progress_callback(0.8, "오디오 통합 중...")
            
            temp_output = self.file_manager.temp_dir / f"{self.file_manager.session_id}_integrated.wav"
            spacing_samples = int(spacing_ms * 48000 / 1000)  # ms to samples
            
            composition_result = self.streaming_composer.compose_from_files(
                normalized_files, 
                str(temp_output),
                spacing_samples=spacing_samples,
                progress_callback=lambda p, msg: progress_callback(0.8 + p * 0.15, msg) if progress_callback else None
            )
            
            # Phase 6: 완료 (95% ~ 100%)
            if progress_callback:
                progress_callback(0.95, "완료 처리 중...")
            
            session_info = self.file_manager.get_session_info()
            session_info['generation_summary'] = {
                'requested_segments': len(segments),
                'generated_segments': len(segment_files),
                'normalized_segments': len(normalized_files),
                'final_duration': composition_result.stats['total_duration'],
                'target_lufs': target_lufs,
                'spacing_ms': spacing_ms
            }
            
            if progress_callback:
                progress_callback(1.0, f"완료! ({composition_result.stats['total_duration']:.1f}초)")
            
            result = MemoryEfficientResult(
                final_audio_path=temp_output,
                integrated_audio_path=temp_output,  # 현재는 동일 (후처리 미구현)
                volume_analysis=volume_analysis,
                composition_result=composition_result,
                file_manager=self.file_manager,
                session_info=session_info
            )
            
            logger.info(
                f"Audio generation complete: {len(segment_files)} segments, "
                f"{composition_result.stats['total_duration']:.1f}s total"
            )
            
            return result
            
        except Exception as e:
            # 오류 발생 시 임시파일 정리
            logger.error(f"Audio generation failed: {e}")
            self.cleanup()
            raise
    
    def load_final_audio_for_ui(
        self, 
        result: MemoryEfficientResult
    ) -> Tuple[int, np.ndarray]:
        """UI 표시용 최종 오디오 로드 (필요시에만)"""
        
        try:
            audio_data, sample_rate = sf.read(str(result.final_audio_path))
            
            logger.debug(f"Loaded final audio: {len(audio_data)} samples @ {sample_rate}Hz")
            return sample_rate, audio_data
            
        except Exception as e:
            logger.error(f"Failed to load final audio: {e}")
            raise
    
    def get_generation_summary(self, result: MemoryEfficientResult) -> str:
        """생성 결과 요약 텍스트 반환"""
        
        stats = result.composition_result.stats
        session_info = result.session_info
        gen_summary = session_info.get('generation_summary', {})
        
        summary = f"""
✅ 메모리 효율적 오디오 생성 완료!

📊 **생성 통계:**
• 요청 세그먼트: {gen_summary.get('requested_segments', 0)}개
• 성공 생성: {gen_summary.get('generated_segments', 0)}개
• 정규화 완료: {gen_summary.get('normalized_segments', 0)}개

🎵 **최종 결과:**
• 총 길이: {stats['total_duration']:.1f}초
• 화자 수: {len(session_info.get('unique_speakers', []))}명
• 간격 설정: {gen_summary.get('spacing_ms', 200)}ms

💾 **메모리 효율성:**
• 임시파일: {session_info['segment_count']}개
• 디스크 사용: {session_info['total_file_size'] / (1024*1024):.1f}MB
• 세션 ID: {session_info['session_id']}

⚡ **볼륨 정규화:**
• 목표 LUFS: {gen_summary.get('target_lufs', -23.0)}
• 평균 RMS: {result.volume_analysis.global_stats.get('mean_rms_db', 0):.1f}dB
• 최대 Peak: {result.volume_analysis.global_stats.get('max_peak_db', 0):.1f}dB
        """.strip()
        
        return summary
    
    def cleanup(self):
        """임시파일 정리"""
        try:
            self.file_manager.cleanup_temp_files()
            logger.info("Cleanup completed")
        except Exception as e:
            logger.warning(f"Cleanup failed: {e}")
    
    def update_speaker_embeddings(self, new_embeddings: Dict[str, torch.Tensor]):
        """화자 임베딩 업데이트"""
        self.speaker_embeddings.update(new_embeddings)
        self.segment_generator.speaker_embeddings = self.speaker_embeddings
        logger.info(f"Updated speaker embeddings: {len(self.speaker_embeddings)} total speakers")

# ==================== 유틸리티 함수 ====================

def convert_legacy_dialogue_format(legacy_parts: List[Tuple]) -> List[DialogueSegment]:
    """기존 대화 형식을 새로운 DialogueSegment 형식으로 변환"""
    
    segments = []
    
    for i, (speaker, text, settings) in enumerate(legacy_parts):
        segments.append(DialogueSegment(
            sequence_id=i,
            speaker_name=speaker,
            text=text,
            settings=settings or {}
        ))
    
    return segments

def estimate_memory_usage(segment_count: int, avg_duration: float) -> Dict[str, float]:
    """메모리 사용량 추정"""
    
    # 대략적인 계산 (48kHz, 16bit 기준)
    avg_samples = int(avg_duration * 48000)
    bytes_per_segment = avg_samples * 4  # float32
    
    # 기존 방식 (모든 세그먼트를 메모리에)
    legacy_memory_mb = (segment_count * bytes_per_segment) / (1024 * 1024)
    
    # 새로운 방식 (배치 처리, 최대 5개 세그먼트)
    efficient_memory_mb = min(5, segment_count) * bytes_per_segment / (1024 * 1024)
    
    return {
        'legacy_memory_mb': legacy_memory_mb,
        'efficient_memory_mb': efficient_memory_mb,
        'memory_savings_percent': ((legacy_memory_mb - efficient_memory_mb) / legacy_memory_mb * 100) if legacy_memory_mb > 0 else 0
    }