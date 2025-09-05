"""
Lazy Loading 화자 관리 시스템

앱 시작 시 화자 목록만 가져오고, 실제 임베딩 로딩은 필요할 때만 수행하는 시스템
"""

import os
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass

import torch
import torchaudio

logger = logging.getLogger(__name__)

@dataclass
class SpeakerInfo:
    """화자 정보 (임베딩은 포함하지 않음)"""
    name: str
    audio_path: str
    created_at: datetime
    last_used: Optional[datetime] = None
    is_loaded: bool = False
    file_size: int = 0
    sample_rate: int = 0
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'SpeakerInfo':
        return cls(
            name=data['name'],
            audio_path=data['audio_path'],
            created_at=datetime.fromisoformat(data.get('created_at', datetime.now().isoformat())),
            last_used=datetime.fromisoformat(data['last_used']) if data.get('last_used') else None,
            is_loaded=False,  # 항상 False로 시작
            file_size=data.get('file_size', 0),
            sample_rate=data.get('sample_rate', 0)
        )
    
    def to_dict(self) -> Dict:
        return {
            'name': self.name,
            'audio_path': self.audio_path,
            'created_at': self.created_at.isoformat(),
            'last_used': self.last_used.isoformat() if self.last_used else None,
            'file_size': self.file_size,
            'sample_rate': self.sample_rate
        }

class LazySpeakerManager:
    """Lazy Loading을 지원하는 화자 관리자"""
    
    def __init__(self, data_dir: str, device: str = "cuda"):
        self.data_dir = Path(data_dir)
        self.speakers_json_path = self.data_dir / "saved_speakers.json"
        self.device = device
        
        # 화자 메타데이터만 저장 (임베딩은 별도)
        self.speakers_info: Dict[str, SpeakerInfo] = {}
        
        # 실제 로드된 임베딩들 (필요시에만 로드)
        self.loaded_embeddings: Dict[str, torch.Tensor] = {}
        
        # 현재 로드된 모델 (임베딩 생성용)
        self.current_model = None
        
        logger.info(f"LazySpeakerManager initialized with data_dir: {data_dir}")
    
    def set_model(self, model):
        """현재 모델 설정 (임베딩 생성용)"""
        self.current_model = model
        logger.info("Model set for embedding generation")
    
    def load_speaker_list_on_startup(self) -> Tuple[bool, str, List[str]]:
        """
        앱 시작시 화자 목록만 빠르게 로드 (임베딩 생성 없음)
        Returns: (success, message, speaker_names)
        """
        try:
            if not self.speakers_json_path.exists():
                logger.info("No saved speakers file found, starting with empty list")
                return True, "저장된 화자 목록이 없습니다. 새로 시작합니다.", []
            
            with open(self.speakers_json_path, "r", encoding="utf-8") as f:
                speaker_data = json.load(f)
            
            self.speakers_info = {}
            available_speakers = []
            missing_files = []
            
            for speaker_name, audio_path in speaker_data.items():
                if isinstance(audio_path, str):
                    # 레거시 형식 (단순 경로)
                    if os.path.exists(audio_path):
                        # 파일 정보 수집
                        file_stat = os.stat(audio_path)
                        
                        speaker_info = SpeakerInfo(
                            name=speaker_name,
                            audio_path=audio_path,
                            created_at=datetime.fromtimestamp(file_stat.st_ctime),
                            file_size=file_stat.st_size
                        )
                        
                        self.speakers_info[speaker_name] = speaker_info
                        available_speakers.append(speaker_name)
                    else:
                        missing_files.append(speaker_name)
                else:
                    # 새로운 형식 (딕셔너리)
                    if os.path.exists(audio_path.get('audio_path', '')):
                        speaker_info = SpeakerInfo.from_dict({
                            'name': speaker_name,
                            **audio_path
                        })
                        self.speakers_info[speaker_name] = speaker_info
                        available_speakers.append(speaker_name)
                    else:
                        missing_files.append(speaker_name)
            
            message_parts = []
            if available_speakers:
                message_parts.append(f"✅ {len(available_speakers)}개 화자 목록 로드 완료")
            
            if missing_files:
                message_parts.append(f"⚠️ {len(missing_files)}개 파일 누락: {', '.join(missing_files[:3])}")
                # 누락된 화자들은 목록에서 제거
                for missing in missing_files:
                    if missing in self.speakers_info:
                        del self.speakers_info[missing]
            
            message = " | ".join(message_parts) if message_parts else "화자 목록 로드 완료"
            
            logger.info(f"Speaker list loaded: {len(available_speakers)} available, {len(missing_files)} missing")
            return True, message, available_speakers
            
        except Exception as e:
            logger.error(f"Failed to load speaker list: {e}")
            return False, f"화자 목록 로드 실패: {str(e)}", []
    
    def get_speaker_names(self) -> List[str]:
        """로드된 화자 이름 목록 반환"""
        return list(self.speakers_info.keys())
    
    def is_speaker_loaded(self, speaker_name: str) -> bool:
        """특정 화자의 임베딩이 로드되어 있는지 확인"""
        return speaker_name in self.loaded_embeddings
    
    def get_loaded_speakers(self) -> List[str]:
        """현재 임베딩이 로드된 화자 목록"""
        return list(self.loaded_embeddings.keys())
    
    def load_speaker_embedding(self, speaker_name: str) -> Optional[torch.Tensor]:
        """
        특정 화자의 임베딩을 지연 로딩
        이미 로드된 경우 캐시에서 반환
        """
        # 이미 로드된 경우
        if speaker_name in self.loaded_embeddings:
            logger.debug(f"Speaker {speaker_name} already loaded, returning cached embedding")
            # 사용 시간 업데이트
            if speaker_name in self.speakers_info:
                self.speakers_info[speaker_name].last_used = datetime.now()
            return self.loaded_embeddings[speaker_name]
        
        # 화자 정보가 없는 경우
        if speaker_name not in self.speakers_info:
            logger.warning(f"Speaker {speaker_name} not found in speaker list")
            return None
        
        # 모델이 설정되지 않은 경우
        if self.current_model is None:
            logger.error("Model not set, cannot generate embedding")
            return None
        
        try:
            speaker_info = self.speakers_info[speaker_name]
            
            # 오디오 파일 로드
            if not os.path.exists(speaker_info.audio_path):
                logger.error(f"Audio file not found: {speaker_info.audio_path}")
                return None
            
            logger.info(f"Loading embedding for speaker: {speaker_name}")
            
            wav, sr = torchaudio.load(speaker_info.audio_path)
            embedding = self.current_model.make_speaker_embedding(wav, sr)
            embedding = embedding.to(self.device, dtype=torch.bfloat16)
            
            # 캐시에 저장
            self.loaded_embeddings[speaker_name] = embedding
            
            # 메타데이터 업데이트
            speaker_info.is_loaded = True
            speaker_info.last_used = datetime.now()
            if speaker_info.sample_rate == 0:
                speaker_info.sample_rate = sr
            
            logger.info(f"Successfully loaded embedding for speaker: {speaker_name}")
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to load speaker embedding for {speaker_name}: {e}")
            return None
    
    def load_multiple_speakers(self, speaker_names: List[str]) -> Dict[str, torch.Tensor]:
        """여러 화자의 임베딩을 한번에 로드"""
        results = {}
        
        for speaker_name in speaker_names:
            embedding = self.load_speaker_embedding(speaker_name)
            if embedding is not None:
                results[speaker_name] = embedding
        
        logger.info(f"Loaded {len(results)} out of {len(speaker_names)} requested speakers")
        return results
    
    def preload_frequent_speakers(self, limit: int = 5) -> int:
        """
        자주 사용되는 화자들의 임베딩을 미리 로드
        최근 사용된 순서대로 limit개만 로드
        """
        # 최근 사용 순으로 정렬
        sorted_speakers = sorted(
            self.speakers_info.items(),
            key=lambda x: x[1].last_used or datetime.min,
            reverse=True
        )
        
        preload_count = 0
        for speaker_name, speaker_info in sorted_speakers[:limit]:
            if not self.is_speaker_loaded(speaker_name):
                if self.load_speaker_embedding(speaker_name) is not None:
                    preload_count += 1
        
        logger.info(f"Preloaded {preload_count} frequent speakers")
        return preload_count
    
    def unload_speaker(self, speaker_name: str) -> bool:
        """특정 화자의 임베딩을 메모리에서 해제"""
        if speaker_name in self.loaded_embeddings:
            del self.loaded_embeddings[speaker_name]
            
            if speaker_name in self.speakers_info:
                self.speakers_info[speaker_name].is_loaded = False
            
            logger.info(f"Unloaded speaker: {speaker_name}")
            return True
        
        return False
    
    def cleanup_unused_embeddings(self, keep_recent: int = 3) -> int:
        """
        오래된 임베딩들을 정리하여 메모리 절약
        최근 사용된 keep_recent개만 유지
        """
        if len(self.loaded_embeddings) <= keep_recent:
            return 0
        
        # 사용 시간 순으로 정렬
        speakers_by_usage = []
        for speaker_name in self.loaded_embeddings.keys():
            if speaker_name in self.speakers_info:
                last_used = self.speakers_info[speaker_name].last_used or datetime.min
                speakers_by_usage.append((last_used, speaker_name))
        
        speakers_by_usage.sort(reverse=True)  # 최근 사용 순
        
        # 오래된 것들 제거
        removed_count = 0
        for _, speaker_name in speakers_by_usage[keep_recent:]:
            if self.unload_speaker(speaker_name):
                removed_count += 1
        
        logger.info(f"Cleaned up {removed_count} unused embeddings")
        return removed_count
    
    def get_memory_usage_info(self) -> Dict[str, any]:
        """현재 메모리 사용 정보 반환"""
        total_speakers = len(self.speakers_info)
        loaded_speakers = len(self.loaded_embeddings)
        
        # 임베딩 크기 추정 (대략적)
        estimated_size_mb = 0
        if self.loaded_embeddings:
            sample_embedding = next(iter(self.loaded_embeddings.values()))
            bytes_per_embedding = sample_embedding.element_size() * sample_embedding.nelement()
            estimated_size_mb = (bytes_per_embedding * loaded_speakers) / (1024 * 1024)
        
        return {
            'total_speakers': total_speakers,
            'loaded_speakers': loaded_speakers,
            'loading_ratio': f"{loaded_speakers}/{total_speakers}",
            'estimated_memory_mb': round(estimated_size_mb, 2),
            'recently_used': [
                name for name, info in sorted(
                    self.speakers_info.items(),
                    key=lambda x: x[1].last_used or datetime.min,
                    reverse=True
                )[:5]
            ]
        }
    
    def save_speaker_metadata(self) -> bool:
        """업데이트된 메타데이터를 파일에 저장"""
        try:
            # 새로운 형식으로 저장 (메타데이터 포함)
            speaker_data = {}
            for name, info in self.speakers_info.items():
                speaker_data[name] = info.to_dict()
            
            # 백업 생성
            if self.speakers_json_path.exists():
                backup_path = self.speakers_json_path.with_suffix('.json.bak')
                import shutil
                shutil.copy2(self.speakers_json_path, backup_path)
            
            with open(self.speakers_json_path, "w", encoding="utf-8") as f:
                json.dump(speaker_data, f, ensure_ascii=False, indent=2)
            
            logger.info("Speaker metadata saved successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save speaker metadata: {e}")
            return False