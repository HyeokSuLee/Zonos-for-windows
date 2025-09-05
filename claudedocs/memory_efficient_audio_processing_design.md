# 메모리 효율적 오디오 처리 시스템 설계

## 🎯 핵심 처리 원칙

**메모리 효율성을 위한 임시파일 기반 처리 파이프라인**

```
세그먼트 생성 → 임시파일 저장 → 메모리 해제
     ↓
모든 세그먼트 완료 → 전체 볼륨 분석 → 정규화 파라미터 결정  
     ↓
배치별 정규화 → 정규화된 임시파일들 저장
     ↓  
스트리밍 통합 → 단일 통합 오디오 파일 생성
     ↓
통합된 전체 오디오 → 무음구간 편집 → 최종 출력
```

## 🗂️ PART I: 임시파일 관리 시스템

### 1. SegmentFileManager

```python
class SegmentFileManager:
    """세그먼트 임시파일 관리 시스템"""
    
    def __init__(self, temp_dir: str = "temp_audio_segments"):
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(exist_ok=True)
        self.session_id = str(uuid.uuid4())[:8]
        self.segment_files: List[SegmentFile] = []
    
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
        
        return segment_file
    
    def load_segment_from_file(self, segment_file: SegmentFile) -> AudioSegment:
        """임시파일에서 세그먼트 로드"""
        
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
    
    def load_segments_batch(
        self, 
        segment_files: List[SegmentFile],
        max_memory_mb: int = 512
    ) -> Generator[List[AudioSegment], None, None]:
        """메모리 제한 내에서 배치 단위로 세그먼트 로드"""
        
        current_batch = []
        current_memory_mb = 0
        
        for seg_file in segment_files:
            estimated_mb = seg_file.file_size / (1024 * 1024) * 2  # 압축 해제 고려
            
            if current_memory_mb + estimated_mb > max_memory_mb and current_batch:
                # 현재 배치 반환 후 초기화
                batch_segments = [self.load_segment_from_file(sf) for sf in current_batch]
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
            batch_segments = [self.load_segment_from_file(sf) for sf in current_batch]
            yield batch_segments
    
    def cleanup_temp_files(self):
        """세션 임시파일 정리"""
        
        for seg_file in self.segment_files:
            try:
                seg_file.audio_path.unlink(missing_ok=True)
                seg_file.metadata_path.unlink(missing_ok=True)
            except Exception as e:
                logger.warning(f"임시파일 삭제 실패: {e}")
        
        self.segment_files.clear()
    
    def get_total_duration(self) -> float:
        """전체 세그먼트 길이 계산"""
        return sum(seg_file.duration for seg_file in self.segment_files)
    
    def get_speaker_segments(self, speaker_name: str) -> List[SegmentFile]:
        """특정 화자의 세그먼트 파일들만 반환"""
        return [
            seg_file for seg_file in self.segment_files 
            if seg_file.speaker_name == speaker_name
        ]

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
class AudioSegment:
    """메모리상의 오디오 세그먼트 (임시 로드용)"""
    audio_data: np.ndarray
    sample_rate: int
    metadata: SegmentMetadata
```

## 🔄 PART II: 메모리 효율적 처리 파이프라인

### 2. 수정된 SegmentGenerator

```python
class MemoryEfficientSegmentGenerator:
    """메모리 효율적 세그먼트 생성기"""
    
    def __init__(
        self, 
        model: Zonos, 
        speaker_loader: LazyEmbeddingLoader,
        file_manager: SegmentFileManager
    ):
        self.model = model
        self.speaker_loader = speaker_loader
        self.file_manager = file_manager
    
    async def generate_segments_to_files(
        self, 
        segments: List[DialogueSegment],
        progress_callback: Optional[Callable] = None
    ) -> List[SegmentFile]:
        """세그먼트들을 생성하여 직접 파일로 저장"""
        
        segment_files = []
        
        for i, dialogue_seg in enumerate(segments):
            if progress_callback:
                progress_callback(i / len(segments), f"세그먼트 {i+1}/{len(segments)} 생성 중...")
            
            # 1. 음성 생성 (메모리 사용)
            speaker_embedding = self.speaker_loader.get_embedding(dialogue_seg.speaker_name)
            audio_data = self.model.generate_speech(
                text=dialogue_seg.text,
                speaker_embedding=speaker_embedding,
                **dialogue_seg.settings
            )
            
            # 2. 메타데이터 준비
            metadata = SegmentMetadata(
                sequence_id=dialogue_seg.sequence_id,
                speaker_name=dialogue_seg.speaker_name,
                text=dialogue_seg.text,
                settings_used=dialogue_seg.settings,
                generation_timestamp=datetime.now()
            )
            
            # 3. 즉시 파일로 저장 (메모리 해제)
            segment_file = self.file_manager.save_segment_to_file(
                audio_data=audio_data,
                sample_rate=48000,  # 또는 self.model.sample_rate
                metadata=metadata
            )
            
            segment_files.append(segment_file)
            
            # 4. 명시적 메모리 해제
            del audio_data, speaker_embedding
            gc.collect()
            
            # 5. GPU 메모리 정리 (필요시)
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
        
        if progress_callback:
            progress_callback(1.0, f"모든 세그먼트 생성 완료! ({len(segment_files)}개)")
        
        return segment_files
```

### 3. 배치 기반 볼륨 분석 및 정규화

```python
class BatchVolumeProcessor:
    """파일 기반 배치 볼륨 처리"""
    
    def __init__(self, file_manager: SegmentFileManager):
        self.file_manager = file_manager
    
    def analyze_all_segments(self) -> GlobalVolumeAnalysis:
        """모든 세그먼트 파일의 볼륨 분석"""
        
        volume_data = []
        
        # 배치 단위로 파일 로드 및 분석
        for batch_segments in self.file_manager.load_segments_batch(max_memory_mb=256):
            for segment in batch_segments:
                volume_metrics = self._analyze_segment_volume(segment.audio_data)
                volume_data.append({
                    'sequence_id': segment.metadata.sequence_id,
                    'speaker_name': segment.metadata.speaker_name,
                    'rms_db': volume_metrics.rms_db,
                    'peak_db': volume_metrics.peak_db,
                    'lufs': volume_metrics.lufs
                })
            
            # 배치 처리 후 메모리 해제
            del batch_segments
            gc.collect()
        
        return GlobalVolumeAnalysis(
            segment_volumes=volume_data,
            global_stats=self._calculate_global_stats(volume_data)
        )
    
    def normalize_all_segments(
        self, 
        volume_analysis: GlobalVolumeAnalysis,
        target_lufs: float = -23.0
    ) -> List[SegmentFile]:
        """모든 세그먼트를 정규화하여 새 파일로 저장"""
        
        normalized_files = []
        normalization_params = self._calculate_normalization_params(
            volume_analysis, target_lufs
        )
        
        # 배치별로 정규화 처리
        for batch_segments in self.file_manager.load_segments_batch(max_memory_mb=256):
            for segment in batch_segments:
                # 정규화 적용
                speaker_params = normalization_params[segment.metadata.speaker_name]
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
        
        return normalized_files
    
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

@dataclass
class GlobalVolumeAnalysis:
    """전체 세그먼트 볼륨 분석 결과"""
    segment_volumes: List[Dict]
    global_stats: Dict[str, float]
```

### 4. 스트리밍 기반 오디오 통합

```python
class StreamingAudioComposer:
    """메모리 효율적 스트리밍 통합"""
    
    def __init__(self, sample_rate: int = 48000):
        self.sample_rate = sample_rate
        self.chunk_size = 1024 * 1024  # 1MB 청크
    
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
            'segments_processed': 0
        }
        
        with sf.SoundFile(
            output_path, 
            'w', 
            samplerate=self.sample_rate, 
            channels=1
        ) as output_file:
            
            for i, seg_file in enumerate(sorted(normalized_files, key=lambda x: x.sequence_id)):
                
                if progress_callback:
                    progress_callback(i / total_segments, f"통합 중: {i+1}/{total_segments}")
                
                # 세그먼트 파일을 청크 단위로 스트리밍
                with sf.SoundFile(str(seg_file.audio_path), 'r') as input_file:
                    
                    while True:
                        chunk = input_file.read(self.chunk_size)
                        if len(chunk) == 0:
                            break
                        
                        output_file.write(chunk)
                        composition_stats['total_samples'] += len(chunk)
                
                # 세그먼트 간 간격 추가 (마지막 세그먼트 제외)
                if i < total_segments - 1:
                    spacing_audio = np.zeros(spacing_samples)
                    output_file.write(spacing_audio)
                    composition_stats['total_samples'] += spacing_samples
                
                composition_stats['segments_processed'] += 1
        
        composition_stats['total_duration'] = composition_stats['total_samples'] / self.sample_rate
        
        if progress_callback:
            progress_callback(1.0, "통합 완료!")
        
        return CompositionResult(
            output_path=output_path,
            stats=composition_stats
        )

@dataclass
class CompositionResult:
    """통합 결과"""
    output_path: str
    stats: Dict[str, Any]
```

## 🎛️ PART III: 통합된 메모리 효율적 파이프라인

### 5. MemoryEfficientAudioSystem

```python
class MemoryEfficientAudioSystem:
    """완전한 메모리 효율적 오디오 처리 시스템"""
    
    def __init__(self, model: Zonos, data_dir: str):
        self.model = model
        
        # 기본 컴포넌트
        self.speaker_registry = SpeakerRegistry(data_dir)
        self.embedding_loader = LazyEmbeddingLoader(model, cache_size=20)
        self.dialogue_parser = DialogueParser()
        
        # 파일 기반 처리 컴포넌트
        self.file_manager = SegmentFileManager()
        self.segment_generator = MemoryEfficientSegmentGenerator(
            model, self.embedding_loader, self.file_manager
        )
        self.volume_processor = BatchVolumeProcessor(self.file_manager)
        self.streaming_composer = StreamingAudioComposer()
        
        # 후처리 (기존 유지)
        self.post_processor = AudioPostProcessor()
    
    async def generate_complete_audio(
        self,
        dialogue_text: str,
        generation_settings: Dict[str, Any],
        post_processing_settings: Optional[EditSettings] = None,
        progress_callback: Optional[Callable] = None
    ) -> MemoryEfficientResult:
        """완전한 메모리 효율적 오디오 생성"""
        
        try:
            # Phase 1: 대화 파싱
            if progress_callback:
                progress_callback(0.1, "대화 파싱 중...")
            
            segments = self.dialogue_parser.parse_dialogue(dialogue_text)
            
            # Phase 2: 세그먼트별 생성 및 임시파일 저장
            if progress_callback:
                progress_callback(0.2, "음성 생성 중...")
            
            segment_files = await self.segment_generator.generate_segments_to_files(
                segments, 
                lambda p, msg: progress_callback(0.2 + p * 0.4, msg) if progress_callback else None
            )
            
            # Phase 3: 전체 볼륨 분석
            if progress_callback:
                progress_callback(0.6, "볼륨 분석 중...")
            
            volume_analysis = self.volume_processor.analyze_all_segments()
            
            # Phase 4: 배치별 정규화
            if progress_callback:
                progress_callback(0.7, "정규화 중...")
            
            normalized_files = self.volume_processor.normalize_all_segments(volume_analysis)
            
            # Phase 5: 스트리밍 통합
            if progress_callback:
                progress_callback(0.8, "오디오 통합 중...")
            
            temp_output = self.file_manager.temp_dir / f"{self.file_manager.session_id}_integrated.wav"
            composition_result = self.streaming_composer.compose_from_files(
                normalized_files, 
                str(temp_output),
                progress_callback=lambda p, msg: progress_callback(0.8 + p * 0.1, msg) if progress_callback else None
            )
            
            # Phase 6: 후처리 (옵션)
            final_output_path = temp_output
            post_processing_result = None
            
            if post_processing_settings:
                if progress_callback:
                    progress_callback(0.9, "후처리 편집 중...")
                
                # 통합된 오디오 로드
                integrated_audio, sample_rate = sf.read(str(temp_output))
                
                # 후처리 적용
                edited_result = self.post_processor.edit_audio(
                    integrated_audio, post_processing_settings
                )
                
                # 최종 파일 저장
                final_output_path = self.file_manager.temp_dir / f"{self.file_manager.session_id}_final.wav"
                sf.write(str(final_output_path), edited_result.audio_data, sample_rate)
                
                post_processing_result = edited_result
                
                # 메모리 해제
                del integrated_audio, edited_result.audio_data
                gc.collect()
            
            if progress_callback:
                progress_callback(1.0, "완료!")
            
            return MemoryEfficientResult(
                final_audio_path=final_output_path,
                integrated_audio_path=temp_output,
                volume_analysis=volume_analysis,
                composition_result=composition_result,
                post_processing_result=post_processing_result,
                file_manager=self.file_manager  # 정리용
            )
            
        except Exception as e:
            # 오류 발생 시 임시파일 정리
            self.cleanup()
            raise e
    
    def load_final_audio_for_ui(self, result: MemoryEfficientResult) -> Tuple[int, np.ndarray]:
        """UI 표시용 최종 오디오 로드 (필요시에만)"""
        audio_data, sample_rate = sf.read(str(result.final_audio_path))
        return sample_rate, audio_data
    
    def cleanup(self):
        """임시파일 정리"""
        self.file_manager.cleanup_temp_files()

@dataclass
class MemoryEfficientResult:
    """메모리 효율적 처리 결과"""
    final_audio_path: Path
    integrated_audio_path: Path  # 후처리 이전 통합본
    volume_analysis: GlobalVolumeAnalysis
    composition_result: CompositionResult
    post_processing_result: Optional[EditedAudio]
    file_manager: SegmentFileManager
```

## 📊 메모리 사용량 비교

### 기존 메모리 기반 방식
```
화자 10명, 세그먼트 50개, 각 10초 = 500초 오디오
메모리 사용량: 500초 × 48kHz × 4byte ≈ 96MB (모든 세그먼트 동시 보관)
+ 정규화 시 임시 복사본: +96MB 
+ 통합 시 최종 오디오: +96MB
총 메모리: ~288MB
```

### 새로운 파일 기반 방식  
```
동일한 조건
메모리 사용량: 현재 처리 배치만 (예: 5세그먼트 × 10초) ≈ 9.6MB
최대 메모리: ~20MB (배치 처리 + 버퍼)
디스크 사용: ~200MB (임시파일들, 처리 완료 후 자동 삭제)
메모리 절약: 93% 절약!
```

## 🎯 핵심 장점

1. **메모리 효율성**: 93% 메모리 사용량 절약
2. **확장성**: 긴 대화나 많은 화자도 일정한 메모리 사용  
3. **안정성**: 메모리 부족으로 인한 크래시 방지
4. **품질 일관성**: 전체 분석 후 정규화로 균일한 품질
5. **복구 가능성**: 중간 결과 파일 보존으로 재시작 가능

<function_calls>
<invoke name="TodoWrite">
<parameter name="todos">[{"content": "\uae30\uc874 \ud1b5\ud569 \uc124\uacc4\uc758 \uba54\ubaa8\ub9ac \uae30\ubc18 \ucc98\ub9ac \ubc29\uc2dd \ubd84\uc11d", "activeForm": "\uae30\uc874 \uc124\uacc4 \ubd84\uc11d \uc911", "status": "completed"}, {"content": "\uc784\uc2dc\ud30c\uc77c \uae30\ubc18 \ucc98\ub9ac \uc544\ud0a4\ud14d\ucc98\ub85c \uc124\uacc4 \uc218\uc815", "activeForm": "\uc784\uc2dc\ud30c\uc77c \uc544\ud0a4\ud14d\ucc98 \uc124\uacc4 \uc911", "status": "completed"}, {"content": "SegmentFileManager \ubc0f \ubc30\uce58 \ucc98\ub9ac \uc2dc\uc2a4\ud15c \uc124\uacc4", "activeForm": "\ud30c\uc77c \uad00\ub9ac \uc2dc\uc2a4\ud15c \uc124\uacc4 \uc911", "status": "completed"}, {"content": "\uba54\ubaa8\ub9ac \ud6a8\uc728\uc801 \ucc98\ub9ac \uc21c\uc11c \uba85\uc138 \uc791\uc131", "activeForm": "\ucc98\ub9ac \uc21c\uc11c \uba85\uc138 \uc791\uc131 \uc911", "status": "completed"}]