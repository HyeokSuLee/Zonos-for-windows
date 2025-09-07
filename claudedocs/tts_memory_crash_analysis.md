Zonos TTS 메모리 크래시 원인 분석 및 해결책

요약: UI 경로가 메모리 절감형 파이프라인을 사용하지 않아 KV-캐시가 과할당되고, DAC 모델의 VRAM 상주와 맞물려 연속 생성 중 GPU 드라이버 리셋(TDR) 또는 시스템 재부팅이 발생할 수 있습니다. 아래 조치를 순서대로 적용하세요.

핵심 원인
- UI 경로(gradio_interface.py)에서 고정 큰 max_new_tokens(≈2580)를 사용하여 Zonos.generate()가 큰 KV-캐시를 매 세그먼트 할당.
- 메모리 절감형 경로(memory_efficient_audio.py)를 UI에서 사용하지 않음(파일 스필, 동적 길이, 세그먼트별 empty_cache/GC 미적용).
- DAC 오토인코더가 GPU 상주로 기본 VRAM 여유 감소.

권장 해결책(우선순위)
- UI 경로를 MemoryEfficientAudioSystem 사용으로 전환: initialize_memory_efficient_system()이 준비되어 있으므로, generate_multi_speaker_audio()에서 MEMORY_EFFICIENT_SYSTEM이 있으면 이를 사용하도록 분기.
- Fallback 경로 개선: 텍스트 길이 기반으로 max_new_tokens를 동적 축소하고, 세그먼트마다 torch.cuda.empty_cache()와 gc.collect() 호출.
- 저메모리 모드: 필요 시 디코더(DAC)만 CPU로 옮겨 VRAM 확보(속도 저하 감수).
- 운영 환경: PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,garbage_collection_threshold:0.9 설정, Windows TdrDelay(10~30초) 적용, 페이지 파일 확보, 드라이버/온도 점검.

코드 근거
- gradio_interface.py: generate_multi_speaker_audio()는 메모리 절감형 경로 미사용, 기본 max_new_tokens=86*30.
- zonos/model.py: generate()는 max_new_tokens에 비례하는 KV-캐시 할당. EOS 조기 종료와 무관하게 캐시는 미리 크게 잡힘.
- zonos/memory_efficient_audio.py: 파일 스필, 동적 길이, 세그먼트별 empty_cache/GC 이미 구현.

빠른 적용 체크리스트
- [ ] UI 경로에서 메모리 절감 파이프라인 사용
- [ ] Fallback 경로에 동적 max_new_tokens + 세그먼트별 empty_cache/GC 추가
- [ ] 필요 시 ZONOS_LOW_MEM_DECODE=1 등 저메모리 옵션 도입
- [ ] PYTORCH_CUDA_ALLOC_CONF 및 Windows TDR/페이지 파일 설정

비고
- 긴 대사/다화자 케이스는 파일 스필과 배치 로드의 효과가 큽니다. memory_efficient_audio.estimate_memory_usage()로 절감폭을 가늠할 수 있습니다.

권장 변경 포인트(파일별)
- `gradio_interface.py`:
  - `generate_multi_speaker_audio()` 초반에 `MEMORY_EFFICIENT_SYSTEM` 존재 시 분기하여 `generate_complete_audio()` 사용. 결과 오디오만 메모리로 불러 UI에 반환.
  - Fallback 경로에서는 `max_new_tokens`를 텍스트 길이로 동적 계산(`min(86*20, 86*(len(text)//10 + 1))`)하고, 세그먼트 종료마다 정리 수행(`torch.cuda.empty_cache()`, `gc.collect()`, 지역 변수 `del`).
- `zonos/autoencoder.py`:
  - 옵션 환경변수 `ZONOS_LOW_MEM_DECODE=1`일 때 디코더만 CPU 사용하도록 분기(속도 저하 감수하고 VRAM 확보).
- 실행 환경:
  - `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,garbage_collection_threshold:0.9` 설정으로 분할 크기와 GC 임계값 상향.
  - Windows TDR 완화(`TdrDelay` 10~30)와 충분한 페이지 파일.

동적 길이 계산 가이드
- TTS 코드북(9개) 기준 초당 약 86스텝을 사용합니다. 과할당을 피하려면 텍스트 길이로 상한을 보수적으로 추정하세요.
- 권장식: `max_new_tokens = min(86*20, 86 * (len(chunk)//10 + 1))`
  - 최대 20초 상한을 두고, 글자수 10당 약 1초 분량으로 가정(사용 중 조정 가능).

세그먼트별 정리 패턴(표준 루프용)
- 세그먼트 생성 후 즉시 다음을 수행:
  - 지역 텐서와 큰 넘파이 배열 참조 해제: `del conditioning, codes, wav_out`
  - `torch.cuda.empty_cache()` 호출로 프리 캐시 반환
  - `gc.collect()` 호출로 파이썬 객체 수집

운영 팁
- `nvidia-smi -l 1`로 VRAM 추이와 프로세스별 사용량을 지속 모니터링.
- 드라이버 업데이트, 케이스 내부 온도/팬 상태 점검, 전원 공급(PSU) 안정성 확인.
