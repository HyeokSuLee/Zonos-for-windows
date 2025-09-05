"""
새로운 메모리 효율적 generate_multi_speaker_audio 함수
"""

import asyncio
import numpy as np
import torch
import torchaudio
from zonos.conditioning import make_cond_dict

def generate_multi_speaker_audio(
    model_choice,
    dialogue_text,
    language,
    prefix_audio,
    e1, e2, e3, e4, e5, e6, e7, e8,
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
    progress,
):
    """
    메모리 효율적 다중 화자 음성 생성 함수
    """
    global SPEAKER_EMBEDDINGS, DEFAULT_SETTINGS, MEMORY_EFFICIENT_SYSTEM

    try:
        # 기본 입력 검증
        if not dialogue_text.strip():
            return (None, None), seed, "대화 텍스트를 입력해주세요."

        if not SPEAKER_EMBEDDINGS:
            return (None, None), seed, "적어도 하나의 화자를 추가해주세요."

        # 모델 로드 확인
        selected_model = load_model_if_needed(model_choice)
        if selected_model is None:
            return (None, None), seed, "모델을 로드할 수 없습니다."

        # UI 설정값 업데이트
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

        # 메모리 효율적 시스템 사용 여부 확인
        if MEMORY_EFFICIENT_SYSTEM is not None:
            print("🚀 메모리 효율적 시스템 사용")
            
            # 메모리 효율적 시스템 설정 업데이트
            MEMORY_EFFICIENT_SYSTEM.dialogue_parser.default_settings = DEFAULT_SETTINGS.copy()
            MEMORY_EFFICIENT_SYSTEM.update_speaker_embeddings(SPEAKER_EMBEDDINGS)
            
            # 진행률 콜백 함수
            def progress_callback(progress_val: float, message: str):
                progress(progress_val)
                
            try:
                # 메모리 효율적 시스템으로 오디오 생성
                result = asyncio.run(MEMORY_EFFICIENT_SYSTEM.generate_complete_audio(
                    dialogue_text=dialogue_text,
                    spacing_ms=200.0,  # 기본 간격
                    target_lufs=-23.0,  # 기본 정규화 목표
                    progress_callback=progress_callback
                ))
                
                # 최종 오디오 로드
                sample_rate, final_audio = MEMORY_EFFICIENT_SYSTEM.load_final_audio_for_ui(result)
                
                # 후처리 적용
                if normalize_audio_toggle:
                    final_audio = normalize_audio(final_audio)
                    
                if volume_adjustment != 0:
                    final_audio = apply_volume_adjustment(final_audio, volume_adjustment)
                
                # 생성 요약
                summary = MEMORY_EFFICIENT_SYSTEM.get_generation_summary(result)
                
                return (
                    (sample_rate, final_audio),
                    None,
                    summary
                )
                
            except Exception as e:
                print(f"메모리 효율적 시스템 오류: {e}")
                # 임시파일 정리
                MEMORY_EFFICIENT_SYSTEM.cleanup()
                return (None, None), seed, f"메모리 효율적 생성 실패: {str(e)}"
        
        else:
            print("⚠️ 레거시 시스템으로 fallback - 메모리 사용량이 높을 수 있습니다")
            
            # 기존 시스템 사용 (간소화된 버전)
            dialogue_parts = parse_dialogue(dialogue_text)

            if not dialogue_parts:
                return (None, None), seed, "유효한 대화를 찾을 수 없습니다."

            # 화자 검증
            unknown_speakers = []
            for speaker, _, _ in dialogue_parts:
                if speaker not in SPEAKER_EMBEDDINGS and speaker != "Unknown":
                    unknown_speakers.append(speaker)

            if unknown_speakers:
                return (
                    (None, None),
                    seed,
                    f"다음 화자를 찾을 수 없습니다: {', '.join(unknown_speakers)}"
                )

            # 간소화된 음성 생성 (청크 분할 없이)
            audio_segments = []
            total_parts = len(dialogue_parts)
            
            for i, (speaker_name, text, speaker_settings) in enumerate(dialogue_parts):
                progress(i / total_parts)
                
                try:
                    # 기본 설정과 화자별 설정 병합
                    merged_settings = DEFAULT_SETTINGS.copy()
                    merged_settings.update(speaker_settings)
                    
                    # 화자 임베딩 가져오기
                    if speaker_name in SPEAKER_EMBEDDINGS:
                        speaker_embedding = SPEAKER_EMBEDDINGS[speaker_name]
                    else:
                        speaker_name = list(SPEAKER_EMBEDDINGS.keys())[0]
                        speaker_embedding = SPEAKER_EMBEDDINGS[speaker_name]
                    
                    # 간단한 조건부 생성 (세부 파라미터 간소화)
                    emotion_tensor = torch.tensor([
                        merged_settings["emotion1"], merged_settings["emotion2"],
                        merged_settings["emotion3"], merged_settings["emotion4"],
                        merged_settings["emotion5"], merged_settings["emotion6"],
                        merged_settings["emotion7"], merged_settings["emotion8"]
                    ], device=device)
                    
                    vq_tensor = torch.tensor([merged_settings["vq_single"]] * 8, device=device).unsqueeze(0)
                    
                    cond_dict = make_cond_dict(
                        text=text[:500],  # 길이 제한
                        language=language,
                        speaker=speaker_embedding,
                        emotion=emotion_tensor,
                        vqscore_8=vq_tensor,
                        fmax=merged_settings["fmax"],
                        pitch_std=merged_settings["pitch_std"],
                        speaking_rate=merged_settings["speaking_rate"],
                        dnsmos_ovrl=merged_settings["dnsmos_ovrl"],
                        speaker_noised=merged_settings["speaker_noised"],
                        device=device,
                        unconditional_keys=list(unconditional_keys),
                    )
                    conditioning = selected_model.prepare_conditioning(cond_dict)
                    
                    # 음성 생성
                    codes = selected_model.generate(
                        prefix_conditioning=conditioning,
                        max_new_tokens=86 * 20,  # 길이 제한
                        cfg_scale=merged_settings["cfg_scale"],
                        batch_size=1,
                        sampling_params=dict(linear=float(linear), conf=float(confidence), quad=float(quadratic))
                    )
                    
                    wav_out = selected_model.autoencoder.decode(codes).cpu().detach()
                    if wav_out.dim() == 2 and wav_out.size(0) > 1:
                        wav_out = wav_out[0:1, :]
                    
                    audio_segments.append(wav_out.squeeze().numpy())
                    
                except Exception as e:
                    print(f"세그먼트 {i} 생성 실패: {e}")
                    audio_segments.append(np.zeros(24000))  # 0.5초 무음
            
            # 오디오 합성
            if audio_segments:
                try:
                    # 간단한 정규화 후 합성
                    normalized_segments = normalize_all_chunks_to_loudest(audio_segments, first_chunk_boost_db=3.0)
                    final_audio = np.concatenate(normalized_segments, axis=0)
                    
                    # 후처리
                    if normalize_audio_toggle:
                        final_audio = normalize_audio(final_audio)
                        
                    if volume_adjustment != 0:
                        final_audio = apply_volume_adjustment(final_audio, volume_adjustment)
                    
                    progress(1.0)
                    
                    return (
                        (48000, final_audio),
                        None,
                        f"레거시 모드로 생성 완료! {len(normalized_segments)}개 세그먼트 ({len(final_audio)/48000:.1f}초)"
                    )
                    
                except Exception as e:
                    return (None, None), seed, f"오디오 합성 실패: {str(e)}"
            else:
                return (None, None), seed, "생성된 오디오가 없습니다."

    except Exception as e:
        print(f"음성 생성 중 오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # 메모리 효율적 시스템 정리 (필요시)
        if MEMORY_EFFICIENT_SYSTEM is not None:
            try:
                MEMORY_EFFICIENT_SYSTEM.cleanup()
            except:
                pass
                
        return (None, None), seed, f"음성 생성 중 오류 발생: {str(e)}"