# Zonos-for-Windows

<div align="center">
<img src="assets/ZonosHeader.png" 
     alt="Alt text" 
     style="width: 500px;
            height: auto;
            object-position: center top;">
</div>

---

Zonos는 200,000시간 이상의 다양한 다국어 음성으로 훈련된 오픈 소스 텍스트-음성 변환(TTS) 모델로, 최고 수준의 TTS 제공업체에 필적하거나 이를 능가하는 표현력과 품질을 제공합니다.

이 모델은 화자 임베딩이나 오디오 프리픽스가 주어졌을 때 텍스트 프롬프트에서 자연스러운 음성을 생성할 수 있으며, 몇 초 분량의 참조 클립이 주어졌을 때 정확한 음성 클로닝을 수행할 수 있습니다. 또한 말하기 속도, 피치 변화, 오디오 품질, 행복, 두려움, 슬픔, 분노와 같은 감정에 대한 세밀한 제어가 가능합니다. 모델은 기본적으로 44kHz로 음성을 출력합니다.

---

Zonos는 eSpeak를 통한 텍스트 정규화 및 음소화, 트랜스포머 또는 하이브리드 백본을 통한 DAC 토큰 예측이라는 간단한 아키텍처를 따릅니다. 아래에서 아키텍처 개요를 확인할 수 있습니다.

<div align="center">
<img src="assets/ArchitectureDiagram.png" 
     alt="Alt text" 
     style="width: 1000px;
            height: auto;
            object-position: center top;">
</div>

---

## Windows Gradio 인터페이스 (권장)
PowerShell에서 `2、run_gradio.ps1`로 실행 (마우스 오른쪽 버튼 클릭 후 '파워쉘로 실행' 선택)

## 주요 기능

- 제로샷 TTS와 음성 클로닝: 원하는 텍스트와 10-30초 화자 샘플을 입력하여 고품질 TTS 출력 생성
- 오디오 프리픽스 입력: 더 풍부한 화자 매칭을 위해 텍스트와 오디오 프리픽스 추가. 화자 임베딩에서 복제하기 어려울 수 있는 속삭임과 같은 동작을 유도하는 데 사용 가능
- 다국어 지원: Zonos-v0.1은 영어, 일본어, 중국어, 프랑스어, 독일어, 한국어 등을 지원
- 오디오 품질 및 감정 제어: Zonos는 생성된 오디오의 많은 측면을 세밀하게 제어할 수 있습니다. 여기에는 말하기 속도, 피치, 최대 주파수, 오디오 품질, 행복, 분노, 슬픔, 두려움과 같은 다양한 감정이 포함됩니다.
- 빠른 속도: 모델은 RTX 4090에서 약 ~2x 실시간 속도로 실행됩니다(즉, 1초 컴퓨팅 시간당 2초 오디오 생성)
- Gradio 웹 UI: Zonos는 음성을 생성하는 사용하기 쉬운 gradio 인터페이스와 함께 제공됩니다.

## Windows 설치
  PowerShell이 스크립트를 실행할 수 있도록 제한 없는 스크립트 액세스 권한 부여:

- 관리자 PowerShell 창 열기
- `Set-ExecutionPolicy Unrestricted` 입력 후 A로 응답
- 관리자 PowerShell 창 닫기

### CUDA
이 저장소는 CUDA 12.4가 필요합니다.
https://developer.nvidia.com/cuda-12-4-1-download-archive?target_os=Windows&target_arch=x86_64

### MSVC
**C++ 컴파일러**가 포함된 [VS Studio 2022](https://visualstudio.microsoft.com/vs/)가 필요합니다.

### 원클릭 설치:
PowerShell에서 `1、install-uv-qinglong.ps1`로 실행 (마우스 오른쪽 버튼 클릭 후 '파워쉘로 실행' 선택) - 자동으로 한 번에 설치됩니다.
