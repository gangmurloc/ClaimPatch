# DECAP

[English README](README.md)

**변화하는 LLM 답변을 위한 의존성 완전 의미 패칭**

DECAP은 기존 장문 답변의 근거가 바뀌었을 때 직접 수정 대상뿐 아니라
그에 의존하는 파생 주장까지 갱신하면서, 여전히 유효한 주장은 보존할 수
있는지를 연구하는 프로토타입입니다. 답변을 버전이 있는 claim graph로
표현하고, 변경 영향을 판정해 실행 가능한 semantic patch를 만든 뒤
transactional executor로 적용합니다.

이 저장소는 연구 작업 공간에서 공개에 필요한 부분만 추린 아티팩트입니다.
실행 코드, 버전 관리된 프롬프트, 합성 벤치마크 생성기, 대표 집계 결과,
53개 테스트를 포함하며 원시 모델 출력·체크포인트·캐시·탐색 로그는
제외합니다.

## 핵심 아이디어

```mermaid
flowchart LR
    E[근거 변경] --> I[직접 영향 탐지]
    I --> G[의존성 폐쇄]
    G --> V[의미 재검증]
    V --> P[실행 가능한 패치]
    P --> T[트랜잭션 실행]
    T --> A[버전 갱신 답변]
```

전체 답변을 다시 생성하면 불필요한 문장까지 바뀔 수 있고, 직접 언급된
문장만 고치면 파생 비교나 인용이 오래된 상태로 남을 수 있습니다. DECAP은
이 문제를 자유 생성이 아니라 의존성과 트랜잭션의 문제로 다룹니다.

## 대표 결과

Qwen2.5-7B-Instruct를 사용해 합성 인스턴스 100개에 순차 근거 변경을
3회씩 적용한 300-step 진단 결과입니다. 이는 통제된 합성 실험이며
현실 세계의 일반화를 증명하지 않습니다.

| 시스템 | DCS | 패치 정밀도 | 패치 재현율 | 불필요 편집 | 미수정 잔존 |
|---|---:|---:|---:|---:|---:|
| DECAP prompted | **0.820** | **0.835** | **0.926** | **0.183** | **0.074** |
| 비구조 선택 편집 | 0.500 | 0.780 | 0.807 | 0.223 | 0.193 |
| 그래프 없는 속성 기반 | 0.250 | 1.000 | 0.662 | 0.000 | 0.338 |
| 모든 후손 갱신 | 1.000 | 0.745 | 1.000 | 0.273 | 0.000 |
| 전체 재생성 | 1.000 | 0.505 | 1.000 | 1.000 | 0.000 |

비구조 선택 편집 대비 DCS 차이는 **+0.320**, 95% bootstrap 구간은
**[0.257, 0.380]**입니다. 모든 후손을 무조건 수정하는 정책은 DCS가 더
높지만 불필요한 편집도 더 많습니다. DECAP의 목표는 의존성 누락과
과잉 수정을 함께 줄이는 것입니다.

## 설치 및 빠른 실행

```bash
git clone https://github.com/gangmurloc/DECAP.git
cd DECAP
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

decap run-p0 --config configs/experiments/p0_rule_based.yaml --limit 4
pytest -q
```

로컬 LLM 실험에는 GPU와 이미 내려받은 모델 가중치가 필요합니다.

```bash
pip install -e ".[dev,local]"
CUDA_VISIBLE_DEVICES=0 sh scripts/run_p1_local_full100x3.sh
```

## 정직한 범위

- 주 벤치마크는 정답 claim graph를 가진 합성 데이터입니다.
- 자연어 prose-to-graph 경로는 파일럿이며 외부 검증이 아닙니다.
- 구조화 메타데이터를 강하게 제거하면 보존 정밀도와 과잉 편집이 악화됩니다.
- Llama 실험은 특정 실패 패턴의 재현이지 전체 방법의 다중 backbone 검증이
  아닙니다.

세부 내용은 [research status](docs/research_status.md)와
[reproducibility](docs/reproducibility.md)를 참고하세요.

## 저자

**Ganggil Lee** — Undergraduate Researcher, NLP Laboratory, Hallym University

## 라이선스

아직 오픈소스 라이선스를 선택하지 않았습니다. 공개 저장소라는 사실만으로
코드의 복제·수정·재배포 권한이 부여되지는 않습니다. 모델 가중치는 포함하지
않으며 각 제공자의 이용 조건을 따라야 합니다.

