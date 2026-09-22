# stable-worldmodel 전수조사 & 활용 분석 (한국어 정리)

> 작성일: 2026-09-22
> 작성: Claude Code (카리나 페르소나) · 요청자: mono7594@gmail.com

---

## 🔗 관련 링크

| 구분 | URL |
|------|-----|
| **내 포크 저장소** | https://github.com/bmshin94/stable-worldmodel |
| **원본(Upstream) 저장소** | https://github.com/galilai-group/stable-worldmodel |
| 공식 문서 | https://galilai-group.github.io/stable-worldmodel/ |
| 논문 (arXiv) | https://arxiv.org/abs/2605.21800 |
| PyPI 패키지 | https://pypi.org/project/stable-worldmodel/ |
| Colab 노트북 | https://colab.research.google.com/github/galilai-group/stable-worldmodel/blob/main/scripts/notebooks/train_from_hf_buckets.ipynb |
| 이슈 트래커 | https://github.com/galilai-group/stable-worldmodel/issues |

---

## 1. 프로젝트 개요

### 한 줄 요약

**`stable-worldmodel`(줄여서 `swm`)은 "월드모델(World Model)" AI 연구를 위한 올인원 실험 플랫폼**이다.

- 이 저장소(`bmshin94/stable-worldmodel`)는 **포크**이며, 원본은 `galilai-group/stable-worldmodel`이다.
- 저자진에 **Yann LeCun**(2018 튜링상, Meta 수석 AI 과학자)이 포함되어 있다.
- 라이선스: **MIT** (상업적 이용 자유)
- 패키지 버전: PyPI `v0.1.1` (2026-06-06 릴리스, 총 14회 릴리스)

### 월드모델이란?

> *"A world model is a learned simulator that predicts how an environment evolves in response to actions,
> enabling agents to plan by imagining future outcomes."* — `docs/index.md`

AI가 머릿속에 만든 **상상 시뮬레이터**. 행동하기 전에 결과를 예측해서 최적의 행동을 고른다.
이 "머릿속 시뮬레이션으로 최적 행동 찾기"를 **MPC(Model Predictive Control, 모델 예측 제어)** 라고 한다.

| 방식 | 학습 과정 | 새로운 환경 대응 |
|------|----------|-----------------|
| 기존 강화학습(RL) | 수백만 번 시행착오 | 처음부터 재학습 |
| **월드모델** | 세상의 동작 원리를 예측 모델로 학습 | 물리법칙 공유 → 빠른 적응 |

---

## 2. 저장소 실측 통계 (2026-09-22 GitHub API 확인)

| 지표 | 수치 |
|------|------|
| ⭐ 스타 | **2,211** |
| 🍴 포크 | **276** |
| 🐛 열린 이슈 | 25 |
| 📅 생성일 | 2025-06-27 (약 1년 3개월) |
| 🔄 마지막 푸시 | 2026-09-08 (활발한 유지보수) |
| 🏷️ 토픽 | `deep-learning`, `jepa`, `model-predictive-control`, `pytorch`, `world-model` |
| 📦 PyPI | v0.1.1 / 총 14 릴리스 / Python ≥3.10 |
| ⚖️ 라이선스 | MIT |

> 월 평균 약 147스타 증가 — 연구용 라이브러리로는 상당히 좋은 성장세.

---

## 3. 코드베이스 전수조사 결과

### 전체 구조

```
stable-worldmodel/
├── stable_worldmodel/     # 핵심 라이브러리 본체
├── scripts/               # 실행 스크립트 (데이터수집/학습/평가/시각화/벤치마크)
├── tests/                 # 테스트 1,163개
├── docs/                  # MkDocs 기반 공식 문서
├── .github/workflows/     # CI (Python 3.10/3.11/3.12 매트릭스)
├── pyproject.toml         # 패키지 설정 (uv 빌드 백엔드)
└── CLAUDE.md              # 카리나 페르소나 설정
```

- 총 파일 수: **351개**
- 코드 규모: 약 **3만 줄** (Python)

### 핵심 모듈별 분석

| 모듈 | 규모 | 역할 |
|------|------|------|
| `world/` | 909줄 | `World` 클래스 — 메인 진입점. N개 환경 병렬 실행(`EnvPool`), 롤아웃 루프 |
| `data/` | 5,723줄 | 5가지 저장 포맷 레지스트리, 정규화, 리플레이 버퍼 |
| `envs/` | 약 16,000줄 | 30개+ 표준 환경 + Atari 100종 |
| `planning/` | 3,149줄 | 7가지 MPC 솔버 + 비용함수(objective) |
| `wm/` | 3,369줄 | 월드모델 베이스라인 7종 구현 |
| `wrapper/` | 1,215줄 | 전처리 파이프라인 + 시각 교란 12종 |
| `spaces.py` | 824줄 | **FoV(Factors of Variation) 시스템** |
| `cli.py` | 1,030줄 | `swm` 터미널 명령어 |

### ① `world/` — 시뮬레이션 실행 엔진

```python
world = swm.World('swm/PushT-v1', num_envs=8)   # 환경 8개 동시 실행
world.set_policy(policy)
world.collect('data.lance', episodes=100)        # 데이터 수집
results = world.evaluate(episodes=50)            # 평가
```

- 끝난 환경은 **마스킹으로 건너뛰어** GPU 낭비 방지
- `evaluate()`는 데이터셋 기반 start/goal 평가도 지원

### ② `data/` — 저장 포맷 레지스트리 (5종)

| 포맷 | 특징 |
|------|------|
| `lance` | **기본값**. LanceDB 컬럼형 DB, append 친화, 고속 인덱스 읽기 |
| `hdf5` | 단일 `.h5` 파일, 이식성 우수 |
| `video` | `.npz` + 에피소드당 MP4, 압축률 최고 |
| `folder` | `.npz` + 스텝당 JPEG, 육안 검사용 |
| `lerobot` | HuggingFace LeRobot 데이터셋 읽기 전용 어댑터 |

**README 벤치마크 (PushT 데이터셋 기준)**

| 포맷 | 소스 | samples/s | 용량 |
|------|------|-----------|------|
| HDF5 | local | 1,416.1 | 43.12 GB |
| **LanceDB** | local | **4,814.8** | **13.31 GB** |
| Video | local | 1,330.6 | **496.29 MB** |
| HDF5 | **s3** | **9.1** | — |
| **LanceDB** | **s3** | **3,183.7** | — |

> 클라우드(S3) 환경에서 HDF5 대비 **약 350배** 처리량 차이. 클라우드 학습 시 결정적 이점.

`register_format()`으로 커스텀 포맷 등록 가능 (플러그인 구조).

### ③ `envs/` — 표준 환경 30종+

- **DeepMind Control Suite**: Cheetah, Walker, Hopper, Humanoid, Quadruped, Finger, Manipulator 등
- **Gymnasium Classic Control**: CartPole, MountainCar, Acrobot, Pendulum
- **Gymnasium Robotics**: FetchReach / Push / Slide / PickAndPlace
- **OGBench**: Cube, Scene
- **Craftax**: 마인크래프트 유사 서바이벌
- **PushT / TwoRoom**: 월드모델 표준 벤치마크
- **ALE**: Atari 100종+
- **자체 제작**: `rocket_landing`, `piecewise`, `image_positioning`

모두 `swm/*` 네이밍으로 통일 → Gymnasium 인터페이스만 맞추면 새 환경 추가 가능.

### ④ FoV (Factors of Variation) — 이 라이브러리의 킬러 기능

`spaces.py`에 구현된, **환경마다 독립적으로 조절 가능한 변수 시스템**.

| 환경 | FoV 개수 |
|------|---------|
| `swm/TwoRoom-v1` | **17** |
| `swm/PushT-v1` | **16** |
| `swm/OGBScene-v0` | 12 |
| `swm/FetchPush-v3` | 11 |
| `swm/AcrobotControl-v1` | 11 |

```python
world.reset(options={'variation': ['color', 'lighting', 'friction']})
```

**왜 중요한가**: AI가 "이해"한 것인지 "암기"한 것인지 검증할 수 있다.
빨간 큐브로만 학습한 모델이 파란 큐브에서 실패하는 일반화 실패를
**한 줄로 Zero-shot / OOD(분포 외) 평가**할 수 있다.

`wrapper/visual.py`에는 시각 교란 12종도 있다:
노이즈, 블러, 가림(Occlusion), 컬러지터, 흑백, 해상도 변경, 이동 패치, 랜덤 시프트, Cutout, RandomConv, 크로마키.

### ⑤ `planning/` — MPC 솔버 7종

| 솔버 | 타입 |
|------|------|
| CEM (Cross-Entropy Method) | 샘플링 |
| iCEM (Improved CEM) | 샘플링 |
| MPPI (Model Predictive Path Integral) | 샘플링 |
| Predictive Sampling | 샘플링 |
| SGD / Adam | 그래디언트 |
| PGD (Projected Gradient Descent) | 그래디언트 |
| Augmented Lagrangian | 제약 최적화 |

**CEM 동작 원리 (비유)**
1. 무작위 행동 후보 300개 샘플링
2. 각각 월드모델로 평가
3. 상위 30개(elite) 평균·분산 계산
4. 그 분포로 다시 샘플링 → 30회 반복 → 최적해 수렴

비용함수(`objective.py`): `GoalMSE`, `ControlPenalty`, `WeightedSum` (가중합 조합 가능)

### ⑥ `wm/` — 월드모델 베이스라인 7종

| 모델 | 타입 | 비고 |
|------|------|------|
| PreJEPA | JEPA | DINO-WM (arXiv:2411.04983) 재현 |
| LeWM | JEPA | LeWorldModel 공식 구현 |
| PLDM | JEPA | Planning with Latent Dynamics |
| TD-MPC2 | 모델기반 RL | SOTA 알고리즘 |
| GCBC | Behavior Cloning | 목표조건부 |
| GCIVL | RL | 목표조건부 |
| GCIQL | RL | 목표조건부 |

백본 선택 가능: **DINOv2, DINOv3, SigLIP2, MAE**

### ⑦ `cli.py` — 터미널 도구

```bash
swm datasets                                        # 캐시된 데이터셋 목록
swm inspect pusht_expert_train                      # 데이터셋 내부 확인
swm preview pusht_expert_train                      # 영상 미리보기
swm envs                                            # 등록된 환경 목록
swm fovs PushT-v1                                   # 해당 환경의 FoV 목록
swm checkpoints                                     # 체크포인트 목록
swm convert pusht_expert_train --dest-format video  # 포맷 변환
swm merge                                           # 데이터셋 병합
```

### 코드 품질 평가

| 항목 | 상태 |
|------|------|
| 테스트 | **1,163개** |
| CI | GitHub Actions, Python 3.10/3.11/3.12 매트릭스 + PyPI 설치 검증 |
| 린터 | Ruff (line-length 79) + pre-commit |
| 문서 | MkDocs(shadcn), mkdocstrings API 자동생성, 튜토리얼 |
| 설계 | Lazy import(PEP 562), Protocol 인터페이스, 포맷 레지스트리, Hydra config |

> 연구용 코드로는 **최상위권 엔지니어링 품질**. "제품" 수준.

---

## 4. 설치 및 사용법

### 설치

```bash
# ① 가벼움 — 솔버 + 모델만 (임베디드 배포용)
pip install stable-worldmodel

# ② 추천 — 기본 + Lance 데이터 I/O (+약 410MB)
pip install 'stable-worldmodel[data]'

# ③ 풀세트 — 학습 + 환경 30종 + 모든 포맷 (수 GB)
pip install 'stable-worldmodel[all]'

# LeRobot 연동 (Python 3.12+ 필요)
pip install 'stable-worldmodel[lerobot]'
```

### 개발용 설치

```bash
git clone https://github.com/galilai-group/stable-worldmodel
cd stable-worldmodel
uv venv --python=3.10 && source .venv/bin/activate
uv sync --extra all --group dev
```

### 데이터 저장 경로 설정 (중요)

```bash
export STABLEWM_HOME=/충분한/디스크/경로   # 기본값: ~/.stable_worldmodel/
```

### 시스템 요구사항

| 항목 | 요구 |
|------|------|
| Python | 3.10 / 3.11 / 3.12 |
| OS | Linux 권장(MuJoCo/OSMesa), macOS 가능, Windows는 WSL |
| GPU | 학습 시 필수. 논문 재현은 H100/H200급 |
| 디스크 | 최소 20GB, 권장 100GB |

### 기본 워크플로우

```python
import stable_worldmodel as swm
from stable_worldmodel.policy import WorldModelPolicy, PlanConfig
from stable_worldmodel.planning import CEMSolver

# 1. 데이터 수집
world = swm.World('swm/PushT-v1', num_envs=8, image_shape=(64, 64))
world.set_policy(expert_policy)
world.collect('data/pusht.lance', episodes=100, seed=0,
              options={'variation': ['all']})

# 2. 데이터 로딩 (포맷 자동 감지)
dataset = swm.data.load_dataset('data/pusht.lance', num_steps=16)
world_model = ...   # 내 모델

# 3. MPC 평가
solver = CEMSolver(cost=world_model, num_samples=300)
policy = WorldModelPolicy(solver=solver,
                          config=PlanConfig(horizon=10, receding_horizon=5))
world.set_policy(policy)
results = world.evaluate(episodes=50)
print(f"Success Rate: {results['success_rate']:.1f}%")
```

### 준비된 학습 스크립트 (Hydra config 기반)

```bash
python scripts/train/lewm.py     data=pusht      # LeWM
python scripts/train/prejepa.py  data=pusht      # DINO-WM 재현
python scripts/train/tdmpc2.py   data=dmc        # TD-MPC2
python scripts/train/gcbc.py     data=ogb        # Behavior Cloning

# 백본 교체
python scripts/train/prejepa.py world_model/backbone=dinov3_small

# 평가
python scripts/plan/eval_wm.py --config-name=pusht solver=cem
```

> 설치 없이 체험하려면 README의 **Colab 배지**를 누르면 된다.
> HuggingFace 버킷에서 데이터 다운로드 없이 바로 스트리밍 학습이 가능하다.

---

## 5. 자주 묻는 질문 정리

### Q. 플러그인? 스킬? MCP?

**셋 다 아니다. 순수 Python 라이브러리(패키지)다.**

| 구분 | 정체 | 사용 주체 | 해당 여부 |
|------|------|----------|----------|
| Python 라이브러리 | `pip install` 후 `import` | 사람 개발자 | ✅ **정답** |
| MCP 서버 | LLM ↔ 외부도구 통신 규약 | AI 에이전트 | ❌ |
| Skill | Claude용 지침 문서(SKILL.md) | Claude | ❌ |
| 플러그인 | 호스트 앱 확장 | 호스트 앱 | ❌ |

근거: `pyproject.toml`에 `mcp` 의존성 없음, `SKILL.md`/`.claude/` 없음, 플러그인 매니페스트 없음.
의존성은 `torch`, `torchvision`, `numpy`, `gymnasium`, `hydra-core` 등 순수 딥러닝 스택.

> 참고: 이름의 `stable-` 접두사는 같은 팀의 `stable-pretraining` 라이브러리와 세트다
> (`[train]` extra에 `stable-pretraining>=0.1.8`이 포함됨).
>
> 단, **이걸 MCP 서버로 감싸는 것은 가능하며, 유망한 수익화 아이디어다.**

### Q. API 토큰이 필요한가?

**기본 사용에는 전혀 필요 없다. 100% 무료, 100% 로컬 실행.**

| 항목 | 필요 여부 |
|------|----------|
| OpenAI / Anthropic API 키 | ❌ 코드에 존재하지 않음 |
| 유료 구독 / 회원가입 | ❌ |
| 클라우드 계정 | ❌ |

**선택적으로 필요할 수 있는 것 (3개)**

1. **`HF_TOKEN`** — 비공개 HuggingFace 데이터셋 읽을 때만.
   실제 구현(`data/formats/lance.py:1391`):
   ```python
   if str(path).startswith('hf://') and os.environ.get('HF_TOKEN'):
       opts['token'] = os.environ['HF_TOKEN']
   ```
   `hf://` 경로 + 환경변수가 둘 다 있을 때만 사용. 공개 데이터셋은 토큰 불필요.
2. **`WANDB_API_KEY`** — 학습 대시보드(선택). `WANDB_MODE=disabled`로 끌 수 있음.
3. **AWS 자격증명** — Lance 데이터를 `s3://`에 둘 때만.

**보안 점검 결과**: 하드코딩된 키 없음, 전부 환경변수로만 읽음, `.gitignore` 설정 양호.

### Q. 왜 GitHub에서 유명한가? (7가지 이유)

1. **Yann LeCun 이름값** — 튜링상 수상자. "LLM으로는 AGI 못 간다, World Model이 답"이라는 그의 주장(JEPA)의 공식 실험 플랫폼 격.
2. **타이밍** — 2024~2026 AI 트렌드가 LLM 스케일링 한계론 → 월드모델/피지컬 AI로 이동 중.
3. **진짜 Pain Point 해결** — `docs/index.md`: *"each new article re-implements over and over the same baselines, evaluation protocols, and data processing logic."*
4. **압도적 엔지니어링 품질** — 테스트 1,163개, 3버전 CI, 완비된 문서.
5. **임팩트 있는 벤치마크** — "S3에서 HDF5 9.1 vs Lance 3,183 samples/s" (재현 스크립트 제공).
6. **README 완성도** — 환경 GIF 30개+(기본 vs 변형 비교), 배지 8개, Colab 원클릭.
7. **후속 연구 존재** — "Built on stable-worldmodel"에 **C-JEPA**, **LeWM** 등재.

### Q. 로컬 에이전트 구축에 도움이 되나?

| 만들려는 것 | 추천도 | 이유 |
|-----------|--------|------|
| 로봇 / 드론 / 제어 에이전트 | ⭐⭐⭐⭐⭐ | 바로 활용 가능 |
| 게임 플레이 AI | ⭐⭐⭐⭐⭐ | Atari / Craftax 준비됨 |
| 시뮬레이션 최적화 | ⭐⭐⭐⭐ | 솔버 7종 활용 |
| LLM 에이전트 (설계 참고) | ⭐⭐⭐ | 아키텍처 패턴 학습용으로 최고 |
| LLM 챗봇 (직접 사용) | ⭐ | 도메인이 완전히 다름 |

**LLM 에이전트를 만들더라도 배워갈 설계 패턴**

- **레지스트리 패턴** (`data/format.py`) — `register_format()` → 에이전트 툴/플러그인 시스템에 응용
- **Protocol 인터페이스** (`planning/solver/solver.py`) — 상속 없는 느슨한 결합
- **Lazy Import (PEP 562)** (`__init__.py`) — CLI 시작 속도 대폭 개선 (torch import만 수 초)
- **콜백 시스템** (`planning/solver/callbacks/`) — 실행 중 로깅/개입/중단
- **Hydra config** (`scripts/*/config/`) — YAML 조합 관리 + CLI 오버라이드

**개념적 연결점**: CEM의 "후보 N개 생성 → 평가 → 상위 선택 → 재탐색"은
LLM 에이전트의 **Tree-of-Thought / Best-of-N / Self-Consistency**와 구조가 동일하다.
`objective.py`의 `WeightedSum`(다중 목적함수 가중합)도 에이전트 보상 설계에 그대로 응용된다.

또한 LanceDB는 **RAG용 벡터 검색 DB로도 널리 쓰이므로**, 여기서 익힌 지식이 직접 재활용된다.

### Q. React나 PHP로 만들 수 있나?

**A-1. 라이브러리 자체를 React/PHP로 포팅 → 현실적으로 불가능하며, 그럴 이유도 없다.**

| 필요 기능 | Python | JavaScript | PHP |
|----------|--------|-----------|-----|
| 딥러닝 프레임워크 | PyTorch ✅ | TF.js (제한적) | ❌ |
| GPU 텐서 연산 | CUDA ✅ | WebGPU 초기 | ❌ |
| 물리 시뮬레이터 | MuJoCo, PyMunk ✅ | 제한적 | ❌ |
| DM Control / Gymnasium | ✅ | ❌ | ❌ |
| DINOv2/v3, SigLIP | ✅ | 변환 필요 | ❌ |

핵심 문제: MuJoCo, DeepMind Control Suite, OGBench, Craftax(JAX)는 전부 C/C++/Python 네이티브이며
JS/PHP 포팅본이 존재하지 않는다. PHP는 GPU 연산 자체가 불가능하다.

**A-2. React/PHP로 감싸는 웹 서비스 → 완전히 가능하며, 최고의 수익화 루트다.**

```
┌─────────────────────────────────────────┐
│  React (프론트엔드)                      │
│  - 실험 대시보드 / 성능 그래프            │
│  - FoV 슬라이더 (색·조명·마찰 실시간 조절)│
│  - 롤아웃 영상 플레이어 / 리더보드        │
└──────────────┬──────────────────────────┘
               │ REST / WebSocket
┌──────────────▼──────────────────────────┐
│  백엔드 API                              │
│  A: FastAPI (Python) ← 추천              │
│  B: PHP(Laravel) + Python 워커           │
└──────────────┬──────────────────────────┘
               │ 작업 큐 (Celery / Redis)
┌──────────────▼──────────────────────────┐
│  Python GPU 워커                         │
│  stable-worldmodel 그대로 실행            │
└─────────────────────────────────────────┘
```

| 언어 | 역할 |
|------|------|
| React | UI/UX 전부 — 대시보드, 차트, 영상, 인터랙티브 컨트롤 |
| Python | 연산 코어 — GPU, 시뮬레이션, 학습 (수정 불필요) |
| PHP | 비즈니스 로직 — 회원관리, 결제, 구독, 관리자 |

> 처음이라면 **React + FastAPI** 조합 권장 (Python 단일 스택이라 단순).
> PHP에 익숙하면 PHP(회원/결제) + FastAPI(실험 API) 하이브리드도 가능.

**React로 만들면 임팩트 있는 기능 3가지**
1. FoV 실시간 슬라이더 — 색/조명/마찰 조절 시 환경과 성공률이 실시간 변화 (데모 바이럴 포인트)
2. 롤아웃 비교 플레이어 — 모델 A vs B 나란히 재생 + 프레임별 비용함수 오버레이
3. 리더보드 — 환경 × 모델 × 솔버 조합별 성공률 매트릭스

---

## 6. 수익화 아이디어 (MIT 라이선스 → 상업적 이용 자유)

> 전제: 이 분야는 **틈새 B2B/연구 시장**이다. 대중 소비자 시장은 아니지만
> **고객 단가가 높고 경쟁이 적다**. 아래 금액은 업계 일반 가격대 기반 **참고 시나리오**이며
> 보장된 수치가 아니다.

### 종합 비교표

| # | 아이디어 | 난이도 | 초기자본 | 수익규모 | 수익화 속도 | 종합 평가 |
|---|---------|--------|---------|---------|-----------|----------|
| 4 | 교육 콘텐츠 | ⭐⭐ | 0원 | 중 | 빠름 | **초보 1순위** |
| 3 | MCP 서버 + AI 연구 에이전트 | ⭐⭐⭐⭐ | 중 | 대 | 중간 | **트렌드 1순위** |
| 2 | 산업 제어 B2B | ⭐⭐⭐⭐ | 낮음 | 대 | 느림 | **수익성 1순위** |
| 5 | 데이터셋/체크포인트 마켓 | ⭐⭐⭐ | 중 | 중~대 | 중간 | 틈새 공략 |
| 1 | SaaS 클라우드 플랫폼 | ⭐⭐⭐⭐⭐ | 높음 | 초대형 | 매우 느림 | 장기 승부 |

---

### 아이디어 1. WorldModel Cloud — 매니지드 실험 플랫폼

> "월드모델 연구계의 Weights & Biases + Modal"

**해결하는 고통**: GPU 클러스터 세팅, MuJoCo/OSMesa 설치 지옥, 실험 관리 수작업, 결과 공유 어려움

**제품 구성**
- 실험 생성 마법사 (환경/모델/솔버 드롭다운)
- FoV 인터랙티브 컨트롤 (슬라이더)
- 실시간 학습 모니터링 (loss, success rate)
- 롤아웃 영상 갤러리 + A/B 비교 뷰
- 베이스라인 6종 대비 리더보드
- 논문용 LaTeX 표 / 고화질 그래프 export

**가격 모델**

| 티어 | 가격 | 대상 |
|------|------|------|
| Free | $0 (CPU, 월 5실험) | 학생, 체험 |
| Researcher | $49/월 | 대학원생 개인 |
| Lab | $499/월 (5석) | 연구실 |
| Enterprise | $5,000+/월 (온프레미스, SLA) | 기업 R&D |
| GPU 사용량 | 시간당 과금 (마진 20~30%) | 전체 |

**3년 목표 시나리오**: 랩 200곳 × $499 = 월 $100K

**기술스택**: React + TypeScript + Recharts / FastAPI + PostgreSQL + Redis + Celery / Kubernetes + GPU 노드
**개발**: 6~12개월, 2~4명

**리스크**: GPU 비용이 매출을 초과할 수 있음. W&B/Comet과 경쟁.
**차별화**: **월드모델 전용 + FoV 일반화 평가** — 범용 MLOps 툴이 제공할 수 없는 영역.

---

### 아이디어 2. 산업 제어 솔루션 (B2B) — 수익성 1순위

> 핵심 통찰: `planning/solver/`의 **MPPI, CEM, Augmented Lagrangian은 원래 산업 제어의 표준 알고리즘**이다.
> 월드모델을 떼고 "공정 예측 모델"을 끼우면 그대로 공장에서 작동한다.

**타겟 산업**

| 산업 | 적용 | 고객 가치 |
|------|------|----------|
| 화학/정유 | 반응기 온도·압력 최적 제어 | 수율 +2% = 연 수십억 |
| 데이터센터 | 냉각 시스템 최적화 | 전력비 10~30% 절감 |
| 배터리 공장 | 공정 파라미터 튜닝 | 불량률 감소 |
| 물류창고 | 로봇 경로 최적화 | 처리량 증가 |
| 스마트빌딩 | HVAC 에너지 관리 | 전기료 절감 |
| 농업 | 자동 수확 로봇 | 인건비 절감 |

> 참고: DeepMind가 구글 데이터센터 냉각에 유사 기술을 적용해 40% 전력 절감을 달성한 사례가 있다.

**비즈니스 모델**
1. PoC (개념검증): 3,000만 ~ 1억원 / 2~3개월
2. 파일럿 구축: 1억 ~ 3억원 / 6개월
3. 본 도입: 3억 ~ 10억원
4. 유지보수 구독: 연 계약금액의 20%
5. 성과 공유(선택): 절감액의 10~20%

**시나리오**: 1년차 1~2억 → 2년차 5~8억 → 3년차 10~20억

**장점**: 초기 자본 거의 불필요(인건비 위주), 고객 단가 높음, 경쟁자 적음
**단점**: 영업 사이클 6~12개월, 도메인 전문가/파트너 필요 (성공의 80%가 영업력)

---

### 아이디어 3. MCP 서버 + AI 연구 자동화 에이전트 — 트렌드 1순위

> "Claude가 직접 실험을 돌려주는 AI 연구 조수"

**동작 예시**
```
사용자: "PushT에서 DINO-WM과 LeWM 성능 비교해줘. 조명/색깔 변화도 테스트하고."
   ↓ Claude가 MCP 툴 호출
   swm_collect(env="swm/PushT-v1", episodes=100)
   swm_train(model="prejepa", data="pusht")
   swm_train(model="lewm", data="pusht")
   swm_evaluate(fov=["color", "lighting"], episodes=50)
   ↓
Claude: "LeWM이 기본 환경에서 3% 높지만, 조명 변화 시 DINO-WM이 12% 더 강건합니다."
```

**3단계 진화**

| 단계 | 제품 | 수익 모델 |
|------|------|----------|
| 1 | 오픈소스 MCP 서버 배포 | $0 (인지도 확보) |
| 2 | 호스팅 버전 (GPU 포함) | $99~499/월 |
| 3 | 자율 AI 연구 에이전트 | $999+/월 |

**시나리오**: 개인 500명 × $99 + 기업 20곳 × $999 = 월 약 $69,500

**난이도** ⭐⭐⭐⭐ / **MCP 서버만이면 1~2개월, 1인 개발 가능** → 빠른 시장 검증에 유리
**매력 포인트**: AI 에이전트 + 월드모델이라는 두 최신 트렌드의 교차점. 마케팅 스토리가 강력함.

---

### 아이디어 4. 교육 콘텐츠 — 초보 1순위 (가장 현실적)

**장점**: 초기 자본 0원, 1인 시작 가능, 실패해도 지식이 남음, 다른 아이디어의 마케팅 자산이 됨

**상품 라인업**

| 상품 | 가격 | 제작 기간 | 플랫폼 |
|------|------|----------|--------|
| YouTube 시리즈 | 무료 (광고+유입) | 지속 | YouTube |
| 전자책 "월드모델 실전 입문" | $29~49 | 2~3개월 | Gumroad, 리디 |
| 온라인 강의 (10~15시간) | $99~299 | 3~4개월 | Udemy, 인프런 |
| 기업 워크샵 (2일) | $3,000~10,000/회 | — | 직접 영업 |
| 유료 뉴스레터 | $10/월 | 주간 | Substack |

**수익 시나리오**

| 시나리오 | 구성 | 연 수익 |
|---------|------|---------|
| 보수적 | 강의 200명 × $99 | 약 2,600만원 |
| 중간 | 강의 500명 + 책 300권 | 약 8,000만원 |
| 낙관적 | 강의 2,000명 + 워크샵 10회 | 약 3억원 |

**마케팅 전략**
1. "FoV 슬라이더 데모" 같은 시각적 콘텐츠로 바이럴
2. 원본 저장소에 PR 기여 → 컨트리뷰터 타이틀로 신뢰도 확보
3. **한국어 콘텐츠가 사실상 전무 → 블루오션**
4. 커뮤니티(r/MachineLearning, 긱뉴스 등) 공유

---

### 아이디어 5. 데이터셋 / 체크포인트 마켓플레이스

> "월드모델계의 HuggingFace Hub"

로봇 데이터 수집에는 시간·GPU·전문가 정책이 모두 필요하다.
`world.collect()` + FoV 시스템으로 **조건이 정밀하게 통제된 고품질 데이터셋**을 대량 생산할 수 있다.

| 상품 | 가격 |
|------|------|
| 프리미엄 데이터셋 (환경 × FoV 조합, 정밀 라벨) | $99~999/개 |
| 사전학습 체크포인트 (DINO-WM, LeWM 등) | $199~1,999/개 |
| 커스텀 데이터 생성 서비스 (주문 제작) | $2,000~20,000/건 |
| 벤치마크 인증 리포트 | $500~5,000/건 |

**숨은 기회: OOD 강건성 인증 서비스**

> "당신의 로봇 AI는 조명이 바뀌어도 작동합니까?"
> → 17가지 변화요인별 강건성 테스트 → 인증서 발급

자율주행·로봇 기업은 안전성 인증이 필수이며, 규제 강화 시 수요가 급증할 영역이다.
FoV 시스템이 있어야만 만들 수 있는 **독점적 상품**.

**시나리오**: 데이터셋 20종 × 월 10건 × $299 + 커스텀 월 2건 × $10,000 = 월 약 $80,000
**웹 스택**: React 마켓 + PHP/Laravel 결제 시스템이 잘 맞음

---

## 7. 추천 로드맵

### 0~3개월: 기반 다지기 (리스크 0)

- 라이브러리 완전 숙달 (직접 실행, 논문 재현)
- 원본 저장소에 PR 1~2개 기여 → 컨트리뷰터 이력 확보
- 한국어 블로그/YouTube 콘텐츠 시작 (주 1회)
- **FoV 시각화 데모 웹앱**을 React + FastAPI로 제작해 무료 공개
  → 실력 + 인지도 + 포트폴리오를 동시에 확보

> 비용 거의 0원.

### 3~9개월: 첫 수익화 (아이디어 4 + 3)

- 전자책 또는 온라인 강의 출시 → 첫 매출
- MCP 서버 오픈소스 공개 → 화제성 확보
- 기업 워크샵 영업 시작

> 목표: 월 200~500만원

### 9~24개월: 본격 사업 (아이디어 2 또는 1)

- 산업 제어 PoC 1건 수주 (아이디어 2) — 초기 자본이 적게 드는 쪽
- 또는 SaaS MVP 출시 (아이디어 1) — 투자 유치 전제

> 목표: 연 1억~5억원

### 핵심 조언

**1번(SaaS)부터 시작하지 말 것.** GPU 비용 부담이 크다.
**4번(교육)으로 실력과 인지도를 먼저 쌓고**, 그 과정에서 생긴 네트워크로
2번(B2B)이나 3번(MCP) 기회를 잡는 것이 현실적인 경로다.

**React 웹앱은 어떤 아이디어를 택하든 필요하므로**,
0~3개월 구간의 FoV 데모 제작이 가장 투자 대비 효율이 높다.

---

## 8. 최종 총평

> **"월드모델 연구계의 scikit-learn"**
>
> 논문마다 베이스라인을 재구현하고 평가 방식이 제각각이라 재현이 안 되던 고질적 문제를,
> **하나의 표준 플랫폼으로 통일**하려는 프로젝트다.
>
> 특히 **FoV(Factors of Variation) 시스템**은 경쟁 라이브러리에 거의 없는 독보적 기능으로,
> "이 AI가 진짜 이해한 것인가, 암기한 것인가"를 정량 검증할 수 있게 해준다.

**나에게 도움이 되는 지점**

| 상황 | 가치 |
|------|------|
| 로보틱스/제어 AI 개발 | 파이프라인 구축 기간을 수개월 → 수일로 단축 |
| "AI가 계획하는 법" 학습 | CEM/MPPI 등 알고리즘의 최고 수준 교재 |
| 대용량 시계열/영상 데이터 처리 | `data/formats/` 설계 + Lance/LanceDB 실전 경험 |
| 모던 Python 프로젝트 설계 학습 | Lazy import, 레지스트리, Protocol, Hydra, uv |
| 포트폴리오 | 스타 2,211개 + LeCun 참여 저장소 기여 이력 |

**주의할 점**

- 웹 개발(React/PHP)과는 직접적 관련이 없는 순수 Python + PyTorch 연구 도구
- 제대로 활용하려면 GPU 필수 (논문 재현은 H100/H200급)
- `[all]` 설치 시 수 GB 용량
- `v0.1.1` 초기 버전 — README에 "API may change between minor versions" 명시됨

---

*이 문서는 저장소 전체(351개 파일, 약 3만 줄)를 조사하고,
GitHub API와 PyPI API로 실측 데이터를 확인해 작성했습니다.*
