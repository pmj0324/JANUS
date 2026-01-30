# train_exp.py 개선 제안

코드 품질과 디퓨전 관점에서의 개선점 정리.

---

## 1. 코드 품질 (Code)

### 1.1 설정/실험 관리
- **현상**: `T`, `beta_start/end`, `batch_size`, `lr`, `num_epochs`, `h5_path`, `output_dir` 등이 전부 스크립트 상단에 하드코딩.
- **제안**: YAML config + argparse (또는 Hydra)로 옮기기. 재현성·실험 추적이 쉬워짐. `train.py`처럼 `-c config.yaml` 패턴 사용 권장.

### 1.2 경로/import
- **현상**: `sys.path.insert(0, os.path.join(os.getcwd(), "GENESIS"))` → 실행 CWD에 의존.
- **제안**: `Path(__file__).resolve().parent` 기준으로 GENESIS 경로 잡기. 다른 디렉터리에서 실행해도 동작하도록.

### 1.3 전역 상태
- **현상**: `device`, `dataset`, `loader`, `betas`, `model`, `optim` 등이 모듈 전역.
- **제안**: `main(config)` 또는 `Trainer` 클래스로 묶어서 config/device를 인자로 넘기기. 테스트·재사용이 쉬워짐.

### 1.4 재현성
- **현상**: `torch.manual_seed`, `np.random.seed` 없음.
- **제안**: 스크립트 초반에 `torch.manual_seed(seed)`, `np.random.seed(seed)` (및 cuDNN 결정론 옵션) 설정. config에서 `seed` 읽도록.

### 1.5 검증/모니터링
- **현상**: 검증 루프 없음. train loss만 기록.
- **제안**: 고정된 val set 또는 일부 배치로 주기적으로 val loss 계산. TensorBoard/W&B 로깅 시 loss·스케줄·학습률 등 한곳에서 보기.

### 1.6 학습률 스케줄
- **현상**: 고정 `lr = 3e-4`.
- **제안**: Cosine decay, warmup + decay 등 스케줄러 도입. 디퓨전 학습에서 흔히 사용.

### 1.7 체크포인트
- **현상**: best 체크포인트에 `betas`, `alphas`, `alphas_cumprod` 텐서 전체 저장.
- **제안**: 샘플링 시 필요한 건 `T`, `beta_start`, `beta_end`(및 스케줄 타입) 정도. 스케줄 파라미터만 저장하고, 로드 시 `compute_alpha_schedule(betas)`로 다시 계산하면 용량·호환성 관리에 유리.

### 1.8 메모리/시각화
- **현상**: `show_event_dual_plot` 후 figure를 명시적으로 닫지 않음.
- **제안**: 저장 후 `plt.close(fig)` 호출. 여러 이벤트·t 값 시각화 시 메모리 증가 완화.

### 1.9 torch.compile
- **현상**: `torch._dynamo.config.suppress_errors = True`로 모든 에러 무시.
- **제안**: 기본은 False로 두고, Triton 미설치 등 알려진 환경에서만 선택적으로 True. 또는 config 플래그로 켜/끄기.

---

## 2. 디퓨전 (Diffusion)

### 2.1 타임스텝 샘플링 (t)
- **현상**: `t ~ Uniform(1, T)`.
- **제안**: Loss가 큰 t를 더 자주 샘플하는 **importance sampling** (또는 t에 대한 가중치) 고려. 논문 “Improved DDPM” 등에서 사용. 구현은 간단히 `p(t) ∝ sqrt(E[L_t])` 추정 또는 sqrt(1/(1-alpha_bar)) 등으로 비균일 샘플링.

### 2.2 노이즈 스케줄 선택
- **현상**: sigmoid만 사용 (`beta_start=1e-4`, `beta_end=2e-2`).
- **제안**: linear, cosine 스케줄을 config로 선택 가능하게. 특히 **cosine**은 많은 이미지 디퓨전에서 선호됨. `train.py`처럼 config에 `schedule.type` + 인자 넣기.

### 2.3 목적함수 (objective)
- **현상**: ε-prediction, MSE.
- **제안**: 현재 선택은 표준적임. 도메인에 따라 **L1/Huber** 실험 가능. x0-prediction, v-prediction은 코드 변경이 크므로 필요 시 별도 브랜치에서 시도.

### 2.4 Classifier-Free Guidance (CFG)
- **현상**: 조건부 생성만 있고, inference 시 guidance 없음.
- **제안**: 학습 시 label을 일정 확률로 dropout (예: 10%)하고, 추론 시  
  `eps_hat = eps_uncond + w * (eps_cond - eps_uncond)`  
  형태의 CFG 적용. 조건부 품질·다양성 조절에 유리.

### 2.5 EMA (Exponential Moving Average)
- **현상**: 모델 파라미터만 저장, EMA 없음.
- **제안**: 파라미터의 EMA를 유지하고, 샘플링/평가 시 EMA 모델 사용. 디퓨전 논문에서 샘플 품질 향상에 자주 사용됨.

### 2.6 역확산 루프 공유
- **현상**: `train_exp.py`와 `sample.py`에 DDPM 역확산이 각각 구현됨.
- **제안**: `diffusion/sampling/ddpm.py` 등에 `sample_ddpm(model, x_T, betas, alphas_cumprod, ...)` 같은 함수를 두고, train_exp와 sample에서 공통 호출. 버그 수정·DDIM 등 변형 추가 시 한 곳만 수정하면 됨.

### 2.7 수치 안정성
- **현상**: `mean = (1/sqrt(alpha_t)) * (x_t - ...)` 등에서 `alpha_t`, `alpha_bar_t`가 매우 작을 수 있음.
- **제안**: t가 T에 가까울 때 sqrt, 나눗셈에서 underflow 방지. 필요 시 스케줄을 float64로 계산 후 float32로 캐스팅하거나, clamp(sqrt(alpha_bar), min=1e-8) 등으로 하한 두기.

### 2.8 샘플링 단계 수 (T)
- **현상**: 학습·샘플링 모두 T=1000.
- **제안**: 학습은 T=1000 유지하고, 추론만 **DDIM** 등으로 step 수 줄이기 (예: 50~100 step). 별도 스크립트/함수로 DDIM 스텝 서브셋 구현하면 됨.

---

## 3. 우선순위 제안

| 우선순위 | 항목 | 이유 |
|----------|------|------|
| 높음 | config(YAML)+seed | 재현성·실험 관리 기본 |
| 높음 | 역확산 로직 공통화 (sample.py와) | 유지보수·일관성 |
| 중간 | LR 스케줄, validation 로깅 | 수렴·과적합 모니터링 |
| 중간 | cosine 스케줄 옵션, t importance sampling | 디퓨전 품질 |
| 낮음 | EMA, CFG | 샘플 품질 추가 개선 |
| 낮음 | DDIM 추론 | 추론 속도 |

원하면 위 항목 중 하나씩 코드 수준으로 패치 제안해 줄 수 있음.
