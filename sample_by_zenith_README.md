# sample_by_zenith.py — 사용 설명

zenith 구간별로 이벤트를 골라, 각 이벤트에 대해 `sample_cfg.py`를 실행하는 스크립트입니다.

---

## Tab 자동완성 (입력할 때 Tab 키)

옵션·경로 입력 시 **Tab**으로 자동완성하려면:

1. **argcomplete 설치**
   ```bash
   pip install argcomplete
   ```

2. **스크립트 등록 (한 번만 하면 됨)**  
   아래 중 하나를 `~/.bashrc` 또는 `~/.zshrc`에 넣고 저장한 뒤 `source ~/.bashrc` (또는 zsh면 `source ~/.zshrc`) 실행.

   ```bash
   # 이 스크립트만 Tab 완성 쓰기
   eval "$(register-python-argcomplete /home/pmj0324/icecube-genesis/0121/GENESIS/sample_by_zenith.py)"
   ```

   또는 **모든** argparse 스크립트에 전역 적용:
   ```bash
   activate-global-python-argcomplete --dest=-  # 출력을 파일로 저장 후 source
   # 또는
   eval "$(activate-global-python-argcomplete --dest=-)"
   ```

3. **사용법**  
   터미널에서 `python sample_by_zenith.py ` 까지 입력한 뒤 **Tab**을 누르면 옵션 목록이 나옵니다. `-` 입력 후 Tab으로 `-c`, `-n`, `-m` 등이 완성됩니다.

---

## 동작 요약

1. **HDF5에서 label 로드** — label의 두 번째 값이 zenith
2. **zenith를 n개 구간으로 분할**
3. **각 구간에서 m개 이벤트 샘플링**
4. **선택된 각 이벤트마다 `sample_cfg.py` 실행**

---

## 필수 인자 (반드시 지정)

| Short | Long | 설명 |
|-------|------|------|
| **-c** | --checkpoint | 체크포인트 파일 경로 (.pt) |
| **-n** | --n_bins | zenith 구간 개수 |
| **-m** | --m_per_bin | 구간당 샘플할 이벤트 수 |

---

## 옵션 전체 (Short / Long / 기본값)

| Short | Long | 기본값 | 설명 |
|-------|------|--------|------|
| -c | --checkpoint | *(필수)* | 체크포인트 .pt 경로 |
| -n | --n_bins | *(필수)* | zenith 구간 개수 |
| -m | --m_per_bin | *(필수)* | 구간당 이벤트 수 |
| -d | --data_dir | `./GENESIS-data` | H5 파일이 있는 디렉터리 |
| -H | --h5_file | `None` | 사용할 H5 파일 (없으면 data_dir에서 첫 .h5 사용) |
| -k | --label_key | `label` | HDF5에서 label 데이터셋 키 |
| -o | --output_dir | `tasks/output_zenith_sampling` | 결과 저장 기준 디렉터리 |
| -N | --num_samples | `1` | 이벤트당 생성할 샘플 수 |
| -g | --gpu | `None` (자동) | GPU 번호 |
| -W | --histogram | **True** | 히스토그램 저장 (끄려면 `--no-histogram`) |
| -p | --cut_npe | `0.0` | nPE 컷 (시각화) |
| -t | --cut_firsttime | `0.0` | FirstTime 컷 (시각화) |
| -C | --cfg_scale | `None` | CFG scale (없으면 체크포인트 값 사용) |
| -s | --seed | `42` | 랜덤 시드 |
| -x | --skip_existing | `False` | 이미 출력 폴더 있으면 해당 이벤트 스킵 |

---

## 실행 예시

```bash
# 최소 실행 (필수 3개만)
python sample_by_zenith.py -c ./checkpoints/model.pt -n 5 -m 3

# 출력 디렉터리 지정
python sample_by_zenith.py -c ./checkpoints/model.pt -n 5 -m 3 -o tasks/my_run

# 히스토그램 끄기
python sample_by_zenith.py -c ./checkpoints/model.pt -n 5 -m 3 --no-histogram

# 이미 돌린 이벤트 스킵 (재실행 시)
python sample_by_zenith.py -c ./checkpoints/model.pt -n 5 -m 3 -x
```

---

## 출력 구조

```
tasks/output_zenith_sampling/   (또는 -o 로 지정한 경로)
├── event_00001/
│   ├── actual_event_1.png
│   ├── sampled_event_1_sample_001.npy
│   ├── sampled_event_1_sample_001.png
│   └── *_histogram.png   (히스토그램 켜져 있을 때)
├── event_00023/
│   └── ...
└── ...
```
