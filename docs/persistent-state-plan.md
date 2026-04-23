
# Persistent State & Infrastructure Plan

## 목표

세션 종료 시에도 파이프라인 결과가 유지되고, 동일 파일 재처리 시 캐시를 활용하며,
병렬 실행 시 진행 상태를 실시간 추적할 수 있는 시스템으로 개선한다.

---

## 현재 문제점

| 문제 | 원인 |
|------|------|
| 세션 종료 시 결과 소실 | `st.session_state`(인메모리)에 의존, checkpoint JSON은 불완전 |
| 병렬 실행 시 진행 상태 불투명 | ThreadPoolExecutor 사용하지만 UI에 progress 미반영 |
| 이전 단계로 복원 불가 | 단방향 파이프라인, step별 스냅샷 없음 |
| 동일 파일 재처리 시 중복 연산 | 파일 해시 기반 캐시 없음 |
| 단일 사용자 한정 | ReviewQueue가 인메모리 싱글턴 |

---

## Phase 1: Azure 인프라 배포 (Bicep) ✅ 완료

> **배포 완료**: 2026-04-21 | **환경**: dev | **리전**: Korea Central

### 배포된 리소스

| 리소스 | 엔드포인트 | SKU |
|--------|-----------|-----|
| **Cosmos DB** (`cosmos-sldbom-dev`) | `https://cosmos-sldbom-dev.documents.azure.com:443/` | Serverless |
| **Blob Storage** (`stsldbomdev`) | `https://stsldbomdev.blob.core.windows.net/` | Standard LRS |
| **Service Bus** (`sb-sldbom-dev`) | `sb-sldbom-dev.servicebus.windows.net` | Standard |

### 컨테이너 구조

**Cosmos DB:**
```
Database: sld-bom
├── Container: pipeline-runs      (partition: /run_id)
│   └── {run_id, pdf_hash, status, created_at, updated_at, steps: [...]}
├── Container: step-states        (partition: /run_id)  
│   └── {run_id, step_num, status, started_at, completed_at, result_ref, error}
└── Container: file-cache         (partition: /pdf_hash)
    └── {pdf_hash, filename, step_num, blob_ref, created_at, ttl}
```

**Blob Storage:**
```
Container: pipeline-artifacts
├── {run_id}/pages/
├── {run_id}/di_detection/
├── {run_id}/missing_detection/
├── {run_id}/crops/
├── {run_id}/bay_results/
└── {run_id}/final_summary.json

Container: file-cache
└── {pdf_hash}/step{n}/...       # 동일 파일 재사용 캐시
```

**Service Bus:**
```
Queue: pipeline-tasks             # 비동기 작업 큐
Topic: pipeline-events            # 진행 상태 브로드캐스트
├── Subscription: ui-updates      # Streamlit UI 구독
└── Subscription: api-updates     # API 클라이언트 구독
```

---

## Phase 2: 상태 관리 계층 (State Management Layer)

### 2.1 StateManager 클래스

```python
class StateManager:
    """Cosmos DB + Blob 기반 영구 상태 관리"""
    
    async def create_run(pdf_path, pdf_hash) -> RunState
    async def get_run(run_id) -> RunState
    async def update_step(run_id, step_num, status, result_ref)
    async def get_step_state(run_id, step_num) -> StepState
    async def rollback_to_step(run_id, target_step)  # 이전 단계로 복원
    async def find_cached_result(pdf_hash, step_num) -> Optional[CachedResult]
    async def save_cache(pdf_hash, step_num, blob_ref)
```

### 2.2 BlobManager 클래스

```python
class BlobManager:
    """Blob Storage 기반 중간 결과물 관리"""
    
    async def upload_step_artifacts(run_id, step_num, local_dir)
    async def download_step_artifacts(run_id, step_num, local_dir)
    async def copy_from_cache(pdf_hash, step_num, run_id)
    async def cleanup_run(run_id)
```

### 2.3 ProgressTracker 클래스

```python
class ProgressTracker:
    """Service Bus 기반 실시간 진행 상태 추적"""
    
    async def emit_progress(run_id, step_num, progress_pct, message)
    async def subscribe_progress(run_id) -> AsyncIterator[ProgressEvent]
    async def emit_step_complete(run_id, step_num, result_summary)
```

---

## Phase 3: 파이프라인 수정

### 3.1 각 Step에 상태 저장 추가

```
Step 실행 전:
  1. pdf_hash로 캐시 확인 → 있으면 Blob에서 복원, skip
  2. Cosmos DB에 step status = "running" 기록
  3. Service Bus에 progress event 발행

Step 실행 중:
  4. ThreadPoolExecutor 각 task 완료 시 progress % 갱신
  5. 중간 결과물을 Blob에 스트리밍 업로드

Step 완료 후:
  6. Cosmos DB에 step status = "completed" + result_ref 기록
  7. 캐시 저장 (pdf_hash + step_num → blob_ref)
  8. Service Bus에 complete event 발행
```

### 3.2 Rollback (이전 단계 복원)

```
롤백 요청 시 (target_step):
  1. Cosmos DB에서 target_step의 StepState 조회
  2. target_step 이후 step들의 status를 "invalidated"로 변경
  3. Blob에서 target_step 결과물을 로컬로 다운로드
  4. Streamlit session_state를 해당 시점으로 복원
  5. target_step부터 재실행 가능
```

### 3.3 동일 파일 캐시 활용

```
새 파일 업로드 시:
  1. SHA-256 해시 계산
  2. file-cache 컨테이너에서 해시로 조회
  3. 캐시 히트 시:
     - 각 step별 캐시된 blob_ref를 새 run_id로 복사
     - UI에서 "이전 결과 복원됨" 표시 + 수정 가능
  4. 캐시 미스 시: 정상 파이프라인 실행
```

---

## Phase 4: Streamlit UI 수정

### 4.1 진행 상태 표시

- 각 Step 실행 중 실시간 progress bar (Service Bus 구독)
- 병렬 처리 시 페이지별 진행률 표시
- 전체 파이프라인 타임라인 뷰

### 4.2 이전/다음 네비게이션

- Step 간 자유로운 이동 (현재는 단방향)
- "이전" 버튼 클릭 시 해당 step의 저장된 상태 복원
- 수정 후 해당 step부터 재실행

### 4.3 실행 이력 관리

- 사이드바에 이전 실행 목록 (Cosmos DB 조회)
- 실행 선택 시 해당 결과 로드
- 실행 비교 뷰

---

## Phase 5: E2E 테스트

### 테스트 시나리오

1. **세션 복원 테스트**: 앱 재시작 후 이전 결과 로드 확인
2. **롤백 테스트**: Step 3 완료 후 Step 2로 롤백, 재실행
3. **캐시 테스트**: 동일 파일 재업로드 시 캐시 활용 확인
4. **병렬 진행 상태 테스트**: Step 2A 병렬 실행 중 progress 표시 확인
5. **Blob 저장 테스트**: 중간 결과물이 Blob에 저장되는지 확인

---

## 작업 순서

| 순서 | 작업 | 의존성 | 난이도 | 상태 |
|------|------|--------|--------|------|
| **1** | **Azure 인프라 Bicep 배포** | 없음 (독립) | ★★☆ | ✅ 완료 (2026-04-21) |
| **2** | **State 모델 정의** (RunState, StepState) | Phase 1 완료 | ★★☆ | 🔲 대기 |
| **3** | **BlobManager 구현** | Phase 1 완료 | ★★☆ | 🔲 대기 |
| **4** | **StateManager 구현** | Phase 1 완료 | ★★★ | 🔲 대기 |
| **5** | **config.py 확장** (Azure 리소스 연결) | Phase 1 완료 | ★☆☆ | 🔲 대기 |
| 6 | ProgressTracker 구현 | Phase 1 완료 | ★★☆ | 🔲 대기 |
| 7 | Executor 수정 (상태 저장 연동) | Phase 2, 3 완료 | ★★★ | 🔲 대기 |
| 8 | Streamlit UI 수정 (네비게이션 + 이력) | Phase 4 완료 | ★★★ | 🔲 대기 |
| 9 | E2E 테스트 작성 및 실행 | Phase 2-5 완료 | ★★☆ | 🔲 대기 |

---

## 파일 구조 (신규/수정)

```
infra/                                  ✅ 생성 완료
├── main.bicep              # 메인 Bicep 템플릿
├── modules/
│   ├── cosmosdb.bicep       # Cosmos DB Serverless + 3 containers
│   ├── storage.bicep        # Blob Storage + lifecycle policy
│   └── servicebus.bicep     # Service Bus + queue + topic + subscriptions
├── parameters/
│   ├── dev.bicepparam       # 개발 환경 파라미터
│   └── prod.bicepparam      # 운영 환경 파라미터
├── deploy.sh                # 배포 스크립트 (RG 생성 → Bicep → .env 생성)
└── teardown.sh              # 인프라 삭제 스크립트

.env.dev.azure               ✅ 자동 생성됨 (배포 출력값)

src/
├── config.py                # (수정 예정) Azure 리소스 연결 설정 추가
├── state/                   # (신규 예정) 상태 관리 계층
│   ├── __init__.py
│   ├── models.py            # RunState, StepState, CachedResult 모델
│   ├── blob_manager.py      # Blob Storage 관리
│   ├── state_manager.py     # Cosmos DB 상태 관리
│   └── progress_tracker.py  # Service Bus 진행 상태
├── hitl/
│   └── streamlit_app.py     # (수정 예정) 영구 상태 연동
└── workflow/
    └── executors.py          # (수정 예정) 상태 저장 훅 추가
```
