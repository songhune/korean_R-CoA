# iCloud + Git 충돌 문제 완전 가이드

## 🔴 문제 상황

매일 아침이나 일정 시간 후, `git status`를 확인하면 수백 개의 파일이 `deleted`로 표시됩니다.

```bash
deleted:    data/gwashi.csv
deleted:    notebook/experiments/1번실험.ipynb
...
(수백 개의 파일)
```

## 🎯 근본 원인

### iCloud Drive의 "Optimize Mac Storage" 기능

1. **iCloud는 공간 절약을 위해 파일을 "evict"합니다**
   - 사용하지 않는 파일을 클라우드로만 저장
   - 로컬에서는 제거하거나 placeholder로 교체

2. **Git은 로컬 파일 시스템만 봅니다**
   - 파일이 로컬에 없으면 → "deleted"로 인식
   - `git add .` 또는 자동 스크립트 실행 시 → 모든 삭제가 staging됨

3. **충돌 발생**
   ```
   iCloud evicts files (밤사이)
   → Git sees deletions (아침)
   → Staged deletions (자동으로)
   → Commit 막힘 (매일 반복)
   ```

## ✅ 해결 방법 (3가지 옵션)

### 옵션 1: iCloud Drive 최적화 끄기 (추천)

**장점**: 문제 완전 해결, 파일 항상 로컬에 존재
**단점**: 디스크 공간 사용

```bash
# 1. 시스템 설정 > Apple ID > iCloud > iCloud Drive 옵션
#    "Optimize Mac Storage" 체크 해제

# 2. 현재 프로젝트의 모든 파일 다운로드
cd "/Users/songhune/Library/Mobile Documents/com~apple~CloudDocs/Workspace/korean_eda"
brctl download .

# 3. 프로젝트를 "Always Keep on This Mac"으로 설정
# Finder에서 폴더 우클릭 → "Always Keep on This Mac"
```

### 옵션 2: 프로젝트를 iCloud 밖으로 이동 (가장 안전)

**장점**: Git과 iCloud 완전 분리, 성능 향상
**단점**: iCloud 백업 없음 (GitHub가 백업 역할)

```bash
# 1. 프로젝트를 로컬 디렉토리로 이동
mv "/Users/songhune/Library/Mobile Documents/com~apple~CloudDocs/Workspace/korean_eda" \
   ~/Projects/korean_eda

# 2. Git remote 확인 (변경 없음)
cd ~/Projects/korean_eda
git remote -v

# 3. 정상 작동 확인
git status
```

### 옵션 3: Git Hook으로 방어 (현재 설정됨)

**장점**: iCloud 사용 가능, 실수 방지
**단점**: 근본 원인 해결 아님

```bash
# 이미 설정된 pre-commit hook이 50개 이상 삭제 감지 시 경고합니다
# .git/hooks/pre-commit 참조
```

## 🚨 긴급 복구 방법 (문제 재발 시)

```bash
cd "/Users/songhune/Library/Mobile Documents/com~apple~CloudDocs/Workspace/korean_eda"

# 1. Staging area에서 모든 삭제 취소
git restore --staged .

# 2. iCloud에서 파일 다운로드 (필요시)
brctl download .

# 3. 상태 확인
git status

# 4. 실제로 삭제를 원하지 않으면 working directory도 복구
git restore .
```

## 📋 예방 체크리스트

- [ ] iCloud "Optimize Mac Storage" 끄기 (옵션 1)
- [ ] 프로젝트를 로컬로 이동 (옵션 2)
- [ ] Git hook 설치 완료 확인 (옵션 3, 이미 완료)
- [ ] `git add .` 대신 `git add [specific files]` 사용
- [ ] 커밋 전 항상 `git status` 확인

## 🔍 문제 진단 명령어

```bash
# iCloud 상태 확인
brctl log

# 파일이 iCloud에만 있는지 확인
ls -l@ .

# Git staged deletions 개수 확인
git diff --cached --name-status | grep "^D" | wc -l
```

## 💡 권장 사항

**프로젝트를 iCloud 밖으로 이동 (옵션 2)을 강력 추천합니다:**

1. **성능**: iCloud 동기화 오버헤드 없음
2. **안정성**: 파일 eviction 걱정 없음
3. **백업**: GitHub가 이미 백업 역할
4. **표준**: 대부분의 개발자가 로컬에서 Git 사용

```bash
# 한 번만 실행
mv "/Users/songhune/Library/Mobile Documents/com~apple~CloudDocs/Workspace/korean_eda" \
   ~/Projects/korean_eda

cd ~/Projects/korean_eda
git status  # 정상 작동 확인
```

이후 iCloud와 Git의 충돌이 완전히 사라집니다.
