# 2-DOF Robot Manipulator Control with Disturbance Observer and Actor-Critic

## 개요
이 프로젝트는 2자유도 로봇 매니퓰레이터의 제어를 위한 외란 관측기 기반 액터-크리틱 제어 시스템을 구현합니다.

## 주요 기능
- 2-DOF 로봇 매니퓰레이터 동역학 모델
- 외란 관측기 (Disturbance Observer, DOB)
- 액터-크리틱 신경망 제어기
- 실시간 시각화
- 성능 비교 분석

## 실행 방법
```bash
python acdob.py
```

### 옵션
1. 실시간 시각화 (DOB+AC)
2. 기본 시뮬레이션 (DOB+AC)
3. AC vs DOB+AC 성능 비교

## 필요 라이브러리
- numpy
- matplotlib
- torch
- scipy
- tqdm

## 설치
```bash
pip install numpy matplotlib torch scipy tqdm
```