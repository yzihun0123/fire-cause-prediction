# 화재 데이터 프로파일링 리포트 생성 (수행내역서 '데이터 프로파일링 리포트' 장에 사용)
# ydata-profiling은 메인 .venv에 없어서, 수업용 환경(C:\YOOJIHUN\ydataprofiling\.venv)으로 실행합니다.
#   실행: C:\YOOJIHUN\ydataprofiling\.venv\Scripts\python.exe make_profile.py

import os
import pandas as pd
from ydata_profiling import ProfileReport

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'dataset', 'fire_2020_2024.csv')
OUT_PATH = os.path.join(BASE_DIR, 'outputs', '10_데이터_프로파일링_리포트.html')

# 전처리 전 데이터 그대로 분석 (191,510행) → 재산피해의 극단적인 치우침 등도 리포트에 드러남
df = pd.read_csv(DATA_PATH, encoding='utf-8-sig', parse_dates=['datetime'])
profile = ProfileReport(df, title='소방청 화재발생 정보 (2020~2024) 데이터 프로파일링', explorative=True)
profile.to_file(OUT_PATH)
print('저장:', OUT_PATH)
