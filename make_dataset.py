# 소방청 화재발생 정보 원본에서 프로젝트용 데이터 파일을 만드는 스크립트
# 원본 CSV(cp949)를 영어 컬럼명 + 엑셀에서 바로 열리는 utf-8-sig CSV로 바꿉니다.
# 분석에 쓰지 않는 컬럼 3개(시군구, 부동산·동산 피해 세부)는 뺍니다.
#   원본 받는 곳: 공공데이터포털 '소방청_화재발생 정보' (https://www.data.go.kr/data/15044003/fileData.do)
#   → 받은 CSV를 dataset/raw/소방청_화재발생정보_20241231.csv 로 저장한 뒤
#   실행: python make_dataset.py

import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_PATH = os.path.join(BASE_DIR, 'dataset', 'raw', '소방청_화재발생정보_20241231.csv')
OUT_PATH = os.path.join(BASE_DIR, 'dataset', 'fire_2020_2024.csv')

# 원본 항목명 → 영어 이름 (값은 원본 그대로, 정제는 노트북에서 함)
# 빼는 컬럼: 시군구(시도만 사용), 부동산·동산(두 값의 합인 재산피해소계만 사용)
COLUMNS = {
    '화재발생년원일': 'datetime',
    '시도': 'sido',
    '화재유형': 'fire_type',
    '발화요인대분류': 'cause',
    '발화요인소분류': 'cause_detail',
    '인명피해(명)소계': 'casualties',
    '사망': 'deaths',
    '부상': 'injuries',
    '재산피해소계': 'damage',
    '장소대분류': 'place_1',
    '장소중분류': 'place_2',
    '장소소분류': 'place_3',
}

# 소방청 CSV는 한글 윈도우 인코딩(cp949)으로 저장되어 있음
raw = pd.read_csv(RAW_PATH, encoding='cp949')
print('원본 크기:', raw.shape)

data = raw.rename(columns=COLUMNS)[list(COLUMNS.values())]
# utf-8-sig: 엑셀에서 열어도 한글이 깨지지 않음
data.to_csv(OUT_PATH, index=False, encoding='utf-8-sig')
print('저장:', OUT_PATH, data.shape, f'({os.path.getsize(OUT_PATH) / 1024 / 1024:.1f} MB)')
