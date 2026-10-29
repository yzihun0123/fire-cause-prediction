# 전국 화재 데이터로 화재 원인 예측 - Streamlit 시연 앱
# 실행: streamlit run app.py
# 노트북(fire_cause_project.ipynb)과 같은 전처리·같은 분할(2020~2023 학습 / 2024 테스트)·같은 저장 모델을 사용합니다.

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import accuracy_score, f1_score, top_k_accuracy_score

# app.py가 있는 폴더 기준으로 경로를 잡아야 클라우드에서도 파일을 찾을 수 있음
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'dataset', 'fire_2020_2024.csv')
# 노트북 4-6에서 저장한 최종 모델 + 인코딩에 필요한 범주 목록
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'fire_cause_model.pkl')

WEEKDAYS = ['월', '화', '수', '목', '금', '토', '일']

# 원인별 일반적인 예방 수칙 (소방청·한국전기안전공사 등에서 널리 안내하는 내용)
TIPS = {
    '부주의': '담배꽁초는 반드시 끄고 버리기, 조리 중 자리 비우지 않기, 쓰레기·논밭두렁 태우기 금지, 용접 작업 시 주변 가연물 치우기',
    '전기적 요인': '멀티탭 문어발 사용 자제, 오래된 전선·콘센트 교체, 먼지 쌓인 플러그 청소(트래킹 예방), 여름철 에어컨 실외기 배선 점검',
    '기계적 요인': '보일러·난방기·차량 정기 점검, 과열되기 쉬운 기계 주변 정리, 오일·연료 누설 확인',
    '기타': '가스 밸브 잠금 확인, 화학물질 보관 기준 지키기, 방화 예방을 위한 출입 관리와 조명·CCTV 설치',
}


def group_cause(c):
    # 노트북 2-1과 같은 원인 묶기
    return c if c in ['부주의', '전기적 요인', '기계적 요인', '미상'] else '기타'


# @st.cache_data: 화면을 조작할 때마다 다시 읽지 않도록 결과를 기억해 둠
@st.cache_data
def load_data():
    # utf-8-sig: 엑셀용으로 저장한 파일 맨 앞의 표시(BOM)를 빼고 읽음
    df = pd.read_csv(DATA_PATH, encoding='utf-8-sig', parse_dates=['datetime'])
    df['cause_group'] = df['cause'].apply(group_cause)
    df['month'] = df['datetime'].dt.month
    df['hour'] = df['datetime'].dt.hour
    df['weekday'] = df['datetime'].dt.day_of_week
    df['year'] = df['datetime'].dt.year
    return df


def prepare(d, keep_place3):
    # 노트북 3단계와 같은 입력 준비 (드문 장소 소분류는 '기타(드문 장소)'로)
    out = d[['month', 'hour', 'weekday', 'sido', 'fire_type', 'place_1', 'place_2']].copy()
    out['place_3'] = d['place_3'].where(d['place_3'].isin(keep_place3), '기타(드문 장소)')
    return out


def encode(d, categories):
    # 범주 목록을 고정한 원-핫 인코딩 → 한 건만 넣어도 학습 때와 같은 컬럼이 만들어짐
    parts = [pd.get_dummies(pd.Categorical(d[col], categories=cats), prefix=col, dtype=int).set_axis(d.index)
             for col, cats in categories.items()]
    return pd.concat(parts, axis=1)


# @st.cache_resource: 모델 파일은 처음 한 번만 불러오고 계속 재사용
@st.cache_resource
def load_model():
    bundle = joblib.load(MODEL_PATH)

    # 노트북 4-1과 같은 테스트 데이터(2024년, 원인 미상 제외)로 성능을 다시 계산해 표시
    df = load_data()
    test = df[(df['year'] == 2024) & (df['cause_group'] != '미상')]
    X = encode(prepare(test, bundle['keep_place3']), bundle['categories'])
    y = test['cause_group'].map({c: i for i, c in enumerate(bundle['classes'])})
    proba = bundle['model'].predict_proba(X)
    scores = {
        'acc': accuracy_score(y, proba.argmax(axis=1)),
        'f1': f1_score(y, proba.argmax(axis=1), average='macro'),
        'top2': top_k_accuracy_score(y, proba, k=2, labels=range(len(bundle['classes']))),
        'baseline': (test['cause_group'] == '부주의').mean(),
    }
    return bundle, scores


def predict(bundle, inputs):
    row = prepare(pd.DataFrame([inputs]), bundle['keep_place3'])
    return pd.Series(bundle['model'].predict_proba(encode(row, bundle['categories']))[0], index=bundle['classes'])


# 대표 상황 버튼 (노트북 5-3의 예시와 같음)
PRESETS = {
    '🍗 1월 저녁 7시 · 서울 치킨집': dict(month=1, hour=19, weekday=4, sido='서울특별시', fire_type='건축,구조물',
                                      place_1='생활서비스', place_2='음식점', place_3='치킨,족발'),
    '🏢 8월 오후 3시 · 경기 아파트': dict(month=8, hour=15, weekday=2, sido='경기도', fire_type='건축,구조물',
                                     place_1='주거', place_2='공동주택', place_3='아파트'),
    '🌾 4월 오후 2시 · 경북 논밭두렁': dict(month=4, hour=14, weekday=5, sido='경상북도', fire_type='임야',
                                       place_1='임야', place_2='들불', place_3='논밭두렁'),
}


def apply_preset(preset):
    # 버튼을 누르면 입력값(session_state)을 대표 상황으로 바꿈
    for k, v in preset.items():
        st.session_state[k] = v


# ---------------- 화면 구성 ----------------
st.set_page_config(page_title='화재 원인 예측', layout='wide')
st.title('전국 화재 데이터로 화재 원인 예측하기')
# 마크다운에서 ~는 취소선 기호로 읽히므로 화면 문장 안에서는 \~로 씀
st.caption('데이터: 소방청 화재발생 정보 (2020\\~2024년 실제 화재 191,510건) / 모델: 2020\\~2023년으로 학습, 2024년으로 평가')

st.info(
    '**화재 원인 조사는 시간이 걸립니다.** 하지만 언제·어디서 난 불인지만 알아도 원인을 어느 정도 좁힐 수 있습니다.\n\n'
    '**이 앱은** 발생 시기(월·시간·요일), 지역, 장소, 화재 유형을 넣으면 **원인이 부주의·전기적 요인·기계적 요인·기타 중 '
    '무엇일 가능성이 높은지** 알려주고, 그 원인에 맞는 예방 수칙을 보여줍니다.\n\n'
    '**같은 장소라도 계절에 따라 원인이 달라집니다.** 여름에는 전기적 요인, 봄에는 부주의가 늘어납니다. 왼쪽에서 월을 바꿔 확인해 보세요.'
)

for path in [DATA_PATH, MODEL_PATH]:
    if not os.path.exists(path):
        st.error(f'파일이 없습니다: {path}')
        st.stop()

data = load_data()
bundle, scores = load_model()

# 입력 처음 값: 첫 번째 대표 상황 (session_state에 한 번만 넣어 두고, 이후에는 버튼·입력이 바꿈)
for k, v in next(iter(PRESETS.values())).items():
    if k not in st.session_state:
        st.session_state[k] = v

# 모델 성능 (노트북 4-3과 같은 2024년 테스트 데이터)
c1, c2, c3, c4 = st.columns(4)
c1.metric('정확도', f"{scores['acc']:.1%}", help='원인을 정확히 맞힌 비율 (2024년 화재)')
c2.metric('찍기 기준 정확도', f"{scores['baseline']:.1%}", help='비교 기준: 가장 많은 원인 하나만 답할 때의 정확도')
c3.metric('상위 2개 적중률', f"{scores['top2']:.1%}", help='모델이 꼽은 원인 2개 안에 정답이 있는 비율')
c4.metric('매크로 F1', f"{scores['f1']:.3f}", help='4개 원인을 똑같은 비중으로 평균한 F1 (0~1)')

# 왼쪽 사이드바: 화재 상황 입력
st.sidebar.header('대표 상황 고르기')
for name, preset in PRESETS.items():
    st.sidebar.button(name, on_click=apply_preset, args=(preset,), use_container_width=True)

st.sidebar.header('직접 입력')
inputs = {
    'month': st.sidebar.slider('월', 1, 12, key='month'),
    'hour': st.sidebar.slider('발생 시각 (시)', 0, 23, key='hour'),
    'weekday': st.sidebar.selectbox('요일', range(7), format_func=lambda i: WEEKDAYS[i], key='weekday'),
    'sido': st.sidebar.selectbox('시도', bundle['categories']['sido'], key='sido'),
    'fire_type': st.sidebar.selectbox('화재 유형', bundle['categories']['fire_type'], key='fire_type'),
}
# 장소 목록은 실제 데이터에서 앞에서 고른 화재 유형·대분류·중분류와 함께 나온 것만 보여줌
places = data[data['fire_type'] == inputs['fire_type']]
inputs['place_1'] = st.sidebar.selectbox('장소 대분류', sorted(places['place_1'].unique()), key='place_1')
places = places[places['place_1'] == inputs['place_1']]
inputs['place_2'] = st.sidebar.selectbox('장소 중분류', sorted(places['place_2'].unique()), key='place_2')
places = places[places['place_2'] == inputs['place_2']]
inputs['place_3'] = st.sidebar.selectbox('장소 소분류', places['place_3'].value_counts().index.tolist(), key='place_3')

# 1) 예측한 원인 확률
proba = predict(bundle, inputs)
top = proba.idxmax()
st.subheader(f'가장 가능성 높은 원인: {top} ({proba[top]:.0%})')
second = proba.drop(top).idxmax()
st.caption(f'두 번째 후보: {second} ({proba[second]:.0%}) — 상위 2개 안에 실제 원인이 있을 확률은 약 {scores["top2"]:.0%}입니다.')
st.bar_chart(proba.rename('확률') * 100, horizontal=True, y_label='확률(%)', x_label='')
st.success(f'**{top} 예방 수칙:** {TIPS[top]}')

# 2) 이 장소의 실제 기록 (2020~2024년)
same_place = data[(data['place_3'] == inputs['place_3']) & (data['place_2'] == inputs['place_2'])]
st.markdown(f'#### 실제 기록: {inputs["place_2"]} > {inputs["place_3"]} 화재 (2020\\~2024년 전국 {len(same_place):,}건)')
k1, k2, k3 = st.columns(3)
k1.metric('인명피해가 난 비율', f"{(same_place['casualties'] > 0).mean():.1%}")
k2.metric('재산피해 중앙값', f"{same_place['damage'].median() / 10:,.0f}만원")
k3.metric('원인 미상 비율', f"{(same_place['cause_group'] == '미상').mean():.1%}")
detail = same_place.loc[same_place['cause_group'] != '미상', 'cause_detail'].value_counts().head(5)
st.dataframe(pd.DataFrame({'세부 원인': detail.index, '건수': detail.values}), hide_index=True)

# 3) 같은 장소·시간에서 월만 바꾸면 원인 확률이 어떻게 변하는지
st.markdown('#### 같은 장소·시간이라도 계절에 따라 원인이 달라질까?')
by_month = pd.DataFrame({m: predict(bundle, {**inputs, 'month': m}) * 100 for m in range(1, 13)}).T
by_month.index.name = '월'
st.line_chart(by_month, x_label='월', y_label='예측 확률(%)')
st.caption('왼쪽에서 고른 조건에서 월만 1\\~12월로 바꿔 예측한 결과입니다.')
st.caption('※ 과거 화재 기록에서 배운 경향으로 계산한 값이며, 실제 원인은 현장 조사로 확인해야 합니다.')
