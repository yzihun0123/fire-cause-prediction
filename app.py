# 전국 화재 데이터로 화재 원인 예측 - Streamlit 시연 앱
# 실행: streamlit run app.py
# 노트북(fire_cause_project.ipynb)과 같은 전처리·같은 분할(2020~2023 학습 / 2024 테스트)·같은 저장 모델을 사용합니다.

import os
import joblib
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
from sklearn.metrics import accuracy_score, f1_score, top_k_accuracy_score

# app.py가 있는 폴더 기준으로 경로를 잡아야 클라우드에서도 파일을 찾을 수 있음
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'dataset', 'fire_2020_2024.csv')
# 노트북 4-6에서 저장한 최종 모델 + 인코딩에 필요한 범주 목록
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'fire_cause_model.pkl')

WEEKDAYS = ['월', '화', '수', '목', '금', '토', '일']

# 원인별 색·아이콘 (노트북 그래프와 같은 색을 써서 화면 전체에서 원인을 같은 색으로 알아보게 함)
CLASSES = ['부주의', '전기적 요인', '기계적 요인', '기타']
COLORS = {'부주의': '#f58518', '전기적 요인': '#4c78a8', '기계적 요인': '#54a24b', '기타': '#9aa5b1'}
ICONS = {'부주의': '🚬', '전기적 요인': '⚡', '기계적 요인': '⚙️', '기타': '❓'}
COLOR_SCALE = alt.Scale(domain=CLASSES, range=[COLORS[c] for c in CLASSES])

# 원인별 일반적인 예방 수칙 (소방청·한국전기안전공사 등에서 널리 안내하는 내용)
TIPS = {
    '부주의': '담배꽁초는 반드시 끄고 버리기, 조리 중 자리 비우지 않기, 쓰레기·논밭두렁 태우기 금지, 용접 작업 시 주변 가연물 치우기',
    '전기적 요인': '멀티탭 문어발 사용 자제, 오래된 전선·콘센트 교체, 먼지 쌓인 플러그 청소(트래킹 예방), 여름철 에어컨 실외기 배선 점검',
    '기계적 요인': '보일러·난방기·차량 정기 점검, 과열되기 쉬운 기계 주변 정리, 오일·연료 누설 확인',
    '기타': '가스 밸브 잠금 확인, 화학물질 보관 기준 지키기, 방화 예방을 위한 출입 관리와 조명·CCTV 설치',
}


# 원인 그룹 설명 (처음 보는 사람도 알 수 있게 쉬운 말과 실제 사례로)
DESCRIPTIONS = {
    '부주의': ('사람의 실수나 방심으로 불이 난 경우입니다.',
              '담배꽁초를 제대로 끄지 않고 버림 · 요리하다 자리를 비움 · 쓰레기나 논밭을 태우다 불이 번짐 · 용접 불티가 튐'),
    '전기적 요인': ('전선·콘센트·전기제품에 이상이 생겨 불꽃이나 열이 난 경우입니다. 대부분 단락(합선)입니다.',
                '낡은 전선 피복이 약해져 합선(절연열화) · 플러그에 쌓인 먼지와 습기로 불꽃(트래킹) · '
                '멀티탭에 전기제품을 많이 꽂아 과부하'),
    '기계적 요인': ('기계나 장비가 고장 나거나 지나치게 뜨거워져 불이 난 경우입니다.',
                '자동차 엔진·보일러 과열 · 새어 나온 오일·연료가 뜨거운 부품에 닿음 · 오래되거나 정비가 안 된 기계'),
    '기타': ('위 세 가지에 속하지 않는 여러 원인을 묶었습니다. 건수는 적지만 인명피해가 가장 잦은 그룹입니다.',
            '방화·방화 의심 · 기름 걸레 등이 저절로 발화(자연발화) · 교통사고 · 가스 누출 폭발 · 낙뢰 같은 자연재해 · 제품 결함'),
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
    '🏠 12월 밤 10시 · 강원 단독주택': dict(month=12, hour=22, weekday=5, sido='강원특별자치도', fire_type='건축,구조물',
                                       place_1='주거', place_2='단독주택', place_3='단독주택'),
}


def apply_preset(preset):
    # 버튼을 누르면 입력값(session_state)을 대표 상황으로 바꿈
    for k, v in preset.items():
        st.session_state[k] = v


def bar_chart(df, x, y, color=None, height=180):
    # 가로 막대 그래프: 원인 순서를 CLASSES로 고정하고, 막대 끝에 값을 표시
    base = alt.Chart(df).encode(
        # 막대 끝 숫자가 잘리지 않도록 가로축을 최댓값보다 20% 넓게 잡음
        x=alt.X(f'{x}:Q', title=None, axis=None, scale=alt.Scale(domain=[0, df[x].max() * 1.2])),
        y=alt.Y(f'{y}:N', title=None, sort=None, axis=alt.Axis(labelFontSize=14, labelLimit=260)),
    )
    bars = base.mark_bar(cornerRadiusEnd=4, height=26).encode(
        color=alt.Color(f'{color}:N', scale=COLOR_SCALE, legend=None) if color else alt.value('#e45756'))
    text = base.mark_text(align='left', dx=6, fontSize=14, fontWeight='bold').encode(text=f'{x}_label:N')
    return (bars + text).properties(height=height)


# ---------------- 화면 구성 ----------------
st.set_page_config(page_title='화재 원인 예측', page_icon='🔥', layout='wide')

# 결과 카드·사이드바 제목에 쓰는 간단한 스타일
st.markdown('''
<style>
.result-card {border-radius: 12px; padding: 18px 22px; color: white; margin-bottom: 8px;}
.result-card .label {font-size: 15px; opacity: 0.9;}
.result-card .cause {font-size: 34px; font-weight: 800; line-height: 1.3;}
.result-card .sub {font-size: 15px; opacity: 0.95;}
.tip-card {border-left: 6px solid; border-radius: 8px; padding: 12px 16px; background: rgba(128,128,128,0.08);}
</style>
''', unsafe_allow_html=True)

st.title('🔥 전국 화재 데이터로 화재 원인 예측하기')
# 마크다운에서 ~는 취소선 기호로 읽히므로 화면 문장 안에서는 \~로 씀
st.caption('데이터: 소방청 화재발생 정보 (2020\\~2024년 실제 화재 191,510건) · 모델: XGBoost (2020\\~2023년 학습, 2024년 평가)')

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

# 왼쪽 사이드바: 화재 상황 입력
with st.sidebar:
    st.header('🎯 대표 상황')
    for name, preset in PRESETS.items():
        st.button(name, on_click=apply_preset, args=(preset,), use_container_width=True)

    st.header('🕒 언제')
    inputs = {
        'month': st.slider('월', 1, 12, key='month'),
        'hour': st.slider('발생 시각 (시)', 0, 23, key='hour'),
        'weekday': st.selectbox('요일', range(7), format_func=lambda i: WEEKDAYS[i] + '요일', key='weekday'),
    }

    st.header('📍 어디서')
    inputs['sido'] = st.selectbox('시도', bundle['categories']['sido'], key='sido')
    inputs['fire_type'] = st.selectbox('화재 유형', bundle['categories']['fire_type'], key='fire_type')
    # 장소 목록은 실제 데이터에서 앞에서 고른 화재 유형·대분류·중분류와 함께 나온 것만 보여줌
    places = data[data['fire_type'] == inputs['fire_type']]
    inputs['place_1'] = st.selectbox('장소 대분류', sorted(places['place_1'].unique()), key='place_1')
    places = places[places['place_1'] == inputs['place_1']]
    inputs['place_2'] = st.selectbox('장소 중분류', sorted(places['place_2'].unique()), key='place_2')
    places = places[places['place_2'] == inputs['place_2']]
    inputs['place_3'] = st.selectbox('장소 소분류', places['place_3'].value_counts().index.tolist(), key='place_3')
    st.caption('일반 가정은 화재 유형 "건축,구조물" → 장소 "주거"에서 고르세요.')

# 1) 예측 결과: 왼쪽 결과 카드 + 오른쪽 원인별 확률 막대
proba = predict(bundle, inputs)[CLASSES]
top = proba.idxmax()
second = proba.drop(top).idxmax()
situation = (f"{inputs['month']}월 · {inputs['hour']}시 · {WEEKDAYS[inputs['weekday']]}요일 · {inputs['sido']} · "
             f"{inputs['place_2']} > {inputs['place_3']}")
st.markdown(f'##### 📝 {situation}')

left, right = st.columns([2, 3], gap='large')
with left:
    st.markdown(f'''
<div class="result-card" style="background:{COLORS[top]}">
  <div class="label">가장 가능성 높은 원인</div>
  <div class="cause">{ICONS[top]} {top} {proba[top]:.0%}</div>
  <div class="sub">두 번째 후보: {ICONS[second]} {second} {proba[second]:.0%}</div>
</div>
''', unsafe_allow_html=True)
    # 받침 유무에 따라 '란/이란'을 고름 (부주의란 / 전기적 요인이란)
    josa = '이란' if (ord(top[-1]) - 0xAC00) % 28 else '란'
    st.caption(f'**{top}{josa}?** ' + DESCRIPTIONS[top][0] + ' 자세한 설명은 아래 "📖 원인 종류 알아보기" 탭에 있습니다.')
    st.markdown(f'<div class="tip-card" style="border-color:{COLORS[top]}"><b>{ICONS[top]} {top} 예방 수칙</b><br>{TIPS[top]}</div>',
                unsafe_allow_html=True)
with right:
    st.markdown('**원인별 예측 확률**')
    prob_df = pd.DataFrame({'원인': CLASSES, '확률': proba.values * 100})
    prob_df['원인_표시'] = [f'{ICONS[c]} {c}' for c in CLASSES]
    prob_df['확률_label'] = prob_df['확률'].map('{:.1f}%'.format)
    st.altair_chart(bar_chart(prob_df.rename(columns={'원인_표시': '항목'}), '확률', '항목', color='원인'),
                    use_container_width=True)
    st.caption(f'상위 2개 후보 안에 실제 원인이 있을 확률은 약 {scores["top2"]:.0%}입니다 (2024년 화재 기준).')

st.divider()

# 2) 자세히 보기: 실제 기록 / 계절별 변화 / 모델 성능
same_place = data[(data['place_3'] == inputs['place_3']) & (data['place_2'] == inputs['place_2'])]
tab1, tab2, tab0, tab3 = st.tabs(['📍 이 장소의 실제 기록', '📅 계절에 따른 변화', '📖 원인 종류 알아보기', '📊 모델 성능'])

with tab0:
    st.markdown('소방청은 화재 원인을 12가지로 나누는데, 이 앱은 건수가 많은 3가지는 그대로 두고 나머지를 **기타**로 묶어 4가지로 예측합니다. '
                '아래 숫자는 2020\\~2024년 전국 실제 화재 건수입니다.')
    known_all = data[data['cause_group'] != '미상']
    cols = st.columns(4)
    for col, c in zip(cols, CLASSES):
        part = known_all[known_all['cause_group'] == c]
        # 기타는 여러 원인을 묶은 그룹이라 원래 원인(대분류)별로, 나머지는 세부 원인별로 건수를 보여줌
        top_items = (part['cause'] if c == '기타' else part['cause_detail']).value_counts().head(6)
        items = ''.join(f'<li>{name} <span style="opacity:0.6">({n:,}건)</span></li>' for name, n in top_items.items())
        with col:
            st.markdown(f'''
<div class="result-card" style="background:{COLORS[c]}; padding:12px 16px">
  <div class="cause" style="font-size:24px">{ICONS[c]} {c}</div>
  <div class="sub">{len(part):,}건 · 원인을 아는 화재의 {len(part) / len(known_all):.0%}</div>
</div>
''', unsafe_allow_html=True)
            st.markdown(DESCRIPTIONS[c][0])
            st.markdown(f'**이런 경우예요:** {DESCRIPTIONS[c][1]}')
            st.markdown(f'**{"포함된 원인" if c == "기타" else "자주 나온 세부 원인"}**<ul style="margin-top:4px">{items}</ul>',
                        unsafe_allow_html=True)
    st.caption('용어: 단락 = 합선 · 절연열화 = 전선 피복이 오래되어 약해짐 · 트래킹 = 플러그·콘센트에 쌓인 먼지와 습기로 전기가 새며 불꽃이 남 · '
               '자연발화 = 기름 묻은 천 등이 스스로 열을 내다 불이 붙음')

with tab1:
    st.markdown(f'**{inputs["place_2"]} > {inputs["place_3"]}** 화재 — 2020\\~2024년 전국 **{len(same_place):,}건**')
    k1, k2, k3 = st.columns(3)
    k1.metric('인명피해가 난 비율', f"{(same_place['casualties'] > 0).mean():.1%}")
    k2.metric('재산피해 중앙값', f"{same_place['damage'].median() / 10:,.0f}만원")
    k3.metric('원인 미상 비율', f"{(same_place['cause_group'] == '미상').mean():.1%}")
    known_place = same_place[same_place['cause_group'] != '미상']
    detail = (known_place.groupby(['cause_detail', 'cause_group']).size().sort_values(ascending=False)
              .head(5).reset_index(name='건수'))
    detail['건수_label'] = detail['건수'].map('{:,}건'.format)
    st.markdown('**자주 나온 세부 원인 TOP 5** (색 = 원인 그룹)')
    st.altair_chart(bar_chart(detail, '건수', 'cause_detail', color='cause_group', height=40 * len(detail) + 20),
                    use_container_width=True)

with tab2:
    st.markdown('왼쪽에서 고른 조건은 그대로 두고 **월만 1\\~12월로 바꿔** 예측했습니다. 빨간 점선이 지금 고른 달입니다.')
    by_month = pd.DataFrame([{'월': m, '원인': c, '확률': p * 100}
                             for m in range(1, 13)
                             for c, p in predict(bundle, {**inputs, 'month': m})[CLASSES].items()])
    lines = alt.Chart(by_month).mark_line(point=True, strokeWidth=3).encode(
        x=alt.X('월:O', axis=alt.Axis(labelAngle=0, labelExpr="datum.value + '월'")),
        y=alt.Y('확률:Q', title='예측 확률(%)', scale=alt.Scale(domain=[0, 100])),
        color=alt.Color('원인:N', scale=COLOR_SCALE, legend=alt.Legend(orient='top', title=None, labelFontSize=13)),
        tooltip=['월', '원인', alt.Tooltip('확률:Q', format='.1f')],
    )
    now = alt.Chart(pd.DataFrame({'월': [inputs['month']]})).mark_rule(color='red', strokeDash=[6, 4]).encode(x='월:O')
    st.altair_chart((lines + now).properties(height=340), use_container_width=True)

with tab3:
    st.markdown('2020\\~2023년 화재로 학습하고 **2024년 화재 34,561건**(원인 미상 제외)으로 평가한 결과입니다.')
    c1, c2, c3, c4 = st.columns(4)
    c1.metric('정확도', f"{scores['acc']:.1%}", f"{(scores['acc'] - scores['baseline']) * 100:+.1f}%p (찍기 대비)")
    c2.metric('찍기 기준 정확도', f"{scores['baseline']:.1%}", help='가장 많은 원인(부주의) 하나만 답할 때의 정확도')
    c3.metric('상위 2개 적중률', f"{scores['top2']:.1%}", help='모델이 꼽은 원인 2개 안에 정답이 있는 비율')
    c4.metric('매크로 F1', f"{scores['f1']:.3f}", help='4개 원인을 똑같은 비중으로 평균한 F1 (0~1)')
    st.info('장소·시간 정보만으로는 원인을 확정할 수 없어서, 이 앱은 **"우선 의심할 원인"을 알려주는 보조 도구**입니다. '
            '건수가 적고 성격이 다양한 "기타"(방화·가스 등)는 특히 맞히기 어렵습니다.')

st.caption('※ 과거 화재 기록에서 배운 경향으로 계산한 값이며, 실제 원인은 현장 조사로 확인해야 합니다.')

