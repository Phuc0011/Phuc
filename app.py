from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import base64
import matplotlib
import platform

# ---------------------------
# 한글 폰트 설정
# ---------------------------
if platform.system() == 'Windows':
    matplotlib.rc('font', family='Malgun Gothic')
elif platform.system() == 'Darwin':
    matplotlib.rc('font', family='AppleGothic')
else:
    matplotlib.rc('font', family='NanumGothic')

# ---------------------------
# 1. 페이지 설정
# ---------------------------
st.set_page_config(
    page_title="🚢 타이타닉 생존자 대시보드",
    layout="wide",
    page_icon="🚢"
)

# ---------------------------
# 2. 배경 이미지 설정
# ---------------------------
def set_background(image_path):
    with open(image_path, "rb") as img:
        encoded = base64.b64encode(img.read()).decode()
    st.markdown(f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-attachment: fixed;
            background-position: center;
        }}
        .glass {{
            background-color: rgba(255, 255, 255, 0.85);
            padding: 2rem;
            border-radius: 1rem;
            backdrop-filter: blur(5px);
            box-shadow: 0 0 15px rgba(0,0,0,0.2);
        }}
        </style>
    """, unsafe_allow_html=True)

set_background("2.jpg")

# ---------------------------
# 3. 데이터 불러오기
# ---------------------------
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
gender_submission = pd.read_csv("gender_submission.csv")

# ---------------------------
# 4. 사이드바 (메뉴 구성)
# ---------------------------
st.sidebar.title("📊 타이타닉 대시보드")

section = st.sidebar.radio("🗂 분석 항목 선택", [
    "데이터 개요",
    "성별 생존율 배우기",
    "나이 분포 배우기",
    "객실 등급별 생존율",
    "탑승 항구별 생존율",
    "상관관계 히트맵 분석",
    "생존 확률 예측하기",
    "이름으로 검색"
])

st.sidebar.markdown("### 📌 데이터 요약")
st.sidebar.markdown(f"- 총 승객 수: `{len(train)}`명")
st.sidebar.markdown(f"- 생존자 수: `{train['Survived'].sum()}`명")
st.sidebar.markdown(f"- 생존률: `{train['Survived'].mean():.2%}`")

# ---------------------------
# 5. 메인 콘텐츠
# ---------------------------
st.title("🚢 타이타닉 생존자 분석 대시보드")
st.markdown("<div class='glass'>", unsafe_allow_html=True)

# --- 데이터 개요 ---
if section == "데이터 개요":
    st.subheader("📄 학습 데이터 미리보기")
    st.dataframe(train.head(10).style.background_gradient(cmap='Blues'), use_container_width=True)

    st.subheader("🧪 테스트 데이터")
    st.dataframe(test.head(10).style.background_gradient(cmap='Purples'), use_container_width=True)

    st.subheader("📁 제출 양식 예시")
    st.dataframe(gender_submission.head(10).style.background_gradient(cmap='Greens'), use_container_width=True)

# --- 성별 생존율 ---
elif section == "성별 생존율 배우기":
    st.subheader("📊 성별에 따른 생존율")
    survival_by_sex = train.groupby("Sex")["Survived"].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(x=survival_by_sex.index, y=survival_by_sex.values, palette="pastel", ax=ax)
    ax.set_ylabel("생존 확률")
    ax.set_ylim(0, 1)
    for i, v in enumerate(survival_by_sex.values):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center")
    st.pyplot(fig)
    st.subheader("🔎 성별 생존율 배우기 요약")
    st.markdown("📝 **결론:** 여성 승객의 생존률은 약 74%, 남성 승객은 약 19%.")

# --- 나이 분포 ---
elif section == "나이 분포 배우기":
    st.subheader("📈 승객 나이 분포")
    fig, ax = plt.subplots()
    sns.histplot(train["Age"].dropna(), bins=20, kde=True, color="skyblue", ax=ax)
    ax.set_xlabel("나이")
    ax.set_ylabel("인원 수")
    st.pyplot(fig)
    st.subheader("🔎 나이 분석 요약")
    st.markdown("📝 **결론:** 대부분의 승객은 20~40세 사이였으며, 평균 나이는 약 29.7세입니다.")

# --- 객실 등급별 생존율 ---
elif section == "객실 등급별 생존율":
    st.subheader("🚪 객실 등급별 생존율")
    survival_by_class = train.groupby("Pclass")["Survived"].mean()
    avg = train["Survived"].mean()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(x=survival_by_class.index.astype(str), y=survival_by_class.values, palette="coolwarm", ax=ax)
    ax.axhline(avg, color='gray', linestyle='--', label=f"전체 평균 생존률: {avg:.2%}")
    for i, v in enumerate(survival_by_class.values):
        ax.text(i, v + 0.02, f"{v:.2%}", ha='center')
    ax.legend()
    st.pyplot(fig)
    st.subheader("🔎 객실 등급별 생존율 요약")
    st.markdown("📝 **결론:** 1등급 객실 승객의 생존률은 약 63%, 3등급은 약 24%입니다.")

# --- 탑승 항구별 생존율 ---
elif section == "탑승 항구별 생존율":
    st.subheader("⚓ 탑승 항구별 생존율")
    embarked_map = {"C": "셰르부르", "Q": "퀸스타운", "S": "사우샘프턴"}
    survival_by_port = train[train["Embarked"].notna()].groupby("Embarked")["Survived"].mean()
    survival_by_port.index = survival_by_port.index.map(embarked_map)
    avg = train["Survived"].mean()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(x=survival_by_port.index, y=survival_by_port.values, palette="Set2", ax=ax)
    ax.axhline(avg, color='gray', linestyle='--', label=f"전체 평균 생존률: {avg:.2%}")
    for i, v in enumerate(survival_by_port.values):
        ax.text(i, v + 0.02, f"{v:.2%}", ha='center')
    ax.legend()
    st.pyplot(fig)
    st.subheader("🔎 탑승 항구별 생존률 요약")
    st.markdown("📝 **결론:** 셰르부르 항구 승객 생존률 약 55%, 사우샘프턴 약 34%.")

# --- 상관관계 히트맵 분석 ---
elif section == "상관관계 히트맵 분석":
    st.subheader("📌 주요 Feature 간 상관관계 히트맵")
    corr_data = train.copy()
    for col in ["Sex", "Embarked"]:
        le = LabelEncoder()
        corr_data[col] = le.fit_transform(corr_data[col].astype(str))
    selected_features = ["Survived", "Pclass", "Sex", "Age", "Fare", "Embarked"]
    corr_matrix = corr_data[selected_features].corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("📈 Feature 상관관계 히트맵")
    st.pyplot(fig)
    st.subheader("🔎 해설")
    st.markdown("""
    - 색상이 진할수록 상관관계가 높음을 의미합니다.
    - 양의 상관: 값이 함께 증가.
    - 음의 상관: 반대로 움직임.
    """)

# --- 생존 확률 예측하기 ---
elif section == "생존 확률 예측하기":
    st.subheader("🧮 나의 생존 확률 예측하기")
    model_data = train.dropna(subset=["Age", "Embarked"])
    le_sex = LabelEncoder()
    le_embarked = LabelEncoder()
    model_data["Sex_enc"] = le_sex.fit_transform(model_data["Sex"])
    model_data["Embarked_enc"] = le_embarked.fit_transform(model_data["Embarked"])
    X = model_data[["Pclass", "Sex_enc", "Age", "Fare", "Embarked_enc"]]
    y = model_data["Survived"]
    model = LogisticRegression(max_iter=500)
    model.fit(X, y)
    st.markdown("**🎯 나의 정보를 입력하세요**")
    col1, col2 = st.columns(2)
    with col1:
        pclass = st.selectbox("좌석 등급 (Pclass)", [1, 2, 3])
        sex = st.selectbox("성별", le_sex.classes_.tolist())
        age = st.slider("나이", 0, 80, 30)
    with col2:
        fare = st.number_input("운임 요금", 0.0, 600.0, 30.0)
        embarked = st.selectbox("출발 항구", le_embarked.classes_.tolist())
    if st.button("예측하기"):
        input_df = pd.DataFrame([{
            "Pclass": pclass,
            "Sex_enc": le_sex.transform([sex])[0],
            "Age": age,
            "Fare": fare,
            "Embarked_enc": le_embarked.transform([embarked])[0]
        }])
        prob = model.predict_proba(input_df)[0][1]
        st.success(f"🚀 예측된 생존 확률: **{prob:.2%}**")
        st.markdown("""
        - ⚠️ 간단한 예측 모델입니다.
        - 학습 데이터의 패턴을 기반으로 합니다.
        """)

# --- 이름 검색 ---
elif section == "이름으로 검색":
    st.subheader("🔍 승객 이름 검색")
    name = st.text_input("이름을 입력하세요 (일부 단어도 가능):")
    if name:
        result = train[train["Name"].str.contains(name, case=False, na=False)]
        if len(result) > 0:
            st.success(f"{len(result)}명의 승객이 검색되었습니다.")
            st.dataframe(result.style.background_gradient(cmap="Oranges"), use_container_width=True)
        else:
            st.warning("검색 결과가 없습니다.")

st.markdown("</div>", unsafe_allow_html=True)
