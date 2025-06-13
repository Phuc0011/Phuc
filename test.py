from sklearn.calibration import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import base64
import matplotlib
import platform
import os

font_path = r"font/NanumGothic-Regular.ttf"

if not os.path.exists(font_path):
    import urllib.request
    urllib.request.urlretrieve(font_url, font_path)

# Đăng ký font
fm.fontManager.addfont(font_path)
plt.rc('font', family='NanumGothic')
plt.rcParams['axes.unicode_minus'] = False
# 한글 폰트 설정
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

set_background("anh titanic.jpg")

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
#객실 등급별 생존율
# --- 성별 생존율 ---
elif section == "성별 생존율 배우기":
    st.subheader("📊 성별에 따른 생존율")
    left, center, right = st.columns([0.3, 2.4, 0.3])
    with center:
        survival_by_sex = train.groupby("Sex")["Survived"].mean().sort_values(ascending=False)
        fig1, ax1 = plt.subplots(figsize=(8, 6), dpi=90)
        sns.barplot(x=survival_by_sex.index, y=survival_by_sex.values, palette="pastel", ax=ax1)
        ax1.set_ylabel("생존 확률")
        ax1.set_ylim(0, 1)
        for i, v in enumerate(survival_by_sex.values):
            ax1.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=10)
        st.pyplot(fig1)
    st.subheader("🔎 성별 생존율 배우기 요약")
    st.markdown("📝 **결론:** 여성 승객의 생존률은 약 74%로 매우 높은 반면, 남성 승객의 생존률은 약 19%에 불과하여 **성별이 생존에 큰 영향을 미친** 것으로 나타납니다.")


# --- 나이 분포 ---
elif section == "나이 분포 배우기":
    st.subheader("📈 승객 나이 분포")
    fig2, ax2 = plt.subplots()
    sns.histplot(train["Age"].dropna(), bins=20, kde=True, color="skyblue", ax=ax2, label="나이 분포")
    ax2.set_xlabel("나이")
    ax2.set_ylabel("인원 수")
    ax2.legend(title="범례")
    st.pyplot(fig2)
    st.subheader("🔎 나이 분석 요약")
    st.markdown("📝 **결론:** 대부분의 승객은 20~40세 사이였으며, 평균 나이는 약 29.7세입니다.")

# --- 객실 등급 생존율 ---
elif section == "객실 등급별 생존율":
    st.subheader("객실 등급별 생존율")
    
    survival_by_class = train.groupby("Pclass")["Survived"].mean()
    avg = train["Survived"].mean()

    fig, ax = plt.subplots(figsize=(8, 6))  # 크기 조정 가능
    sns.barplot(x=survival_by_class.index.astype(str), y=survival_by_class.values, palette="coolwarm", ax=ax)
    
    ax.axhline(avg, color='gray', linestyle='--', label=f"전체 평균 생존률: {avg:.2%}")
    for i, v in enumerate(survival_by_class.values):
        ax.text(i, v + 0.02, f"{v:.2%}", ha='center')
    
    ax.set_xlabel("객실 등급")
    ax.set_ylabel("생존 확률")
    ax.set_title("객실 등급에 따른 생존율")
    ax.legend()
    st.pyplot(fig)

    st.subheader("🔎 객실 등급별 생존율 요약")
    st.markdown("📝 **결론:** 1등급 객실 승객의 생존률은 약 63%로 가장 높았으며, 3등급 객실 승객의 생존률은 약 24%로 가장 낮았습니다. 이는 **객실 등급이 높을수록 생존 확률이 높아지는 경향**을 보여줍니다.")

# --- 탑승 항구 생존율 ---
elif section == "탑승 항구별 생존율":
    st.subheader("탑승 항구별 생존율")

    # 항구 코드와 한글 매핑
    embarked_map = {"C": "셰르부르", "Q": "퀸스타운", "S": "사우샘프턴"}

    # ✅ NaN 제거 후 groupby 수행
    survival_by_port = train[train["Embarked"].notna()].groupby("Embarked")["Survived"].mean()

    # ✅ 인덱스 수동 치환 (NaN은 이미 제거됨)
    survival_by_port.index = survival_by_port.index.map(embarked_map)

    # 평균 생존률
    avg = train["Survived"].mean()

    # 시각화
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(x=survival_by_port.index, y=survival_by_port.values, palette="Set2", ax=ax)

    ax.axhline(avg, color='gray', linestyle='--', label=f"전체 평균 생존률: {avg:.2%}")
    for i, v in enumerate(survival_by_port.values):
        ax.text(i, v + 0.02, f"{v:.2%}", ha='center')

    ax.set_ylabel("생존 확률")
    ax.set_title("탑승 항구에 따른 생존률")
    ax.legend()
    st.pyplot(fig)

    st.subheader("🔎 탑승 항구별 생존률 요약")
    st.markdown("📝 **결론:** 셰르부르 항구에서 탑승한 승객의 생존률은 약 55%로 가장 높았으며, 사우샘프턴 항구에서 탑승한 승객은 약 34%로 가장 낮았습니다. 이는 **탑승 항구에 따라 생존률에 차이가 있었음을 시사**합니다.")


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
