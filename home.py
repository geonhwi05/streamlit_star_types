import streamlit as st

def run_home():
    st.subheader('	:star:별 유형 데이터를 분석하고 예측합니다!!	:star:')
    st.write('데이터는 캐글에 있는 6 class csv.csv 파일을 사용했습니다.')
    st.link_button('링크로 이동', 'https://www.kaggle.com/datasets/deepu1109/star-dataset')
    st.image('./image/star1.jpg', use_column_width=True)