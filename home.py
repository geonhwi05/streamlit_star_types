import streamlit as st

def run_home():
    st.subheader('별 유형 데이터를 분석하고 예측합니다! ')
    st.write('데이터는 캐글에 있는 6 class csv.csv 파일을 사용했습니다.')
    st.image('./image/star.jpg', use_column_width=True)
    st.link_button('데이터 출처', 'https://www.kaggle.com/datasets/deepu1109/star-dataset')