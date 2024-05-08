import streamlit as st
import pandas as pd
import plotly.express as px

df = pd.read_csv('./data/star_type_data.csv')

# HR 다이어그램 생성
def create_hr_diagram(df):
    fig = px.scatter(df, x=df['Temperature(K)'], y=df['Absolute magnitude(Mv)'], text='Star', 
                     color='Star color', title='Hertzsprung-Russell Diagram')
    fig.update_traces(marker=dict(size=10))
    fig.update_layout(xaxis_title='Temperature (K)', yaxis_title='Absolute Magnitude (Mv)')
    return fig

# 메인 코드
def run_eda():
    # 사이드바에서 메뉴 선택
    menu = ['HR Diagram']
    choice = st.sidebar.selectbox('메뉴', menu)

    # HR 다이어그램 표시
    if choice == 'HR Diagram':
        st.header('Hertzsprung-Russell Diagram')
        st.plotly_chart(create_hr_diagram(df), use_container_width=True)
