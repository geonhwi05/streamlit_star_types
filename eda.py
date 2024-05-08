import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import plotly.express as px

df = pd.read_csv('./data/star_type_data.csv')

def create_hr_diagram(df):
    fig = px.scatter(df, x='Temperature (K)', y='Absolute magnitude(Mv)', 
                     color='Star color', title='Hertzsprung-Russell Diagram',
                     color_discrete_map={'Red': 'red', 
                    'Blue White': 'lightblue', 
                    'Blue white': 'deepskyblue',
                    'Blue white ': 'dodgerblue', 
                    'Blue White ': 'skyblue', 
                    'yellow-white': 'lightyellow',
                    'Yellowish White': 'lemonchiffon', 
                    'white': 'lightgray', 
                    'yellowish': 'gold',
                    'Yellowish': 'khaki', 
                    'Orange': 'darkorange', 
                    'Whitish': 'lightgray',
                    'White-Yellow': 'lightgoldenrodyellow', 
                    'white-Yellow': 'papayawhip',
                    'Pale yellow orange': 'moccasin',
                    'Orange-Red': 'orangered', 
                    'Blue ': 'royalblue', 
                    'Blue': 'cornflowerblue', 
                    'Blue-White': 'lightsteelblue',
                    'yellow white': 'goldenrod'})
    fig.update_traces(marker=dict(size=10))
    fig.update_layout(xaxis_title='Temperature (K)', yaxis_title='Absolute Magnitude (Mv)')
    return fig

def run_eda():
    st.subheader('Exploratory Data Analysis (EDA)')

    st.text('데이터프레임 보기 / 통계치 보기를 할 수 있습니다.')

    radio_menu = ['데이터프레임', '통계치']
    choice_radio = st.radio('선택하세요.', radio_menu)

    df = pd.read_csv('./data/star_type_data.csv')

    if choice_radio == radio_menu[0]:
        st.dataframe(df)
    elif choice_radio == radio_menu[1]:
        st.dataframe(df.describe())

    st.subheader('Hertzsprung-Russell Diagram')
    st.info('헤르츠스프룽-러셀 도표 : 항성천문학에서 항성의 절대등급과 표면온도의 관계를 나타낸 산점도')
    st.text('이 그래프는 별의 온도와 밝기를 나타내는 공간에 별을 표시합니다!')
    st.text('오른쪽 색깔을 한번 클릭하면 해당 색상을 제외하고 더블 클릭하면 해당 색상만 보여줍니다.')
    st.plotly_chart(create_hr_diagram(df), use_container_width=True)

    st.text('')
    st.text('')
    st.subheader('최대 / 최소 데이터')
    st.text('컬럼을 선택하면, 컬럼별 최대/최소 데이터를 보여드립니다.')
    column_list = df.columns.drop(['Star type', 'Star color', 'Spectral Class'])
    choice_column = st.selectbox('컬럼을 선택하세요.', column_list) 

    st.info(f'선택하신 {choice_column} 의 최대 데이터는 다음과 같습니다.')
    st.dataframe(df.loc[df[choice_column] == df[choice_column].max(), ])

    st.info(f'선택하신 {choice_column} 의 최소 데이터는 다음과 같습니다.')
    st.dataframe(df.loc[df[choice_column] == df[choice_column].min(), ])

    col = ['Temperature (K)', 'Luminosity(L/Lo)', 'Radius(R/Ro)',
       'Absolute magnitude(Mv)','Star color', 'Spectral Class']

    selected_column = st.selectbox("Select a column to compare:", col, key="column_selector")
    fig = px.box(df, x=selected_column, color='Star type')
    st.plotly_chart(fig)
    fig2 = px.histogram(df, x=selected_column, color='Star type', template='plotly_white', opacity=0.7)
    st.plotly_chart(fig2)

    def plot_comparison(x, y):
        if y == "":
            # y 선택 안했을 때
            if x in ['Star color', 'Spectral Class', 'Star type']:
                d = df[x].value_counts()
                fig_pie = px.pie(d, values=d.values, names=d.index, hole=0.4, opacity=0.6, title=f"{x} distribution")
                fig_pie.update_traces(textposition='outside', textinfo='percent+label')
                st.plotly_chart(fig_pie)
            else:
                fig = px.histogram(df, x=x, title=x)
                st.plotly_chart(fig)
        else:
            # y 선택 했을 때
            if y in ['Star color', 'Spectral Class']:
                # 'Star type', 'Spectral Class' 선택한 경우
                d = df[y].value_counts()
                fig_pie = px.pie(d, values=d.values, names=d.index, hole=0.4, opacity=0.6, title=f"{x} VS {y}")
                fig_pie.update_traces(textposition='outside', textinfo='percent+label')
                st.plotly_chart(fig_pie)
            else:
                # 다른 경우에는 히스토그램으로 표시
                fig = px.histogram(df, x=x, y=y, title=f"{x} VS {y}")
                st.plotly_chart(fig)

    # Streamlit UI
    selected_column_x = st.selectbox("Select x:", df.columns)
    # selected_column_x를 제외한 열들을 선택 목록에 추가
    selected_column_y = st.selectbox("Select y:", [""] + df.columns[df.columns != selected_column_x].tolist())

    # 함수 호출
    plot_comparison(selected_column_x, selected_column_y)





    
    
