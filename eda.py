import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import plotly.express as px

df = pd.read_csv('./data/star_type_data.csv')

def create_hr_diagram(df):
    fig = px.scatter(df, x='Temperature (K)', y='Absolute magnitude(Mv)', 
                     color='Star color', title='Hertzsprung-Russell Diagram',
                     color_discrete_map = {
    'Red': 'red',
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
    'Whitish': 'lightgray',  # 'White'는 'lightgray'로 매핑합니다.
    'White-Yellow': 'lightgoldenrodyellow',
    'white-Yellow': 'lightgray',
    'Pale yellow orange': 'moccasin',
    'Orange-Red': 'orangered',
    'Blue ': 'royalblue',
    'Blue': 'cornflowerblue',
    'Blue-White': 'lightsteelblue',
    'yellow white': 'goldenrod'
})
    fig.update_traces(marker=dict(size=10))
    fig.update_layout(xaxis_title='Temperature (K)', yaxis_title='Absolute Magnitude (Mv)')
    return fig

def plot_comparison(x, y):
    pie_title = ['Star color', 'Spectral Class', 'Star type']
    
    if y == "":
        if x in pie_title:
            d = df[x].value_counts()
            fig_pie = px.pie(d, values=d.values, names=d.index, hole=0.4, opacity=0.6, title=f"{x} distribution")
            fig_pie.update_traces(textposition='outside', textinfo='percent+label')
            st.plotly_chart(fig_pie)
        else:
            fig = px.histogram(df, x=x, title=x)
            st.plotly_chart(fig)
    else:
        if x in pie_title and y in pie_title:
            fig = px.histogram(df, x=x, y=y, title=f"{x} VS {y}")
            st.plotly_chart(fig)
        elif y in pie_title:
            d = df[y].value_counts()
            fig_pie = px.pie(df, names=y, values=df[x], hole=0.4, opacity=0.6, title=f"{x} VS {y}")
            fig_pie.update_traces(textposition='outside', textinfo='percent+label')
            st.plotly_chart(fig_pie)
        else:
            fig = px.histogram(df, x=x, y=y, title=f"{x} VS {y}")
            st.plotly_chart(fig)


def draw_heatmap_and_pairplot(selected_columns):
    if len(selected_columns) < 2:
        st.write("최소 2개의 컬럼을 선택해주세요!")
        return

    selected_df = df[selected_columns]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    sb.heatmap(selected_df.corr(), annot=True, cmap='Blues', ax=ax)
    ax.set_title("Correlation Heatmap")
    
    pairplot = sb.pairplot(selected_df, palette='Greens')
    
    st.pyplot(fig)
    st.pyplot(pairplot)

def run_eda():
    st.subheader('탐색적 데이터 분석 (EDA)')
    button_key = 'info_button'
    button = st.button('데이터에 대한 설명 보기', key=button_key)

    if 'show_info' not in st.session_state:
        st.session_state.show_info = False

    if button:
        st.session_state.show_info = not st.session_state.show_info

    if st.session_state.show_info:
        st.info('''Temperature (K) : 온도 (K)\n
Luminosity(L/L☉) : 휘도, 광도(L/L☉)\n
                ☉ : 태양 기호
    휘도는 별이 방출하는 총 에너지의 양을 나타내는 물리적인 측정치입니다.
    이는 별의 밝기를 나타내며, 일반적으로 태양의 휘도를 기준으로 표현됩니다.
    태양의 휘도는 'L☉'로 표기되며, 다른 별들의 휘도는 이에 대한 비율로 나타납니다.
    따라서 별의 휘도가 1이면 태양과 같은 밝기를 가진다는 의미입니다.
    
Radius(R/Ro) : 반지름(R/Ro)\n
Absolute magnitude(Mv) : 절대 등급(Mv)\n
                절대 등급(Mv)은 별의 밝기를 나타내는 지표 중 하나입니다
    별의 절대 등급은 별이 10파섹(약 32.6 광년) 떨어진 거리에서 보이는 등급을 나타냅니다.
    다시 말해, 별이 지구와 어떤 거리에 있든 간에, 별이 얼마나 밝게 보이는지를 표현하는 값입니다.
    각 별자리를 동일한 조건이라 가정할 때 빛을 밝기를 비교하는 척도가 절대등급 입니다.
Star type : 별 유형\n 
                0 => 갈색 왜성(Brown Dwarf) :  갈색왜성으로 알려진 이 유형의 별은 충분히 크지 않아서 핵 융합이 시작되지 않았습니다.
                                  따라서 별과 플라넷 사이의 중간 크기를 가지며, 주로 적색왜성과 비슷한 특성을 가지고 있습니다.

    1 => 적색 왜성(Red Dwarf) : 붉은 색깔의 적색왜성입니다. 
                               이러한 유형의 별들은 주로 주변의 빛을 흡수하고 붉은 빛을 방출합니다.

    2 => 백석 왜성(White Dwarf) : 백색 왜성은 별의 최종 단계로, 핵융합 과정이 멈추고 남은 것입니다.
                                 이러한 별들은 주로 매우 높은 밀도를 가지고 있으며, 표면 온도는 높지만 크기가 작습니다.

    3 => 주계열성(Main Sequence) : 주계열 별은 수소 원소를 헬륨으로 핵융합하는 과정에서 에너지를 생성하는 별의 주요 단계입니다.
                                 이러한 별들은 별의 생애 주기 중에 가장 안정적인 단계에 있습니다.

    4 => 초거성(Supergiants) : 초거성은 주계열에서 벗어난 대형 별들을 지칭합니다. 
                              이러한 별들은 주로 매우 크고 밝으며, 대부분의 경우, 생애의 끝에 폭발적인 슈퍼노바로 종료됩니다.

    5=> 극대거성(Hypergiants) : 극대거성은 초거성보다 더 크고 밝은 별들을 나타냅니다. 
                              이러한 별들은 매우 희귀하며, 대부분의 경우, 매우 짧은 시간 동안만 안정적으로 존재합니다.

Star color : 별 색\n 
Spectral Class : 스펙트럼 등급\n
            O: 매우 뜨거운 별 (온도가 매우 높음)
    B: 뜨거운 별
    A: 조금 덜 뜨거운 별
    F: 태양과 비슷한 온도를 가진 별
    G: 태양과 비슷한 별 (태양과 유사한 별들을 G 타입 별이라고도 합니다)
    K: 상대적으로 차가운 별
    M: 매우 차가운 별 (온도가 매우 낮음)''')


        
    

    st.markdown('------')
    
    st.text('데이터프레임 보기 / 통계치 보기를 할 수 있습니다.')

    radio_menu = ['데이터프레임', '통계치']
    choice_radio = st.radio('선택하세요.', radio_menu)

    

    df = pd.read_csv('./data/star_type_data.csv')

    if choice_radio == radio_menu[0]:
        st.dataframe(df)
    elif choice_radio == radio_menu[1]:
        st.dataframe(df.describe())
    st.markdown('------')
    st.subheader('Hertzsprung-Russell Diagram')
    st.text('')
    st.info('헤르츠스프룽-러셀 도표 : 항성천문학에서 항성의 절대등급과 표면온도의 관계를 나타낸 산점도')
    st.text('이 그래프는 별의 온도와 밝기를 나타내는 공간에 별을 표시합니다!')
    st.plotly_chart(create_hr_diagram(df), use_container_width=True)

    st.markdown('------')
    st.subheader('최대 / 최소 데이터')
    st.text('')
    st.text('컬럼을 선택하면, 컬럼별 최대/최소 데이터를 보여드립니다.')
    column_list = df.columns.drop(['Star type', 'Star color', 'Spectral Class'])
    choice_column = st.selectbox('컬럼을 선택하세요.', column_list) 

    st.info(f'선택하신 {choice_column} 의 최대 데이터는 다음과 같습니다.')
    st.dataframe(df.loc[df[choice_column] == df[choice_column].max(), ])

    st.info(f'선택하신 {choice_column} 의 최소 데이터는 다음과 같습니다.')
    st.dataframe(df.loc[df[choice_column] == df[choice_column].min(), ])

    st.markdown('------')
    st.subheader('선택한 컬럼과 Star type 간의 상관 관계를 나타냅니다.')
    st.text('')
    col = ['Temperature (K)', 'Luminosity(L/Lo)', 'Radius(R/Ro)',
       'Absolute magnitude(Mv)','Star color', 'Spectral Class']

    selected_column = st.selectbox("비교할 열을 선택하세요:", col, key="column_selector")
    fig = px.box(df, x=selected_column, color='Star type')
    fig.update_layout(title=f"{selected_column}에 따른 Star Type의 박스 플롯")
    st.plotly_chart(fig)

    fig2 = px.histogram(df, x=selected_column, color='Star type', template='plotly_white', opacity=0.7)
    fig2.update_layout(title=f"{selected_column}에 따른 Star Type의 히스토그램")
    st.plotly_chart(fig2)


    st.markdown('------')
    st.subheader('두 컬럼 간의 분포를 비교하여 시각화합니다.')
    st.text('')
    # Streamlit UI
    selected_column_x = st.selectbox("X축을 선택하세요!:", df.columns)
    # selected_column_x를 제외한 열들을 선택 목록에 추가
    selected_column_y = st.selectbox("Y축을 선택하세요!:", [""] + df.columns[df.columns != selected_column_x].tolist())
    
    # 함수 호출
    plot_comparison(selected_column_x, selected_column_y)

    st.markdown('------')

    st.subheader('선택한 컬럼에 대한 상관 관계를')
    st.subheader('히트맵과 페어플롯을 통해 시각화합니다.') 
    st.text('')
    selected_columns = st.multiselect('컬럼을 선택하세요!', df.columns.drop(['Star color', 'Spectral Class']))

    # 사용자가 선택한 컬럼에 대한 히트맵 그리기
    if selected_columns:
        draw_heatmap_and_pairplot(selected_columns)
    else:
        pass

