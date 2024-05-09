import streamlit as st
from streamlit_option_menu import option_menu
from home import run_home
from eda import run_eda
from ml import run_ml

def main():

    st.title('별 유형 예측:star2:')

    menu = ['홈', 'EDA', '별 유형 예측']

    with st.sidebar:
        # st.sidebar.title("별의 유형:star2:")
        choice = option_menu("메뉴", menu,
                         icons=['house', 'bi bi-file-bar-graph', 'bi bi-star-fill', 'bi bi-star-fill'],
                         menu_icon="bi bi-list", default_index=0,
                         styles={
        "container": {"padding": "4!important", "background-color": "#fafafa"},
        "icon": {"color": "black", "font-size": "25px"},
        "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#fafafa"},
        "nav-link-selected": {"background-color": "#38B3E3"},
        }
        )

    if choice == menu[0]:
        run_home()
    elif choice == menu[1]:
        run_eda()
    elif choice == menu[2]:
        run_ml()

if __name__ == '__main__':
    main()