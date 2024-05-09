import streamlit as st
import joblib
import numpy as np
from sklearn.preprocessing import OneHotEncoder

def run_ml():
    st.subheader('별의 유형 예측하기!')
    st.text('')
    st.text('기본값은 태양을 기준으로 입력되어 있습니다!')
    st.text('')
    temperature = st.text_input(
    "온도(Temperature (K))를 입력하세요",
    value="5778",
    help=("항성 분류와 대응되는 표면온도(K)를 입력하세요.\n"
          "- 태양의 온도는 약 5778K 입니다.\n"
          "- B: 3만K (청백색)\n"
          "- A: 1만K (백색)\n"
          "- F: 7500K (황백색)\n"
          "- G: 6000K (노란색)\n"
          "- K: 4500K (주황색)\n"
          "- M: 3500K (붉은색)")
)
    temperature = int(temperature) if temperature else 6000
    st.write("선택된 표면온도:", temperature, "K")


    st.text('')
    luminosity = st.text_input(
    "광도(Luminosity(L/L☉))를 입력하세요",
    value="1.0",
    help=("값의 범위는 0에서 900000입니다.\n"
        "- L☉(태양 광도) 은/는 태양에 대한 밝기를 나타냅니다.\n"
        "- 태양의 광도는 1L☉ 입니다.\n"
        "- 만약 어떠한 별의 광도가 2L☉ 이라면 2×태양의 밝기와 같습니다.")
)

    luminosity = float(luminosity) if luminosity else 450000
    st.write("선택된 광도:", luminosity)

    st.text('')
    radius = st.text_input(
    "반지름 (Radius(R/R☉))을 입력하세요",
    value="1.0",
    help=("값의 범위는 0에서 2000입니다.\n"
          "- 태양의 반지름은 1R☉ 입니다.\n"
          "- 1R☉ = 6.957 x 10⁸ m 입니다.\n"
          "- 지구의 반지름은 대략적으로 0.009155 R☉입니다.")

)
    radius = float(radius) if radius else 1000
    st.write("선택된 반지름:", radius)

    st.text('')
    absolute_magnitude = st.text_input(
    "절대 등급(Absolute magnitude(Mv))을 입력하세요",
    value="4.83",
    help=("- Absolute magnitude(Mv) 값의 범위는 -11에서 20입니다.\n"
          "- 태양의 절대 등급은 4.83 입니다.\n"
          "- 절대 등급은 별의 실제 밝기를 나타내는 등급입니다."
          "절대 등급은 별이 지구로부터 10파섹(32.6광년) 떨어져 있을 때의 밝기를 기준으로 합니다."
          "절대 등급이 낮을수록 별은 밝고, 절대 등급이 높을수록 별은 어둡습니다")
)
    st.text('')
    absolute_magnitude = float(absolute_magnitude) if absolute_magnitude else 10
    st.write("선택된 절대 등급:", absolute_magnitude)

    star_colors = [
    'Red', 'Blue White', 'White', 'Yellowish White', 'Blue white',
    'Pale yellow orange', 'Blue', 'Blue-white', 'Whitish',
    'yellow-white', 'Orange', 'White-Yellow', 'white', 'Blue ',
    'yellowish', 'Yellowish', 'Orange-Red', 'Blue white ', 'Blue-White'
]
    

    st.text('')
    selected_color = st.selectbox("별의 색상을 선택하세요", star_colors)

    st.write("선택된 별의 색상:", selected_color)

    encoder = OneHotEncoder(categories=[star_colors], sparse=False)
    selected_color_encoded = encoder.fit_transform([[selected_color]])

    st.text('')
    spectral_classes = ['M', 'B', 'A', 'F', 'O', 'K', 'G']

    selected_class = st.selectbox(
        "스펙트럼 등급(Spectral Class)을 선택하세요",
        spectral_classes,
        index=spectral_classes.index('G'),
        help=("별의 스펙트럼 클래스를 선택하세요.\n"
            "다음 중 하나를 선택할 수 있습니다:\n"
            "- M: 적색 별\n"
            "- B: 청색 별\n"
            "- A: 백색 별\n"
            "- F: 황백색 별\n"
            "- O: 푸른색 별\n"
            "- K: 주황색 별\n"
            "- G: 노란색 별")
    )
    
    st.write("선택된 스펙트럼 등급:", selected_class)

    selected_class_mapping = {'M': 0, 'K': 1, 'G': 2, 'F': 3, 'A': 4, 'B': 5, 'O': 6}
    selected_class_encoded = selected_class_mapping[selected_class]

    st.text('')
    st.text('')
    if st.button('예측하기'):
        regressor = joblib.load('./model/regressor.pkl')

        new_data = [temperature,
                    luminosity,
                    radius,
                    absolute_magnitude,
                    selected_class_encoded]

        new_data.extend(selected_color_encoded[0])
        new_data = np.array(new_data).reshape(1,-1)

        y_pred = regressor.predict(new_data)

        class_labels = [
            '갈색 왜성',
            '적색 왜성',
            '백석 왜성',
            '주계열성',
            '초거성',
            '극대거성'
        ]

        explanations = [
            "갈색왜성으로 알려진 이 유형의 별은 충분히 크지 않아서 핵 융합이 시작되지 않았습니다. 따라서 별과 플라넷 사이의 중간 크기를 가지며, 주로 적색왜성과 비슷한 특성을 가지고 있습니다.",
            "붉은 색깔의 적색왜성입니다. 이러한 유형의 별들은 주로 주변의 빛을 흡수하고 붉은 빛을 방출합니다.",
            "백색 왜성은 별의 최종 단계로, 핵융합 과정이 멈추고 남은 것입니다. 이러한 별들은 주로 매우 높은 밀도를 가지고 있으며, 표면 온도는 높지만 크기가 작습니다.",
            "주계열 별은 수소 원소를 헬륨으로 핵융합하는 과정에서 에너지를 생성하는 별의 주요 단계입니다. 이러한 별들은 별의 생애 주기 중에 가장 안정적인 단계에 있습니다. 태양은 G형 주계열성입니다.",
            "초거성은 주계열에서 벗어난 대형 별들을 지칭합니다. 이러한 별들은 주로 매우 크고 밝으며, 대부분의 경우, 생애의 끝에 폭발적인 슈퍼노바로 종료됩니다.",
            "극대거성은 초거성보다 더 크고 밝은 별들을 나타냅니다. 이러한 별들은 매우 희귀하며, 대부분의 경우, 매우 짧은 시간 동안만 안정적으로 존재합니다."
        ]

        images = [
            "data:image/jpeg;base64,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",
            "data:image/jpeg;base64,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",
            "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSxSs2MuSkxHFLZQ3VYK004bEHDEBRg0SQKig&s",
            "data:image/jpeg;base64,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",
            "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wCEAAkGBxMSEhUTExMVFhUVGBcWFxcXFxgXFxUYFRgYFxUXGhgYHSggGBolHRcXITEhJSkrLi4uGB8zODMtNygtLisBCgoKDg0OGhAQGy0lHR0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLf/AABEIAOEA4AMBIgACEQEDEQH/xAAbAAACAwEBAQAAAAAAAAAAAAADBAECBQAGB//EADIQAAEEAQIEBAUEAgMBAAAAAAEAAgMRITFBBFFhcRKBkfATIqGxwTLR4fFScgWiskL/xAAaAQADAQEBAQAAAAAAAAAAAAACAwQFAQAG/8QAHxEAAwEBAAMBAQEBAAAAAAAAAAECAxEEEiETMRRB/9oADAMBAAIRAxEAPwD5PH4bFg1jxURZ51YwuCsWUpjbnljvtp5laSIWyW376KwZuuYiht+6TEhboqzH7q4GylqL4cm9b1vfuExSLdEMYN78tb21V42ZF3XTlzVnXodvvofsrBqdMCnRMDi04cRYIJFjDhRBrZcGokbeYtEbHf3TpzFOgIYrhiKGojWYpOWYt2B8N/f+KGikxJhkYXBiP8gHoAYxQY02yPKgxLv5fDn6fRX4aGWJ0swF3wxm8dtBn9rQvIJaCJjUFmet6UmzH/SoRttt56oHmGrFCxD8Kac1U8CVUBqxYsVCmXNUOABBaTijmtd66JVQMVC7KFEgOzZFkWNwa07goLxZ5JuZ1gXrnYAVtpvrk9EHwpLkaqAFvvsqOCYr6oTgltBphIoXOaXVhpAJ5F11foVQjG3Pry1UQO59e/RFDUKR1vjKMaiNaoITXxz4AzYEmrxdAXXOhVpsoXTBfDIo873GyuGLmtTA8vruqJkVVFGssojWKzGI7GaqmIEVRQM7IrWK7Y0ZkZPvkqJjgirKfCwNM56j9lYNRWxovw01SJrQD8EjXHu1eKMX8wNZ0rWsfVMGM7+6UtjXuC3oLxxZVTH0T0cedFX4S7wH9BIxKvgT0jOQobDWulqnw17gS0EHMVJI/efynnRob4x5IfUYtBAsQ3sT0jOiE6NA4HTYkY8XshFidc1BexT1mOmxR7EMtTT2qjgp6gcqAv7oD2pos7KjWiwHYG5GtJNSMVCcBog8q+/p6orShNRxj0r1SZQ2mXJur7dldrM0M/lRk1Z2rOwGyJGE+JE0y4j6GuqKxqtGNEVrVVnJPVHNajsYpazA1/CO2PNUrJkluyGRozGKzGJiONM4TXoDbGjCNHihRmQr3UT1oLMjRPhJoRK7YkPsKegrFDkKnwloww/MO6oYVz3XQfcQ+F6LpOHINEZx9dE66JRJGd76fhd9gloJTAeFoDaIuzf6rOMbUk3xrSfEgyRol8GzoZz40B7FovjS72Lo+LEHMVAwWL03rB+yeYxt/NdUdBeaxqdLS8jffNLpdKZsSIFEVnFHlWuOqC9icez+uSCQkXBRNC7mEajrpsfwgvaE4+71vb8DVLPUtyOlmewJtr/E5xe5xJz4tSTeSb13SzQjMbj+fwpZRTTDREjINdjnFEfhEY37bdRhV4doseKw3mACfQkJvheGLmudbR4QDRIBNmvlB1OVTCJ6Z0PVHY1Dibrp75JmNitzkkthI2euyZjYqRNTUbFRwj0ssxiZiYuijTccS42R3YbgpSwOoD5mluQDr30UshtGhhT0UCmq1LbR5KrSQm3h0xBw9EGro6c07HCjiFT1sPjx/wDoizh/mBqrOiGeFWtFBkd1Uwpa1+jX43wxn8MgSQLddAgP4dNnYRfjNGDJCl3RrblgSMsKpjTpNUuTKkjS8sXn1Wm+MY16pSRioT6MizNkYgPYtCRiWe1dK4oTnjrcGwDja9u4QHM6Jx7UFzUukUzQm9qWkC0ZSK0yKAoVYzdnmkntUlyUxRmAI8WM/dQ9rb+UkihkgDNC9CcXeVdiihFdMLCy/wAJwx+EluMGjWbI3vkl4XVsno5DVZrWutaqzNEulMhjU1G1UYxMwt3+isRHdBY2pyNiFC3ROQts8vwi6Q6UFjYtKFhJuh5Ch6JaBi1eHjU2tCoTpl4YeidihV4osDI7bpuKNZ96Grh44JkKO2FMRxI7Y1JWxqZ+KKxQ5CqYE+1igsSv2+lD8ZcM10KXfEtZ0SDLEnRsS6+KY0kSTni5jqtmaJJSxKyLMnfDhgzxitOf8JGVi2uJiWZKxaGV9Mul6szZGJWRq0ZGpSUKlDs6EHtQpG41/hNPCWkXGWQwkH/GSPY+RtVGATkCs9dVjytWgZSARZo6pKQCrzd6eWt99lNoirJsymIwFY/lCjCYYL0WfCLrDRsxabjZi6IBwOVir+/1SsX0TUDbI0z1pXZok0Y1AE5CErCtDgvD4h478O9VddLVP8RDoGiCdgCVhT8DVyv4Q2/o7wsa1YGJLhWrUgas/Wirx4GYWJ6JiBA1PRtWZrZ9D42RdjUQNXNVlFVGnM8OUUpXIQipahPajqjgjmgLnohNGkJmLWlakJ2q/GzI8rJGRNGsziI1szhJEDxC6rroFo53z6YG+f3hizx0kpQvT/8APTxuawNaLAzRu15ziFZhbuetcFVCzv1T6Z0oSrwnZRulJE5lObE5Eu47dd+abfVHW9uXW0s9t/3z7pGhbmzJaPfZNB+bHy4xXpk9QgNf3/f6I0bzzWdBdbYeMZ/J/KagKBE84ymo3nmVfmR6dGIhphORCtdkrFK7mnYpXXklPRDoNwBaPDBIQzO/yK0uGncDkn1Qad4RvnfppcM1akDVn8NKeZ9VpQSHmfVZmvTS8ZSPwtTjAlIXdU2wrM16fQ+PzgVSoUqQtOXKFK8eOUEKVBXUeYGQJGdh5FOylITlW49M7yucEOIjPI+hWbxMTv8AE+hWhOVl8WVp5dPnPJ4ITxO/xPokJo3cim5DWcGtjv5ckhIVox0hn+i8sZ5FLPh0/wC2NBevVFlSkqYy3NMDJGc4vX+0s9hRZClpDySbLI6ZrT17JiJ9A9devJKtKPEfP1x1WbBfQ0zbqmYylI3c+VDonOFkLSHCsZzkHoQdVdmSaIZjKciSMZTrARg7KlEWiHoQn+GKzYXJ6HuhohtfTa4V61IHLE4QrUges7WfpX418NiFycjcsuGRPRPWbrB9D42g60qUJrkQOUVTw0prpZcoXIAjlVxXFyFI5HM9YF0kikrkhOUxK9ITvWhjBk+VoKzFZXFuWhO9ZfEvWlij57yK6ITFKujLrrNZrehklMSFJyOV6Xz4IzFJSl5pAQBQBF2c2eV+90eVJyjGvkiZdmLyOSz0w8+9UE4Dh4Q4kA3n5aySKxpg2p9GWwjMEZoHGSRqLxV2NQM/dXYrngyIhL4mEFxbXiHisAHLdazqhArOhl1DUZH7fymodEo28AnS8DNI0blZlRNpJoMs+n2/pNwCzqB326rOY7OE4JLrAGAMY0FX3VaZFcj0Tk9G5ZkTk3C/+v5XWQ6Sa/DyrU4aRYEUq0OHmU2uYuL9Wb8T07FKsSGZOxSrP0zNbDfhsMkRmvWWyZGbMo7xNXPyTRD1Bek2TZUOmSvx+j/9K4MukQXyoDpkvJMnRiS6+SXllSU0imWRJTzK3ODJ336D4iVZ75BRsG9s6G98ZV55ElI9X55mVVdYKVyWkcRfUctjnyV5HpWVyp4NzkBIRXVLPJOPIBGlKVevMuzQCerNXXb8bJZ0lCuevXNo0hS8nlspNS3NGdeNfJELjQzeK10A26apUd/47ojHLNll9SNNITUTtrCQDkxG5UxQi5NCN/v9kyxyz4z7+yYY5WxfSO4NGN/p0TrHGvFtdXjWuSyo3f0mo3ppJcGnE5NxPWXFLomYpF5rpFccNqCZPRcQsFkqai4hT3l36ejRybsc6K2ZYrOJR2cSpqxKY8k2I5sjuFV02VnR8Rkd1D+IyUv8vo7/AE/B90yA+ZJP4hAdPabORPfkdGpZ0lNMqOlwHYOaq8/0lHyKjPMmu2wsn6fF4hrVb9+yTkkxSl0l49Eq96olHZkrI9B4hpFWCLyNdDoQoe9AleTuTXul1lUSVdzB3qt89OSXnFcsgHBv8q0hQHFDTK4QInPspaVHeUtIpNWVQjHDkZp8/wCff0SocjwtBNWBqbPQXWNzostM06Q43NkkCgBVUTVDAA152rxSkAjY1y2uko1yZ4eUi9dCOeCDYo/fZPmiepGI3JhjkmHIzHquLJrkfif75JqBxOBv1oLNYcDqjxSKqb6ia4NNh12rY3zqkxE76LPbITbibzkk5JNn90eOTrv7ympklwaTJUUSUs5kiK2Re4TVmaLZUVsyzhKiNkXHKFOGaMfEZCiSfJ7pBsuitLJ8xrn2Qei6c9WNPlKGJyDYNEZHRKmZUdIj9UeUMYdIhOfYJx75IJehh1mtb5a+SIbOYV0pBwcjQg/YpaR6rK4gkGwRghAe9eHzBaRyE8qhcqPkvU6YF8uSFsomSrjnp6pdzkR5NXsgymsEURrz873SboomSC4keHa789ErIdUVrC7AGxPk0WT6BLvd75qTRlMIxQUVpQ2gZz66qQVmo06Q5BGXAkAmtaF0M6+iK2S9ToKH7KOE4v4bCWPe17vE11YaY3CiLuzZ2qkJr02WIpDQdsiNclA5F8SdNCKkd8Q2wORNosb0oH7WSBdbedIjXYVMWJqR+NyYjf12Wc16PG9VzfSaoHmvR2yCjre38rPa9EEib0nqB8SK/wARJscKv1/dEnkZ4j4C7w2a8VXW10vdFvMZ+JnVWkkOvU/RI/EypfJnK6D+Y1JKMUhmVKGRR8ReCWY0ZFT4tGwl/iKDIvNhrMNNOXGzqdTdknmUB71UuzqhOcgdDZguXqjnit7+nX8KjneyhuKTVDZknW+gtClkJNkkk7nJJ3Ro5w0O+RpsEW6zVirFECxrm9EoXaHrzz17JFUOmSHOIH3Qg8i6NYz2KlxQ3ZU1vo+UY/iRWv0OMbdkGt6xz2tWaN9r+tKBM0mg7pLJOlkmhpn8Irdq2u/39EuMCzvp+SrFwvGnVMTFtDDTnCLFJWwPdKtKPJeDgA6AEYrGl2PNGmJcjDH7evXfHLkih6TY7qjNdyT4oTUjTDjX+UVjkqCiB/NPm+CXIy16OxwrXPJIserB6ctBTgfbIKqs2M3is3j09FPio/RKRvG90j8Y6MPcIy4sv5S7BrqAmrQW4Csl+uqkuyc+9km2TKl0uV39Dn5h3ScleRjm0XNI8Qtt4scxzCUvkofKTVk8ha9+h30G5vlJFg0atpsHqCpaAWlxJBvGLDqGl7bJN0l5JVfEuOzvoGc9Dc5ULlQlLqw1ITxKpKG4qC5JdhqSShvdqMVg+7zuuc5Ce5JqhsySRpp23FIblYuFVvet7cq/KrNKXVZJoACzoBoOwSqYxIxyTVZ7eln3yXNon8rmO1PiIO2tm9c7KMeah6aJcnN/0rAqn3VrHXfP2wjTBaDMeKqhd3efSkaMHNNNjpdXiqI1SzW4uxqRW+xuuWdURp72jTFtB2/RFaUFprUXYPSjseqswpssTSHJGUGmwfECcXijRBvfQ45hVvugAnVXJRqhTkYc8YrlnN2fwpadhZzp9vPVLWiMeRkGiN9wbwQdimKwXIcPNfx+VzXWgtkzdA50z/a4OTFYDkZLs8lzn5QPief7DRS59E2EXuc9A7M3kChed+g9VUSIJfZvzUEr3ue9QviXByF41N++697nvUMT+q7sad7zflap4/t9wqOwSOXUHI6obnIXYSkuXLrvf1QnPXOcaB209P7SXQSkK5zdr0G4138kM17Pu1VUKU6DSJcVS1xNH6FDJQug0jMGoTLtH9mrlylRcdxv63/7FVH6R/sfsFy5GgSY1duq5cjQujU/5XWP/Rn/AJCUC5cmP+iF/A8P6XeSoNVy5EgWGd+kdyobofL8rlyNAlQrLlyJHAkOp7FH4zfv+y5cvHBYbKRofe4XLkZ4quXLl44Qrv8A0N/2d9mrlyFnUBapj/8Ar/X8hcuSmGQ39J7j8oZXLkDCR0uvkP8AyFMv6W+9yuXIWEj/2Q==",
            "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSj3LkjpCi4A1he5csIEYiHAeFeDW2EMyUnHQ&s"
        ]

        st.success(f'예측된 별의 유형은 {class_labels[y_pred[0]]} 입니다!')
        st.info(f'{explanations[y_pred[0]]}')
        st.image(images[y_pred[0]], caption=class_labels[y_pred[0]], use_column_width=True)
        