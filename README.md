# 별 유형 예측 프로젝트
우주 유튜브 보다보니까 너무 재밌어서 별에 대해 검색하다가 관심이 좀 많아졌습니다.  
그래서 별에 관한 데이터로 프로젝트를 해보려고 맘 먹었습니다.  

---
## 1. 프로젝트 소개
이 프로젝트는 별의 유형을 예측하는 예제 프로젝트입니다.  
데이터를 다양한 측면에서 탐색하고 분석하여 데이터를 시각화합니다.  
사용자는 주어진 별의 특징을 입력하고, 모델이 해당 별의 유형을 예측합니다.  

------

## 2. 기획
- 데이터 수집: 캐글 등의 온라인 데이터 플랫폼에서 관련 데이터셋을 찾아봤습니다.
-  https://www.kaggle.com/datasets/deepu1109/star-dataset
- 프로젝트 목표 설정:  
1. 데이터를 분석하고 시각화 합니다.  
2. 별 유형을 예측하는 머신러닝 모델을 개발하여 사용자가 입력한 별의 특징을 기반으로 유형을 예측합니다.

------

## 3. 설계
- 사용자가 원하는 데이터를 데이터프레임에서 추출해 제공합니다.
- 다양한 측면에서 데이터를 탐색하고 분석해서 데이터를 시각화합니다.  
- 데이터를 불러와서 결측치 처리, 이상치 제거, 데이터 스케일링 등의 전처리를 수행하는 방법을 설계합니다.  
- 별 유형을 분류하기 위한 적합한 머신러닝 알고리즘을 선택합니다.

------

## 4. 개발

### 4-1. 깃허브 레포지토리 생성

- 깃허브 레포지토리를 생성합니다.
- 좀 더 간편한 작업을 위해 깃허브 데스크탑을 사용했습니다.

### 4-2. 로컬에서 클론

- 깃허브 데스크탑에서 파일 경로를 복사하여 클론 했습니다.
- 클론한 경로로 이동한 후 데이터를 저장할 새 폴더(data)를 생성해 kaggle에서 가져온 데이터셋을 담아주었습니다.

### 4-3. 데이터 가공  

#### 세팅 
- 아나콘다 프롬프트를 열고 가상 환경을 세팅
```
conda create -n st_310 python=3.10 openssl numpy scipy matplotlib ipython scikit-learn pandas pillow jupyter seaborn
```
- 가상 환경을 활성화
```
conda activate st_310
```
- 클론한 디렉토리의 경로로 들어가 주피터 노트북 실행

#### 개발 과정
- 작업은 주피터 노트북 환경에서 Python을 사용하여 진행했습니다.
- pandas 로 작업할 데이터를 불러와서 작업을 시작했습니다.

##### EDA 설정
- 각 별의 특성(온도, 광도, 반지름 등)에 대한 박스 플롯과 히스토그램 그려서, 각 별의 유형에 따라 분포를 비교할 수 있도록 했습니다.
matplotlib 을 사용했습니다.
- 각 컬럼 간의 상관 관계가 어떻게 되는지 쉽게 확인할 수 있게 seaborn의 heatmap과 pairplot 을 이용해 컬럼 간의 상관 관계를 한 눈에 파악할 수 있도록 시각화 했습니다.

##### ML
- 학습에 사용할 X에서 문자열로 구성된 컬럼의 고유 값 개수를 nunique()로 확인한 결과 3 이상이었기 때문에 적합한 원핫인코딩을 사용하여 인코딩했습니다.
- 다른 하나의 문자열로 구성된 컬럼은 해당하는 숫자의 값으로 매핑 했습니다.  
```
X['Spectral Class'].map({'M': 0, 'K': 1, 'G': 2, 'F': 3, 'A': 4, 'B': 5, 'O': 6})
```
- 학습에 사용할 X와 y를 sklearn의 train_test_split 함수를 사용하여 학습용과 테스트용으로 나눠 데이터를 준비했습니다.
- 학습은 sklearn의 LogisticRegression을 사용하여 데이터를 기반으로 모델을 학습하고 분류 작업을 수행했습니다.
- 학습한 모델을 테스트 하고 성능을 확인했습니다.
1. sklearn 의 accuracy_score를 사용하여 정확도를 확인했습니다.
2. sklearn 의 confusion_matrix를 통해 모델이 각 클래스를 얼마나 정확하게 분류했는지 확인했습니다.
3. sklearn 의 classification_report를 사용하여 각 클래스에 대한 정밀도, 재현율, F1 점수, 그리고 지원 수를 확인하여 모델의 분류 성능을 상세하게 평가했습니다.
- 모델을 저장하기 위해 먼저 os 모듈의 makedir 함수를 사용하여 model 폴더를 만들고, 그 후에 joblib 모듈의 dump 함수를 이용하여 모델을 model 폴더 안에 저장했습니다.

- 모델과 star_type.ipynb 파일을 커밋한 후 푸시했습니다.


### streamlit 을 이용한 대시보드 개발
- 대시보드 개발은 Visual Studio Code 에서 작업했습니다.
- 깃허브 데스크탑을 통해 연동된 레포지토리를 VS Code에서 실행했습니다.
- 활용한 라이브러리 : streamlit, numpy, pandas, matplotlib, seaborn, plotly, joblib

##### Main :
1. streamlit 의 title 함수를 사용하여 제목을 설정했습니다.  
"별 유형 예측"이라는 제목을 설정하고, ":star2:" 아이콘을 추가하여 제목을 꾸며주었습니다.  
2. streamlit의 option_menu를 import하고, 사용자에게 표시될 사이드바 메뉴의 항목들을 menu 리스트에 담아주었습니다.  
사이드바에 요소를 추가하는 with st.sidebar 내부에서 option_menu 함수를 사용하여 사이드바를 디자인했습니다.  

##### Home : 
사이드바의 "홈" 항목을 선택했을 때 나타나는 화면을 만들었습니다.
1. streamlit 의 subheader 를 활용해 프로젝트의 기능을 간단하게 설명했습니다.
2. streamlit 의 image 함수를 활용해 프로젝트와 어울리는 이미지를 삽입했습니다.
3. 데이터의 출처를 작성하고 streamlit 의 link_button 을 활용해 출처의 링크로 이동하는 버튼을 만들었습니다.

##### EDA :
사이드바의 "EDA' 항목을 선택했을 때 나타나는 화면을 만들었습니다.  

1. streamlit의 button을 활용하여 클릭하면 설명이 나타나고 다시 한 번 클릭하면 닫히는 기능을 구현했습니다.
이를 위해 조건문을 사용했고 그 안에 데이터에 대한 설명을 작성했습니다.
2. 데이터 / 통계치 확인:  
streamlit 의 radio와 dataframe 함수를 활용하여 데이터프레임과 통계치(describe)를 분리하여,
사용자가 선택한 메뉴에 따라 제공할 수 있도록 구성했습니다.
3. Hertzsprung-Russell Diagram(헤르츠스프룽-러셀 도표):  
plotly 의 scatter를 이용해서 별의 온도와 밝기를 나타내는 산점도를 그려 별의 분포와 유형을 시각화했습니다.
점은 데이터에 있는 Star color 컬럼의 데이터를 이용해서 온도와 밝기에 따른 별의 색깔을 토대로 산점도를 나타냈습니다.    
4. 최대/최소 데이터 확인:    
streamlit의 selectbox를 이용하여 사용자가 선택한 컬럼에 대한 최대값과 최소값을 포함하는 행을 슬라이싱한 후
이를 dataframe 함수를 통해 사용자에게 제공했습니다.
5. 사용자가 선택한 컬럼과 별의 유형 간의 상관 관계 분석:  
streamlit의 selectbox를 활용하여 사용자가 선택한 컬럼에 따른 별의 유형 분포를 plotly의 box plot과 histogram을 이용해 시각화했습니다.
선택한 컬럼에 따라 각각 박스 플롯과 히스토그램을 통해 각 유형의 특징을 비교하였습니다.
6. 사용자가 선택한 두 컬럼간의 분포를 비교하여 시각화:  
streamlit 의 selectbox를 활용하여 사용자에게 두 개의 컬럼을 선택하도록 안내한 후  
선택한 컬럼을 기반으로 plotly를 이용해 박스 플롯과 히스토그램을 그렸습니다.  
사용자가 선택한 컬럼에 따라 다른 시각화가 나타나도록 조건문을 활용했습니다.  
7. 히트맵(Heatmap), 페어플롯(Pair plot):  
streamlit 의 multiselect를 활용하여 사용자가 여러 개의 컬럼을 선택할 수 있도록 안내한 후,  
선택된 컬럼들 간의 상관 관계를 seabron의 heatmap과 pairplot으로 시각화하여 변수들 간의 상관성과 분포를 시각화 했습니다.

##### ML :  
사이드바의 "별 유형 예측" 항목을 선택했을 때 나타나는 화면을 만들었습니다.  

1. 사용자로부터 예측에 필요한 데이터를 입력받기 위해 streamlit의 text_input과 selectbox를 활용했습니다.
입력된 데이터는 새로운 리스트에 저장했습니다.
3. 데이터 전처리 단계에서 적용한 원핫인코딩과 매핑과 동일한 처리를 사용자가 입력한 데이터에도 적용하여 새로운 변수에 저장했습니다.
4. 입력한 데이터가 담긴 리스트를 모델에 적합한 형태로 변환하기 위해
NumPy 배열로 변환한 후 reshape하여 1행과 여러 열을 가진 2차원 배열로 변경했습니다.
5. 학습한 모델을 불러와서 입력된 데이터에 대한 예측을 수행했습니다. joblib의 load를 활용했습니다.
6. 예측된 값을 사용자에게 보여주기 위해 streamlit의 info와 f-string을 활용하여 결과를 출력했습니다.

------

## 5. 배포
AWS의 EC2를 활용하여 배포했습니다.
http://ec2-43-201-149-238.ap-northeast-2.compute.amazonaws.com:8505/

### 배포 과정
1. AWS EC2 인스턴스를 생성했습니다. (Amazon Linux, 키 페어는 ppk)
2. EC2에 접속하기 위해 PuTTy 를 사용했습니다.
3. SSH > Auth > Credentials에서 Public-key에 EC2 생성할때 키페어 생성했던 파일을 입력하고
Session 에서 Saved Sessions에 Save 했습니다.
4. ec2-user 를 입력하고 실행
5. EC2 Linux 에 아나콘다를 설치했습니다.
```
wget https://repo.anaconda.com/archive/Anaconda3-2022.10-Linux-x86_64.sh 
```
6. 완료 후 실행했습니다.
```
sh Anaconda3-2022.10-Linux-x86_64.sh
```
7. 설치 완료 후 작업 환경과 같은 가상환경을 만들고 라이브러리를 설치했습니다.
8. git 을 설치하고 repository 를 연동했습니다.
```
sudo yum intstall git    # git 설치

mkdir Github     # Github 디렉토리 생성

cd Github     # 디렉토리 안으로 이동

git clone repository HTTP 주소    # 입력하여 클론해서 해당 파일 경로로 이동해 사용
```
9. 터미널을 종료해도 페이지가 종료되지 않게 백그라운드에서 실행했습니다.
```
nohup streamlit run app.py --server.port 8505 &
```

------

## 6. 이슈 개선 :
#### 1. 애플리케이션 실행이 안되는 이슈 발생
EC2에서 Streamlit을 실행할 때는 해당 포트를 열어야 하고   
보안 그룹 설정을 통해 해당 포트에 대한 액세스를 허용해야 하는데 하지 않음

#### 해결
AWS EC2 인스턴스의 보안 > 보안 그룹 > 인바운드 규칙 편집 > 규칙 추가  
유형은 사용자 지정 TCP  
포트범위는 8505  
소스 유형은 Anywhere-IPv4 로 지정하고 규칙을 저장하여  
해당 포트를 열고 포트에 대한 엑세스를 허용했습니다.

------
## 작성자
- 김건휘
- e-mail : iwhnoegmik@gmail.com








