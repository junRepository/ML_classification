# 머신러닝 분류 모델
##### 딥러닝과 머신러닝 성능 차이를 비교하기 위해 머신러닝 분류 모델의 정확도를 확인

### 1. 학습 데이터 전처리
```py
data = pd.read_csv('drive/MyDrive/lab/financial_data.csv', encoding='cp949')
data_X = data.iloc[:,[1,3,4,6,8]].values
```
* pandas를 이용하여 CSV 파일 불러오기

```py
#데이터 정규화
scaler = MinMaxScaler()
scaler.fit(data_X)
X = scaler.transform(data_X)
Y_t = data['Y'].replace(['AAA','AA','A','BBB','BB','B','CCC','CC','C','D'],[0,0,0,0,1,1,1,1,1,1])
X_train, X_test, Y_train, Y_test = train_test_split(X, Y_t, test_size=0.2)
```
* 학습데이터 전처리 및 데이터 분할
