# 머신러닝 분류 예측 모델
##### 딥러닝과 머신러닝 성능 차이를 비교하기 위해 머신러닝 분류 모델의 정확도를 확인

### 1. 학습 데이터 전처리
* pandas를 이용하여 CSV 파일 불러오기
```py
data = pd.read_csv('drive/MyDrive/lab/financial_data.csv', encoding='cp949')
data_X = data.iloc[:,[1,3,4,6,8]].values
```

* 학습데이터 전처리 및 데이터 분할
```py
#데이터 정규화
scaler = MinMaxScaler()
scaler.fit(data_X)
X = scaler.transform(data_X)
Y_t = data['Y'].replace(['AAA','AA','A','BBB','BB','B','CCC','CC','C','D'],[0,0,0,0,1,1,1,1,1,1])
X_train, X_test, Y_train, Y_test = train_test_split(X, Y_t, test_size=0.2)
```

### 2. 머신러닝 모델 학습
* 선언한 머신러닝 프레임워크를 `.fit()`를 이용하여 train 데이터로 학습
```py
#SVM
svm = SVC()
svm.fit(X_train, Y_train)
```

### 3. 학습된 모델 성능 확인
* 학습된 모델 test 데이터로 성능 확인하기
* `.score()`함수를 이용하여 X_test로 예측한 값을 Y_test와 비교하여 정확로를 return
* 정확도가 가장 높은값을 첫 번째에 출력하기 위해 list 형태로 저장하였다.
* `sorted()`함수를 이용하여 정확도가 가장 높은 순부터 정렬
```py
acc_list = []
acc_list.append(["Random Forest", rf.score(X_test, Y_test)])
sorted_array = sorted(acc_list, key=lambda x: x[1], reverse=True)
```
