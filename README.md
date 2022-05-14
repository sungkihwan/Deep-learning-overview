## 딥러닝 개론

### Perceptron
> y = activation(w1x1 + w2x2 + b)

### AND gate, OR gate, NAND gate, NOR gate 구현
> 진리표에 맞게 단층 퍼셉트론 구현

### 다층 퍼셉트론으로 XOR gate 구현
> X1 -> NAND -> AND -> Y     
> X2 ->  OR  -> 

## 딥러닝 모델의 학습 방법

### Gradient descent
> 최적의 가중치는 모든 경우의 수를 체크 하는 브루트 포스 방식이 불가능합니다.           
> Gradient descent 알고리즘은 손실 함수(loss function)의 미분값인 gradient를 이용하여             
> 모델에게 맞는 최적의 가중치(weight), 즉 손실 함수의 값을 최소화 하는 가중치를 구할 수 있는 알고리즘입니다.

### Back propagation
> 손실 함수(loss function)의 gradient 값을 역전파해서 받은 후,         
> 그 값을 참고하여 손실 함수값을 최소화 하는 방향으로 가중치(weight)를 업데이트 합니다.

## TensorFlow

### 텐서플로우 자료형
- tf.float32 : 32-bit float
- tf.float64 : 64-bit float
- tf.int8 : 8-bit integer
- tf.int16 : 16-bit integer
- tf.int32 : 32-bit integer
- tf.uint8 : 8-bit unsigned integer
- tf.string : String
- tf.bool : Boolean

### 텐서 플로우 연산
- tf.add(a, b)
- tf.subtract(a, b)
- tf.multiply(a, b)
- tf.truediv(a, b)

### 텐서 플로우 활용
```
model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(64, input_dim = 1 ,activation='sigmoid'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='softmax')
])

model.compile(
  loss='mean_squared_error',
  optimizer='adam',
  metrics = ['accuracy']
)

history = model.fit(x_train, y_train, epochs=30, batch_size = 500, verbose=2)

model.evaluate(x_test, y_test)
```

### Keras 사용법
- [keras 사용법](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense)

## 학습속도 문제와 최적화 알고리즘

### gradient descent vs SGD
> GD는 전체 데이터 셋을 가지고 학습하여 안정적이지만 계산량과 학습 비용이 많다.
> SGD는 무작위로 뽑은 데이터들에 대한 GD를 진행

### Momentum
> SGD는 손실 함수의 최솟값에 도달하는 동안 Gradient가 진동하여 최적값에 도달하기까지의 시간이 오래걸린다.      
> 이를 보완하기 위해 사용되는 모멘텀 기법은 관성의 개념을 이용해 최적값에 좀 더 빠르게 도달할 수 있도록 도와줍니다.

### Adagrad, RMSprop, Adam optimize
> Adagrad(Adaptive Gradient)는 최적의 가중치를 찾아내기 위해 learning rate를 조절해 하강하는 방법 중 하나입니다  .     
> RMSprop는 학습이 진행될수록 가중치 업데이트 강도가 약해지는 Adagrad의 단점을 보완하고자 제안된 방법입니다.        
> RMSprop는 과거의 gradient 값은 잊고 새로운 gradient 값을 크게 반영해서 가중치를 업데이트합니다.       
> Adam은 RMSProp과 Momentum을 함께 사용함으로써, 진행 방향과 learning rate를 적절하게 유지하면서 학습할 수 있습니다.

## 기울기 소실 문제와 방지 기법

### vanishing gradient descent
> 깊은 층의 모델에선 역전파시 전달되는 loss function의 gradient 값에 활성화 함수인 sigmoid 함수의 0에 가까운 기울기 값이       
> 계속해서 곱해지면서 결국 가중치 업데이트가 잘 안되는 문제가 생기는데, 이것이 바로 기울기 소실 문제(Vanishing Gradient)입니다.       
> relu와 tanh 활성화 함수를 사용하여 해결하는 방안이 있습니다.

## 초기 설정 문제와 방지 기법

### Xavier 초기화 방법
> Xavier 초기화 방법은 앞 레이어의 노드가 n개일 때 표준 편차가 1 / sqrt(n) 분포를 사용합니다.    
> 표준 정규 분포를 입력 개수의 제곱근으로 나누어주면 됩니다.

### He 초기화 방법
> He 초기화 방법은 앞 레이어의 노드가 n개일 때 표준 편차가 sqrt(2) / sqrt(n) 분포를 사용합니다.       
> 표준 정규 분포를 입력 개수 절반의 제곱근으로 나누어주면 됩니다.

## 과적합 문제와 방지 기법
![과적합](https://t1.daumcdn.net/cfile/tistory/216C294D572DE7E904)

### Overfitting
> 과적합(Overfitting)은 모델이 학습 데이터에만 너무 치중되어 학습 데이터에 대한 예측 성능은 좋으나 
> 테스트 데이터에 대한 예측 성능이 떨어지는 경우를 말하는데, 과적합이 발생하는 이유는 아래와 같다.
- 데이터의 퍼진 정도, 즉 분산(variance)이 높은 경우
- 너무 많이 학습 데이터를 학습시킨 경우 (epochs가 매우 큰 경우)
- 학습에 사용된 파라미터가 너무 많은 경우
- 데이터에 비해 모델이 너무 복잡한 경우
- 데이터에 노이즈 & 이상치(outlier)가 너무 많은 경우

### Regularization
> L1 정규화는 모델 내의 일부 가중치를 0으로 만들어 의미있는 가중치만 남도록 만들어줍니다.          
> 이를 통해 모델을 일반화시킬 수 있습니다. 다른 말로 Sparse Model을 만든다라고도 합니다.       
> L2 정규화는 학습이 진행될 때 가중치의 값이 0에 가까워지도록 만들어줍니다.       
> 가중치를 0으로 만들어주는 L1 정규화와는 차이가 있습니다.

### Drop out
> 드롭 아웃은 데이터를 학습할 때, 일부 퍼셉트론(뉴런)을 랜덤하게 0으로 만들어     
> 모델 내부의 특정 가중치(Weight)에 치중되는 것을 막습니다.

### Batch Normalization
> 배치 정규화를 적용하면 매 층마다 정규화를 진행하므로 가중치 초기값에 크게 의존하지 않습니다.     
> 가중치 초기화의 중요도 감소 및 과적합을 억제합니다.        
> 드롭 아웃(Drop out)과 L1, L2 정규화의 필요성이 감소하고 학습 속도도 빨라집니다.

## CNN

## RNN

## 이미지 처리

## 자연어 처리

## Deep Q Learning

### 강화학습이란?

### 벨만 방정식

### 마르코프 의사결정

### 리빙페널티

### 시간적 차이

### 경험 리플레이

## A3C
