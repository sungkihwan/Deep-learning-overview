## 딥러닝 개론

### Perceptron

### AND gate, OR gate, NAND gate, NOR gate 구현

### 다층 퍼셉트론으로 XOR gate 구현

### 다층 퍼셉트론(MLP) 모델로 2D 데이터 분류

## 딥러닝 모델의 학습 방법

### Gradient descent

### Back propagation

## TensorFlow

### Tensor

### 텐서 플로우 활용

### Keras와 다층퍼셉트론 모델
- [keras 사용법](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense)

### 텐서플로우로 XOR 문제 해결하기

## 학습속도 문제와 최적화 알고리즘

### gradient descent vs SGD

### Momentum

### Adagrad, RMSprop, Adam optimize

## 기울기 소실 문제와 방지 기법

### vanishing gradient descent

### 활성화 함수 다르게 적용하기

## 초기 설정 문제와 방지 기법

### Naive한 가중치 초기화 방법

### Xavier 초기화 방법

### He 초기화 방법

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

### Batch Normalization

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
