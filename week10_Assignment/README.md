
baseline 모델은 첨부된 파일의 hidden layer가 한개인 MLP모델로 했다.

# 성능 향상
- 모든 모델은 baseline 모델size를 넘지 않도록 하였다.
- torch,np,tf의 난수값을 123으로 고정했다.
![image](https://github.com/mindang/KDT-AI-Assinment/assets/91838563/ae1dba1e-1c86-4ccc-8660-7983fd6ede48)

0. baseline모델은 첨부된 MLP모델이며 10epoch학습
- 이후 Test는 train/valid/test로 나눠서 충분한 학습 진행

1. Test1은 적절한 하이퍼 파라미터를 찾기 위해 lr조절을 위한 스케줄러 , 과적합방지를 위한 조기종료를 사용했다.

2. Test2는 과적합방지를 위한 규제 방법인 dropout을 사용하였다.

3. Test3은 다양한 옵티마이저를 사용하였다.

4. Test4는 mixup방식을 사용하였다.

5. Test5-1와 5-2는 모델 size를 각각 절반 , 동일하게(10%낮음) 사용하여 효율적인 구조를 사용하였다.

# TF
- 모델 size & 학습 시간이 효율적인 Test5.1 model을 사용했다.
