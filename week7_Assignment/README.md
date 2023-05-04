
# 음식배달에 걸리는 시간 예측하기
## 모델은 MLP를 사용

두가지 방법 시도 (범주형 특성은 임베딩으로 통일)
1. 11가지 수치형 특성을 그대로 모델에 넣음
2. PCA를 통해 정보량을 90%이상 유지하고 11차원 -> 6차원으로 줄인 후 모델에 넣음

## 손실함수
빨리 예측한경우 : 늦게 예측한 경우 = 2 : 1 가중치를 줌 (루트 고려하여 4)

class MyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        mse_loss = nn.MSELoss(reduction='none')                                            # 각 샘플별로 MSE 계산
        result = torch.where(y_pred < y_true, torch.tensor(4.0), torch.tensor(1.0))        # 빨리 예측한 경우 가중치 2배
        weighted_loss = result * mse_loss(y_pred, y_true)                                  # 각 샘플별로 가중치 곱
        summed_loss = torch.sum(weighted_loss)                                             # 손실 값들을 합산
        rmse_loss = torch.sqrt(summed_loss)                                                

        return rmse_loss

## 결과
1. 첫번째 모든 수치형 특성을 넣은 방법
    최적의 epoch는 37이고
    
    RMSE값은 956.9
    
    up비율은 27%
    
![result](https://user-images.githubusercontent.com/91838563/235207459-1b0be7f7-6a65-4aaa-9fb0-2fcbafd0fac2.png)   

2. 두번째 PCA를 통해 차원축소 후 넣는 방법

![스크린샷 2023-05-01 011727](https://user-images.githubusercontent.com/91838563/235364110-a5ed78c0-5ac0-48e3-9581-6a08c2ab879c.png)

    최적의 epoch는 14이고
    
    RMSE값은 975.34
    
    up비율은 27%
    

결론 , 수치형 특성을 11 -> 6차원 , 범주형 특성 임베딩을 (148->89)으로 줄였는데 거의 동일한 성능



## 모델 구조 : MLP
![KakaoTalk_20230428_190657296](https://user-images.githubusercontent.com/91838563/235179653-eb3c6bf9-509c-4e5b-9782-735a5aaaf96c.jpg)

## PCA 방법
공분산행렬을 구한 후 고윳값 분해 후 고유벡터로 선형 변환

![pca1](https://user-images.githubusercontent.com/91838563/235336989-c5808e9c-6bb4-41b3-81b7-5e8950f0303c.png)

## 학습 그래프
- 손실함수 : 가중치를 적용한 Custom Loss

![cl](https://user-images.githubusercontent.com/91838563/235207454-065863e1-947d-45f7-bb46-6b17cff2542c.png)

- RMSELoss

![rlpng](https://user-images.githubusercontent.com/91838563/235207457-ee02cca8-ff4f-45dd-80a2-66a278870dd9.png)

- UP-rate

![up](https://user-images.githubusercontent.com/91838563/235207448-da68e69b-2ef9-4e8f-ae94-91a68081a440.png)
