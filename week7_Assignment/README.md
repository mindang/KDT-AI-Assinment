
# 음식배달에 걸리는 시간 예측하기

## 모델 구조 : MLP
![KakaoTalk_20230428_190657296](https://user-images.githubusercontent.com/91838563/235179653-eb3c6bf9-509c-4e5b-9782-735a5aaaf96c.jpg)

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
최적의 epoch는 37이고
RMSE값은 956.9
up비율은 27%


![result](https://user-images.githubusercontent.com/91838563/235207459-1b0be7f7-6a65-4aaa-9fb0-2fcbafd0fac2.png)



## 학습 그래프
- 손실함수 : 가중치를 적용한 Custom Loss

![cl](https://user-images.githubusercontent.com/91838563/235207454-065863e1-947d-45f7-bb46-6b17cff2542c.png)

- RMSELoss

![rlpng](https://user-images.githubusercontent.com/91838563/235207457-ee02cca8-ff4f-45dd-80a2-66a278870dd9.png)

- UP-rate

![up](https://user-images.githubusercontent.com/91838563/235207448-da68e69b-2ef9-4e8f-ae94-91a68081a440.png)
