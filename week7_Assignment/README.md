
# 음식배달에 걸리는 시간 예측하기

## 모델 구조 : MLP
![KakaoTalk_20230428_190657296](https://user-images.githubusercontent.com/91838563/235179653-eb3c6bf9-509c-4e5b-9782-735a5aaaf96c.jpg)

## 손실함수
빨리 예측한경우 : 늦게 예측한 경우 = 2 : 1 가중치를 줌
class MyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        mse_loss = nn.MSELoss(reduction='none')                                            # 각 샘플별로 MSE 계산
        result = torch.where(y_pred < y_true, torch.tensor(2.0), torch.tensor(1.0))        # 빨리 예측한 경우 가중치 2배
        weighted_loss = result * mse_loss(y_pred, y_true)                                  # 각 샘플별로 가중치 곱
        summed_loss = torch.sum(weighted_loss)                                             # 손실 값들을 합산
        rmse_loss = torch.sqrt(summed_loss)                                                

        return rmse_loss

## 결과
최적의 epoch는 21이고
RMSE값은 907.73
up비율은 33%
![스크린샷 2023-04-28 235945](https://user-images.githubusercontent.com/91838563/235182767-67912dbd-ce10-42f8-b7f4-8b4c58487ef0.png)



## 학습 그래프
- 손실함수 : 가중치를 적용한 Custom Loss

![CL](https://user-images.githubusercontent.com/91838563/235181765-c0462af9-6c78-4664-a202-159334350a9c.png)

- RMSELoss

![RL](https://user-images.githubusercontent.com/91838563/235181788-30d3a0d5-aa00-46e7-bede-fe90b60fe011.png)

- UP-rate

![UR](https://user-images.githubusercontent.com/91838563/235181848-724de248-016c-4839-abc1-ba1ea7a0e273.png)
