import numpy as np
import torch

class ECGAugment:
    def __init__(self, mode='weak'):
        self.mode = mode

    def __call__(self, x):
        # x shape: (12, 1000) or (1000, 12) -> numpy array
        # 처리를 위해 (Channel, Time) 형태로 가정하고 진행
        if x.shape[0] != 12: 
            x = x.T 
        
        if self.mode == 'weak':
            return self.weak_aug(x)
        elif self.mode == 'strong':
            return self.strong_aug(x)
        else:
            return x

    def weak_aug(self, x):
        # 1. Scaling (크기 조절)
        sigma = 0.1
        factor = np.random.normal(loc=1.0, scale=sigma, size=(12, 1))
        x = x * factor
        
        # 2. Jittering (노이즈 추가)
        noise = np.random.normal(loc=0, scale=0.01, size=x.shape)
        return x + noise

    def strong_aug(self, x):
        # 1. Permutation (구간 섞기) - 가끔 적용
        if np.random.rand() < 0.5:
            segments = 5
            seg_len = x.shape[1] // segments
            indices = np.random.permutation(segments)
            new_x = []
            for i in indices:
                new_x.append(x[:, i*seg_len : (i+1)*seg_len])
            x = np.concatenate(new_x, axis=1)
        
        # 2. Masking (일부 구간 0으로)
        if np.random.rand() < 0.5:
            mask_len = np.random.randint(50, 200) # 1000개 중 50~200개 마스킹
            start = np.random.randint(0, x.shape[1] - mask_len)
            x[:, start:start+mask_len] = 0
            
        # 3. Weak Aug도 기본적으로 포함
        return self.weak_aug(x)