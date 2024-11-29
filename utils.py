import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as T
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
import torchvision
from torchvision.utils import save_image
import os
import cv2

from color_space import *
from torchvision.models import vgg16

def load_model_weights(model, path):
    pretrained_dict = torch.load(path, map_location=lambda storage, loc: storage)
    model_dict = model.state_dict()
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if 'preprocessing' not in k}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    model.load_state_dict(pretrained_dict, strict=False)

def denorm(x):
    """Convert the range from [-1, 1] to [0, 1]."""
    out = (x + 1) / 2
    return out.clamp_(0, 1)

def label2onehot(labels, dim):
    """Convert label indices to one-hot vectors."""
    batch_size = labels.size(0)
    out = torch.zeros(batch_size, dim)
    out[np.arange(batch_size), labels.long()] = 1
    return out

def create_labels(c_org, c_dim=5, dataset='CelebA', selected_attrs=None):
    """Generate target domain labels for debugging and testing."""
    # Get hair color indices.
    if dataset == 'CelebA':
        hair_color_indices = []
        for i, attr_name in enumerate(selected_attrs):
            if attr_name in ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair']:
                hair_color_indices.append(i)

    c_trg_list = []
    for i in range(c_dim):
        if dataset == 'CelebA':
            c_trg = c_org.clone()
            if i in hair_color_indices:  # Set one hair color to 1 and the rest to 0.
                c_trg[:, i] = 1
                for j in hair_color_indices:
                    if j != i:
                        c_trg[:, j] = 0
            else:
                c_trg[:, i] = (c_trg[:, i] == 0)  # Reverse attribute value.
        elif dataset == 'RaFD':
            c_trg = label2onehot(torch.ones(c_org.size(0)) * i, c_dim)

        c_trg_list.append(c_trg.cuda())
    return c_trg_list

def random_transform(img):
    T_list = [
        T.RandomHorizontalFlip(p=0.5),
        #T.RandomErasing(p=1, scale=(0.03, 0.10)),
        T.RandomRotation(degrees=(-15, 15)),
        T.RandomVerticalFlip(p=0.5),
        T.RandomCrop((192,192)),
    ]

    T_compose = T.Compose([
        T.RandomChoice(T_list),
        T.Resize((256, 256)),
    ])

    return T_compose(img)

def compare(img1,img2):
    """input tensor, translate to np.array"""
    img1_np = img1.squeeze(0).cpu().numpy()
    img2_np = img2.squeeze(0).cpu().numpy()
    img1_np = np.transpose(img1_np, (1, 2, 0))
    img2_np = np.transpose(img2_np, (1, 2, 0))

    
    #img1_np = np.nan_to_num(img1_np, nan=0.0)
    #img2_np = np.nan_to_num(img2_np, nan=0.0)

    print(f'img1 shape :{img1_np.shape}, img2 shape : {img2_np.shape}')
    #print(f'img1 nan replace후:{np.min(img1_np)},{np.max(img2_np)}')
    #ssim = structural_similarity(img1_np,img2_np,multichannel=True)
    ssim = structural_similarity(img1_np,img2_np,channel_axis=2,data_range=1.0)
    
    psnr = peak_signal_noise_ratio(img1_np,img2_np)

    return ssim, psnr

def lab_attack(X_nat, c_trg, model, epsilon=0.01, iter = 100):

    criterion = nn.MSELoss().cuda()
    #pert_a = torch.zeros(X_nat.shape[0], 2, X_nat.shape[2], X_nat.shape[3]).cuda().requires_grad_()
    print(f'X_nat shape :{X_nat.shape}')
    pert_a = (torch.ones(X_nat.shape[0], 2, X_nat.shape[2], X_nat.shape[3]) * 0.01).cuda().requires_grad_()


    optimizer = torch.optim.Adam([pert_a], lr=1e-4, betas=(0.9, 0.999))

    r = torch.ones_like(pert_a)

    X = denorm(X_nat.clone())

    for i in range(iter):
        X_lab = rgb2lab(X).cuda()
       
        pert = torch.clamp(pert_a, min=-epsilon, max=epsilon)
        print(f'pert requires grad?:{pert.requires_grad}')

        X_lab[:, 1:, :, :] = X_lab[:, 1:, :, :] + pert
        
        
        X_new = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])(lab2rgb(X_lab))
        

        #X_new = random_transform(X_new)

        with torch.no_grad():
            gen_noattack, gen_feats_noattack = model(X_nat, c_trg[i%len(c_trg)])

        gen_stargan, gen_feats_stargan = model(X_new, c_trg[i%5])

        print(f"result requires grad: {gen_stargan.requires_grad},{gen_noattack.requires_grad}")
        loss = -criterion(gen_stargan, gen_noattack)
        print(f'loss:{loss.item()}')

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return X_new, X_new - X



def lab_attack_faceswap(X_src, X_tgt, model, epsilon=0.07, iter=100, output_dir='./output'):
    """
    Faceswapping 적대적 공격 함수
    - X_src: 소스 이미지 (Face Swapping에 사용할 기준 얼굴)
    - X_tgt: 타겟 이미지 (Face Swapping 대상 얼굴)
    - model: Face Swapping 모델
    - epsilon: perturbation 크기 제한
    - iter: 공격 반복 횟수
    """

    os.makedirs(output_dir,exist_ok=True)
    # 손실 함수 및 초기화
    criterion = nn.MSELoss().cuda()
    #pert_a = torch.zeros(X_tgt.shape[0], 2, X_tgt.shape[2], X_tgt.shape[3]).cuda().requires_grad_(True)
    pert_a = (torch.ones(X_tgt.shape[0], 2, X_tgt.shape[2], X_tgt.shape[3]) * 0.01).cuda().requires_grad_()

    optimizer = torch.optim.Adam([pert_a], lr=1e-2, betas=(0.9, 0.999))
    X_tgt_denorm = denorm(X_tgt.clone())

    # Latent ID 생성
    X_src_downsample = F.interpolate(X_src, size=(112, 112))  # Arcface 입력 크기에 맞게 downsample
    latend_id = model.netArc(X_src_downsample)
    latend_id = latend_id.detach().to('cpu')
    latend_id = latend_id / np.linalg.norm(latend_id, axis=1, keepdims=True)
    #latend_id = latend_id / torch.norm(latend_id, dim=1, keepdim=True)
    latend_id = latend_id.to('cuda')

    X_tgt_adv_list=[]
  
    for i in range(1,iter+1):
        # LAB 색상 공간으로 변환 및 공격 적용
        X_lab = rgb2lab(X_tgt_denorm).cuda()
        
        pert = torch.clamp(pert_a, min=-epsilon, max=epsilon)
        print(f'pert requires grad : {pert.requires_grad}')
        X_lab[:, 1:, :, :] = X_lab[:, 1:, :, :] + pert
        X_tgt_adv = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])(lab2rgb(X_lab))
        
        
        with torch.no_grad():
            img_fake_noattack = model(X_src, X_tgt, latend_id, latend_id, True)
        img_fake_attack = model(X_src, X_tgt_adv, latend_id, latend_id, True)

        print(f'pert a grad_fn{pert.grad_fn}')

        # 손실 계산
        loss = -criterion(img_fake_attack, img_fake_noattack)
        
        print(f'iter {i} loss a : {loss.item()}')

        # Gradient Update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        
        # 10,20,..,100 번째 iteration에서 저장
        if i== iter:
            # perturbation 이미지 저장
            
            #pert_np = pert[0].detach().cpu().numpy()  # 첫 번째 배치만 저장
            #pert_img = (pert_np - pert_np.min()) / (pert_np.max() - pert_np.min()) * 255  # 정규화
            #pert_img = pert_img.transpose(1, 2, 0).astype(np.uint8)  # (C, H, W) -> (H, W, C)

            # RGB로 저장하기 위해 BGR 변환
            #if pert_img.shape[2] == 2:  # 채널이 2개일 경우, 빈 채널 추가
            #    pert_img = cv2.merge([pert_img[:, :, 0], pert_img[:, :, 1], np.zeros_like(pert_img[:, :, 0])])
            #cv2.imwrite(os.path.join(output_dir, f"perturbation_iter_{i+1}_tr2.jpg"), pert_img)
            


            # 적대적 입력 이미지 저장
           
            X_tgt_adv_np = X_tgt_adv[0].detach().permute(1, 2, 0).cpu().numpy()  # (C, H, W) -> (H, W, C)
            X_tgt_adv_np = (X_tgt_adv_np - X_tgt_adv_np.min()) / (X_tgt_adv_np.max() - X_tgt_adv_np.min()) * 255  # 정규화
            X_tgt_adv_np = X_tgt_adv_np.astype(np.uint8)
            cv2.imwrite(os.path.join(output_dir, f"adversarial_input_iter_{i}_e0.07.jpg"), cv2.cvtColor(X_tgt_adv_np, cv2.COLOR_RGB2BGR))
            
            # 모델 출력 이미지 저장
            
            #img_fake_attack_np = img_fake_attack[0].detach().permute(1, 2, 0).cpu().numpy()  # (C, H, W) -> (H, W, C)
            #img_fake_attack_np = (img_fake_attack_np - img_fake_attack_np.min()) / (img_fake_attack_np.max() - img_fake_attack_np.min()) * 255  # 정규화
            #img_fake_attack_np = img_fake_attack_np.astype(np.uint8)
            #cv2.imwrite(os.path.join(output_dir, f"output_iter_{i}_tr2.jpg"), cv2.cvtColor(img_fake_attack_np, cv2.COLOR_RGB2BGR))
        
    return X_tgt_adv, X_tgt_adv - X_tgt

def lab_attack_faceswap_video(X_src, X_tgt_batch, model, epsilon=0.07, iter=100, output_dir='./output'):
    """
    Adversarial attack for video frames.
    - X_src: Source image (Face Swapping baseline identity).
    - X_tgt_batch: Batch of target frames.
    - model: Face Swapping model.
    - epsilon: Perturbation magnitude.
    - iter: Number of attack iterations.
    """
    os.makedirs(output_dir, exist_ok=True)
    criterion = nn.MSELoss().cuda()
    pert_a = (torch.ones(X_tgt_batch.shape[0], 2, X_tgt_batch.shape[2], X_tgt_batch.shape[3]) * 0.01).cuda().requires_grad_()
    optimizer = torch.optim.Adam([pert_a], lr=1e-2, betas=(0.9, 0.999))
    X_tgt_batch_denorm = denorm(X_tgt_batch.clone())

    # Generate latent ID
    X_src_downsample = F.interpolate(X_src, size=(112, 112))
    latend_id = model.netArc(X_src_downsample)
    latend_id = F.normalize(latend_id, p=2, dim=1).cuda()

    for i in range(1, iter + 1):
        X_lab = rgb2lab(X_tgt_batch_denorm).cuda()
        pert = torch.clamp(pert_a, min=-epsilon, max=epsilon)
        X_lab[:, 1:, :, :] += pert
        X_tgt_adv_batch = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])(lab2rgb(X_lab))

        with torch.no_grad():
            img_fake_noattack = model(X_src, X_tgt_batch, latend_id, latend_id, True)
        img_fake_attack = model(X_src, X_tgt_adv_batch, latend_id, latend_id, True)

        loss = -criterion(img_fake_attack, img_fake_noattack)
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

        """
        # Save adversarial frames every 10 iterations
        if i % 10 == 0:
            for frame_idx, X_tgt_adv in enumerate(X_tgt_adv_batch):
                frame_adv_np = X_tgt_adv.detach().permute(1, 2, 0).cpu().numpy()
                frame_adv_np = (frame_adv_np - frame_adv_np.min()) / (frame_adv_np.max() - frame_adv_np.min()) * 255
                frame_adv_np = frame_adv_np.astype(np.uint8)
                cv2.imwrite(os.path.join(output_dir, f"adversarial_frame_{frame_idx}_iter_{i}.jpg"),
                            cv2.cvtColor(frame_adv_np, cv2.COLOR_RGB2BGR))
        """

    return X_tgt_adv_batch



