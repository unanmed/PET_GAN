import argparse
import os, time
import shutil
import sys
from pydicom import Dataset
from scipy import io
import torch
import numpy as np
from datetime import datetime
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from model.generator import PETUNet
from model.critic import PETCritic
from model.loss import WGANLoss, validate_loss

EPOCHS = 100
IMAGE_SIZE = 256
BATCH_SIZE = 6

disable_tqdm = not sys.stdout.isatty()

def log_layer_gradients(model, prefix=""):
    grad_log = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            # 计算该参数梯度的L2范数
            grad_norm = torch.linalg.vector_norm(param.grad, 2).item()
            # 按层名记录（如"conv1.weight", "bn2.bias"）
            grad_log[f"{prefix}{name}"] = grad_norm
        else:
            grad_log[f"{prefix}{name}"] = 0.0  # 标记无梯度
    return grad_log

class TrainDataset(Dataset):
    def __init__(self, input, target):
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(30),
        ])
        
        imput_img = np.array([input +"/"+ x  for x in os.listdir(input)])
        target_img = np.array([target +"/"+ x  for x in os.listdir(target)])
        
        assert len(imput_img) == len(target_img)
        
        imput_img.sort()
        target_img.sort()

        self.data = {'input': imput_img, 'target': target_img}
            
    def np2tensor(self, array):
        return torch.Tensor(array).permute(2,0,1)

    def __len__(self):
        return len(self.data['target'])

    def __getitem__(self, idx):
        input_path = self.data['input'][idx]
        target_path = self.data['target'][idx]
        
        input_img = io.loadmat(input_path)['img'].astype('float32')
        target_img = io.loadmat(target_path)['img'].astype('float32')
        
        input_img = np.expand_dims(input_img, axis=0)  # (1, H, W)
        target_img = np.expand_dims(target_img, axis=0)  # (1, H, W)
        
        input_img = torch.from_numpy(input_img).float()
        target_img = torch.from_numpy(target_img).float()

        if self.transform:  # 应用数据增强
            seed = torch.randint(0, 2**32, (1,)).item()
            torch.manual_seed(seed)
            input_img = self.transform(input_img)
            torch.manual_seed(seed)  # 保证input和target应用相同的变换
            target_img = self.transform(target_img)
        
        sample = {
            'input_img': input_img,
            'target_img': target_img,
        }
        return sample

def parse_arguments():
    parser = argparse.ArgumentParser(description="training codes")
    parser.add_argument("--output", type=str, default="../models/model_default")
    parser.add_argument("--input", type=str, default="../mat/NAC_train_diffusion")
    parser.add_argument("--target", type=str, default="../mat/CTAC_train_diffusion")
    parser.add_argument("--resume", type=bool, default=False)
    parser.add_argument("--from_state", type=str, default="../models/model_default/checkpoint/latest.pth")
    args = parser.parse_args()
    return args

def init_status(args):
    os.makedirs(args.output, exist_ok=True)
    os.makedirs(args.output+"/checkpoint", exist_ok=True)

def main(args):
    device = torch.device('cuda:1')
    
    # 载入数据
    dataset = TrainDataset(input=args.input, target=args.target)
    dataset_val = TrainDataset(input="../mat/NAC_test", target="../mat/CTAC_test")
    dataset_val.transform = None
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    dataloader_val = DataLoader(dataset_val, batch_size=BATCH_SIZE, shuffle=True)

    # 定义模型
    gen = PETUNet().to(device)
    critic = PETCritic().to(device)

    # 优化器和调度器
    criterion = WGANLoss()
    optimizer_gen = torch.optim.Adam(gen.parameters(), lr=1e-4, betas=(0.0, 0.9))
    optimizer_critic = torch.optim.Adam(critic.parameters(), lr=1e-4, betas=(0.0, 0.9))
    # scheduler_gen = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_gen, T_max=EPOCHS)
    # scheduler_critic = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_critic, T_max=EPOCHS)
    
    # 初始化训练
    loss_all = []
    num_batches = len(dataloader)
    
    g_steps = 1
    c_steps = 5
    
    if args.resume:
        checkpoint_path = args.from_state
        data = torch.load(checkpoint_path, map_location=device)
        gen.load_state_dict(data["gen_state"])
        critic.load_state_dict(data["critic_state"])
        optimizer_gen.load_state_dict(data["gen_optim"])
        optimizer_critic.load_state_dict(data["critic_optim"])
        print("Train from loaded state.")
    
    print(f"[INFO {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Start to train")

    for epoch in tqdm(range(EPOCHS), leave=False, desc="WGAN Training", disable=disable_tqdm):
        gen.train()
        critic.train()
        
        dis_total = torch.Tensor([0]).to(device)
        gen_loss_total = torch.Tensor([0]).to(device)
        critic_loss_total = torch.Tensor([0]).to(device)
        
        for sample_batched in tqdm(dataloader, leave=False, desc="Epoch progress", disable=disable_tqdm):
            input = sample_batched['input_img'].to(device, non_blocking=True)
            target = sample_batched['target_img'].to(device, non_blocking=True)
            
            # Train critic
            for _ in range(c_steps):
                optimizer_critic.zero_grad()
                fake_data = gen(input).detach()
                
                dis, loss_d = criterion.loss_critic(critic, input, target, fake_data)
                loss_d.backward()
                optimizer_critic.step()
                
                dis_total += dis
                critic_loss_total += loss_d
                
            # gradients = log_layer_gradients(critic, prefix="critic/")
            # print(gradients)
            # total_norm = torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(p.grad) for p in critic.parameters()]), 2)
            # print("Critic 梯度范数:", total_norm.item())
            
            # Train generator
            for _ in range(g_steps):
                optimizer_gen.zero_grad()
                fake_data = gen(input)
                
                loss_g = criterion.loss_generator(critic, input, fake_data, target)
                loss_g.backward()
                optimizer_gen.step()
                
                gen_loss_total += loss_g
            
        dis_avg = dis_total.item() / num_batches / c_steps
        gen_loss_avg = gen_loss_total.item() / num_batches / g_steps
        critic_loss_avg = critic_loss_total.item() / num_batches / c_steps
        loss_all.append([dis_avg, gen_loss_avg, critic_loss_avg])
        
        tqdm.write(
            f"[INFO {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] " +
            f"Epoch: {epoch + 1} | " +
            # 应该分别在 -2~2, -10~10, -10~10 范围内
            f"W Loss: {dis_avg:.8f} | G Loss: {gen_loss_avg:.8f} | D Loss: {critic_loss_avg:.8f} | " +
            f"G lr: {(optimizer_gen.param_groups[0]['lr']):.8f}"
        )
        
        # scheduler_gen.step()
        # scheduler_critic.step()
        
        if (epoch + 1) % 5 == 0:
            state = {
                "gen_state": gen.state_dict(),
                "gen_optim": optimizer_gen.state_dict(),
                "critic_state": critic.state_dict(),
                "critic_optim": optimizer_critic.state_dict()
            }
            path1 = os.path.join(args.output, f"checkpoint/{epoch + 1}.pth")
            torch.save(state, path1)
            shutil.copy2(path1, os.path.join(args.output, "checkpoint/latest.pth"))
            
            # 每若干轮验证一次
            gen.eval()
            critic.eval()
            ssim_total = 0
            mse_total = 0
            nmse_total = 0
            me_total = 0
            rmse_total = 0
            mae_total = 0
            n = 0
            for batch in dataloader_val:
                input = batch['input_img'].to(device)
                target = batch['target_img'].to(device)
                with torch.no_grad():
                    output_img = gen(input)
                output_img: np.ndarray = output_img.cpu().numpy()
                target: np.ndarray = target.cpu().numpy()
                for idx in range(target.shape[0]):
                    pred = output_img[idx].squeeze(0)
                    tar = target[idx].squeeze(0)
                    ssim, mse, nmse, me, rmse, mae = validate_loss(pred, tar)
                    ssim_total += ssim
                    mse_total += mse
                    nmse_total += nmse
                    me_total += me
                    rmse_total += rmse
                    mae_total += mae
                    n += 1

            tqdm.write(f"[INFO {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Validation info:")
            tqdm.write(f"[INFO {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] SSIM: {(ssim_total / n):.12f}")
            tqdm.write(f"[INFO {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] MSE: {(mse / n):.12f}")
            tqdm.write(f"[INFO {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] NMSE: {(nmse_total / n):.12f}")
            tqdm.write(f"[INFO {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ME: {(me_total / n):.12f}")
            tqdm.write(f"[INFO {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] RMSE: {(rmse_total / n):.12f}")
            tqdm.write(f"[INFO {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] MAE: {(mae_total / n):.12f}")
    
    state = {
        "gen_state": gen.state_dict(),
        "gen_optim": optimizer_gen.state_dict(),
        "critic_state": critic.state_dict(),
        "critic_optim": optimizer_critic.state_dict()
    }
    torch.save(state, os.path.join(args.output, "checkpoint/result.pth"))
    print(f"[INFO {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Train finished.")

if __name__ == '__main__':
    args = parse_arguments()
    init_status(args)
    torch.set_num_threads(4)
    main(args)
