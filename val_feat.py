import argparse
import os
import torch
from torch.utils.data import DataLoader
import numpy as np
import cv2
from tqdm import tqdm
from model.generator import PETModel
from dataset import TrainDataset2

def parse_arguments():
    parser = argparse.ArgumentParser(description="training codes")
    parser.add_argument("--output", type=str, default="../results/model_1200")
    parser.add_argument("--input", type=str, default="../mat/NAC_test_WGAN")
    parser.add_argument("--target", type=str, default="../mat/CTAC_test_WGAN")
    parser.add_argument("--model", type=str, default="../models/model_1200/checkpoint/latest.pth")
    args = parser.parse_args()
    return args

def init_status(args):
    os.makedirs(args.output, exist_ok=True)
    os.makedirs(os.path.join(args.output, "predict_png"), exist_ok=True)
    os.makedirs(os.path.join(args.output, "target"), exist_ok=True)
    os.makedirs(os.path.join(args.output, "origin"), exist_ok=True)
    
def process_image(img):
    img = np.expand_dims(img, axis=0)
    img = torch.from_numpy(img).unsqueeze(0).float().cuda()
    return img

def eval(args):
    # 🧠 2. 加载训练好的模型权重
    model = PETModel()
    checkpoint = torch.load(args.model, map_location='cuda')
    
    new_state_dict = {k[7:] if k.startswith('module.') else k: v for k, v in checkpoint.items()}
    model.load_state_dict(new_state_dict["gen_state"])
    model = model.cuda()
    model.eval()  # 设为推理模式
    
    output_folder = args.output
    
    dataset = TrainDataset2(input=args.input, target=args.target)
    dataloader = DataLoader(dataset, batch_size=1)
    dataset.transform = None
    
    i = 0
    
    for data in tqdm(dataloader):
        input_img, target_img, input = data
        input_img = input_img.cuda()
        target_img = target_img
        input = input.cuda()
        
        with torch.no_grad():
            output = model(input_img, input)
        
        # 处理输出
        output = output.squeeze(0).squeeze(0).cpu().numpy()
        input_img = input_img.squeeze(0).squeeze(0).cpu().numpy()
        target_img = target_img.squeeze(0).squeeze(0).cpu().numpy()
        
        # 归一化到 0-255（用于 PNG 保存）
        output_png = (output - output.min()) / (output.max() - output.min()) * 255
        output_png = output_png.astype(np.uint8)
                
        # 处理 input_img 和 target_img 以匹配输出
        input_img = (input_img - input_img.min()) / (input_img.max() - input_img.min()) * 255
        input_img = input_img.astype(np.uint8)

        target_img = (target_img - target_img.min()) / (target_img.max() - target_img.min()) * 255
        target_img = target_img.astype(np.uint8)

        # 拼接三张图像
        combined_img = np.concatenate([input_img, output_png, target_img], axis=1)
        
        # 保存拼接后的图像
        output_path = os.path.join(output_folder, "predict_png", f"{i}.png")
        cv2.imwrite(output_path, combined_img)
        i += 1
        
    print(f"All processed.")
    
if __name__ == "__main__":
    args = parse_arguments()
    init_status(args)
    eval(args)