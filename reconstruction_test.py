import argparse, torch, os
from data_utils import data_loader
from Network import HACF_Net
from trainer import rgb_to_ycbcr
from torchvision import transforms
from utils import psnr, ssim

parser = argparse.ArgumentParser(description='Test Models')


parser.add_argument('--subrate', default=0.125, type=float, help='sampling sub rate')
parser.add_argument('--image_size', default=256, type=int, help='image size used for training (default: 96)')
parser.add_argument('--batchSize', default=2, type=int, help='mini-batch size (default: 128)')
parser.add_argument('--recon_dir', default='./result', type=str, help='mini-batch size (default: 128)')
parser.add_argument('--save_path', default='./parameter', type=str, help = 'resume training or not')
parser.add_argument('--data_path', default=r"../hy-tmp/train5k/",help='datasets root')
parser.add_argument('--set11_path', default=r"../hy-tmp/set11/",help='datasets root')
opt = parser.parse_args()

recon_dir = os.path.join(opt.recon_dir, str(opt.subrate), "resonstruction_img")
if not os.path.exists(recon_dir):
    os.makedirs(recon_dir)

save_folder = os.path.join(opt.save_path, str(opt.subrate), "models")

def imread_CS_py(inputs):
    block_size = 64
    _, _, H, W = inputs.shape

    H_pad = 0
    W_pad = 0

    if H % block_size != 0:
        H_pad = block_size - (H % block_size)

    if W % block_size != 0:
        W_pad = block_size - (W % block_size)

    Ipad = inputs

    if H_pad > 0 or W_pad > 0:
        Ipad = torch.cat((inputs, torch.zeros(1, 1, H, W_pad)), dim=3)
        Ipad = torch.cat((Ipad, torch.zeros(1, 1, H_pad, W + W_pad)), dim=2)

    [B, C, H_new, W_new] = Ipad.shape

    return [H, W, Ipad, H_new, W_new]


def main_test():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    _,test_loader = data_loader(opt)
    net = HACF_Net(sensing_rate=opt.subrate, img_size=256, window_size=8,
                   depths=[4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4], num_heads=[4, 8, 4, 8, 4, 8, 8, 4, 8, 4, 8, 4, 8],
                   mlp_ratio=4.)
    save_path = os.path.join(save_folder, f'HACF_Net_{opt.subrate}.pth')
    checkpoint = torch.load(save_path)
    net.load_state_dict(checkpoint['net'])
    net = net.to(device)
    net.eval()
    psnr_total = 0
    ssim_total = 0

    with torch.no_grad():
        for iters, (inputs, _) in enumerate(test_loader):
            inputs_ycbcr = rgb_to_ycbcr(inputs)[:, 0, :, :].unsqueeze(1) / 255.
            [H, W, Ipad, H_new, W_new] = imread_CS_py(inputs_ycbcr)
            Ipad = Ipad.to(device)
            inputs_ycbcr = inputs_ycbcr.to(device)
            B, C, _, _ = Ipad.shape
            recon_img1, recon_img2 = net(Ipad)
            recon_img = recon_img1[0, 0, 0:H, 0:W]

            psnr_value = psnr(inputs_ycbcr, recon_img.unsqueeze(0).unsqueeze(1))
            print(f"PSNR: {psnr_value:.4f}")
            psnr_total += psnr_value

            ssim_value = ssim(inputs_ycbcr, recon_img.unsqueeze(0).unsqueeze(1))
            print(f"SSIM: {ssim_value:.4f}")
            ssim_total += ssim_value

            recon_img = recon_img.data.cpu()
            recon_img = torch.squeeze(recon_img, 0)
            recon_img = transforms.ToPILImage()(recon_img)
            recon_img_name = os.path.join(recon_dir, f'reconstructed_{iters}.png')
            recon_img.save(recon_img_name)
    # Print average PSNR and SSIM
    print(f'Average PSNR: {psnr_total / len(test_loader):.4f}')
    print(f'Average SSIM: {ssim_total / len(test_loader):.4f}')


if __name__ == "__main__":
    main_test()




