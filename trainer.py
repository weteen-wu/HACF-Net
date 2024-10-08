from torch.autograd import Variable

from utils import psnr,ssim
from tqdm import tqdm

def rgb_to_ycbcr(input):
    # input is mini-batch N x 3 x H x W of an RGB image
    output = Variable(input.data.new(*input.size()))
    output[:, 0, :, :] = input[:, 0, :, :] * 65.481 + input[:, 1, :, :] * 128.553 + input[:, 2, :, :] * 24.966 + 16
    return output

def train(device,train_loader,model,criterion,optimizer,lr,epoch,args):
    model.train()
    dic = {"rate": args.subrate, "epoch": epoch,
               "device": device, "lr": lr}
    for i,(feature,label) in enumerate(tqdm(train_loader, desc="Now training: ", postfix=dic)):
        feature,label = feature.to(device),label.to(device)

        feature = rgb_to_ycbcr(feature)[:,0,:,:].unsqueeze(1) / 255.
        optimizer.zero_grad()
        y_hat = model(feature)
        loss = criterion(y_hat[0],feature) + criterion(y_hat[1],feature)
        loss.backward()
        optimizer.step()

        psnr_ = psnr(feature, y_hat[0])
        ssim_ = ssim(feature, y_hat[0])
        j = i + 1
        if j%50 == 0:
            output = (
        f"CS_ratio: {args.subrate:.6f} | Iteration: {j} | Loss: {loss:.4f} | "
        f"LR: {lr:.7f} | PSNR: {psnr_:.4f} | SSIM: {ssim_:.4f}"
    )
            tqdm.write(output)



