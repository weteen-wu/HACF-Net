import torch,os
import argparse
import torch.optim as optim
import torch.nn as nn
from Network import HACF_Net
from trainer import train
from data_utils import data_loader

parser = argparse.ArgumentParser(description='Compressed sensing with HACF_Net')
parser.add_argument('--lr',default=2*(1e-4), type=float, help='initial learning rate')
parser.add_argument('--nEpochs',default=100,type=int,help='number of total epochs to run')
parser.add_argument('--batchSize',default=2,type=int,help='mini-batch size (default: 128)')
parser.add_argument('--image_size',default=256,type=int,help='image size used for training (default: 96)')
parser.add_argument('--save_epoch',default=2,type=int,help='image size used for training (default: 96)')
parser.add_argument('--subrate',default=0.125,type=float,
                    choices=[0.50000, 0.25000, 0.12500, 0.06250, 0.03125, 0.015625],help='set sensing rate')
parser.add_argument('--save_path', default='./parameter', type=str, help = 'resume training or not')
parser.add_argument('--data_path', default=r"../hy-tmp/train5k/",help='datasets root')
parser.add_argument('--set11_path', default=r"../hy-tmp/set11/",help='datasets root')
args = parser.parse_args()

save_folder = os.path.join(args.save_path, str(args.subrate), "models")
if not os.path.exists(save_folder):
    os.makedirs(save_folder)


def loss_fn(outputs,inputs):
    mse = ((inputs-outputs)**2).mean(-1).mean(-1).squeeze()
    loss = torch.sqrt((torch.sqrt(mse) ** 2).mean())
    return loss

def main(net):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = net.to(device)

    criterion = loss_fn
    optimizer = optim.Adam(model.parameters(),args.lr,betas=(0.9,0.999))
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [30, 60, 90], gamma=0.1)

    train_loader,_ = data_loader(args)


    start_epoch = 0
    save_path = os.path.join(save_folder, f'HACF_Net_{args.subrate}.pth')
    if os.path.exists(save_path):
        checkpoint = torch.load(save_path)
        model.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch']

    for epoch in range(start_epoch+1,args.nEpochs+1):
        print("Epoch: " + str(epoch))
        print('current lr {:.5e}'.format(scheduler.get_lr()[0]))
        train(device,train_loader,model,
              criterion,optimizer,optimizer.state_dict()['param_groups'][0]['lr'],epoch,args)


        scheduler.step()

        if epoch % args.save_epoch ==0:
            checkpoint = {
                "net":model.state_dict(),
                "optimizer":optimizer.state_dict(),
                "scheduler":scheduler.state_dict(),
                "epoch":epoch
            }
            torch.save(checkpoint, save_path)


if __name__ == "__main__":
    net = HACF_Net(sensing_rate=args.subrate, img_size=256, window_size=8,
                   depths=[4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4], num_heads=[4, 8, 4, 8, 4, 8, 8, 4, 8, 4, 8, 4, 8],
                   mlp_ratio=4.)
    main(net)


