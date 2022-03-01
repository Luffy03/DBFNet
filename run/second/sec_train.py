import torch.optim as optim
from torchvision import transforms
from torch.autograd import Variable
import shutil
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import OrderedDict
from options import *
from utils import *
from dataset import *


print("PyTorch Version: ", torch.__version__)
print('cuda', torch.version.cuda)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device("cuda:0")
print('Device:', device)


def main():
    opt = Sec_Options().parse()
    save_path = os.path.join(opt.save_path, opt.dataset)
    log_path, checkpoint_path, _, _, _ = create_save_path(opt)
    train_txt_path, val_txt_path, _ = create_data_path(opt)

    # Define logger
    logger, tensorboard_log_dir = create_logger(log_path)
    logger.info(opt)
    # Define Transformation
    train_transform = transforms.Compose([
        trans.RandomHorizontalFlip(),
        trans.RandomVerticleFlip(),
        trans.RandomRotate90(),
    ])

    # load train and val dataset
    train_dataset = Dataset_sec(opt, save_path, train_txt_path, flag='train', transform=train_transform)
    val_dataset = Dataset_sec(opt, save_path, val_txt_path, flag='val', transform=None)

    # Create training and validation dataloaders
    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True,
                              num_workers=opt.num_workers, pin_memory=opt.pin, drop_last=True)

    model = Seg_Net(opt)
    if opt.resume:
        checkpoint = torch.load(checkpoint_path+'/model_best.pth', map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['state_dict'])
        print('resume success')

    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=opt.base_lr, amsgrad=True)
    best_metric = -1e16

    for epoch in range(opt.num_epochs):
        time_start = time.time()

        train(opt, epoch, model, train_loader, optimizer, logger)
        metric = validate(opt, epoch, model, val_dataset, logger)

        logger.info('best_val_metric:%.4f current_val_metric:%.4f' % (best_metric, metric))
        if metric > best_metric:
            logger.info('epoch:%d Save to model_best' % (epoch))
            torch.save({'state_dict': model.state_dict()},
                       os.path.join(checkpoint_path, 'model_best.pth'))
            best_metric = metric

        end_time = time.time()
        time_cost = end_time - time_start
        logger.info("Epoch %d Time %d ----------------------" % (epoch, time_cost))
        logger.info('\n')


def train(opt, epoch, model, loader, optimizer, logger):
    start_iter_epoch = 15

    model.train()
    loss_m = AverageMeter()
    soft_loss_m = AverageMeter()
    last_idx = len(loader) - 1

    for batch_idx, batch in enumerate(loader):
        # step = epoch * len(loader) + batch_idx
        # adjust_learning_rate(opt.base_lr, optimizer, step, len(loader))
        lr = get_lr(optimizer)

        last_batch = batch_idx == last_idx
        img, label, name = batch
        input, label = img.cuda(non_blocking=True), label.cuda(non_blocking=True)

        output = model(input)

        criterion = nn.CrossEntropyLoss(ignore_index=255)
        loss = criterion(output, label)
        # if epoch <= start_iter_epoch:
        #     loss = criterion(output, label)
        #
        # else:
        #     soft_loss = get_soft_loss(output, soft_label)
        #     loss = criterion(output, label) + soft_loss
        #     soft_loss_m.update(soft_loss.item(), input.size(0))

        loss_m.update(loss.item(), input.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()

        log_interval = len(loader) // 3
        if last_batch or batch_idx % log_interval == 0:
            logger.info('Train:{} [{:>4d}/{} ({:>3.0f}%)] '
                    'Loss:({loss.avg:>6.4f}) '
                    #'soft:({soft.avg:>6.4f}) '
                    'LR:{lr:.3e} '.format(
                        epoch,
                        batch_idx, len(loader),
                        100. * batch_idx / last_idx,
                        loss=loss_m,
                        lr=lr))

    # if epoch >= start_iter_epoch:
    #     logger.info('predict_train')
    #     predict_train(opt, model)

    return OrderedDict([('loss', loss_m.avg)])


def predict_train(opt, model):
    train_txt_path, val_txt_path, test_txt_path = create_data_path(opt)
    save_path = os.path.join(opt.save_path, opt.dataset)
    train_dataset = Dataset_soft(opt, save_path, train_txt_path, flag='train', transform=None)
    loader = DataLoader(train_dataset, batch_size=1, num_workers=opt.num_workers,
                        pin_memory=opt.pin)

    model.eval()
    for batch_idx, batch in enumerate(tqdm(loader)):
        img, label, cls_label, soft_label, filename = batch
        with torch.no_grad():
            input, cls_label = img.cuda(non_blocking=True), cls_label.cuda(non_blocking=True)
            coarse_mask = model.forward(input)
            mask = F.softmax(coarse_mask, dim=1)
            mask = clean_mask(mask, cls_label)
            mask = mask.squeeze(0).permute(1, 2, 0).data.cpu().numpy()

            save_mask_path = save_path + '/Second/soft_label'
            check_dir(save_mask_path)
            path = save_mask_path + '/' + filename[0][:-4] + '.npy'

            if os.path.isfile(path):
                soft_label = np.load(path)
                new_soft_label = (soft_label + mask) / 2
                np.save(path, new_soft_label)
            else:
                np.save(path, mask)


def validate(opt, epoch, model, val_dataset, logger):
    compute_metric = IOUMetric_tensor(num_classes=opt.num_classes)
    hist = torch.zeros([opt.num_classes, opt.num_classes]).cuda()

    val_loader = DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers, pin_memory=opt.pin)
    model.eval()

    for batch_idx, batch in enumerate(val_loader):
        with torch.no_grad():
            img, label, name = batch
            input, label = img.cuda(non_blocking=True), label.cuda(non_blocking=True)
            output = model(input)
            torch.cuda.synchronize()

        output = torch.argmax(output, dim=1)
        hist += compute_metric.get_hist(output, label)

    iou = torch.diag(hist) / (hist.sum(dim=1) + hist.sum(dim=0) - torch.diag(hist))
    miou = torch.mean(iou).float()

    logger.info('epoch:%d current_miou:%.4f current_iou:%s' % (epoch, miou, str(iou)))

    return miou


if __name__ == '__main__':
   main()