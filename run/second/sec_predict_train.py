from utils import *
from dataset import *
from torchvision import transforms
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
from datafiles.color_dict import *
from models.MyModel import clean_mask
import tifffile
import random
import ttach as tta

print("PyTorch Version: ", torch.__version__)
print('cuda', torch.version.cuda)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device("cuda:0")
print('Device:', device)


def predict_val():
    opt = Sec_Options().parse()
    log_path, checkpoint_path, predict_test_path, predict_train_path, predict_val_path = create_save_path(opt)
    train_txt_path, val_txt_path, test_txt_path = create_data_path(opt)

    # load val dataset
    val_dataset = Dataset_point(opt, val_txt_path, flag='val', transform=None)
    loader = DataLoader(val_dataset, batch_size=1, num_workers=opt.num_workers,
                        pin_memory=opt.pin)

    model = Seg_Net(opt)
    checkpoint = torch.load(checkpoint_path + '/model_best_0.8184.pth', map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['state_dict'])
    print('resume success')
    model = model.to(device)
    model.eval()

    for batch_idx, batch in enumerate(tqdm(loader)):
        img, label, cls_label, filename = batch
        with torch.no_grad():
            input, label, cls_label = img.cuda(non_blocking=True), label.cuda(non_blocking=True), \
                                      cls_label.cuda(non_blocking=True)
            coarse_mask = model.forward(input)
            torch.cuda.synchronize()

        # save train predict after softmax and clean
        coarse_mask = F.softmax(coarse_mask, dim=1)
        mask_clean = clean_mask(coarse_mask, cls_label)
        mask_clean = torch.argmax(mask_clean, dim=1)
        save_pred_anno(mask_clean, predict_val_path, filename, dict=postdam_color_dict, flag=False)


def main():
    opt = Sec_Options().parse()
    save_path = os.path.join(opt.save_path, opt.dataset)

    log_path, checkpoint_path, predict_test_path, predict_train_path, predict_val_path = create_save_path(opt)
    train_txt_path, val_txt_path, test_txt_path = create_data_path(opt)

    # load train and val dataset
    train_dataset = Dataset_point(opt, train_txt_path, flag='predict_train', transform=None)
    loader = DataLoader(train_dataset, batch_size=1, num_workers=opt.num_workers,
                        pin_memory=opt.pin)

    model = Seg_Net(opt)
    checkpoint = torch.load(checkpoint_path+'/model_best_0.8184.pth', map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['state_dict'])
    print('resume success')
    model = model.to(device)
    model.eval()

    compute_metric = IOUMetric_tensor(num_classes=opt.num_classes)
    hist = torch.zeros([opt.num_classes, opt.num_classes]).cuda()

    for batch_idx, batch in enumerate(tqdm(loader)):
        img, label, cls_label, filename = batch
        with torch.no_grad():
            input, label, cls_label = img.cuda(non_blocking=True), label.cuda(non_blocking=True),\
                                      cls_label.cuda(non_blocking=True)
            tta_model = tta.SegmentationTTAWrapper(model, tta.aliases.d4_transform(),
                                                   merge_mode='mean')
            coarse_mask = tta_model.forward(input)
            torch.cuda.synchronize()

        output = torch.argmax(coarse_mask, dim=1)

        # save train predict after softmax and clean
        coarse_mask = F.softmax(coarse_mask, dim=1)
        mask_clean = clean_mask(coarse_mask, cls_label)
        mask_clean = torch.argmax(mask_clean, dim=1)
        save_pred_anno(mask_clean, predict_train_path, filename, dict=postdam_color_dict, flag=True)

        hist += compute_metric.get_hist(output, label)

    hist = hist.data.cpu().numpy()
    iou, miou, kappa, acc, acc_cls, f_score, m_f_score = eval_hist(hist)

    print('miou:%.4f iou:%s kappa:%.4f' % (miou, str(iou), kappa))
    print('acc:%.4f acc_cls:%s' % (acc, str(acc_cls)))
    print('mfscore:%.4f fscore:%s' % (m_f_score, str(f_score)))

    result_path = log_path[:-4] + '/result_predict_train.txt'
    result_txt = open(result_path, 'a')
    result_txt.write('miou:%.4f iou:%s kappa:%.4f' % (miou, str(iou), kappa) + '\n')
    result_txt.write('acc:%.4f acc_cls:%s' % (acc, str(acc_cls)) + '\n')
    result_txt.write('mfscore:%.4f fscore:%s' % (m_f_score, str(f_score)) + '\n')
    result_txt.write('---------------------------' + '\n')
    result_txt.close()


def show(predict_path, test_path):
    path = predict_path + '/label_vis'
    list = os.listdir(path)

    img_path = test_path + '/img'
    point_label_path = test_path + '/point_label_vis'
    label_path = test_path + '/label_vis'

    for i in list:
        predict = read(os.path.join(path, i))

        img = read(os.path.join(img_path, i))[:, :, :3]
        label = read(os.path.join(label_path, i))
        point_label = read(os.path.join(point_label_path, i))

        fig, axs = plt.subplots(1, 4, figsize=(14, 4))

        axs[0].imshow(img.astype(np.uint8))
        axs[0].axis("off")
        axs[1].imshow(label.astype(np.uint8))
        axs[1].axis("off")
        axs[2].imshow(predict.astype(np.uint8))
        axs[2].axis("off")
        axs[3].imshow(point_label.astype(np.uint8))
        axs[3].axis("off")

        plt.suptitle(os.path.basename(i), y=0.94)
        plt.tight_layout()
        plt.show()
        plt.close()


if __name__ == '__main__':
    main()

    # opt = Point_iter_second_Options().parse()
    # log_path, checkpoint_path, predict_path, predict_train_path, _ = create_save_path(opt)
    #
    # label_path = '/home/ggm/WLS/semantic/dataset/potsdam/train'
    # show(predict_train_path, label_path)

    predict_val()