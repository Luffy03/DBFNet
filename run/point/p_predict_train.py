from utils import *
from dataset import *
from torchvision import transforms
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
from datafiles.color_dict import *
from models.MyModel import clean_mask
from models.tools import get_crf
import tifffile
import random
import ttach as tta

print("PyTorch Version: ", torch.__version__)
print('cuda', torch.version.cuda)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device("cuda:0")
print('Device:', device)


def predict_val():
    opt = Point_Options().parse()
    log_path, checkpoint_path, predict_test_path, predict_train_path, predict_val_path = create_save_path(opt)
    train_txt_path, val_txt_path, test_txt_path = create_data_path(opt)

    # load train and val dataset
    val_dataset = Dataset_point(opt, val_txt_path, flag='predict_val', transform=None)
    loader = DataLoader(val_dataset, batch_size=1, num_workers=opt.num_workers,
                        pin_memory=opt.pin)

    model = Net(opt, flag='test')
    checkpoint = torch.load(checkpoint_path + '/model_best_0.8553.pth', map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['state_dict'])
    print('resume success')
    model = model.cuda()
    model.eval()

    for batch_idx, batch in enumerate(tqdm(loader)):
        img, label, cls_label, filename = batch
        with torch.no_grad():
            input, label, cls_label = img.cuda(non_blocking=True), label.cuda(non_blocking=True), \
                                      cls_label.cuda(non_blocking=True)
            tta_model = tta.SegmentationTTAWrapper(model, tta.aliases.d4_transform(),
                                                   merge_mode='mean')
            coarse_mask = tta_model(input)
            torch.cuda.synchronize()

            # save train predict after softmax and clean
            coarse_mask = F.softmax(coarse_mask, dim=1)
            mask_clean = clean_mask(coarse_mask, cls_label)
            mask = mask_clean.squeeze(0).permute(1, 2, 0).cpu().numpy()
            mask = mask.astype(np.float32)

            # get crf
            img = img.squeeze(0).permute(1, 2, 0).cpu().numpy()
            img = Normalize_back(img, flag=opt.dataset)
            crf_out = get_crf(opt, mask, img.astype(np.uint8))

            # # save crf out
            predict_path_crf = predict_val_path + '/crf'
            check_dir(predict_path_crf)
            save_pred_anno_numpy(crf_out, predict_path_crf, filename, dict=postdam_color_dict, flag=False)


def main():
    opt = Point_Options().parse()
    log_path, checkpoint_path, predict_test_path, predict_train_path, predict_val_path = create_save_path(opt)
    train_txt_path, val_txt_path, test_txt_path = create_data_path(opt)

    # load train and val dataset
    train_dataset = Dataset_point(opt, train_txt_path, flag='predict_train', transform=None)
    loader = DataLoader(train_dataset, batch_size=1, num_workers=opt.num_workers,
                        pin_memory=opt.pin)

    model = Net(opt, flag='test')
    checkpoint = torch.load(checkpoint_path+'/model_best_0.8553.pth', map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['state_dict'])
    print('resume success')
    model = model.to(device)
    model.eval()

    # numpy for crf
    compute_metric_ny = IOUMetric(num_classes=opt.num_classes)
    hist_ny = np.zeros([opt.num_classes, opt.num_classes])

    for batch_idx, batch in enumerate(tqdm(loader)):
        img, label, cls_label, filename = batch
        with torch.no_grad():
            input, label, cls_label = img.cuda(non_blocking=True), label.cuda(non_blocking=True),\
                                      cls_label.cuda(non_blocking=True)
            tta_model = tta.SegmentationTTAWrapper(model, tta.aliases.d4_transform(),
                                                   merge_mode='mean')
            coarse_mask = tta_model(input)
            torch.cuda.synchronize()

        # save train predict after softmax and clean
        coarse_mask = F.softmax(coarse_mask, dim=1)
        mask_clean = clean_mask(coarse_mask, cls_label)

        mask = mask_clean.squeeze(0).permute(1, 2, 0).cpu().numpy()
        mask = mask.astype(np.float32)

        # get crf
        img = img.squeeze(0).permute(1, 2, 0).cpu().numpy()
        img = Normalize_back(img, flag=opt.dataset)
        crf_out = get_crf(opt, mask, img.astype(np.uint8))
        # # # save crf out
        predict_path_crf = predict_train_path + '/crf'
        check_dir(predict_path_crf)
        save_pred_anno_numpy(crf_out, predict_path_crf, filename, dict=postdam_color_dict, flag=False)

        label_ny = label.squeeze(0).data.cpu().numpy()
        hist_ny += compute_metric_ny.get_hist(crf_out, label_ny)

    # crf out's metric
    iou, miou, kappa, acc, acc_cls, f_score, m_f_score = eval_hist(hist_ny)

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
    predict_val()