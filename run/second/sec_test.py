from utils import *
from dataset import *
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
from datafiles.color_dict import *
from models.tools import get_crf
import random
import ttach as tta

print("PyTorch Version: ", torch.__version__)
print('cuda', torch.version.cuda)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device("cuda:0")
print('Device:', device)


def main():
    opt = Sec_Options().parse()
    save_path = os.path.join(opt.save_path, opt.dataset)
    log_path, checkpoint_path, predict_path, _,  _ = create_save_path(opt)
    train_txt_path, val_txt_path, test_txt_path = create_data_path(opt)

    # load train and val dataset
    test_dataset = Dataset_point(opt, test_txt_path, flag='test', transform=None)
    loader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=opt.num_workers,
                        pin_memory=opt.pin)

    model = Seg_Net(opt)
    checkpoint = torch.load(checkpoint_path+'/model_best.pth', map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['state_dict'])
    print('resume success')
    model = model.to(device)
    model.eval()

    compute_metric = IOUMetric_tensor(num_classes=opt.num_classes)
    hist = torch.zeros([opt.num_classes, opt.num_classes]).cuda()

    for batch_idx, batch in enumerate(tqdm(loader)):
        img, label, cls_label, filename = batch
        with torch.no_grad():
            input, label = img.cuda(non_blocking=True), label.cuda(non_blocking=True)
            tta_model = tta.SegmentationTTAWrapper(model, tta.aliases.d4_transform(),
                                                   merge_mode='mean')
            output = model(input)
            torch.cuda.synchronize()

        if opt.dataset == 'vaihingen':
            cls_label = cls_label.cuda(non_blocking=True)
            coarse_mask = F.softmax(output, dim=1)
            mask_clean = clean_mask(coarse_mask, cls_label)
            output = torch.argmax(mask_clean, dim=1)
        else:
            output = torch.argmax(output, dim=1)

        # save_pred_anno(output, predict_path, filename, dict=postdam_color_dict, flag=True)
        hist += compute_metric.get_hist(output, label)

    hist = hist.data.cpu().numpy()
    iou, miou, kappa, acc, acc_cls, f_score, m_f_score = eval_hist(hist)

    print('miou:%.4f iou:%s kappa:%.4f' % (miou, str(iou), kappa))
    print('acc:%.4f acc_cls:%s' % (acc, str(acc_cls)))
    print('mfscore:%.4f fscore:%s' % (m_f_score, str(f_score)))

    # result_path = log_path[:-4] + '/result_test.txt'
    # result_txt = open(result_path, 'a')
    # result_txt.write('miou:%.4f iou:%s kappa:%.4f' % (miou, str(iou), kappa) + '\n')
    # result_txt.write('acc:%.4f acc_cls:%s' % (acc, str(acc_cls)) + '\n')
    # result_txt.write('mfscore:%.4f fscore:%s' % (m_f_score, str(f_score)) + '\n')
    # result_txt.write('---------------------------' + '\n')
    # result_txt.close()


def p_test_withcrf():
    opt = Sec_Options().parse()
    log_path, checkpoint_path, predict_path, _, _ = create_save_path(opt)
    train_txt_path, val_txt_path, test_txt_path = create_data_path(opt)

    # load train and val dataset
    test_dataset = Dataset_point(opt, test_txt_path, flag='test', transform=None)
    loader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=opt.num_workers,
                        pin_memory=opt.pin)

    model = Seg_Net(opt)
    checkpoint = torch.load(checkpoint_path + '/model_best.pth', map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['state_dict'])
    print('resume success')
    model = model.to(device)
    model.eval()

    compute_metric = IOUMetric(num_classes=opt.num_classes)
    hist = np.zeros([opt.num_classes, opt.num_classes])

    for batch_idx, batch in enumerate(tqdm(loader)):
        img, label, cls_label, filename = batch
        with torch.no_grad():
            input, label = img.cuda(non_blocking=True), label.cuda(non_blocking=True)
            tta_model = tta.SegmentationTTAWrapper(model, tta.aliases.d4_transform(),
                                                   merge_mode='mean')
            output = tta_model(input)
            torch.cuda.synchronize()

        if opt.dataset == 'vaihingen':
            cls_label = cls_label.cuda(non_blocking=True)
            coarse_mask = F.softmax(output, dim=1)
            mask = clean_mask(coarse_mask, cls_label)
        else:
            mask = F.softmax(output, dim=1)

        mask = mask.squeeze(0).permute(1, 2, 0).cpu().numpy()
        mask = mask.astype(np.float32)

        img = img.squeeze(0).permute(1, 2, 0).cpu().numpy()
        img = Normalize_back(img, flag='potsdam')

        crf_out = get_crf(opt, mask, img.astype(np.uint8))
        label = label.squeeze(0).data.cpu().numpy()

        # save crf out
        predict_path_crf = predict_path + '/crf'
        check_dir(predict_path_crf)
        # save_pred_anno_numpy(crf_out, predict_path_crf, filename, dict=nuclear_color_dict, flag=True)

        hist += compute_metric.get_hist(crf_out, label)

    iou, miou, kappa, acc, acc_cls, f_score, m_f_score = eval_hist(hist)

    print('miou:%.4f iou:%s kappa:%.4f' % (miou, str(iou), kappa))
    print('acc:%.4f acc_cls:%s' % (acc, str(acc_cls)))
    print('mfscore:%.4f fscore:%s' % (m_f_score, str(f_score)))


def show(predict_path, test_path):
    # crf_path = predict_path[:-10] + '/crf/label_vis'
    img_path = test_path + '/img'
    label_path = test_path + '/label_vis'

    list = os.listdir(predict_path)
    random.shuffle(list)

    for i in list:
        predict = read(os.path.join(predict_path, i))
        img = read(os.path.join(img_path, i))[:, :, :3]
        # crf = read(os.path.join(crf_path, i))
        label = read(os.path.join(label_path, i))

        fig, axs = plt.subplots(1, 3, figsize=(14, 4))

        axs[0].imshow(img.astype(np.uint8))
        axs[0].axis("off")
        axs[1].imshow(label.astype(np.uint8))
        axs[1].axis("off")

        axs[2].imshow(predict.astype(np.uint8))
        axs[2].axis("off")

        # axs[3].imshow(crf.astype(np.uint8))
        # axs[3].axis("off")

        plt.suptitle(os.path.basename(i), y=0.94)
        plt.tight_layout()
        plt.show()
        plt.close()


if __name__ == '__main__':
    main()
    p_test_withcrf()

    # predict_path = '/home/ggm/WLS/semantic/PointAnno/save/nuclear/Second/predict_test/label_vis'
    # test_path = '/home/ggm/WLS/semantic/dataset/nuclear/test'
    # show(predict_path, test_path)