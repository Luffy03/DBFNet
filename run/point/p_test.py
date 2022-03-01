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
    opt = Point_Options().parse()
    log_path, checkpoint_path, predict_path, _, _ = create_save_path(opt)
    train_txt_path, val_txt_path, test_txt_path = create_data_path(opt)

    # load train and val dataset
    test_dataset = Dataset_point(opt, test_txt_path, flag='test', transform=None)
    loader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=opt.num_workers,
                        pin_memory=opt.pin)

    model = Net(opt, flag='test')
    checkpoint = torch.load(checkpoint_path + '/model_best_0.8553.pth', map_location=torch.device('cpu'))
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

        mask = F.softmax(output, dim=1)
        mask = mask.squeeze(0).permute(1, 2, 0).cpu().numpy()
        mask = mask.astype(np.float32)

        img = img.squeeze(0).permute(1, 2, 0).cpu().numpy()
        img = Normalize_back(img, flag=opt.dataset)

        crf_out = get_crf(opt, mask, img.astype(np.uint8))
        # save_pred_anno_numpy(crf_out, predict_path, filename, dict=postdam_color_dict, flag=True)
        label = label.squeeze(0).data.cpu().numpy()

        hist += compute_metric.get_hist(crf_out, label)

    iou, miou, kappa, acc, acc_cls, f_score, m_f_score = eval_hist(hist)

    print('miou:%.4f iou:%s kappa:%.4f' % (miou, str(iou), kappa))
    print('acc:%.4f acc_cls:%s' % (acc, str(acc_cls)))
    print('mfscore:%.4f fscore:%s' % (m_f_score, str(f_score)))


if __name__ == '__main__':
    main()

