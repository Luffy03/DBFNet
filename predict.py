from utils import *
from dataset import *
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
from datafiles.color_dict import *
from models.tools import get_crf
import random
from dataset.data_utils import value_to_rgb
# from models.tools import get_crf
import ttach as tta

print("PyTorch Version: ", torch.__version__)
print('cuda', torch.version.cuda)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device("cuda:0")
print('Device:', device)


def save(visual, name, path):
    check_dir(path)
    imsave(path+'/'+name[:-4]+'.png', visual)


def eval(opt, out, label, name, stride=128):
    compute_metric = IOUMetric(num_classes=opt.num_classes)
    hist = np.zeros([opt.num_classes, opt.num_classes])

    h, w, _ = label.shape
    num_h, num_w = h//stride, w//stride
    for i in range(num_h):
        for j in range(num_w):
            o = out[i*stride:(i+1)*stride, j*stride:(j+1)*stride]
            l = label[i*stride:(i+1)*stride, j*stride:(j+1)*stride, 0]
            hist += compute_metric.get_hist(o, l)
    # # evaluate
    iou, miou, kappa, acc, acc_cls, f_score, m_f_score = eval_hist(hist)
    print(name)
    print('mfscore:%.4f fscore:%s' % (m_f_score, str(f_score)))
    print('------'*5)
    return hist


def predict_tile(opt, model, img, label, label_vis, name,
                 save_path, dataset='potsdam', size=256):
    im = img.copy()
    with torch.no_grad():
        model.eval()
        h, w, c = img.shape
        img = Normalize(img, flag=dataset)
        img = img.astype(np.float32).transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0).float().cuda()
        output = pre_slide(model, img, num_classes=5, tile_size=(size, size), tta=True)

    output = output.squeeze(0).permute(1, 2, 0).data.cpu().numpy().astype(np.float32)

    output = output[:h, :w, :]
    output = np.argmax(output, axis=-1)
    hist = eval(opt, output, label, name)

    # output = np.expand_dims(output, axis=-1)
    # predict = value_to_rgb(output, flag=opt.dataset)
    # fig, axs = plt.subplots(1, 3, figsize=(20, 8))
    # axs[0].imshow(im.astype(np.uint8))
    # axs[0].axis("off")
    # axs[1].imshow(label_vis.astype(np.uint8))
    # axs[1].axis("off")
    # axs[2].imshow(predict.astype(np.uint8))
    # axs[2].axis("off")
    # plt.suptitle(os.path.basename(name), y=0.94)
    # plt.tight_layout()
    # plt.show()
    # plt.close()

    # save(predict, name)

    return hist


def run_potsdam(root_path, final=True):
    img_path = root_path + '/4_Ortho_RGBIR'
    label_path = root_path + '/Labels'
    vis_path = root_path + '/5_Labels_all'
    save_path = './save/potsdam/predict_masks'
    train_name = [7, 8, 9, 10, 11, 12]

    opt = Sec_Options().parse()
    model = Seg_Net(opt)
    checkpoint = torch.load('./save/potsdam_0.8719.pth', map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['state_dict'])
    model = model.cuda()
    model.eval()

    hist = np.zeros([opt.num_classes, opt.num_classes])
    list = os.listdir(label_path)
    for i in list:
        if int(i[14:-10]) in train_name:
            pass
        else:
            img = read(os.path.join(img_path, i[:-9] + 'RGBIR.tif'))[:, :, :3]
            label = read(os.path.join(label_path, i))
            vis = read(os.path.join(vis_path, i))
            hist += predict_tile(opt, model, img, label, vis, i, save_path, dataset='potsdam')

    iou, miou, kappa, acc, acc_cls, f_score, m_f_score = eval_hist(hist)
    print('total')
    print(acc)
    print('mfscore:%.4f fscore:%s' % (m_f_score, str(f_score)))
    print('------' * 5)


def run_vaihingen(root_path):
    save_path = './save/vaihingen/predict_masks'
    img_path = root_path + '/image'
    label_path = root_path + '/gts_noB'
    vis_path = root_path + '/vis_noB'
    test_name = [2, 4, 6, 8, 10, 12, 14, 16, 20, 22, 24, 27, 29, 31, 33, 35, 38]

    opt = Sec_Options().parse()
    model = Seg_Net(opt)

    checkpoint = torch.load('./save/vaihingen_0.8867.pth', map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['state_dict'])
    model = model.cuda()
    model.eval()

    hist = np.zeros([opt.num_classes, opt.num_classes])
    list = os.listdir(label_path)
    for i in list:
        if int(i[20:-4]) not in test_name:
            pass
        else:
            img = read(os.path.join(img_path, i))
            label = read(os.path.join(label_path, i))
            vis = read(os.path.join(vis_path, i))
            hist += predict_tile(opt, model, img, label, vis, i, save_path, dataset='vaihingen')

    iou, miou, kappa, acc, acc_cls, f_score, m_f_score = eval_hist(hist)
    print('total')
    print('mfscore:%.4f fscore:%s' % (m_f_score, str(f_score)))
    print('------' * 5)


if __name__ == '__main__':
    potsdam_path = '/media/hlf/Luffy/WLS/semantic/dataset/potsdam/dataset_origin'
    run_potsdam(potsdam_path)

    # vaihingen_path = '/media/hlf/Luffy/WLS/semantic/dataset/vaihingen/dataset_origin'
    # run_vaihingen(vaihingen_path)

