import time
import os
import cv2
from datasets.dataset import CrackData
import argparse
import cfg
from os.path import splitext, join
import logging

from model import *

def createDataList(inputDir, outputFileName='data.lst', supportedExtensions=['.png', '.jpg', '.jpeg']):

    out = open(join(inputDir, outputFileName), "w")
    res = []
    for root, directories, files in os.walk(inputDir):
        for f in files:
            for extension in supportedExtensions:
                fn, ext = splitext(f.lower())

                if extension == ext:
                    out.write('%s %s\n' % (f, f))
                    res.append(f)

    out.close()
    return res

def onescale_test(model, args):
    test_root = cfg.config_test[args.dataset]['data_root']

    logging.info('Processing: %s' % test_root)
    test_lst = cfg.config_test[args.dataset]['data_lst']

    imageFileNames = createDataList(test_root, test_lst)

    test_img = CrackData(test_root, test_lst)
    testloader = torch.utils.data.DataLoader(test_img, batch_size=1, shuffle=False, num_workers=8)

    save_dir = args.res_dir
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    if args.cuda:
        model.cuda()

    model.eval()
    start_time = time.time()
    # all_t = 0
    timeRecords = open(join(save_dir, 'timeRecords.txt'), "w")
    timeRecords.write('# filename time[ms]\n')

    scale = [1]
    for idx, (image, _) in enumerate(testloader):
        image = image[0]
        image_in = image.numpy().transpose((1, 2, 0))
        _, H, W = image.shape

        for k in range(0, len(scale)):
            im_ = image_in.transpose((2, 0, 1))
            tm = time.time()
            results = model(torch.unsqueeze(torch.from_numpy(im_).cuda(), 0))
            result = F.sigmoid(results[-1]).cpu().data.numpy()[0, :, :]

        # elapsedTime = time.time() - tm
        # timeRecords.write('%s %f\n' % (imageFileNames[idx], elapsedTime * 1000))
        # cv2.imwrite(os.path.join(save_dir, '%s.jpg' % imageFileNames[idx][:-4]), 255 - result * 255)

        cv2.imwrite(os.path.join(save_dir, '%s.png' % imageFileNames[idx][:-4]), result * 255)
        print("Running test [%d/%d]" % (idx + 1, len(testloader)))    ### lk 2022.02.28

    # timeRecords.write('Overall Time use: %f \n' % (time.time() - start_time))    ### lk 2022.02.26
    # timeRecords.close()
    # print(all_t)
    # print('Overall Time use: ', time.time() - start_time)
    print(time.time() - start_time)

def main():
    import time
    print(time.localtime())
    args = parse_args()

    args.cuda = True  ######### lk
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    logging.info('Loading model...')

    model = CarNet34(
        Encoder=Encoder_v0_762,
        dp=DownsamplerBlock,
        block=BasicBlock_encoder,
        channels=[3, 16, 64, 128, 256],
        decoder_block=non_bottleneck_1d_2,
        num_classes=1
    )

    logging.info('Loading state...')
    model.load_state_dict(torch.load('%s' % (args.model)))
    logging.info('Start image processing...')

    onescale_test(model, args)

def parse_args():
    parser = argparse.ArgumentParser('test model performance')
    # parser.add_argument('-d', '--dataset', type=str, choices=cfg.config_test.keys(),
    #                     default='BJN260', help='The dataset to train')
    # parser.add_argument('-d', '--dataset', type=str, choices=cfg.config_test.keys(),
    #                     default='Crack360', help='The dataset to train')
    # parser.add_argument('-d', '--dataset', type=str, choices=cfg.config_test.keys(),
    #                     default='Rain365', help='The dataset to train')
    parser.add_argument('-d', '--dataset', type=str, choices=cfg.config_test.keys(),
                        default='Sun520', help='The dataset to train')
    parser.add_argument('-i', '--inputDir', type=str, default=None, help='Input image directory for testing.')
    parser.add_argument('-c', '--cuda', action='store_true', help='whether use gpu to train network')
    parser.add_argument('-g', '--gpu', type=str, default='0', help='the gpu id to train net')
    parser.add_argument('-m', '--model', type=str,
                        default='Sun520_aug_CarNet/CarNet_15000.pth',
                        help='the model to test')
    parser.add_argument('--res-dir', type=str,
                        default='Sun520_aug_CarNet/onescale_test_15e3',
                        help='the dir to store result')
    return parser.parse_args()

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s %(levelname)s:\t%(message)s', level=logging.INFO)
main()
