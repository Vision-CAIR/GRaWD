import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"

from time import gmtime, strftime, time
import argparse
import random
import glob
import copy
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, help='dataset to be used: CUB/NAB/SUN', default='CUB')
parser.add_argument('--splitmode', type=str, help='the way to split train/test data: easy/hard', default='hard')
parser.add_argument('--exp_name', default='Reproduce', type=str, help='Experiment Name')
parser.add_argument('--main_dir', default='./', type=str, help='Main Directory including data folder')
parser.add_argument('--gpu', default='0', type=str, help='index of GPU to use')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--resume', type=str, help='the model to resume')
parser.add_argument('--disp_interval', type=int, default=20)
parser.add_argument('--save_interval', type=int, default=200)
parser.add_argument('--evl_interval', type=int, default=40)
parser.add_argument('--rw_config_path', default='rw_config.yml', help='Path to RW loss configuration file')
parser.add_argument('--num_iterations', default=3000, type=int, help='For how many iterations to run.')
parser.add_argument('--is_val', action='store_true', help='Should we evaluate on a validation set?')

opt, _ = parser.parse_known_args()
print(opt)
# os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.autograd as autograd
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.autograd import Variable
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import scipy.integrate as integrate
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from firelab.config import Config

from dataset import FeatDataLayer, LoadDataset, LoadDataset_NAB
from models import _netD, _netG, _param
from losses.gan_losses import compute_gradient_penalty, compute_centroid_loss
from losses.prototype_loss import compute_prototype_loss
from losses.rw_loss import compute_rw_real_loss, compute_rw_imitative_loss, compute_rw_creative_loss
from losses.ausuc import compute_ausuc

rw_config = Config.load(opt.rw_config_path, frozen=False)
# rw_config_cli = Config.read_from_cli(config_arg_prefix='--rw_config.')
# rw_config = rw_config.overwrite(rw_config_cli)
rw_config.freeze()
print('<=========== Random Walk config ===========>')
print(rw_config)

""" hyper-parameter for training """
opt.GP_LAMBDA = 10  # Gradient penalty lambda
opt.CENT_LAMBDA = 1
opt.REG_W_LAMBDA = 0.001
opt.REG_Wz_LAMBDA = 0.0001

opt.lr = rw_config.lr
opt.batchsize = 1000

""" hyper-parameter for testing"""
opt.nSample = 60  # number of fake feature for each class
opt.Knn = 20  # knn: the value of K
max_accuracy = -1

if opt.manualSeed is None:
    # opt.manualSeed = random.randint(1, 10000)
    opt.manualSeed = 42
print("Random Seed: ", opt.manualSeed)

random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
torch.cuda.manual_seed_all(opt.manualSeed)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

main_dir = opt.main_dir

class ListModule(nn.Module):
    def __init__(self, *args):
        super(ListModule, self).__init__()
        idx = 0
        for module in args:
            self.add_module(str(idx), module)
            idx += 1

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self._modules):
            raise IndexError('index {} is out of range'.format(idx))
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __iter__(self):
        return iter(self._modules.values())

    def __len__(self):
        return len(self._modules)


class Scale(nn.Module):
    def __init__(self, num_scales):
        super(Scale, self).__init__()
        self.layers = []
        self.layers.append(nn.Linear(1, num_scales, bias=False))
        self.layer_module = ListModule(*self.layers)

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer(out)
        return out


def train(is_val=True):
    param = _param()
    if opt.dataset == 'CUB':
        dataset = LoadDataset(opt, main_dir, is_val)
        exp_info = 'CUB_EASY' if opt.splitmode == 'easy' else 'CUB_HARD'
    elif opt.dataset == 'NAB':
        dataset = LoadDataset_NAB(opt, main_dir, is_val)
        exp_info = 'NAB_EASY' if opt.splitmode == 'easy' else 'NAB_HARD'
    elif opt.dataset == 'SUN':
        dataset = LoadDataset_SUN(opt, main_dir, is_val)
        exp_info = 'SUN'
    else:
        print('No Dataset with that name')
        sys.exit(0)
    param.X_dim = dataset.feature_dim
    data_layer = FeatDataLayer(dataset.labels_train, dataset.pfc_feat_data_train, opt)
    result = Result()
    ones = Variable(torch.Tensor(1, 1))
    ones.data.fill_(1.0)

    netG = _netG(dataset.text_dim, dataset.feature_dim, rw_config=rw_config).cuda()
    netD = _netD(dataset.train_cls_num, dataset.feature_dim).cuda()

    netG.apply(weights_init)
    netD.apply(weights_init)

    exp_params = f'RWZSL_{rw_config.compute_hash()}'

    out_subdir = main_dir + 'out/{:s}/{:s}'.format(exp_info, exp_params)
    if not os.path.exists(out_subdir):
        os.makedirs(out_subdir)

    log_dir = out_subdir + '/log_{:s}.txt'.format(exp_info)
    with open(log_dir, 'a') as f:
        f.write('Training Start:')
        f.write(strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime()) + '\n')

    start_step = 0

    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            netG.load_state_dict(checkpoint['state_dict_G'])
            netD.load_state_dict(checkpoint['state_dict_D'])
            start_step = checkpoint['it']
            print(checkpoint['log'])
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))

    nets = [netG, netD]
    tr_cls_centroid = Variable(torch.from_numpy(dataset.tr_cls_centroid.astype('float32'))).cuda()

    optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(0.5, 0.9))
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(0.5, 0.9))

    # Initializing Random Walk loss config
    for it in range(start_step, opt.num_iterations + 1):
        # Creative Loss
        blobs = data_layer.forward()
        labels = blobs['labels'].astype(int)
        new_class_labels = Variable(torch.from_numpy(np.ones_like(labels) * dataset.train_cls_num)).cuda()
        text_feat_1 = np.array([dataset.train_text_feature[i, :] for i in labels])
        text_feat_2 = np.array([dataset.train_text_feature[i, :] for i in labels])
        np.random.shuffle(text_feat_1)  # Shuffle both features to guarantee different permutations
        np.random.shuffle(text_feat_2)
        alpha = (np.random.random(len(labels)) * (.8 - .2)) + .2

        text_feat_mean = np.multiply(alpha, text_feat_1.transpose())
        text_feat_mean += np.multiply(1. - alpha, text_feat_2.transpose())
        text_feat_mean = text_feat_mean.transpose()
        text_feat_mean = normalize(text_feat_mean, norm='l2', axis=1)
        text_feat_Creative = Variable(torch.from_numpy(text_feat_mean.astype('float32'))).cuda()
        z_creative = Variable(torch.randn(opt.batchsize, param.z_dim)).cuda()
        G_creative_sample = netG(z_creative, text_feat_Creative)

        """ Discriminator """
        for _ in range(5):
            blobs = data_layer.forward()
            feat_data = blobs['data']  # image data
            labels = blobs['labels'].astype(int)  # class labels

            text_feat = np.array([dataset.train_text_feature[i, :] for i in labels])
            text_feat = Variable(torch.from_numpy(text_feat.astype('float32'))).cuda()
            X = Variable(torch.from_numpy(feat_data)).cuda()
            y_true = Variable(torch.from_numpy(labels.astype('int'))).cuda()
            z = Variable(torch.randn(opt.batchsize, param.z_dim)).cuda()

            # GAN's D loss
            D_real, C_real = netD(X)
            D_loss_real = torch.mean(D_real)
            C_loss_real = F.cross_entropy(C_real, y_true)
            DC_loss = -D_loss_real + C_loss_real
            DC_loss.backward()

            # GAN's D loss
            G_sample = netG(z, text_feat).detach()
            D_fake, C_fake = netD(G_sample)
            D_loss_fake = torch.mean(D_fake)
            C_loss_fake = F.cross_entropy(C_fake, y_true)

            DC_loss = D_loss_fake + C_loss_fake
            DC_loss.backward()

            # train with gradient penalty (WGAN_GP)
            grad_penalty = compute_gradient_penalty(netD, X.data, G_sample.data, opt)
            grad_penalty.backward()

            # Imitative fake RW loss for Discriminator
            discr_rw_imitative_walker_loss, discr_rw_imitative_visit_loss = compute_rw_imitative_loss(rw_config, data_layer, dataset, netD, netG)
            discr_rw_imitative_loss = discr_rw_imitative_walker_loss + rw_config.loss_weights.get('visit_loss', 1.0) * discr_rw_imitative_visit_loss
            discr_rw_imitative_loss = rw_config.loss_weights.discr.imitative * discr_rw_imitative_loss
            discr_rw_imitative_loss.backward()

            # Imitative real RW loss for Discriminator
            discr_rw_real_walker_loss, discr_rw_real_visit_loss = compute_rw_real_loss(rw_config, data_layer, dataset, netD, netG)
            discr_rw_real_loss = discr_rw_real_walker_loss + rw_config.loss_weights.get('visit_loss', 1.0) * discr_rw_real_visit_loss
            discr_rw_real_loss = rw_config.loss_weights.discr.real * discr_rw_real_loss
            discr_rw_real_loss.backward()

            Wasserstein_D = D_loss_real - D_loss_fake
            optimizerD.step()
            reset_grad(nets)

        """ Generator """
        for _ in range(1):
            blobs = data_layer.forward()
            feat_data = blobs['data']  # image data
            labels = blobs['labels'].astype(int)  # class labels
            text_feat = np.array([dataset.train_text_feature[i, :] for i in labels])
            text_feat = Variable(torch.from_numpy(text_feat.astype('float32'))).cuda()

            X = Variable(torch.from_numpy(feat_data)).cuda()
            y_true = Variable(torch.from_numpy(labels.astype('int'))).cuda()
            z = Variable(torch.randn(opt.batchsize, param.z_dim)).cuda()

            G_sample = netG(z, text_feat)
            D_fake, C_fake = netD(G_sample)
            _, C_real = netD(X)

            # GAN's G loss
            G_loss = torch.mean(D_fake)
            # Auxiliary classification loss
            C_loss = (F.cross_entropy(C_real, y_true) + F.cross_entropy(C_fake, y_true)) / 2
            GC_loss = -G_loss + C_loss

            # Centroid loss
            centroid_loss = compute_centroid_loss(opt, dataset, G_sample, y_true, tr_cls_centroid)

            # ||W||_2 regularization
            reg_loss = Variable(torch.Tensor([0.0])).cuda()
            if opt.REG_W_LAMBDA != 0:
                for name, p in netG.named_parameters():
                    if 'weight' in name:
                        reg_loss += p.pow(2).sum()
                reg_loss.mul_(opt.REG_W_LAMBDA)

            # ||W_z||21 regularization, make W_z sparse
            reg_Wz_loss = Variable(torch.Tensor([0.0])).cuda()
            if opt.REG_Wz_LAMBDA != 0:
                Wz = netG.rdc_text.weight
                reg_Wz_loss = Wz.pow(2).sum(dim=0).sqrt().sum().mul(opt.REG_Wz_LAMBDA)

            # D(C| GX_fake)) + Classify GX_fake as real
            D_creative_fake, C_creative_fake = netD(G_creative_sample)
            G_fake_C = F.softmax(C_creative_fake, dim=1)

            # Imitative RW loss for Generator
            gen_rw_imitative_walker_loss, gen_rw_imitative_visit_loss = compute_rw_imitative_loss(rw_config, data_layer, dataset, netD, netG)
            gen_rw_imitative_loss = gen_rw_imitative_walker_loss + rw_config.loss_weights.get('visit_loss', 1.0) * gen_rw_imitative_visit_loss
            gen_rw_imitative_loss = rw_config.loss_weights.gen.imitative * gen_rw_imitative_loss

            # Creative RW loss for Generator
            gen_rw_creative_walker_loss, gen_rw_creative_visit_loss = compute_rw_creative_loss(rw_config, data_layer, dataset, netD, G_creative_sample, netG)
            gen_rw_creative_loss = gen_rw_creative_walker_loss + rw_config.loss_weights.get('visit_loss', 1.0) * gen_rw_creative_visit_loss
            gen_rw_creative_loss = rw_config.loss_weights.gen.creative * gen_rw_creative_loss

            total_gen_loss = GC_loss \
                + centroid_loss \
                + reg_loss \
                + reg_Wz_loss \
                + gen_rw_imitative_loss \
                + gen_rw_creative_loss

            reset_grad(nets)
            total_gen_loss.backward()
            if rw_config.get('grad_clip_value', -1) > 0:
                clip_grad_norm_(netG.parameters(), rw_config.grad_clip_value)
            optimizerG.step()
            reset_grad(nets)

        if it % opt.disp_interval == 0 and it:
            acc_real = (np.argmax(C_real.data.cpu().numpy(), axis=1) == y_true.data.cpu().numpy()).sum() / float(y_true.data.size()[0])
            acc_fake = (np.argmax(C_fake.data.cpu().numpy(), axis=1) == y_true.data.cpu().numpy()).sum() / float(y_true.data.size()[0])

            log_text = \
                f'Iter-{it};' \
                f'  Was_D: {Wasserstein_D.data.item():.4};' \
                f'  Euc_ls: {centroid_loss.data.item():.4};' \
                f'  reg_ls: {reg_loss.data.item():.4};' \
                f'  Wz_ls: {reg_Wz_loss.data.item():.4};' \
                f'  G_loss: {G_loss.data.item():.4};' \
                f'  D_loss_real: {D_loss_real.data.item():.4};' \
                f'  D_loss_fake: {D_loss_fake.data.item():.4};' \
                f'  G_rw_imitative_walker_loss: {gen_rw_imitative_walker_loss:.4};' \
                f'  G_rw_imitative_visit_loss: {gen_rw_imitative_visit_loss:.4};' \
                f'  G_rw_creative_walker_loss: {gen_rw_creative_walker_loss:.4};' \
                f'  G_rw_creative_visit_loss: {gen_rw_creative_visit_loss:.4};' \
                f'  D_rw_imitative_walker_loss: {discr_rw_imitative_walker_loss:.4};' \
                f'  D_rw_imitative_visit_loss: {discr_rw_imitative_visit_loss:.4};' \
                f'  D_rw_real_walker_loss: {discr_rw_real_walker_loss:.4};' \
                f'  D_rw_real_visit_loss: {discr_rw_real_visit_loss:.4};' \
                f'  rl: {acc_real * 100:.4}%;' \
                f'  fk: {acc_fake * 100:.4}%'
            print(log_text)
            with open(log_dir, 'a') as f:
                f.write(log_text + '\n')
        else:
            log_text = ''

        if it % opt.evl_interval == 0 and it >= 100:
            netG.eval()

            start_time = time()
            cur_acc = eval_fakefeat_test(it, netG, dataset, param, result)
            # print('`eval_fakefeat_test` call took time:', time() - start_time)
            cur_auc = eval_fakefeat_GZSL(netG, dataset, param, out_subdir, result)
            print('Evaluation took time:', time() - start_time)
            # print("{}: Accuracy is {:.4}%, and Generalized AUC is {:.4}%".format(it, cur_acc, cur_auc))

            if cur_acc > result.best_acc:
                result.best_acc = cur_acc

            if cur_auc > result.best_auc:
                result.best_auc = cur_auc

                files2remove = glob.glob(out_subdir + '/Best_model*')
                for _i in files2remove:
                    os.remove(_i)
                torch.save({
                    'it': it + 1,
                    'state_dict_G': netG.state_dict(),
                    'state_dict_D': netD.state_dict(),
                    'random_seed': opt.manualSeed,
                    'log': log_text,
                }, out_subdir + '/Best_model_AUC_{:.2f}.tar'.format(cur_auc))
            netG.train()
    return result


def eval_fakefeat_GZSL(netG, dataset, param, plot_dir, result):
    gen_feat = np.zeros([0, param.X_dim])

    for i in range(dataset.train_cls_num):
        text_feat = np.tile(dataset.train_text_feature[i].astype('float32'), (opt.nSample, 1))
        text_feat = Variable(torch.from_numpy(text_feat)).cuda()
        z = Variable(torch.randn(opt.nSample, param.z_dim)).cuda()
        G_sample = netG(z, text_feat)
        gen_feat = np.vstack((gen_feat, G_sample.data.cpu().numpy()))

    for i in range(dataset.test_cls_num):
        text_feat = np.tile(dataset.test_text_feature[i].astype('float32'), (opt.nSample, 1))
        text_feat = Variable(torch.from_numpy(text_feat)).cuda()
        z = Variable(torch.randn(opt.nSample, param.z_dim)).cuda()
        G_sample = netG(z, text_feat)
        gen_feat = np.vstack((gen_feat, G_sample.data.cpu().numpy()))

    visual_pivots = [gen_feat[i * opt.nSample:(i + 1) * opt.nSample].mean(0) for i in range(dataset.train_cls_num + dataset.test_cls_num)]
    visual_pivots = np.vstack(visual_pivots)
    """collect points for gzsl curve"""

    acc_S_T_list, acc_U_T_list = list(), list()
    seen_sim = cosine_similarity(dataset.pfc_feat_data_train, visual_pivots)
    unseen_sim = cosine_similarity(dataset.pfc_feat_data_test, visual_pivots)

    # plt.plot(acc_S_T_list, acc_U_T_list)
    # plt.title("{:s}-{:s}: {:.4}%".format(opt.dataset, opt.splitmode, auc_score))
    # plt.savefig(plot_dir + '/best_plot.png')
    # plt.clf()
    # plt.close()
    # np.savetxt(plot_dir + '/best_plot.txt', np.vstack([acc_S_T_list, acc_U_T_list]))

    logits = np.vstack([seen_sim, unseen_sim])
    seen_classes_mask = np.hstack([np.ones(dataset.train_cls_num), np.zeros(dataset.test_cls_num)]).astype(bool)
    targets = np.hstack([
        np.asarray(dataset.labels_train),
        np.asarray(dataset.labels_test) + dataset.train_cls_num
    ])
    auc_score = compute_ausuc(logits, targets, seen_classes_mask)

    result.auc_list += [auc_score]

    return auc_score


def eval_fakefeat_test(it, netG, dataset, param, result):
    gen_feat = np.zeros([0, param.X_dim])
    for i in range(dataset.test_cls_num):
        text_feat = np.tile(dataset.test_text_feature[i].astype('float32'), (opt.nSample, 1))
        text_feat = Variable(torch.from_numpy(text_feat)).cuda()
        z = Variable(torch.randn(opt.nSample, param.z_dim)).cuda()
        G_sample = netG(z, text_feat)
        gen_feat = np.vstack((gen_feat, G_sample.data.cpu().numpy()))

    # cosince predict K-nearest Neighbor
    sim = cosine_similarity(dataset.pfc_feat_data_test, gen_feat)
    idx_mat = np.argsort(-1 * sim, axis=1)
    label_mat = (idx_mat[:, 0:opt.Knn] / opt.nSample).astype(int)
    preds = np.zeros(label_mat.shape[0])
    for i in range(label_mat.shape[0]):
        (values, counts) = np.unique(label_mat[i], return_counts=True)
        preds[i] = values[np.argmax(counts)]

    # produce acc
    label_T = np.asarray(dataset.labels_test)
    acc = (preds == label_T).mean() * 100

    result.acc_list += [acc]
    return acc


class Result(object):
    def __init__(self):
        self.best_acc = 0.0
        self.best_auc = 0.0
        self.best_iter = 0.0
        self.acc_list = []
        self.auc_list = []


def weights_init(m):
    classname = m.__class__.__name__
    if 'Linear' in classname:
        init.xavier_normal_(m.weight.data)
        init.constant_(m.bias, 0.0)


def reset_grad(nets):
    for net in nets:
        net.zero_grad()


def label2mat(labels, y_dim):
    c = np.zeros([labels.shape[0], y_dim])
    for idx, d in enumerate(labels):
        c[idx, d] = 1
    return c


if __name__ == "__main__":
    # Inference
    """
    Values of Cross-validation CIZSL loss weight (Sharma-Entropy)
    CUB-EASY: 0.0001
    CUB-HARD: 0.1
    NAB-EASY: 1
    NAB-HARD: 0.1
    """
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    torch.cuda.manual_seed_all(opt.manualSeed)
    result = train(opt.is_val)
    print('=' * 15)
    print('=' * 15)
    print(opt.exp_name, opt.dataset, opt.splitmode)
    print("Accuracy is {:.4}%, and Generalized AUC is {:.4}%".format(result.best_acc, result.best_auc))
