import torch
from torch import autograd


def compute_gradient_penalty(netD, real_data, fake_data, opt):
    alpha = torch.rand(opt.batchsize, 1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.cuda()

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    interpolates = interpolates.cuda()
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates, _ = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * opt.GP_LAMBDA

    return gradient_penalty


def compute_centroid_loss(opt, dataset, G_sample, y_true, tr_cls_centroid):
    """
    Computes centroid loss to make generators prediction to cluster around the data mean
    """
    loss = torch.Tensor([0.0]).cuda()

    if opt.REG_W_LAMBDA != 0:
        for i in range(dataset.train_cls_num):
            sample_idx = (y_true == i).data.nonzero().squeeze()

            if sample_idx.numel() == 0:
                loss += 0.0
            else:
                G_sample_cls = G_sample[sample_idx, :]
                loss += (G_sample_cls.mean(dim=0) - tr_cls_centroid[i]).pow(2).sum().sqrt()
        loss *= 1.0 / dataset.train_cls_num * opt.CENT_LAMBDA

    return loss
