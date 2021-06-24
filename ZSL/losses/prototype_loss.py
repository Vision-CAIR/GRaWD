import torch
import torch.nn.functional as F


def compute_prototype_loss(samples, classes):
    assert classes.dim() == 1
    assert len(samples) == len(classes)

    # Grouping samples by classes
    groups = [samples[classes == c] for c in set(classes.tolist())]
    prototypes = torch.stack([g.mean(dim=0) for g in groups])
    losses = [compute_prototype_loss_for_group(g, prototypes, i) for i, (g, p) in enumerate(zip(groups, prototypes))]
    loss = torch.cat(losses).mean()

    return loss


def compute_prototype_loss_for_group(group, prototypes, group_idx: int):
    """
    Given features (of the same group) and prototypes (of all groups) computes
    prototype loss

    Arguments:
        - group: tensor of size [N x D]
        - prototypes: tensor of size [P x D]
        - group_idx:int — index of the given group (so we know a proper prototype for it)
    """
    N = len(group)
    P = len(prototypes)

    if opt.prototype_loss == 'cross_entropy':
        logits = torch.mm(group, prototypes.transpose(0, 1)) # tensor of size [N x P]
        targets = torch.full((N,), group_idx, device=logits.device).long() # tensor of size [N]
        losses = F.cross_entropy(logits, targets, reduction='none') # tensor of size [N]

        assert logits.shape == (N, P)
        assert losses.shape == (N,)

        return losses
    elif opt.prototype_loss == 'triplet_loss':
        # TODO: should we just take the closest prototype as negative example?
        distances = (group.unsqueeze(1) - prototypes.unsqueeze(0)).pow(2).sum(dim=2) # tensor of size [N x P]
        distances_total = distances.sum(dim=0) # tensor of size [P]

        closest_neg_idx = distances_total[torch.arange(P) != group_idx].argmin()
        # Correcting `closest_neg_idx` since it was computed for a vector of length [P - 1]
        closest_neg_idx = (closest_neg_idx + 1) if closest_neg_idx >= group_idx else closest_neg_idx

        losses = distances[group_idx] - distances[closest_neg_idx] + opt.prototype_triplet_loss_margin
        losses = losses.clamp(0)

        assert distances.shape == (N, P)
        assert distances_total.shape == (P)
        assert losses.shape == (N,)

        return losses
    elif opt.prototype_loss == 'l2_distance':
        distances = (group.unsqueeze(1) - prototypes.unsqueeze(0)).pow(2).sum(dim=2) # tensor of size [N x P]
        losses = distances[:, group_idx] - distances[torch.arange(P) != group_idx].mean(dim=1) # tensor of size [N]

        assert distances.shape == (N, P)
        assert losses.shape == (N,)

        return losses
    else:
        raise NotImplementedError(f'Unknown prototype loss {opt.prototype_loss}')
