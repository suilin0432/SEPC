import numpy as np
import torch

from .random_sampler import RandomSampler


# Libra R-CNN 中提出的方法
class IoUBalancedNegSampler(RandomSampler):
    """IoU Balanced Sampling

    arXiv: https://arxiv.org/pdf/1904.02701.pdf (CVPR 2019)
    根据 IoU 去进行 proposals 的 sample.
    Sampling proposals according to their IoU. `floor_fraction` of needed RoIs
    are sampled from proposals whose IoU are lower than `floor_thr` randomly.
    The others are sampled from proposals whose IoU are higher than
    `floor_thr`. These proposals are sampled from some bins evenly, which are
    split by `num_bins` via IoU evenly.

    Args:
        num (int): number of proposals.
        pos_fraction (float): fraction of positive proposals.
        floor_thr (float): threshold (minimum) IoU for IoU balanced sampling,
            set to -1 if all using IoU balanced sampling.
        floor_fraction (float): sampling fraction of proposals under floor_thr.
        num_bins (int): number of bins in IoU balanced sampling.
    """

    def __init__(self,
                 num,
                 pos_fraction,
                 floor_thr=-1,
                 floor_fraction=0,
                 num_bins=3,
                 **kwargs):
        """

        :param num: proposal的数量
        :param pos_fraction: 正负(positive/negative)样本的比例
        :param floor_thr: 进行 IoU balanced sampling的最低的 IoU 比例, 默认使用 -1, 即对所有都进行 IoU balanced sampling
        :param floor_fraction: ???
        :param num_bins: 划分为 num_bins 个区域进行 Libra R-CNN 中 IoU balanced sampling
        """
        super(IoUBalancedNegSampler, self).__init__(num, pos_fraction,
                                                    **kwargs)
        assert floor_thr >= 0 or floor_thr == -1
        assert 0 <= floor_fraction <= 1
        assert num_bins >= 1

        self.floor_thr = floor_thr
        self.floor_fraction = floor_fraction
        self.num_bins = num_bins

    def sample_via_interval(self, max_overlaps, full_set, num_expected):
        max_iou = max_overlaps.max()
        iou_interval = (max_iou - self.floor_thr) / self.num_bins
        per_num_expected = int(num_expected / self.num_bins)

        sampled_inds = []
        for i in range(self.num_bins):
            start_iou = self.floor_thr + i * iou_interval
            end_iou = self.floor_thr + (i + 1) * iou_interval
            tmp_set = set(
                np.where(
                    np.logical_and(max_overlaps >= start_iou,
                                   max_overlaps < end_iou))[0])
            tmp_inds = list(tmp_set & full_set)
            if len(tmp_inds) > per_num_expected:
                tmp_sampled_set = self.random_choice(tmp_inds,
                                                     per_num_expected)
            else:
                tmp_sampled_set = np.array(tmp_inds, dtype=np.int)
            sampled_inds.append(tmp_sampled_set)

        sampled_inds = np.concatenate(sampled_inds)
        if len(sampled_inds) < num_expected:
            num_extra = num_expected - len(sampled_inds)
            extra_inds = np.array(list(full_set - set(sampled_inds)))
            if len(extra_inds) > num_extra:
                extra_inds = self.random_choice(extra_inds, num_extra)
            sampled_inds = np.concatenate([sampled_inds, extra_inds])

        return sampled_inds

    # PS: 这里只重写了 _sample_neg, 并没有重写 _sample_pos, 应该是 positive 的 anchor 仍然都要(大部分情况是都要的)，negative 的
    #     anchor 才是按照 IoU-balanced sampling 方法来选择性的去要
    # 这么看来，实际上 IoU-balanced sampling 方法是保留了所有的positive的(相对来说通常都是高IoU的)proposal的情况下，
    # 尽可能的去多 sample 高 IoU 的 anchor. 因为高 IoU proposal(通常来说 > 0.7) 以及与其 sample 出的框有最高 IoU 的那个 proposal
    # 才被视为 positive anchor. 但是其实有很多框有相对较高的 IoU, 却被过多的 低 IoU proposal 所淹没, 所以在采样的时候比较低.
    def _sample_neg(self, assign_result, num_expected, **kwargs):
        neg_inds = torch.nonzero(assign_result.gt_inds == 0)
        if neg_inds.numel() != 0:
            neg_inds = neg_inds.squeeze(1)
        if len(neg_inds) <= num_expected:
            return neg_inds
        else:
            max_overlaps = assign_result.max_overlaps.cpu().numpy()
            # balance sampling for negative samples
            neg_set = set(neg_inds.cpu().numpy())

            if self.floor_thr > 0:
                floor_set = set(
                    np.where(
                        np.logical_and(max_overlaps >= 0,
                                       max_overlaps < self.floor_thr))[0])
                iou_sampling_set = set(
                    np.where(max_overlaps >= self.floor_thr)[0])
            elif self.floor_thr == 0:
                floor_set = set(np.where(max_overlaps == 0)[0])
                iou_sampling_set = set(
                    np.where(max_overlaps > self.floor_thr)[0])
            else:
                floor_set = set()
                iou_sampling_set = set(
                    np.where(max_overlaps > self.floor_thr)[0])
                # for sampling interval calculation
                self.floor_thr = 0

            floor_neg_inds = list(floor_set & neg_set)
            iou_sampling_neg_inds = list(iou_sampling_set & neg_set)
            num_expected_iou_sampling = int(num_expected *
                                            (1 - self.floor_fraction))
            if len(iou_sampling_neg_inds) > num_expected_iou_sampling:
                if self.num_bins >= 2:
                    iou_sampled_inds = self.sample_via_interval(
                        max_overlaps, set(iou_sampling_neg_inds),
                        num_expected_iou_sampling)
                else:
                    iou_sampled_inds = self.random_choice(
                        iou_sampling_neg_inds, num_expected_iou_sampling)
            else:
                iou_sampled_inds = np.array(
                    iou_sampling_neg_inds, dtype=np.int)
            num_expected_floor = num_expected - len(iou_sampled_inds)
            if len(floor_neg_inds) > num_expected_floor:
                sampled_floor_inds = self.random_choice(
                    floor_neg_inds, num_expected_floor)
            else:
                sampled_floor_inds = np.array(floor_neg_inds, dtype=np.int)
            sampled_inds = np.concatenate(
                (sampled_floor_inds, iou_sampled_inds))
            if len(sampled_inds) < num_expected:
                num_extra = num_expected - len(sampled_inds)
                extra_inds = np.array(list(neg_set - set(sampled_inds)))
                if len(extra_inds) > num_extra:
                    extra_inds = self.random_choice(extra_inds, num_extra)
                sampled_inds = np.concatenate((sampled_inds, extra_inds))
            sampled_inds = torch.from_numpy(sampled_inds).long().to(
                assign_result.gt_inds.device)
            return sampled_inds
