"""
Dataset setup and loaders
"""
from datasets import cityscapes
from datasets import nullloader

from datasets import multi_loader
from datasets.sampler import DistributedSampler

import torchvision.transforms as standard_transforms

import transforms.joint_transforms as joint_transforms
import transforms.transforms as extended_transforms
from torch.utils.data import DataLoader, ConcatDataset
import torch


num_classes = 19
ignore_label = 255


def get_train_joint_transform(args, dataset):
    """
    Get train joint transform
    Args:
        args: input config arguments
        dataset: dataset class object

    return: train_joint_transform_list, train_joint_transform
    """
    # Geometric image transformations
    train_joint_transform_list = []
    train_joint_transform_list += [
        joint_transforms.RandomSizeAndCrop(args.crop_size,
                                           crop_nopad=args.crop_nopad,
                                           pre_size=args.pre_size,
                                           scale_min=args.scale_min,
                                           scale_max=args.scale_max,
                                           ignore_index=dataset.ignore_label),
        joint_transforms.Resize(args.crop_size),
        joint_transforms.RandomHorizontallyFlip()]

    if args.rrotate > 0:
        train_joint_transform_list += [joint_transforms.RandomRotate(
            degree=args.rrotate,
            ignore_index=dataset.ignore_label)]

    train_joint_transform = joint_transforms.Compose(train_joint_transform_list)

    # return the raw list for class uniform sampling
    return train_joint_transform_list, train_joint_transform


def get_input_transforms(args, dataset):
    """
    Get input transforms
    Args:
        args: input config arguments
        dataset: dataset class object

    return: train_input_transform, val_input_transform
    """
    # Image appearance transformations
    train_input_transform = []
    val_input_transform = []
    if args.color_aug > 0.0:
        train_input_transform += [standard_transforms.RandomApply([
            standard_transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.5)]

    if args.bblur:
        train_input_transform += [extended_transforms.RandomBilateralBlur()]
    elif args.gblur:
        train_input_transform += [extended_transforms.RandomGaussianBlur()]

    train_input_transform += [
                              standard_transforms.ToTensor()
    ]
    val_input_transform += [
                            standard_transforms.ToTensor()
    ]
    train_input_transform = standard_transforms.Compose(train_input_transform)
    val_input_transform = standard_transforms.Compose(val_input_transform)

    return train_input_transform, val_input_transform


def get_target_transforms(args, dataset):
    """
    Get target transforms
    Args:
        args: input config arguments
        dataset: dataset class object

    return: target_transform, target_train_transform, target_aux_train_transform
    """
    target_transform = extended_transforms.MaskToTensor()
    if args.jointwtborder:
        target_train_transform = extended_transforms.RelaxedBoundaryLossToTensor(
                dataset.ignore_label, dataset.num_classes)
    else:
        target_train_transform = extended_transforms.MaskToTensor()

    target_aux_train_transform = extended_transforms.MaskToTensor()

    return target_transform, target_train_transform, target_aux_train_transform


def create_extra_val_loader(args, dataset, val_input_transform, target_transform, val_sampler):
    """
    Create extra validation loader
    Args:
        args: input config arguments
        dataset: dataset class object
        val_input_transform: validation input transforms
        target_transform: target transforms
        val_sampler: validation sampler

    return: validation loaders
    """
    if dataset == 'cityscapes':
        val_set = cityscapes.CityScapes('fine', 'val', 0,
                                        transform=val_input_transform,
                                        target_transform=target_transform,
                                        cv_split=args.cv)
    elif dataset == 'null_loader':
        val_set = nullloader.nullloader(args.crop_size)
    else:
        raise Exception('Dataset {} is not supported'.format(dataset))

    if args.syncbn:
        from datasets.sampler import DistributedSampler
        val_sampler = DistributedSampler(val_set, pad=False, permutation=False, consecutive_sample=False)

    else:
        val_sampler = None

    val_loader = DataLoader(val_set, batch_size=args.val_batch_size,
                            num_workers=args.num_workers // 2 , shuffle=False, drop_last=False,
                            sampler = val_sampler)
    return val_loader


def setup_loaders(args):
    """
    Setup Data Loaders[Currently supports Cityscapes]
    input: argument passed by the user
    return:  training data loader, validation data loader loader,  train_set
    """

    args.train_batch_size = args.bs_mult * args.ngpu
    if args.bs_mult_val > 0:
        args.val_batch_size = args.bs_mult_val * args.ngpu
    else:
        args.val_batch_size = args.bs_mult * args.ngpu

    # Readjust batch size to mini-batch size for syncbn
    if args.syncbn:
        args.train_batch_size = args.bs_mult
        args.val_batch_size = args.bs_mult_val


    args.num_workers = 8 #1 * args.ngpu
    if args.test_mode:
        args.num_workers = 1

    train_sets = []
    val_sets = []
    val_dataset_names = []

    if 'cityscapes' in args.dataset:
        dataset = cityscapes
        city_mode = args.city_mode #'train' ## Can be trainval
        city_quality = 'fine'
        train_joint_transform_list, train_joint_transform = get_train_joint_transform(args, dataset)
        train_input_transform, val_input_transform = get_input_transforms(args, dataset)
        target_transform, target_train_transform, target_aux_train_transform = get_target_transforms(args, dataset)

        if args.class_uniform_pct:
            if args.coarse_boost_classes:
                coarse_boost_classes = \
                    [int(c) for c in args.coarse_boost_classes.split(',')]
            else:
                coarse_boost_classes = None

            train_set = dataset.CityScapesUniform(
                city_quality, city_mode, args.maxSkip,
                joint_transform_list=train_joint_transform_list,
                transform=train_input_transform,
                target_transform=target_train_transform,
                target_aux_transform=target_aux_train_transform,
                dump_images=args.dump_augmentation_images,
                cv_split=args.cv,
                class_uniform_pct=args.class_uniform_pct,
                class_uniform_tile=args.class_uniform_tile,
                test=args.test_mode,
                coarse_boost_classes=coarse_boost_classes,
                image_mode=args.image_mode)
        else:
            train_set = dataset.CityScapes(
                city_quality, city_mode, 0,
                joint_transform=train_joint_transform,
                transform=train_input_transform,
                target_transform=target_train_transform,
                target_aux_transform=target_aux_train_transform,
                dump_images=args.dump_augmentation_images,
                image_mode=args.image_mode)

        val_set = dataset.CityScapes('fine', 'val', 0,
                                     transform=val_input_transform,
                                     target_transform=target_transform,
                                     cv_split=args.cv)
        train_sets.append(train_set)
        val_sets.append(val_set)
        val_dataset_names.append('cityscapes')

    if 'null_loader' in args.dataset:
        train_set = nullloader.nullloader(args.crop_size)
        val_set = nullloader.nullloader(args.crop_size)

        train_sets.append(train_set)
        val_sets.append(val_set)
        val_dataset_names.append('null_loader')

    if len(train_sets) == 0:
        raise Exception('Dataset {} is not supported'.format(args.dataset))

    if len(train_sets) != len(args.dataset):
        raise Exception('Something went wrong. Please check your dataset names are valid')

    # Define new train data set that has all the train sets
    # Define new val data set that has all the val sets
    val_loaders = {}
    if len(args.dataset) != 1:
        if args.image_uniform_sampling:
            train_set = ConcatDataset(train_sets)
        else:
            train_set = multi_loader.DomainUniformConcatDataset(args, train_sets)

    for i, val_set in enumerate(val_sets):
        if args.syncbn:
            val_sampler = DistributedSampler(val_set, pad=False, permutation=False, consecutive_sample=False)
        else:
            val_sampler = None
        val_loader = DataLoader(val_set, batch_size=args.val_batch_size,
                                num_workers=args.num_workers // 2 , shuffle=False, drop_last=False,
                                sampler = val_sampler)
        val_loaders[val_dataset_names[i]] = val_loader

    if args.syncbn:
        train_sampler = DistributedSampler(train_set, pad=True, permutation=True, consecutive_sample=False)
    else:
        train_sampler = None

    train_loader = DataLoader(train_set, batch_size=args.train_batch_size,
                              num_workers=args.num_workers, shuffle=(train_sampler is None), drop_last=True, sampler = train_sampler)

    extra_val_loader = {}
    for val_dataset in args.val_dataset:
        extra_val_loader[val_dataset] = create_extra_val_loader(args, val_dataset, val_input_transform, target_transform, val_sampler)

    return train_loader, val_loaders, train_set, extra_val_loader

