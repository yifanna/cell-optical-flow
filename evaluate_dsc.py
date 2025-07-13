import sys

sys.path.append('core')

from PIL import Image
import argparse
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

import datasets
from core.datasets import FlyingChairs
from core.datasets import KITTI
from core.utils import flow_viz
from core.utils import frame_utils
from core.datasets import MpiSintel
from core.datasets import CELL_dataset

from core.CELL2 import CELL
from core.utils.utils import InputPadder, forward_interpolate



@torch.no_grad()
def create_sintel_submission(model, iters=32, warm_start=False, output_path='sintel_submission'):
    """ Create submission for the Sintel leaderboard """
    model.eval()
    for dstype in ['clean', 'final']:
        test_dataset = MpiSintel(split='test', aug_params=None, dstype=dstype)

        flow_prev, sequence_prev = None, None
        for test_id in range(len(test_dataset)):
            image1, image2, (sequence, frame) = test_dataset[test_id]
            if sequence != sequence_prev:
                flow_prev = None

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1[None].cuda(), image2[None].cuda())

            flow_low, flow_pr = model(image1, image2, iters=iters, flow_init=flow_prev, test_mode=True)
            flow = padder.unpad(flow_pr[0]).permute(1, 2, 0).cpu().numpy()

            if warm_start:
                flow_prev = forward_interpolate(flow_low[0])[None].cuda()

            output_dir = os.path.join(output_path, dstype, sequence)
            output_file = os.path.join(output_dir, 'frame%04d.flo' % (frame + 1))

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            frame_utils.writeFlow(output_file, flow)
            sequence_prev = sequence


@torch.no_grad()
def create_kitti_submission(model, iters=24, output_path='kitti_submission'):
    """ Create submission for the Sintel leaderboard """
    model.eval()
    test_dataset = datasets.KITTI(split='testing', aug_params=None)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for test_id in range(len(test_dataset)):
        image1, image2, (frame_id,) = test_dataset[test_id]
        padder = InputPadder(image1.shape, mode='kitti')
        image1, image2 = padder.pad(image1[None].cuda(), image2[None].cuda())

        _, flow_pr = model(image1, image2, iters=iters, test_mode=True)
        flow = padder.unpad(flow_pr[0]).permute(1, 2, 0).cpu().numpy()

        output_filename = os.path.join(output_path, frame_id)
        frame_utils.writeFlowKITTI(output_filename, flow)


@torch.no_grad()
def validate_chairs(model, iters=24):
    """ Perform evaluation on the FlyingChairs (test) split """
    model.eval()
    epe_list = []

    val_dataset = FlyingChairs(split='validation')
    for val_id in range(len(val_dataset)):
        image1, image2, flow_gt, _ = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        _, flow_pr = model(image1, image2, iters=iters, test_mode=True)
        epe = torch.sum((flow_pr[0].cpu() - flow_gt) ** 2, dim=0).sqrt()
        epe_list.append(epe.view(-1).numpy())

    epe = np.mean(np.concatenate(epe_list))
    print("Validation Chairs EPE: %f" % epe)
    return {'chairs': epe}


@torch.no_grad()
def validate_sintel(model, iters=32):
    """ Peform validation using the Sintel (train) split """
    model.eval()
    results = {}
    # 添加保存光流场的路径
    check_dir = 'check'
    flow_dir = os.path.join(check_dir, 'flow')
    if not os.path.exists(flow_dir):
        os.makedirs(flow_dir)

    for dstype in ['clean', 'final']:
        val_dataset = MpiSintel(split='test', dstype=dstype)
        epe_list = []

        for val_id in range(len(val_dataset)):
            # image1, image2, flow_gt, _ = val_dataset[val_id]
            sample = val_dataset[val_id]
            image1, image2, flow_gt, _ = sample
            # 没有真实光流值
            # image1, image2 = sample['image1'], sample['image2']

            image1 = image1[None].cuda()
            image2 = image2[None].cuda()

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            flow_low, flow_pr = model(image1, image2, iters=iters, test_mode=True)
            flow = padder.unpad(flow_pr[0]).cpu()

            epe = torch.sum((flow - flow_gt) ** 2, dim=0).sqrt()  # 没有真实光流值注释
            epe_list.append(epe.view(-1).numpy())  #

            # Save flow predictions to .flo file
            output_dir = os.path.join(flow_dir, dstype)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            output_file = os.path.join(output_dir, f'frame{val_id:04d}.flo')
            frame_utils.writeFlow(output_file, flow.permute(1, 2, 0).numpy())
        del sample, image1, image2, flow_low, flow_pr, flow
        torch.cuda.empty_cache()

        epe_all = np.concatenate(epe_list)
        epe = np.mean(epe_all)
        px1 = np.mean(epe_all < 1)
        px3 = np.mean(epe_all < 3)
        px5 = np.mean(epe_all < 5)

        print("Validation (%s) EPE: %f, 1px: %f, 3px: %f, 5px: %f" % (dstype, epe, px1, px3, px5))
        results[dstype] = np.mean(epe_list)

    return results


@torch.no_grad()
def validate_kitti(model, iters=24):
    """ Peform validation using the KITTI-2015 (train) split """
    model.eval()
    val_dataset = KITTI(split='testing')

    out_list, epe_list = [], []
    for val_id in range(len(val_dataset)):
        image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        padder = InputPadder(image1.shape, mode='kitti')
        image1, image2 = padder.pad(image1, image2)

        flow_low, flow_pr = model(image1, image2, iters=iters, test_mode=True)
        flow = padder.unpad(flow_pr[0]).cpu()

        epe = torch.sum((flow - flow_gt) ** 2, dim=0).sqrt()
        mag = torch.sum(flow_gt ** 2, dim=0).sqrt()

        epe = epe.view(-1)
        mag = mag.view(-1)
        val = valid_gt.view(-1) >= 0.5

        out = ((epe > 3.0) & ((epe / mag) > 0.05)).float()
        epe_list.append(epe[val].mean().item())
        out_list.append(out[val].cpu().numpy())

    epe_list = np.array(epe_list)
    out_list = np.concatenate(out_list)

    epe = np.mean(epe_list)
    f1 = 100 * np.mean(out_list)

    print("Validation KITTI: %f, %f" % (epe, f1))
    return {'kitti-epe': epe, 'kitti-f1': f1}

def compute_dsc(Ms, Mt, Fs_to_t):
    # Ensure all tensors are on the same device (cuda)
    Ms = Ms.cuda()
    Mt = Mt.cuda()
    Fs_to_t = Fs_to_t.cuda()

    intersection = torch.sum(Ms * Mt)
    union = torch.sum(Ms) + torch.sum(Mt)
    dsc = (2.0 * intersection) / union
    return dsc

def warp_mask(mask, flow):
    h, w = mask.size()

    # Create a normalized grid
    grid_y, grid_x = torch.meshgrid(torch.arange(h), torch.arange(w))
    grid_x = grid_x.float().cuda()
    grid_y = grid_y.float().cuda()  # Explicitly move grid_y to CUDA

    # Apply the flow field to the grid
    flow_x, flow_y = flow[0, 0].cuda(), flow[0, 1].cuda()

    # Ensure that flow_x and flow_y have the same dimensions
    flow_x = F.interpolate(flow_x.unsqueeze(0).unsqueeze(0), (h, w), mode='bicubic', align_corners=True).squeeze(0).squeeze(0)
    flow_y = F.interpolate(flow_y.unsqueeze(0).unsqueeze(0), (h, w), mode='bicubic', align_corners=True).squeeze(0).squeeze(0)

    # In-place operations to reduce memory usage
    grid_x = grid_x + flow_x
    grid_y = grid_y + flow_y

    # Clamp the coordinates to a reasonable range
    grid_x.clamp_(0, w - 1)
    grid_y.clamp_(0, h - 1)

    # Merge the coordinates into a grid
    grid = torch.stack((grid_x, grid_y), dim=-1)

    # Use grid_sample to interpolate the mask
    warped_mask = F.grid_sample(mask.unsqueeze(0).unsqueeze(0).cuda(), grid.unsqueeze(0), align_corners=True)

    # Adjust the shape as needed
    warped_mask = warped_mask.squeeze(0).squeeze(0)

    return warped_mask

@torch.no_grad()
def validate_cell(model, iters=32, batch_size=1, dsc_threshold=0.75):
    model.eval()
    results = {}

    overall_epe_list = []
    overall_px1_list = []
    overall_px3_list = []
    overall_px5_list = []
    overall_dsc_list = []

    for dstype in ['clean', 'final']:
        val_dataset = CELL_dataset(split='training', dstype=dstype)
        epe_list = []
        px1_list = []
        px3_list = []
        px5_list = []
        dsc_list = []

        for start in range(0, len(val_dataset), batch_size):
            end = min(start + batch_size, len(val_dataset))

            batch_epe_list = []
            batch_px1_list = []
            batch_px3_list = []
            batch_px5_list = []
            batch_dsc_list = []

            for val_id in range(start, end):
                sample = val_dataset[val_id]

                image1, image2, flow_gt, mask_gt = sample

                image1 = image1[None].cuda()
                image2 = image2[None].cuda()

                padder = InputPadder(image1.shape)
                image1, image2 = padder.pad(image1, image2)

                flow_low, flow_pr = model(image1, image2, iters=iters, test_mode=True)
                flow = padder.unpad(flow_pr[0]).cpu()
                flow_gt = flow_gt[:, :flow.shape[1], :flow.shape[2]]

                predicted_mask = warp_mask(mask_gt, forward_interpolate(flow_low[0])[None].cuda())

                epe = torch.sum((flow - flow_gt) ** 2, dim=0).sqrt()
                batch_epe_list.append(epe.view(-1).numpy())

                dsc = compute_dsc(mask_gt, predicted_mask, forward_interpolate(flow_low[0])[None].cuda())
                batch_dsc_list.append(dsc.item())

                # Memory cleanup
                del sample, image1, image2, flow_low, flow_pr, flow
                torch.cuda.empty_cache()

                batch_epe_all = np.concatenate(batch_epe_list)
                batch_px1_list.append(np.mean(batch_epe_all < 1))
                batch_px3_list.append(np.mean(batch_epe_all < 3))
                batch_px5_list.append(np.mean(batch_epe_all < 5))
                batch_dsc_list.append(np.mean(batch_dsc_list))

            results[dstype] = {
                'EPE': np.mean(batch_epe_list),
                '1px': np.mean(batch_px1_list),
                '3px': np.mean(batch_px3_list),
                '5px': np.mean(batch_px5_list),
                'DSC': np.mean(batch_dsc_list)
            }
            epe_list.extend(batch_epe_list)
            px1_list.extend(batch_px1_list)
            px3_list.extend(batch_px3_list)
            px5_list.extend(batch_px5_list)
            dsc_list.extend(batch_dsc_list)

        overall_epe_list.extend(epe_list)
        overall_px1_list.extend(px1_list)
        overall_px3_list.extend(px3_list)
        overall_px5_list.extend(px5_list)
        overall_dsc_list.extend(dsc_list)

        overall_results = {
            'EPE': np.mean(overall_epe_list),
            '1px': np.mean(overall_px1_list),
            '3px': np.mean(overall_px3_list),
            '5px': np.mean(overall_px5_list),
            'DSC': np.mean(overall_dsc_list)
        }

        print(f"Overall Results ({dstype}): "
              f"EPE: {overall_results['EPE']}, "
              f"1px: {overall_results['1px']}, "
              f"3px: {overall_results['3px']}, "
              f"5px: {overall_results['5px']}, "
              f"DSC: {overall_results['DSC']}")

    return overall_results

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--dataset', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()

    model = torch.nn.DataParallel(CELL(args))
    # model.load_state_dict(torch.load(args.model))
    model.load_state_dict(torch.load(args.model, map_location='cpu'))

    model.cuda()
    model.eval()

    # create_sintel_submission(model.module, warm_start=True)
    # create_kitti_submission(model.module)

    with torch.no_grad():
        if args.dataset == 'chairs':
            validate_chairs(model.module)

        elif args.dataset == 'sintel':
            validate_sintel(model.module)

        elif args.dataset == 'kitti':
            validate_kitti(model.module)

        elif args.dataset == 'CELL_dataset':
            validate_cell(model.module)




