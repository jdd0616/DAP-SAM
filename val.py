import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
from einops import repeat
import os
import cv2
import time
from thop import profile

def calculate_metric_percase(pred, gt):
    # pred[pred > 0] = 1
    # gt[gt > 0] = 1

    intersection = (pred * gt).sum()
    union = pred.sum() + gt.sum()
    dice = (2. * intersection + 1e-6) / (union + 1e-6)

    # dice = metric.binary.dc(pred, gt)
    precision = metric.binary.precision(pred, gt)

    union_iou = pred.sum() + gt.sum() - intersection  # 正确计算并集
    iou = (intersection + 1e-6) / (union_iou + 1e-6)  # 添加平滑项避免除零
    return dice, precision, iou


def calculate_binary_metric_safe(pred, gt):
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    intersection = np.logical_and(pred, gt).sum()
    pred_sum = pred.sum()
    gt_sum = gt.sum()
    union = np.logical_or(pred, gt).sum()
    dice = (2.0 * intersection + 1e-6) / (pred_sum + gt_sum + 1e-6)
    precision = (intersection + 1e-6) / (pred_sum + 1e-6)
    iou = (intersection + 1e-6) / (union + 1e-6)
    return float(dice), float(precision), float(iou)


def normalize_to_uint8(image):
    image = image.astype(np.float32)
    image = image - image.min()
    denom = image.max()
    if denom > 0:
        image = image / denom
    return (image * 255.0).clip(0, 255).astype(np.uint8)


def colorize_mask(mask):
    mask = mask.astype(np.uint8)
    color = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    color[mask == 1] = (0, 0, 255)
    color[mask == 2] = (0, 255, 0)
    color[mask == 3] = (255, 0, 0)
    return color


def add_panel_title(panel, title):
    titled = cv2.copyMakeBorder(panel, 28, 0, 0, 0, cv2.BORDER_CONSTANT, value=(25, 25, 25))
    cv2.putText(titled, title, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return titled


def save_comparison_image(path, image_slice, gt, pred, num_classes):
    if image_slice.ndim == 3:
        image_slice = image_slice[0]
    gray = normalize_to_uint8(image_slice)
    base = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    gt = gt.astype(np.uint8)
    pred = pred.astype(np.uint8)
    gt_color = colorize_mask(gt)
    pred_color = colorize_mask(pred)

    gt_panel = cv2.addWeighted(base, 0.72, gt_color, 0.28, 0)
    pred_panel = cv2.addWeighted(base, 0.72, pred_color, 0.28, 0)

    overlay = base.copy()
    gt_fg = gt > 0
    pred_fg = pred > 0
    overlay[np.logical_and(gt_fg, pred_fg)] = (0, 180, 0)
    overlay[np.logical_and(~gt_fg, pred_fg)] = (0, 0, 255)
    overlay[np.logical_and(gt_fg, ~pred_fg)] = (255, 0, 0)
    overlay = cv2.addWeighted(base, 0.50, overlay, 0.50, 0)

    panels = [
        add_panel_title(base, "Image"),
        add_panel_title(gt_panel, "GT"),
        add_panel_title(pred_panel, "Pred"),
        add_panel_title(overlay, "TP/FP/FN"),
    ]
    comparison = cv2.hconcat(panels)

    if num_classes == 1:
        dice, precision, iou = calculate_binary_metric_safe(pred == 1, gt == 1)
        metric_text = f"Lesion Dice: {dice:.4f}  Precision: {precision:.4f}  IoU: {iou:.4f}"
    else:
        class_metrics = [calculate_binary_metric_safe(pred == cls, gt == cls) for cls in range(1, num_classes + 1)]
        dice = float(np.mean([m[0] for m in class_metrics]))
        precision = float(np.mean([m[1] for m in class_metrics]))
        iou = float(np.mean([m[2] for m in class_metrics]))
        dice_text = "  ".join([f"C{idx + 1} Dice: {m[0]:.3f}" for idx, m in enumerate(class_metrics)])
        metric_text = f"FG mean Dice: {dice:.4f}  Precision: {precision:.4f}  IoU: {iou:.4f}    {dice_text}"

    comparison = cv2.copyMakeBorder(comparison, 42, 0, 0, 0, cv2.BORDER_CONSTANT, value=(10, 10, 10))
    cv2.putText(comparison, metric_text, (10, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(path, comparison)


def test_single_PROMPT(image, label, net, num_classes, PSEUDO_MASK, case_batch, save_path, device,
                       point_numbers, prompt_mode, multimask_output=True, patch_size=[256, 256]):
    image, label = image.squeeze(0).cpu().detach(
    ).numpy(), label.squeeze(0).cpu().detach().numpy()
    prediction = np.zeros_like(label)

    net.eval()
    total_time = 0.0
    total_flops = 0.0
    count = 0


    if PSEUDO_MASK ==None:
        save_dir = os.path.join(save_path, 'visualization_NOPROMPT')
        comparison_dir = os.path.join(save_path, 'comparison_NOPROMPT')
    else:
        save_dir = os.path.join(save_path, 'visualization')
        comparison_dir = os.path.join(save_path, 'comparison')

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(comparison_dir, exist_ok=True)
    per_sample_names = []  # 对应的样本名字（如果有传 case_batch）

    for ind in range(image.shape[0]):
        with torch.no_grad():
            slice = image[ind, :]
            x, y = 256, 256
            inputs = torch.from_numpy(slice).unsqueeze(
                0).float().to(device)

            if PSEUDO_MASK == None:
                pp = PSEUDO_MASK
            else:
                pp = PSEUDO_MASK[ind, :].unsqueeze(0).to(device)
                pp[pp > 0] = 1

            if ind == 0:
                if PSEUDO_MASK is not None:
                    flops, params = profile(net, inputs=(inputs, multimask_output, patch_size[0],  'val',pp, -1, prompt_mode, 'GT', 1))
                else:
                    flops, params = profile(net, inputs=(inputs, multimask_output, patch_size[0], 'val', None, prompt_mode))
                # 计算FLOPs
                total_flops = flops / 1e9  # 转换为GFLOPs

            start_time = time.time()

            outputs = net(inputs, multimask_output, patch_size[0], mode = 'val',pseudo=pp, prompt_mode=prompt_mode)
            # 添加：推理时间计算结束
            end_time = time.time()
            inference_time = end_time - start_time
            total_time += inference_time
            count += 1

            output_masks = outputs[0]['low_res_logits']

            out = torch.argmax(torch.softmax(output_masks, dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy().astype(np.uint8)
            out_h, out_w = out.shape
            if x != out_h or y != out_w:
                pred = zoom(out, (x / out_h, y / out_w), order=0)
            else:
                pred = out

            prediction[ind] = pred
            pred_for_comparison = pred.copy()

            if num_classes == 1:
                cv2.imwrite(os.path.join(save_dir, case_batch[ind]), pred * 255)
            elif num_classes == 3:
                pred_save = pred.copy()
                pred_save[pred_save == 1] = 100
                pred_save[pred_save == 2] = 200
                pred_save[pred_save == 3] = 255
                cv2.imwrite(os.path.join(save_dir, case_batch[ind]), pred_save)

            save_comparison_image(
                os.path.join(comparison_dir, case_batch[ind]),
                slice,
                label[ind],
                pred_for_comparison,
                num_classes,
            )

    # dice, iou = calculate_metric_percase(prediction, label)
    # return dice, iou

    metric_list = []
    for i in range(num_classes + 1):
        metric_list.append(calculate_metric_percase(
            prediction == i, label == i))
    # print(metric_list)
        # 添加：在函数末尾打印结果
    if count > 0:
        avg_time = total_time / count
        print(f"Average inference time: {avg_time:.4f}s per slice")
        if total_flops > 0:
            print(f"Model FLOPs: {total_flops:.2f}G")


    per_sample_dice = []  # list of [dice_c0, dice_c1, ...] for each sample in this batch
    batch_size = prediction.shape[0]
    for ind in range(batch_size):
        # 对单个样本，按类别计算 dice（calculate_metric_percase 返回 (dice, precision, iou)）
        per_class = []
        for cls in range(num_classes + 1):
            dice_val, _, _ = calculate_metric_percase(prediction[ind] == cls, label[ind] == cls)
            per_class.append(float(dice_val))
        per_sample_dice.append(per_class)
    return metric_list, per_sample_dice




def val_single_PROMPT(image, label, net, iter_numbers, num_classes, PSEUDO_MASK, case_batch, save_path,
                      device, prompt_mode, multimask_output=True, patch_size=[256, 256], point_numbers=1):
    image, label = image.squeeze(0).cpu().detach(
    ).numpy(), label.squeeze(0).cpu().detach().numpy()
    prediction = np.zeros_like(label)

    if PSEUDO_MASK ==None:
        save_dir = os.path.join(save_path, 'visualization_NOPROMPT', str(iter_numbers))
    else:
        save_dir = os.path.join(save_path, 'visualization', str(iter_numbers))

    os.makedirs(save_dir, exist_ok=True)

    for ind in range(image.shape[0]):
        slice = image[ind, :]
        x, y = 256, 256
        inputs = torch.from_numpy(slice).unsqueeze(
            0).float().to(device)

        if PSEUDO_MASK == None:
            pp = PSEUDO_MASK
        else:
            pp = PSEUDO_MASK[ind, :].unsqueeze(0).to(device)
        pp[pp>0] =1

        net.eval()

        with torch.no_grad():
            outputs = net(inputs, multimask_output, patch_size[0], mode = 'val',pseudo=pp, prompt_mode= prompt_mode, point_numbers= point_numbers)
            output_masks = outputs[0]['low_res_logits']

            out = torch.argmax(torch.softmax(output_masks, dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy().astype(np.uint8)
            out_h, out_w = out.shape
            if x != out_h or y != out_w:
                pred = zoom(out, (x / out_h, y / out_w), order=0)
            else:
                pred = out
            prediction[ind] = pred

            if num_classes == 1:
                cv2.imwrite(os.path.join(save_dir, case_batch[ind]), pred * 255)
            elif num_classes == 3:
                pred[pred == 1] = 100
                pred[pred == 2] = 200
                pred[pred == 3] = 255
                cv2.imwrite(os.path.join(save_dir, case_batch[ind]), pred)

    metric_list = []
    for i in range(num_classes + 1):
        metric_list.append(calculate_metric_percase(
            prediction == i, label == i))
    return metric_list
