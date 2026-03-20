import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import torch
from torchvision import transforms
import segmentation_models_pytorch as smp
import numpy as np
from skimage.morphology import erosion, disk, remove_small_objects, opening
from skimage.measure import label, regionprops
from scipy.ndimage import binary_fill_holes
import joblib
import matplotlib.pyplot as plt
from skimage.morphology import binary_dilation, binary_erosion
from skimage.measure import label as sk_label
from skimage.measure import regionprops as sk_regionprops


def segment_and_classify(
    image_path: str,
    unet_weights_path: str,
    pipeline_path: str,
    device: torch.device = torch.device('cpu'),
    image_size: tuple = (256, 256),
    erode_radius: int = 2,
    min_size: int = 100,
    open_radius: int = 2,
    hole_fill: bool = True
):
    
    unet = smp.Unet(
        encoder_name="resnet18",
        encoder_weights=None,
        in_channels=3,
        classes=1,
        activation=None
    )
    unet.load_state_dict(torch.load(unet_weights_path, map_location=device))
    unet.to(device)
    unet.eval()

    clf = joblib.load(pipeline_path)

    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    elif img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)

    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    orig_h, orig_w = img.shape[:2]
    img_resized = cv2.resize(img, image_size)
    tensor = transforms.ToTensor()(img_resized).unsqueeze(0).to(device)

    with torch.no_grad():
        out = unet(tensor)
        mask = torch.sigmoid(out).squeeze().cpu().numpy()

    mask = (mask > 0.5).astype(np.uint8) * 255
    mask = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    binary = mask > 0
    eroded = erosion(binary, disk(erode_radius))
    cleaned = remove_small_objects(eroded, min_size=min_size)
    sep = opening(cleaned, disk(open_radius))
    if hole_fill:
        sep = binary_fill_holes(sep)
    labels_img = label(sep)

    binary_mask_u8 = (sep.astype(np.uint8) * 255)

    props = regionprops(labels_img)
    if len(props) == 0:
        return np.array([], dtype=int), [], labels_img, np.array([], dtype=float), [], binary_mask_u8

    total_area = sum(p.area for p in props) or 1
    clf_features, object_ids, table_features = [], [], []

    for p in props:
        area = p.area
        maj = p.major_axis_length
        min_ax = p.minor_axis_length if p.minor_axis_length > 0 else 1e-6
        ecc = p.eccentricity
        ratio = maj / min_ax
        perim = p.perimeter if p.perimeter > 0 else 1e-6
        circularity = 4 * np.pi * area / (perim ** 2)
        area_frac = area / total_area
        clf_features.append([area_frac, ecc, ratio])

        table_features.append([area, maj, min_ax, circularity, ecc, ratio])
        object_ids.append(p.label)

    X_clf = np.array(clf_features)
    preds = clf.predict(X_clf)
    probs = clf.predict_proba(X_clf).max(axis=1)

    return preds, object_ids, labels_img, probs, table_features, binary_mask_u8


def overlay_segmentation(raw_img, labels, obj_ids, preds, alpha=0.4):
    
    pixel_classes = np.zeros_like(labels, dtype=int)
    for obj_id, cls in zip(obj_ids, preds):
        pixel_classes[labels == obj_id] = int(cls)

    base_cmap = plt.get_cmap('tab10', 10)
    overlay_rgb = base_cmap(np.clip(pixel_classes - 1, 0, 9))[..., :3] * 255
    overlay_rgb[pixel_classes == 0] = [0, 0, 0]

    blended = (raw_img * (1 - alpha) + overlay_rgb * alpha).astype(np.uint8)
    return blended


def overlay_binary_mask(raw_img, binary_mask_u8, alpha=0.4, color_rgb=(255, 0, 0)):
    if binary_mask_u8.ndim == 3:
        binary_mask_u8 = binary_mask_u8[..., 0]
    mask = binary_mask_u8 > 0
    overlay = np.zeros_like(raw_img, dtype=np.uint8)
    overlay[mask] = np.array(color_rgb, dtype=np.uint8)
    blended = raw_img.copy().astype(np.float32)
    blended[mask] = blended[mask] * (1 - alpha) + overlay[mask].astype(np.float32) * alpha
    return blended.astype(np.uint8)


def overlay_label_mask(raw_img, class_img, alpha=0.4, cmap_name='tab10'):
    base_cmap = plt.get_cmap(cmap_name, 10)
    overlay_rgb = base_cmap(np.clip(class_img - 1, 0, 9))[..., :3] * 255
    overlay_rgb[class_img == 0] = [0, 0, 0]
    blended = (raw_img * (1 - alpha) + overlay_rgb * alpha).astype(np.uint8)
    return blended


def build_class_label_image(labels_img, obj_ids, preds):
    class_img = np.zeros_like(labels_img, dtype=np.int32)
    for obj_id, cls in zip(obj_ids, preds):
        class_img[labels_img == obj_id] = int(cls)
    return class_img


def _keep_largest_component(binary):
    lbl = sk_label(binary.astype(bool))
    if lbl.max() == 0:
        return binary.astype(bool)
    props = sk_regionprops(lbl)
    largest = max(props, key=lambda r: r.area)
    return (lbl == largest.label)


def _farthest_points(binary_image, align_lr=True):
    binary_u8 = (binary_image.astype(np.uint8) * 255)
    contours, _ = cv2.findContours(binary_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError("No contour found for a vein band.")

    cnt = contours[0][:, 0, :] 

    pts = cnt.astype(np.float32)
    diffs = pts[:, None, :] - pts[None, :, :]
    d2 = (diffs ** 2).sum(axis=2)
    i, j = np.unravel_index(np.argmax(d2), d2.shape)

    p1 = cnt[i]
    p2 = cnt[j]

    if align_lr:
        if p1[0] <= p2[0]:
            left, right = p1, p2
        else:
            left, right = p2, p1
    else:
        left, right = p1, p2

    xL, yL = int(left[0]), int(left[1])
    xR, yR = int(right[0]), int(right[1])
    return xL, yL, xR, yR


def vein_normal(class_img, se_radius=14):
    
    se = disk(se_radius)

    def morph_band(mask):
        d = binary_dilation(mask, se)
        e = binary_erosion(d, se)
        band = e & (~mask)
        band = _keep_largest_component(band)
        return band

    veins = []

    mask = np.isin(class_img, [1, 2])
    veins.append(morph_band(mask))

    mask34 = np.isin(class_img, [3, 4])
    d = binary_dilation(mask34, se)
    e = binary_erosion(d, se)
    mask2 = (e != 0) | (class_img == 2)
    veins.append(morph_band(mask2))

    mask34 = np.isin(class_img, [3, 4])
    mask56 = np.isin(class_img, [5, 6])
    join1 = binary_erosion(binary_dilation(mask34, se), se)
    join2 = binary_erosion(binary_dilation(mask56, se), se)
    join = join1 | join2
    veins.append(morph_band(join))

    d = binary_dilation(mask56, se)
    e = binary_erosion(d, se)
    mask4 = (e != 0) | (class_img == 7)
    veins.append(morph_band(mask4))

    veins.append(morph_band(np.isin(class_img, [3, 4])))

    veins.append(morph_band(np.isin(class_img, [5, 6])))

    vpos = np.zeros((6, 4), dtype=int)
    for i, v in enumerate(veins):
        align = (i < 4)  
        vpos[i, :] = _farthest_points(v, align_lr=align)

    return vpos


def compute_axes_and_metrics(class_img, se_radius=14, scale_um_per_px=None):
    
    present = np.unique(class_img)
    present = present[present > 0]
    if not np.array_equal(np.sort(present), np.arange(1, 8)):
        raise ValueError("Axis computation requires a normal wing with unique labels 1..7 only.")

    mask3 = (class_img == 3)
    lbl3 = sk_label(mask3)
    props3 = sk_regionprops(lbl3)
    if not props3:
        raise ValueError("Could not find label 3 region for centroid reference.")
    cy, cx = props3[0].centroid
    centroidI3 = np.array([cx, cy], dtype=float)

    vpos = vein_normal(class_img, se_radius=se_radius)

    L2a = np.array([vpos[0, 0], vpos[0, 1]], dtype=float)
    L2b = np.array([vpos[0, 2], vpos[0, 3]], dtype=float)
    L5a = np.array([vpos[3, 0], vpos[3, 1]], dtype=float)
    L5b = np.array([vpos[3, 2], vpos[3, 3]], dtype=float)

    L2PointAP = L2a if np.linalg.norm(centroidI3 - L2a) > np.linalg.norm(centroidI3 - L2b) else L2b
    L5PointAP = L5a if np.linalg.norm(centroidI3 - L5a) > np.linalg.norm(centroidI3 - L5b) else L5b
    ap_line = (tuple(L2PointAP.astype(int)), tuple(L5PointAP.astype(int)))

    L3a = np.array([vpos[1, 0], vpos[1, 1]], dtype=float)
    L3b = np.array([vpos[1, 2], vpos[1, 3]], dtype=float)
    L4a = np.array([vpos[2, 0], vpos[2, 1]], dtype=float)
    L4b = np.array([vpos[2, 2], vpos[2, 3]], dtype=float)

    if np.linalg.norm(centroidI3 - L3a) > np.linalg.norm(centroidI3 - L3b):
        L3Right, L3Left = L3a, L3b
    else:
        L3Right, L3Left = L3b, L3a

    if np.linalg.norm(centroidI3 - L4a) > np.linalg.norm(centroidI3 - L4b):
        L4Right, L4Left = L4a, L4b
    else:
        L4Right, L4Left = L4b, L4a

    l3l4_right_line = (tuple(L3Right.astype(int)), tuple(L4Right.astype(int)))
    d_l3_l4_px = float(np.linalg.norm(L3Right - L4Right))

    left_mid = 0.5 * (L3Left + L4Left)
    right_mid = 0.5 * (L3Right + L4Right)
    pd_line = (tuple(left_mid.astype(int)), tuple(right_mid.astype(int)))

    out = {
        "ap_line_px": ap_line,
        "pd_line_px": pd_line,
        "l3l4_line_px": l3l4_right_line,
        "d_l3_l4_px": d_l3_l4_px,
    }
    if scale_um_per_px is not None:
        out["d_l3_l4_um"] = d_l3_l4_px * float(scale_um_per_px)
    return out

