import os
import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
from tkinter import messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import cv2
import numpy as np
import csv
import threading
from datetime import datetime
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from source_mapper_v2 import (
    segment_and_classify,
    build_class_label_image,
    compute_axes_and_metrics,
    overlay_binary_mask,
    overlay_label_mask,
)


WING_CLASSIFIER_PATH = "trained-models/wing_classifier.pt"
WING_IMG_SIZE = 224

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_eval_transform(img_size: int):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    return transforms.Compose([
        transforms.Resize(int(img_size * 1.15)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        normalize
    ])


def build_resnet18_classifier(num_classes: int):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


@torch.no_grad()
def predict_single_image(model, pil_img, tfm, device, class_names):
    model.eval()
    x = tfm(pil_img).unsqueeze(0).to(device)
    logits = model(x)
    probs = torch.softmax(logits, dim=1).squeeze(0).detach().cpu().numpy()

    pred_idx = int(np.argmax(probs))
    pred_class = class_names[pred_idx]
    pred_conf = float(probs[pred_idx])

    pairs = [(class_names[i], float(probs[i])) for i in range(len(class_names))]
    pairs.sort(key=lambda t: t[1], reverse=True)
    return pred_class, pred_conf, pairs


def _largest_component_binary_u8(binary_u8: np.ndarray) -> np.ndarray:
    if binary_u8 is None:
        return None
    bw = (binary_u8 > 0).astype(np.uint8)
    if bw.sum() == 0:
        return (bw * 255).astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    if num_labels <= 1:
        return (bw * 255).astype(np.uint8)

    areas = stats[1:, cv2.CC_STAT_AREA]
    max_idx = 1 + int(np.argmax(areas))
    out = (labels == max_idx).astype(np.uint8) * 255
    return out


def build_whole_wing_mask_from_interveins(binary_intervein_u8: np.ndarray, radius: int) -> np.ndarray:
    if binary_intervein_u8 is None:
        return None
    r = int(radius)
    if r < 0:
        r = 0

    bw = (binary_intervein_u8 > 0).astype(np.uint8)
    if bw.sum() == 0:
        return (bw * 255).astype(np.uint8)

    if r == 0:
        return _largest_component_binary_u8((bw * 255).astype(np.uint8))

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    dil = cv2.dilate(bw, k, iterations=1)
    ero = cv2.erode(dil, k, iterations=1)

    wing_u8 = (ero.astype(np.uint8) * 255)
    wing_u8 = _largest_component_binary_u8(wing_u8)
    return wing_u8


def centroid_xy_from_binary_u8(binary_u8: np.ndarray):
    if binary_u8 is None:
        return None
    ys, xs = np.where(binary_u8 > 0)
    if len(xs) == 0:
        return None
    return float(xs.mean()), float(ys.mean())


def centroid_xy_from_class(class_img: np.ndarray, cls: int):
    ys, xs = np.where(class_img == int(cls))
    if len(xs) == 0:
        return None
    return float(xs.mean()), float(ys.mean())


class WingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("MAPPER 2.0")

        self.image_path = None
        self.main_canvas = None

        self._raw_rgb = None
        self._binary_mask_u8 = None       
        self._class_img = None            
        self._axes_cache = None
        self._wing_mask_u8 = None         

        self.show_binary_overlay = tk.BooleanVar(value=False)
        self.show_labeled_overlay = tk.BooleanVar(value=False)
        self.show_wing_overlay = tk.BooleanVar(value=False)

        self.show_ap_axis = tk.BooleanVar(value=False)
        self.show_pd_axis = tk.BooleanVar(value=False)
        self.show_l3l4 = tk.BooleanVar(value=False)

        self.device = get_device()
        self.wing_clf_model = None
        self.wing_clf_class_names = None
        self.wing_clf_tfm = build_eval_transform(WING_IMG_SIZE)
        self._init_wing_classifier()

        legend_frame = ttk.Frame(root, padding=5)
        legend_frame.grid(row=0, column=0, sticky="ew")

        base_cmap = plt.get_cmap('tab10', 10)
        for i in range(10):
            color = (np.array(base_cmap(i)[:3]) * 255).astype(int)
            hex_color = f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"
            swatch = tk.Label(legend_frame, width=2, background=hex_color)
            swatch.pack(side="left", padx=2)
            label = tk.Label(legend_frame, text=str(i+1))
            label.pack(side="left", padx=5)

        control_frame = ttk.Frame(root, padding=10)
        control_frame.grid(row=1, column=0, sticky="ew")

        ttk.Button(control_frame, text="Load Image", command=self.load_image)\
            .grid(row=0, column=0, padx=5, pady=4)

        ttk.Button(control_frame, text="Segment & Classify", command=self.run_segmentation)\
            .grid(row=0, column=1, padx=5, pady=4)

        ttk.Button(control_frame, text="Batch Process Folder", command=self.batch_process_folder)\
            .grid(row=0, column=2, padx=5, pady=4)

        ttk.Checkbutton(control_frame, text="Overlay Binary",
                        variable=self.show_binary_overlay, command=self._toggle_binary_overlay)\
            .grid(row=1, column=0, padx=5, pady=4, sticky="w")

        ttk.Checkbutton(control_frame, text="Overlay Labeled",
                        variable=self.show_labeled_overlay, command=self._toggle_labeled_overlay)\
            .grid(row=1, column=1, padx=5, pady=4, sticky="w")

        ttk.Checkbutton(control_frame, text="Overlay Whole Wing",
                        variable=self.show_wing_overlay, command=self._toggle_wing_overlay)\
            .grid(row=1, column=2, padx=5, pady=4, sticky="w")

        ttk.Checkbutton(control_frame, text="Show AP Axis",
                        variable=self.show_ap_axis, command=self._toggle_axis_overlay)\
            .grid(row=2, column=0, padx=5, pady=4, sticky="w")

        ttk.Checkbutton(control_frame, text="Show PD Axis",
                        variable=self.show_pd_axis, command=self._toggle_axis_overlay)\
            .grid(row=2, column=1, padx=5, pady=4, sticky="w")

        ttk.Checkbutton(control_frame, text="Show d(L3–L4)",
                        variable=self.show_l3l4, command=self._toggle_axis_overlay)\
            .grid(row=2, column=2, padx=5, pady=4, sticky="w")

        ttk.Label(control_frame, text="Scale (µm/pixel):")\
            .grid(row=3, column=0, sticky="w", padx=5)
        self.scale_var = tk.DoubleVar(value=1.0)
        ttk.Entry(control_frame, textvariable=self.scale_var, width=8)\
            .grid(row=3, column=1, sticky="w")

        ttk.Label(control_frame, text="Erode radius:")\
            .grid(row=3, column=2, sticky="w", padx=5)
        self.erode_radius_var = tk.StringVar(value="2")
        ttk.Entry(control_frame, textvariable=self.erode_radius_var, width=6)\
            .grid(row=3, column=3, sticky="w")

        ttk.Label(control_frame, text="Open radius:")\
            .grid(row=4, column=0, sticky="w", padx=5)
        self.open_radius_var = tk.StringVar(value="2")
        ttk.Entry(control_frame, textvariable=self.open_radius_var, width=6)\
            .grid(row=4, column=1, sticky="w")

        ttk.Label(control_frame, text="Wing mask radius:")\
            .grid(row=4, column=2, sticky="w", padx=5)
        self.wing_mask_radius_var = tk.StringVar(value="20")
        ttk.Entry(control_frame, textvariable=self.wing_mask_radius_var, width=6)\
            .grid(row=4, column=3, sticky="w")

        progress_frame = ttk.Frame(root, padding=(10, 0, 10, 10))
        progress_frame.grid(row=2, column=0, sticky="ew")

        self.batch_status_var = tk.StringVar(value="")
        ttk.Label(progress_frame, textvariable=self.batch_status_var).grid(row=0, column=0, sticky="w")

        self.batch_progress = ttk.Progressbar(progress_frame, orient="horizontal", mode="determinate", length=400)
        self.batch_progress.grid(row=0, column=1, padx=10, sticky="ew")

        progress_frame.grid_columnconfigure(1, weight=1)

        container = ttk.Frame(root)
        container.grid(row=3, column=0, sticky="nsew")

        root.grid_rowconfigure(3, weight=1)
        root.grid_columnconfigure(0, weight=1)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=4)
        container.grid_columnconfigure(1, weight=1)

        self.plot_frame = ttk.Frame(container)
        self.plot_frame.grid(row=0, column=0, sticky="nsew")

        self.right_panel = ttk.Frame(container, padding=10)
        self.right_panel.grid(row=0, column=1, sticky="nsew")
        self._build_right_panel()

    def _batch_wing_class_probs(self, img_path: str):
        if self.wing_clf_model is None or self.wing_clf_class_names is None:
            return {"wing_pred_class": "", "wing_pred_conf": 0.0}

        try:
            pil_img = Image.open(img_path).convert("RGB")
            pred_class, pred_conf, pairs = predict_single_image(
                self.wing_clf_model,
                pil_img,
                self.wing_clf_tfm,
                self.device,
                self.wing_clf_class_names
            )

            out = {"wing_pred_class": pred_class, "wing_pred_conf": float(pred_conf)}

            for cname, p in pairs:
                safe = str(cname).strip().replace(" ", "_").replace("/", "_")
                out[f"wing_prob_{safe}"] = float(p)

            return out

        except Exception:
            return {"wing_pred_class": "", "wing_pred_conf": 0.0}

    def _list_images_recursive(self, root_dir: str):
        exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
        paths = []
        for dirpath, _, filenames in os.walk(root_dir):
            for fn in filenames:
                if fn.lower().endswith(exts):
                    paths.append(os.path.join(dirpath, fn))
        return sorted(paths)

    def _mask_paths_for_image(self, img_path: str):
        base, _ = os.path.splitext(img_path)
        return (
            base + "_binary_mask.png",
            base + "_labeled_mask.png",
            base + "_wing_mask.png",
        )

    def _load_existing_masks_if_present(self, img_path: str):
        bin_p, lbl_p, wing_p = self._mask_paths_for_image(img_path)
        if not (os.path.exists(bin_p) and os.path.exists(lbl_p) and os.path.exists(wing_p)):
            return None, None, None, False

        b = cv2.imread(bin_p, cv2.IMREAD_GRAYSCALE)
        c = cv2.imread(lbl_p, cv2.IMREAD_GRAYSCALE)
        w = cv2.imread(wing_p, cv2.IMREAD_GRAYSCALE)

        if b is None or c is None or w is None:
            return None, None, None, False

        b = (b > 0).astype(np.uint8) * 255
        w = (w > 0).astype(np.uint8) * 255
        c = c.astype(np.uint8) 
        return b, c, w, True

    def _compute_simple_features_dict(self, class_img, intervein_binary_u8, wing_mask_u8, scale_um_per_px: float):
        scale = float(scale_um_per_px)

        wing_center = centroid_xy_from_binary_u8(wing_mask_u8) if wing_mask_u8 is not None else None
        if wing_center is None:
            wing_center = centroid_xy_from_binary_u8(intervein_binary_u8)
        if wing_center is None:
            wing_center = (0.0, 0.0)

        wing_cx, wing_cy = wing_center

        if wing_mask_u8 is not None and (wing_mask_u8 > 0).sum() > 0:
            wing_area_px = float((wing_mask_u8 > 0).sum())
        else:
            wing_area_px = float((intervein_binary_u8 > 0).sum())
        wing_area_um2 = wing_area_px * (scale ** 2)

        ap_len_um = 0.0
        pd_len_um = 0.0
        d_l3l4_um = 0.0
        try:
            tmp_axes = compute_axes_and_metrics(class_img, se_radius=14, scale_um_per_px=scale)
            (ap_p1, ap_p2) = tmp_axes["ap_line_px"]
            (pd_p1, pd_p2) = tmp_axes["pd_line_px"]

            ap_len_px = float(np.linalg.norm(np.array(ap_p1, float) - np.array(ap_p2, float)))
            pd_len_px = float(np.linalg.norm(np.array(pd_p1, float) - np.array(pd_p2, float)))

            ap_len_um = ap_len_px * scale
            pd_len_um = pd_len_px * scale
            d_l3l4_um = float(tmp_axes.get("d_l3_l4_um", float(tmp_axes["d_l3_l4_px"]) * scale))
        except Exception:
            pass

        out = {
            "total_wing_area_um2": wing_area_um2,
            "ap_axis_length_um": ap_len_um,
            "pd_axis_length_um": pd_len_um,
            "dL3L4_um": d_l3l4_um,
        }

        for lab in range(1, 11):
            area_px = float((class_img == lab).sum())
            area_um2 = area_px * (scale ** 2)

            c = centroid_xy_from_class(class_img, lab)
            if c is None:
                dist_um = 0.0
            else:
                cx, cy = c
                dist_px = float(np.sqrt((cx - wing_cx) ** 2 + (cy - wing_cy) ** 2))
                dist_um = dist_px * scale

            out[f"intervein_{lab}_area_um2"] = area_um2
            out[f"intervein_{lab}_centroid_dist_um"] = dist_um

        return out

    def batch_process_folder(self):
        folder = filedialog.askdirectory(title="Select master folder to batch process")
        if not folder:
            return

        img_paths = self._list_images_recursive(folder)
        if len(img_paths) == 0:
            messagebox.showinfo("No images found", "No image files found in the selected folder.")
            return

        try:
            erode_radius = int(str(self.erode_radius_var.get()).strip() or "2")
            open_radius = int(str(self.open_radius_var.get()).strip() or "2")
            wing_r = int(str(self.wing_mask_radius_var.get()).strip() or "0")
        except ValueError:
            messagebox.showerror("Invalid input", "Radii must be integers.")
            return

        scale = float(self.scale_var.get())

        self.batch_progress["value"] = 0
        self.batch_progress["maximum"] = len(img_paths)
        self.batch_status_var.set(f"Starting batch: 0 / {len(img_paths)} processed")

        t = threading.Thread(
            target=self._batch_worker,
            args=(folder, img_paths, erode_radius, open_radius, wing_r, scale),
            daemon=True
        )
        t.start()

    def _batch_worker(self, folder, img_paths, erode_radius, open_radius, wing_r, scale):
        rows = []
        total = len(img_paths)

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_csv = os.path.join(folder, f"wing_features_batch_{stamp}.csv")

        for i, img_path in enumerate(img_paths, start=1):
            rel = os.path.relpath(img_path, folder)

            try:
                intervein_binary_u8, class_img, wing_mask_u8, ok = self._load_existing_masks_if_present(img_path)

                if ok:
                    status = "skipped_model_used_existing_masks"
                else:
                    preds, obj_ids, lbl_img, confs, table_features, binary_mask_u8 = segment_and_classify(
                        image_path=img_path,
                        unet_weights_path="trained-models/unet_weights.pth",
                        pipeline_path="trained-models/KNN_pipeline.pkl",
                        erode_radius=erode_radius,
                        open_radius=open_radius
                    )

                    intervein_binary_u8 = binary_mask_u8.astype(np.uint8)
                    class_img = build_class_label_image(lbl_img, obj_ids, preds).astype(np.uint8)
                    wing_mask_u8 = build_whole_wing_mask_from_interveins(intervein_binary_u8, wing_r)

                    bin_p, lbl_p, wing_p = self._mask_paths_for_image(img_path)
                    cv2.imwrite(bin_p, intervein_binary_u8)
                    cv2.imwrite(lbl_p, class_img)
                    if wing_mask_u8 is not None:
                        cv2.imwrite(wing_p, wing_mask_u8)

                    status = "ok_ran_model"

                feats = self._compute_simple_features_dict(
                    class_img=class_img,
                    intervein_binary_u8=intervein_binary_u8,
                    wing_mask_u8=wing_mask_u8,
                    scale_um_per_px=scale
                )

                row = {
                    "filename": os.path.basename(img_path),
                    "relative_path": rel,
                    "status": status,
                }
                row.update(self._batch_wing_class_probs(img_path))
                row.update(feats)
                rows.append(row)

            except Exception as e:
                rows.append({
                    "filename": os.path.basename(img_path),
                    "relative_path": rel,
                    "status": f"error: {str(e)}",
                })

            self.root.after(0, self._update_batch_progress, i, total, rel)

        self.root.after(0, self._write_batch_csv, out_csv, rows)

    def _update_batch_progress(self, done, total, current_rel_path):
        self.batch_progress["value"] = done
        remaining = total - done
        self.batch_status_var.set(
            f"Processed {done} / {total} | Remaining: {remaining} | Current: {current_rel_path}"
        )
        self.root.update_idletasks()

    def _write_batch_csv(self, out_csv, rows):
        keys = set()
        for r in rows:
            keys.update(r.keys())
        keys = sorted(keys)

        try:
            with open(out_csv, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=keys)
                w.writeheader()
                for r in rows:
                    w.writerow(r)

            self.batch_status_var.set(f"Batch complete. Saved: {out_csv}")
            messagebox.showinfo("Batch complete", f"Saved features CSV:\n{out_csv}")

        except Exception as e:
            messagebox.showerror("CSV write failed", str(e))

    def _init_wing_classifier(self):
        if not os.path.exists(WING_CLASSIFIER_PATH):
            self.wing_clf_model = None
            self.wing_clf_class_names = None
            return
        try:
            ckpt = torch.load(WING_CLASSIFIER_PATH, map_location=self.device)
            class_names = ckpt.get("class_names", None)
            state_dict = ckpt.get("state_dict", None)
            if class_names is None or state_dict is None:
                raise ValueError("wing_classifier.pt must contain keys: 'class_names' and 'state_dict'.")
            num_classes = len(class_names)
            model = build_resnet18_classifier(num_classes)
            model.load_state_dict(state_dict)
            model.to(self.device)
            model.eval()
            self.wing_clf_model = model
            self.wing_clf_class_names = class_names
        except Exception as e:
            self.wing_clf_model = None
            self.wing_clf_class_names = None
            messagebox.showwarning("Wing classifier load failed", str(e))

    def _build_right_panel(self):
        title = ttk.Label(self.right_panel, text="Wing Type Classification", font=("Arial", 12, "bold"))
        title.pack(anchor="w", pady=(0, 8))

        self.wing_status_var = tk.StringVar(value="Model: not loaded")
        ttk.Label(self.right_panel, textvariable=self.wing_status_var).pack(anchor="w", pady=(0, 8))

        sep = ttk.Separator(self.right_panel, orient="horizontal")
        sep.pack(fill="x", pady=6)

        self.wing_pred_var = tk.StringVar(value="Predicted: —")
        self.wing_conf_var = tk.StringVar(value="Confidence: —")
        ttk.Label(self.right_panel, textvariable=self.wing_pred_var, font=("Arial", 11, "bold")).pack(anchor="w")
        ttk.Label(self.right_panel, textvariable=self.wing_conf_var).pack(anchor="w", pady=(0, 8))

        ttk.Label(self.right_panel, text="Class probabilities:", font=("Arial", 10, "bold")).pack(anchor="w")
        self.wing_probs_text = tk.Text(self.right_panel, width=32, height=18, wrap="none")
        self.wing_probs_text.pack(fill="both", expand=True, pady=(4, 0))
        self.wing_probs_text.configure(state="disabled")
        self._refresh_classifier_status()

    def _refresh_classifier_status(self):
        if self.wing_clf_model is None:
            self.wing_status_var.set(f"Model: not loaded\nExpected: {WING_CLASSIFIER_PATH}")
        else:
            self.wing_status_var.set(f"Path to trained model:\n{WING_CLASSIFIER_PATH}")

    def _set_probs_text(self, lines):
        self.wing_probs_text.configure(state="normal")
        self.wing_probs_text.delete("1.0", "end")
        self.wing_probs_text.insert("1.0", "\n".join(lines))
        self.wing_probs_text.configure(state="disabled")

    def _run_wing_classifier(self):
        self._refresh_classifier_status()

        if self.wing_clf_model is None or self.wing_clf_class_names is None:
            self.wing_pred_var.set("Predicted: —")
            self.wing_conf_var.set("Confidence: —")
            self._set_probs_text(["(wing classifier model not loaded)"])
            return

        if not self.image_path:
            self.wing_pred_var.set("Predicted: —")
            self.wing_conf_var.set("Confidence: —")
            self._set_probs_text(["(no image loaded)"])
            return

        try:
            pil_img = Image.open(self.image_path).convert("RGB")
            pred_class, pred_conf, pairs = predict_single_image(
                self.wing_clf_model, pil_img, self.wing_clf_tfm, self.device, self.wing_clf_class_names
            )
            self.wing_pred_var.set(f"Predicted: {pred_class}")
            self.wing_conf_var.set(f"Confidence: {pred_conf:.4f}")
            lines = [f"{name:<10s} : {p:.4f}" for (name, p) in pairs]
            self._set_probs_text(lines)
        except Exception as e:
            self.wing_pred_var.set("Predicted: —")
            self.wing_conf_var.set("Confidence: —")
            self._set_probs_text([f"(classification failed)", str(e)])

    def load_image(self):
        self.image_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.tif *.tiff")]
        )
        if not self.image_path:
            return

        raw = cv2.imread(self.image_path, cv2.IMREAD_UNCHANGED)
        if raw is None:
            messagebox.showerror("Load error", "Could not read image.")
            return

        if raw.ndim == 2:
            self._raw_rgb = cv2.cvtColor(raw, cv2.COLOR_GRAY2RGB)
        elif raw.ndim == 3 and raw.shape[2] == 4:
            self._raw_rgb = cv2.cvtColor(raw, cv2.COLOR_BGRA2RGB)
        else:
            self._raw_rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)

        self._binary_mask_u8 = None
        self._class_img = None
        self._axes_cache = None
        self._wing_mask_u8 = None

        self.show_binary_overlay.set(False)
        self.show_labeled_overlay.set(False)
        self.show_wing_overlay.set(False)
        self.show_ap_axis.set(False)
        self.show_pd_axis.set(False)
        self.show_l3l4.set(False)

        self._render_display()
        self._run_wing_classifier()

    
    def run_segmentation(self):
        if not self.image_path or self._raw_rgb is None:
            messagebox.showwarning("No image", "Please load an image first.")
            return

        try:
            erode_radius = int(str(self.erode_radius_var.get()).strip() or "2")
            open_radius = int(str(self.open_radius_var.get()).strip() or "2")
        except ValueError:
            messagebox.showerror("Invalid input", "Erode radius and Open radius must be integers.")
            return
        if erode_radius < 0 or open_radius < 0:
            messagebox.showerror("Invalid input", "Erode radius and Open radius must be >= 0.")
            return

        try:
            preds, obj_ids, lbl_img, confs, table_features, binary_mask_u8 = segment_and_classify(
                image_path=self.image_path,
                unet_weights_path="trained-models/unet_weights.pth",
                pipeline_path="trained-models/KNN_pipeline.pkl",
                erode_radius=erode_radius,
                open_radius=open_radius
            )
        except Exception as e:
            messagebox.showerror("Segmentation error", str(e))
            return

        self._binary_mask_u8 = binary_mask_u8.astype(np.uint8)
        self._class_img = build_class_label_image(lbl_img, obj_ids, preds).astype(np.uint8)
        self._axes_cache = None

        try:
            wing_r = int(str(self.wing_mask_radius_var.get()).strip() or "0")
        except ValueError:
            wing_r = 0

        self._wing_mask_u8 = build_whole_wing_mask_from_interveins(self._binary_mask_u8, wing_r)

        base, _ = os.path.splitext(self.image_path)
        intervein_binary_path = base + "_binary_mask.png"
        labeled_path = base + "_labeled_mask.png"
        wing_mask_path = base + "_wing_mask.png"

        cv2.imwrite(intervein_binary_path, self._binary_mask_u8)
        cv2.imwrite(labeled_path, self._class_img)
        if self._wing_mask_u8 is not None:
            cv2.imwrite(wing_mask_path, self._wing_mask_u8)

        messagebox.showinfo(
            "Saved masks",
            "Created mask files:\n"
            f"1) {intervein_binary_path}\n"
            f"2) {labeled_path}\n"
            f"3) {wing_mask_path}"
        )

        self.show_binary_overlay.set(False)
        self.show_labeled_overlay.set(True)
        self.show_wing_overlay.set(False)

        self._render_display()
        self.show_simple_feature_table()
        self._run_wing_classifier()

    def _toggle_binary_overlay(self):
        if self.show_binary_overlay.get():
            self.show_labeled_overlay.set(False)
            self.show_wing_overlay.set(False)
        self._render_display()

    def _toggle_labeled_overlay(self):
        if self.show_labeled_overlay.get():
            self.show_binary_overlay.set(False)
            self.show_wing_overlay.set(False)
        self._render_display()

    def _toggle_wing_overlay(self):
        if self.show_wing_overlay.get():
            self.show_binary_overlay.set(False)
            self.show_labeled_overlay.set(False)
        self._render_display()

    
    def _ensure_axes_cache(self):
        if self._class_img is None:
            raise ValueError("Please run 'Segment & Classify' first (axes require labels 1..7).")
        if self._axes_cache is None:
            scale = self.scale_var.get()
            self._axes_cache = compute_axes_and_metrics(
                self._class_img,
                se_radius=14,
                scale_um_per_px=scale
            )

    def _toggle_axis_overlay(self):
        try:
            if self.show_ap_axis.get() or self.show_pd_axis.get() or self.show_l3l4.get():
                self._ensure_axes_cache()
                if self.show_l3l4.get():
                    dpx = self._axes_cache["d_l3_l4_px"]
                    dum = self._axes_cache.get("d_l3_l4_um", None)
                    msg = f"d(L3–L4) = {dpx:.2f} px" if dum is None else f"d(L3–L4) = {dpx:.2f} px  ({dum:.2f} µm)"
                    messagebox.showinfo("d(L3–L4)", msg)
        except Exception as e:
            messagebox.showwarning("Axes not applicable", str(e))
            self.show_ap_axis.set(False)
            self.show_pd_axis.set(False)
            self.show_l3l4.set(False)
            self._axes_cache = None

        self._render_display()

    
    def _compose_rgb(self):
        if self._raw_rgb is None:
            return None

        img = self._raw_rgb.copy()

        if self.show_binary_overlay.get():
            if self._binary_mask_u8 is None:
                messagebox.showwarning("No mask", "Please run 'Segment & Classify' first.")
                self.show_binary_overlay.set(False)
            else:
                img = overlay_binary_mask(img, self._binary_mask_u8, alpha=0.4, color_rgb=(255, 0, 0))

        elif self.show_labeled_overlay.get():
            if self._class_img is None:
                messagebox.showwarning("No mask", "Please run 'Segment & Classify' first.")
                self.show_labeled_overlay.set(False)
            else:
                img = overlay_label_mask(img, self._class_img, alpha=0.4, cmap_name='tab10')

        elif self.show_wing_overlay.get():
            if self._wing_mask_u8 is None:
                messagebox.showwarning("No wing mask", "Please run 'Segment & Classify' first.")
                self.show_wing_overlay.set(False)
            else:
                img = overlay_binary_mask(img, self._wing_mask_u8, alpha=0.35, color_rgb=(0, 255, 0))

        if (self.show_ap_axis.get() or self.show_pd_axis.get() or self.show_l3l4.get()) and self._axes_cache is not None:
            img2 = img.copy()
            if self.show_ap_axis.get():
                p1, p2 = self._axes_cache["ap_line_px"]
                cv2.line(img2, p1, p2, (255, 0, 0), 2)
            if self.show_pd_axis.get():
                p1, p2 = self._axes_cache["pd_line_px"]
                cv2.line(img2, p1, p2, (0, 255, 0), 2)
            if self.show_l3l4.get():
                p1, p2 = self._axes_cache["l3l4_line_px"]
                cv2.line(img2, p1, p2, (0, 0, 255), 2)
            img = img2

        return img

    def _render_display(self):
        img = self._compose_rgb()
        if img is None:
            return

        fig, ax = plt.subplots(figsize=(7, 7))
        ax.imshow(img)
        ax.axis("off")

        if self.main_canvas:
            self.main_canvas.get_tk_widget().destroy()

        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.LEFT, fill="both", expand=True)
        self.main_canvas = canvas

    def show_simple_feature_table(self):
        if self._class_img is None or self._binary_mask_u8 is None:
            messagebox.showwarning("No data", "Please run 'Segment & Classify' first.")
            return

        scale = float(self.scale_var.get())
        feats = self._compute_simple_features_dict(self._class_img, self._binary_mask_u8, self._wing_mask_u8, scale)

        rows = []
        for lab in range(1, 11):
            rows.append((f"I{lab} area (µm²)", f"{feats[f'intervein_{lab}_area_um2']:.2f}"))
            rows.append((f"I{lab} centroid→wing center (µm)", f"{feats[f'intervein_{lab}_centroid_dist_um']:.2f}"))

        rows.append(("Total wing area (µm²)", f"{feats['total_wing_area_um2']:.2f}"))
        rows.append(("AP axis length (µm)", f"{feats['ap_axis_length_um']:.2f}"))
        rows.append(("PD axis length (µm)", f"{feats['pd_axis_length_um']:.2f}"))
        rows.append(("d(L3–L4) (µm)", f"{feats['dL3L4_um']:.2f}"))

        win = tk.Toplevel(self.root)
        win.title("Wing Feature Table")
        win.geometry("280x500")

        cols = ("Feature", "Value")
        tree = ttk.Treeview(win, columns=cols, show="headings", height=20)
        tree.heading("Feature", text="Feature")
        tree.heading("Value", text="Value")
        tree.column("Feature", width=180, anchor="w")
        tree.column("Value", width=100, anchor="center")

        yscroll = ttk.Scrollbar(win, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=yscroll.set)

        tree.grid(row=0, column=0, sticky="nsew")
        yscroll.grid(row=0, column=1, sticky="ns")

        win.grid_rowconfigure(0, weight=1)
        win.grid_columnconfigure(0, weight=1)

        for feat, val in rows:
            tree.insert("", "end", values=(feat, val))

        def save_csv():
            file_path = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv")]
            )
            if file_path:
                with open(file_path, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(cols)
                    writer.writerows(rows)
                messagebox.showinfo("Export Successful", f"Table saved to {file_path}")

        ttk.Button(win, text="Save as CSV", command=save_csv).grid(row=1, column=0, pady=8, sticky="w")


if __name__ == "__main__":
    root = tk.Tk()
    app = WingApp(root)
    root.mainloop()
