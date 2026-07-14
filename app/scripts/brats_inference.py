"""
brats_inference.py
End-to-end inference for the brain-tumor pipeline.

Given the 4 MRI modalities of a patient BraTS2021 this module:

  1. resolves and loads the four modalities;         
  2. runs the fine-tuned SwinUNETR --> segmentation; 
  3. extracts the frozen features and ratios;      
  4. builds the structured clinical description;       
  5. LLM explanation can be generated in the app;        
  6. builds an overlay figure of the prediction.
"""

from __future__ import annotations
import os
import re
import json
import math
import tempfile
import copy
import re
import torch
from monai.networks.nets import SwinUNETR
from monai.transforms import (Compose, LoadImaged, EnsureChannelFirstd,
                                   NormalizeIntensityd, EnsureTyped)
from monai.inferers import sliding_window_inference

import numpy as np
import pandas as pd
import nibabel as nib
from scipy import ndimage
from radiomics import featureextractor
import SimpleITK as sitk
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# The user supplies the patient MRI folder, everything else is bundled.
# brats_inference.py

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
APP_DIR = os.path.dirname(SCRIPT_DIR)
ASSETS_DIR = os.path.join(APP_DIR, "assets")

# Configuration
# Phase 2: segmentation
MODEL_KWARGS = dict(in_channels=4, out_channels=3, feature_size=48, use_checkpoint=True)
ROI_SIZE = (64, 64, 64)
SW_OVERLAP = 0.5
SW_BATCH_SIZE = 1
LABEL_NCR, LABEL_ED, LABEL_ET = 1, 2, 4
FALLBACK_CHANNEL_ORDER = ["flair", "t1ce", "t1", "t2"]   

# Phase 3: radiomics
SUBREGIONS   = {"ET": "t1ce", "TC": "t1ce", "WT": "flair"}
SUBREGION_LABELS = {"ET": [LABEL_ET], "TC": [LABEL_NCR, LABEL_ET], "WT": [LABEL_NCR, LABEL_ED, LABEL_ET]}
RADIOMICS_SETTINGS = dict(binWidth=25, normalize=False, resampledPixelSpacing=None, interpolator="sitkBSpline", label=1, minimumROIDimensions=2, minimumROISize=None, padDistance=5)
MIN_ROI_VOXELS = 10

# volume feature keys used for the ratios
ET_VOL = "ET_T1CE_original_shape_VoxelVolume"
TC_VOL = "TC_T1CE_original_shape_VoxelVolume"
WT_VOL = "WT_FLAIR_original_shape_VoxelVolume"
RATIO_NAMES = ["ET_WT_ratio", "ET_TC_ratio", "TC_WT_ratio", "necrosis_fraction", "edema_index"]

_TOKEN_TO_ROLE = {
    "t1ce": "t1ce", "t1c": "t1ce", "t1gd": "t1ce", "gd": "t1ce",
    "t1": "t1", "t1n": "t1",
    "t2": "t2", "t2w": "t2",
    "flair": "flair", "t2f": "flair", "fla": "flair",
}

REPORT_SCHEMA = {
    "size": {
        "title": "Tumor Size",
        "header": "Source: SwinUNETR Segmentation",
        "body_template": (
            "<b>Directly Measured (Voxel Count / 1000):</b><br>"
            "ET (Enhancing Tumor), TC (Tumor Core), WT (Whole Tumor).<br><br>"
            "<b>Derived via Subtraction:</b><br>"
            "NCR (Necrosis) = TC - ET ({tc} - {et} = {ncr} cm³).<br>"
            "ED (Edema) = WT - TC ({wt} - {tc} = {ed} cm³)."
        )
    },
    "composition": {
        "title": "Composition",
        "header": "Metric Derivation",
        "body_template": (
            "<b>Necrosis Fraction:</b> NCR / TC ({ncr} / {tc} = {nec_f}).<br>"
            "<b>Enhancing Fraction:</b> ET / WT ({et} / {wt} = {enh_f}).<br>"
            "<i>Ranked vs cohort tertiles (33rd/66th pct).</i>"
        )
    },
    "morphology": {
        "title": "Morphology",
        "header": "Source: ET Mask (T1ce)",
        "body_template": (
            "<b>Logic:</b> Sphericity measures how closely the shape resembles a sphere (1.0).<br>"
            "<b>Classification:</b> {sph} indicates {desc} geometry.<br>"
            "<i>Ranked vs cohort tertiles (33rd/66th pct).</i>"
        )
    }
}

# MODALITY RESOLUTION

def modality_token(filename: str):
    """Return the canonical role for a modality filename, or None.

    Works for BraTS2021 ('..._t1ce.nii.gz') and BraTS-Africa ('...-t1c.nii.gz').
    The modality token is the last '_' or '-' separated piece before the extension.
    """
    base = os.path.basename(filename).lower()
    base = re.sub(r"\.nii(\.gz)?$", "", base)
    token = re.split(r"[_\-]", base)[-1]
    return _TOKEN_TO_ROLE.get(token)


def resolve_modalities(input_dir: str):
    """Map a patient folder to {role: filepath} for t1, t1ce, t2, flair. Raises a clear error if any of the four modalities is missing."""
    found = {}
    for fn in sorted(os.listdir(input_dir)):
        if not fn.lower().endswith((".nii", ".nii.gz")):
            continue
        role = modality_token(fn)
        if role and role not in found:           
            found[role] = os.path.join(input_dir, fn)
    missing = [m for m in ("t1", "t1ce", "t2", "flair") if m not in found]
    if missing:
        raise FileNotFoundError(
            f"Could not find these modalities in {input_dir}: {missing}. "
            f"Found: { {k: os.path.basename(v) for k, v in found.items()} }")
    return found


def detect_channel_order(folds_json: str | None):
    """Derive the model's input channel order (list of roles) from the training folds JSON, so inference matches training exactly. Falls back to
    FALLBACK_CHANNEL_ORDER with a warning if the JSON is unavailable."""
    if folds_json and os.path.exists(folds_json):
        with open(folds_json) as f:
            splits = json.load(f)
        entry = splits["training"][0] if isinstance(splits, dict) else splits[0]
        order = [modality_token(p) for p in entry["image"]]
        if all(order) and len(order) == 4:
            return order, True
    print("WARNING: folds JSON not found / unreadable - falling back to "
          f"{FALLBACK_CHANNEL_ORDER}. VERIFY this matches your training order!")
    return list(FALLBACK_CHANNEL_ORDER), False


# RATIOS and FEATURE SUBSETTING

def compute_ratios(features: dict):
    """Volumetric ratios (div-by-zero -> NaN)."""
    et, tc, wt = features.get(ET_VOL), features.get(TC_VOL), features.get(WT_VOL)
    def safe(n, d):
        if n is None or d is None or d == 0 or not math.isfinite(d):
            return float("nan")
        v = n / d
        return v if math.isfinite(v) else float("nan")
    et0 = 0.0 if et is None else et
    return {
        "ET_WT_ratio": safe(et, wt),
        "ET_TC_ratio": safe(et, tc),
        "TC_WT_ratio": safe(tc, wt),
        "necrosis_fraction": safe((tc - et0) if tc is not None else None, tc),
        "edema_index": safe((wt - tc) if (wt is not None and tc is not None) else None, tc),
    }


def subset_to_frozen(features: dict, ratios: dict, interp_features: list):
    """Keep only the frozen 15 features (+ ratios), as a flat dict."""
    row = {f: features.get(f) for f in interp_features}
    row.update(ratios)
    return row


# 3. ARTEFACT LOADING

def load_artifacts(feature_dir):
    """Load the frozen Phase 3 reporting feature set (the 15 stable features)."""
    final_path = os.path.join(feature_dir, "05_report_features_final.json")
    with open(final_path) as f:
        frozen = json.load(f)
    report_features = frozen["report_features"]
    ratio_keys = ["ET_WT_ratio", "ET_TC_ratio", "TC_WT_ratio", "necrosis_fraction", "edema_index"]
    groups = {"shape": [], "firstorder": [], "ratio": []}
    for f in report_features:
        if f in ratio_keys:
            groups["ratio"].append(f)
        elif "shape" in f:
            groups["shape"].append(f)
        elif "firstorder" in f:
            groups["firstorder"].append(f)
    icc = frozen.get("icc", {})        # ICC scores if present in the JSON, else empty
    return {"report_features": report_features, "groups": groups, "icc": icc}


# 4. STRUCTURED CLINICAL DESCRIPTION
# Descriptive bands = cohort tertiles (33rd/66th percentile), defined for this work.
# Cohort-relative descriptive labels, not diagnostic cutoffs.
NECROSIS_BANDS = [(0.007, "minimal"), (0.298, "moderate"), (1.01, "extensive")]
ENHANCING_BANDS = [(0.176, "minimal"), (0.339, "moderate"), (1.01, "predominant")]
SPHERICITY_BANDS = [(0.364, "markedly irregular"), (0.685, "moderately irregular"), (1.01, "relatively regular, well-circumscribed")]

def _band(value, bands):
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "indeterminate"
    for edge, label in bands:
        if value < edge:
            return label
    return bands[-1][1]

def disjoint_volumes_cm3(row):
    """Disjoint tissue volumes (cm3) from the nested-region volumes. ET = enhancing, NCR = TC - ET, ED = WT - TC. These partition the whole tumor."""
    def g(k):
        v = row.get(k)
        return float("nan") if v is None else float(v)
    wt = g(WT_VOL) / 1000.0
    tc = g(TC_VOL) / 1000.0
    et = g(ET_VOL) / 1000.0
    et = 0.0 if math.isnan(et) else et
    ncr = max(tc - et, 0.0) if not math.isnan(tc) else float("nan")
    ed  = max(wt - tc, 0.0) if not (math.isnan(wt) or math.isnan(tc)) else float("nan")
    rnd = lambda x: None if (isinstance(x, float) and math.isnan(x)) else round(x, 1)
    return {"enhancing_cm3": rnd(et), "necrotic_cm3": rnd(ncr), "edema_cm3": rnd(ed), "whole_tumor_cm3": rnd(wt), "tumor_core_cm3": rnd(tc)}


def describe_patient(row, groups):
    """Phase 4: one patient's feature row --> structured clinical findings. Description only, no prediction."""
    vols = disjoint_volumes_cm3(row)

    def num(k):
        v = row.get(k)
        return float("nan") if v is None else float(v)

    necrosis = num("necrosis_fraction")
    et_wt = num("ET_WT_ratio")
    composition = {
        "necrosis_fraction": None if math.isnan(necrosis) else round(necrosis, 3),
        "necrosis_category": _band(necrosis, NECROSIS_BANDS),
        "enhancing_fraction_of_wt": None if math.isnan(et_wt) else round(et_wt, 3),
        "enhancing_category": _band(et_wt, ENHANCING_BANDS),
    }

    sph = num("ET_T1CE_original_shape_Sphericity")
    morphology = {
        "et_sphericity": None if math.isnan(sph) else round(sph, 3),
        "margin_descriptor": _band(sph, SPHERICITY_BANDS),
    }

    firstorder = {}
    for f in groups.get("firstorder", []):
        v = row.get(f)
        firstorder[f] = None if (v is None or (isinstance(v, float) and not math.isfinite(v))) else round(float(v), 3)

    return {
        "patient_id": row.get("patient_id", ""),
        "size_cm3": vols,
        "composition": composition,
        "morphology": morphology,
        "firstorder_table": firstorder,
        "disclaimer": "Descriptive only; not a prediction of grade, molecular status or outcome.",
    }

def add_provenance(findings):
    f = copy.deepcopy(findings)
    
    # --- DATA LAYER: Collect the live patient values ---
    v = f.get("size_cm3", {})
    c = f.get("composition", {})
    m = f.get("morphology", {})
    
    # Create a simple "Patient Data Map"
    data = {
        "tc": v.get("tumor_core_cm3", 0),
        "et": v.get("enhancing_cm3", 0),
        "wt": v.get("whole_tumor_cm3", 0),
        "ncr": v.get("necrotic_cm3", 0),
        "ed": v.get("edema_cm3", 0),
        "nec_f": c.get("necrosis_fraction"),
        "enh_f": c.get("enhancing_fraction_of_wt"),
        "sph": f"{m.get('et_sphericity', 0):.3f}",
        "desc": m.get("margin_descriptor")
    }

    # ASSEMBLY LAYER
    report = []
    
    # Loop through the dictionary keys to build the report
    for key in ["size", "composition", "morphology"]:
        cfg = REPORT_SCHEMA[key]
        
        # Build section-specific finding bullets
        if key == "size":
            findings_list = [f"Tumor core (TC): <b>{data['tc']} cm³</b>",
                             f"Enhancing (ET): <b>{data['et']} cm³</b>",
                             f"Whole tumor (WT): <b>{data['wt']} cm³</b>"]
        elif key == "composition":
            findings_list = [f"Necrosis: <b>{c.get('necrosis_category')}</b> ({data['nec_f']})",
                             f"Enhancement: <b>{c.get('enhancing_category')}</b> ({data['enh_f']})"]
        else:
            findings_list = [f"Margins: <b>{data['desc']}</b>", 
                             f"Sphericity: <b>{data['sph']}</b>"]

        # FILL THE TEMPLATE DYNAMICALLY
        report.append({
            "section": cfg["title"],
            "findings": findings_list,
            "prov_h": cfg["header"],
            "prov_b": cfg["body_template"].format(**data)
        })

    f["dynamic_report"] = report
    return f

def features_dataframe(row, art):
    
    # 1. Mapping for clean ratio names 
    RATIO_MAP = {
        "ET_WT_ratio": "Enhancing Fraction (ET/WT)",
        "ET_TC_ratio": "Enhancing Fraction (ET/TC)",
        "TC_WT_ratio": "Core Fraction (TC/WT)",
        "necrosis_fraction": "Necrotic Fraction (NCR/TC)",
        "edema_index": "Edema Index (ED/TC)"
    }

    # 2. Mapping for units and pretty feature names
    FEATURE_DISPLAY_NAMES = {
        "VoxelVolume": "Voxel Volume",
        "SurfaceVolumeRatio": "Surface Volume Ratio",
        "Sphericity": "Sphericity",
        "Flatness": "Flatness",
        "Elongation": "Elongation",
        "Skewness": "Skewness",
    }

    rows = []
    for f in art["report_features"]:
        v = row.get(f)
        val = round(float(v), 3) if isinstance(v, (int, float)) and v == v else None
        
        region = "Unknown"
        modality = "-"
        fname = f
        
        # LOGIC FOR RATIOS
        if f in art["groups"]["ratio"]:
            region = "Ratio"
            modality = "-"
            fname = RATIO_MAP.get(f, f)
        
        # LOGIC FOR RADIOMIC FEATURES
        else:
            if "_original_" in f:
                try:
                    left, right = f.split("_original_")
                    reg, mod = left.split("_", 1)
                    
                    # Clean up region names
                    region = {"WT": "Whole Tumor", "TC": "Tumor Core", "ET": "Enhancing"}.get(reg, reg)
                    modality = mod.upper()
                    
                    # Extract the feature name
                    raw_feature = right.split("_", 1)[1]
                    
                    # Apply names and units
                    if raw_feature in FEATURE_DISPLAY_NAMES:
                        fname = FEATURE_DISPLAY_NAMES[raw_feature]
                    else:
                        # Fallback: Split CamelCase
                        fname = re.sub(r"(\w)([A-Z])", r"\1 \2", raw_feature)
                except:
                    fname = f
            else:
                fname = f
                
        fname = fname[0].upper() + fname[1:] if fname else fname

        rows.append({
            "Region": region,
            "Modality": modality,
            "Feature": fname,
            "Value": val
        })
        
    return pd.DataFrame(rows)


# ML PIPELINE

def load_model(weights_path, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SwinUNETR(**MODEL_KWARGS)
    ckpt = torch.load(weights_path, map_location=device, weights_only=False)
    if   "state_dict" in ckpt: model.load_state_dict(ckpt["state_dict"])
    elif "model"      in ckpt: model.load_state_dict(ckpt["model"])
    else:                      model.load_state_dict(ckpt)
    return model.to(device).eval()


def preprocess(modality_paths, channel_order):
    """Replicate the Phase 2 validation transforms on the 4 ordered modalities. Returns a [1,4,H,W,D] float tensor and the reference nibabel image."""
   
    ordered = [modality_paths[r] for r in channel_order]
    tfm = Compose([
        LoadImaged(keys="image"),
        EnsureChannelFirstd(keys="image"),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        EnsureTyped(keys="image", data_type="tensor", dtype=torch.float32),
    ])
    data = tfm({"image": ordered})
    img = data["image"].unsqueeze(0)                     # [1,4,H,W,D]
    ref = nib.load(modality_paths[channel_order[0]])    
    return img, ref


def predict_label_map(model, image_tensor, device=None, overlap=None):
    """Sliding-window inference --> label map {0,1,2,4}."""
    if device is None:
        device = next(model.parameters()).device
    ov = SW_OVERLAP if overlap is None else float(overlap)
    with torch.no_grad():
        out = sliding_window_inference(image_tensor.to(device), ROI_SIZE,
                                       SW_BATCH_SIZE, model, overlap=ov)
        pred = (torch.sigmoid(out)[0] > 0.5)       # [3,H,W,D] : 0=TC,1=WT,2=ET
    tc, wt, et = pred[0], pred[1], pred[2]
    lab = torch.zeros(pred.shape[1:], dtype=torch.uint8, device=pred.device)
    lab[wt] = LABEL_ED       # edema base
    lab[tc] = LABEL_NCR       # necrotic core overwrites
    lab[et] = LABEL_ET       # enhancing overwrites
    return lab.cpu().numpy()

def clean_label_map(label_map):
    """Post-process the predicted label map so the overlay, the volumes and the features all reflect the SAME cleaned mask. 
    For each disjoint region (NCR, ED, ET) keep the largest connected component and fill internal holes, then reassemble."""
    cleaned = np.zeros_like(label_map)
    for lab in (LABEL_NCR, LABEL_ED, LABEL_ET):
        binary = (label_map == lab).astype(np.uint8)
        if binary.sum() == 0:
            continue
        # largest connected component
        comp, n = ndimage.label(binary)
        if n > 1:
            sizes = ndimage.sum(binary, comp, range(1, n + 1))
            largest = int(np.argmax(sizes)) + 1
            binary = (comp == largest).astype(np.uint8)
        # fill internal holes
        binary = ndimage.binary_fill_holes(binary).astype(np.uint8)
        cleaned[binary == 1] = lab
    return cleaned

def clean_binary_mask(binary):
    """Post-process a binary ROI: keep the largest connected component and fill internal holes. This removes fragmentation (stray voxels,
    holes) that distorts shape features."""
    if binary.sum() == 0:
        return binary
    # largest connected component
    labeled, n = ndimage.label(binary)
    if n > 1:
        sizes = ndimage.sum(binary, labeled, range(1, n + 1))
        largest = int(np.argmax(sizes)) + 1
        binary = (labeled == largest).astype(np.uint8)
    # fill internal holes
    binary = ndimage.binary_fill_holes(binary).astype(np.uint8)
    return binary

def build_extractor(params_file=None):
    """RadiomicsFeatureExtractor uses Params.yaml if given, else reconstructs the equivalent settings (Original + shape + firstorder)."""
    if params_file and os.path.exists(params_file):
        return featureextractor.RadiomicsFeatureExtractor(params_file)
    ex = featureextractor.RadiomicsFeatureExtractor()
    ex.settings.update(RADIOMICS_SETTINGS)
    ex.disableAllImageTypes(); ex.enableImageTypeByName("Original")
    ex.disableAllFeatures()
    ex.enableFeatureClassByName("shape")
    ex.enableFeatureClassByName("firstorder")
    return ex


def extract_features(label_map, ref_nib, modality_paths, params_file=None):
    """Phase 3 extraction on the predicted mask. Returns the flat feature dict (all shape/firstorder per subregion) before subsetting."""
    extractor = build_extractor(params_file)

    with tempfile.TemporaryDirectory() as td:
        mask_path = os.path.join(td, "pred_mask.nii.gz")
        nib.save(nib.Nifti1Image(label_map.astype(np.uint8), ref_nib.affine, ref_nib.header), mask_path)
        raw = sitk.ReadImage(mask_path)

        feats = {}
        for sub, modality in SUBREGIONS.items():
            arr = sitk.GetArrayFromImage(raw)
            binary = np.isin(arr, SUBREGION_LABELS[sub]).astype(np.uint8)
            binary = clean_binary_mask(binary)          # Phase 3 post-processing
            if int(binary.sum()) < MIN_ROI_VOXELS:
                continue
            bin_sitk = sitk.GetImageFromArray(binary); bin_sitk.CopyInformation(raw)
            try:
                res = extractor.execute(str(modality_paths[modality]), bin_sitk)
            except Exception as e:
                print(f"  [{sub}] extraction failed: {e}")
                continue
            for k, v in res.items():
                if k.startswith("diagnostics_") or isinstance(v, str):
                    continue
                try:
                    vv = float(v)
                except (TypeError, ValueError):
                    continue
                if np.isfinite(vv):
                    feats[f"{sub}_{modality.upper()}_{k}"] = vv
    return feats


# VISUALISATION

def overlay_figure(modality_paths, label_map, findings=None, base_modality="flair"):
    """3-panel summary figure: segmentation overlay + composition donut + nested bars."""
    matplotlib.use("Agg")
    try:
        import report_figures as rf
    except Exception:
        rf = None

    base = nib.load(modality_paths[base_modality]).get_fdata()

    if rf is not None and findings is not None:
        v = findings["size_cm3"]
        et  = v.get("enhancing_cm3") or 0.0
        ncr = v.get("necrotic_cm3") or 0.0
        ed  = v.get("edema_cm3") or 0.0
        return rf.figure_unified_summary(base, label_map, et, ncr, ed,
                                         pid=findings.get("patient_id", ""))

    # fallback report_figures unavailable
    seg = label_map
    tps = (seg > 0).sum(axis=(0, 1))
    z = int(tps.argmax()) if tps.sum() > 0 else seg.shape[2] // 2
    remap = np.zeros_like(seg, dtype=np.uint8)
    remap[seg == LABEL_NCR] = 1; remap[seg == LABEL_ED] = 2; remap[seg == LABEL_ET] = 3
    cmap = ListedColormap([(0,0,0,0),(0.20,0.40,1.0,0.55),(1.0,0.85,0.0,0.45),(1.0,0.15,0.15,0.6)])
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(np.rot90(base[:, :, z]), cmap="gray")
    ax.imshow(np.rot90(remap[:, :, z]), cmap=cmap, vmin=0, vmax=3, interpolation="nearest")
    ax.set_title(f"Predicted segmentation (slice {z})"); ax.axis("off")
    return fig


# ORCHESTRATOR

def pipeline_config(overlap=None):
    """The pipeline settings for the 'config' panel."""
    return {
        "model": f"SwinUNETR(in={MODEL_KWARGS['in_channels']}, out={MODEL_KWARGS['out_channels']}, "
                 f"feature_size={MODEL_KWARGS['feature_size']})  weights: fine_tuning_best_model.pt",
        "preprocessing": "z-score on non-zero voxels, channel-wise; no resampling (BraTS is 1 mm); "
                         "no skull-strip / N4 (already preprocessed)",
        "inference": f"sliding-window roi={ROI_SIZE}, overlap={overlap if overlap is not None else SW_OVERLAP}, "
                     "sigmoid>0.5 -> labels {1:NCR, 2:ED, 4:ET}",
        "post_processing": "largest connected component + fill holes",
        "radiomics": f"PyRadiomics Original image, binWidth={RADIOMICS_SETTINGS['binWidth']}, "
                     "normalize=False; ET/TC on T1ce, WT on FLAIR",
        "interpretation": "Phase 4 structured description (sizes in cm3, cited composition/morphology bands)",
    }


def run_inference(input_dir, project_root=None, weights_path=None, feature_dir=None,
                  folds_json=None, params_file=None,
                  output_dir=None, make_figure=True, overlap=None, progress_cb=None):
    """Full pipeline for one patient. Returns a dict with mask, features, structured findings (Phase 4) with provenance and a summary figure."""
    # All assets bundled with the app. The user only supplies the MRI folder.
    feature_dir = feature_dir  or ASSETS_DIR
    weights_path = weights_path or os.path.join(ASSETS_DIR, "fine_tuning_best_model.pt")
    folds_json = folds_json   or os.path.join(ASSETS_DIR, "brats21_folds.json")
    params_file = params_file  or os.path.join(ASSETS_DIR, "Params.yaml")
    ov = SW_OVERLAP if overlap is None else float(overlap)

    total = 7 + (1 if make_figure else 0)
    def step(i, msg):
        if progress_cb:
            progress_cb(i, total, msg)

    patient_id = os.path.basename(os.path.normpath(input_dir))
    if output_dir is None:
        output_dir = os.path.join(input_dir, "neuroseg_output")
    step(1, "Loading frozen feature set (Phase 3)")
    art = load_artifacts(feature_dir)
    step(2, "Resolving the 4 MRI modalities")
    modality_paths = resolve_modalities(input_dir)
    channel_order, detected = detect_channel_order(folds_json)

    step(3, "Loading the segmentation model")
    model = load_model(weights_path)
    step(4, "Preprocessing (z-score normalisation)")
    image_tensor, ref = preprocess(modality_paths, channel_order)
    step(5, "Segmenting (sliding-window inference)")
    label_map = predict_label_map(model, image_tensor, overlap=ov)
    label_map = clean_label_map(label_map)  

    step(6, "Extracting radiomic features")
    raw_feats = extract_features(label_map, ref, modality_paths, params_file)
    ratios = compute_ratios(raw_feats)
    row = {f: raw_feats.get(f) for f in art["report_features"]}
    row.update(ratios)
    
    for vk in (ET_VOL, TC_VOL, WT_VOL):
        if vk in raw_feats:
            row[vk] = raw_feats[vk]
    row["patient_id"] = patient_id

    step(7, "Building structured clinical description (Phase 4)")
    findings = describe_patient(row, art["groups"])
    findings = add_provenance(findings)
    table = features_dataframe(row, art)

    result = {"patient_id": patient_id, "modality_paths": modality_paths,
              "label_map": label_map, "ref": ref, "features_row": row,
              "features_table": table, "findings": findings,
              "channel_order": channel_order, "channel_order_detected": detected}

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        nib.save(nib.Nifti1Image(label_map.astype(np.uint8), ref.affine, ref.header),
                 os.path.join(output_dir, f"pred_{patient_id}.nii.gz"))
        with open(os.path.join(output_dir, f"findings_{patient_id}.json"), "w") as f:
            json.dump(findings, f, indent=2)
        table.to_csv(os.path.join(output_dir, f"features_{patient_id}.csv"), index=False)

    if make_figure:
        step(8, "Rendering the summary figure")
        result["figure"] = overlay_figure(modality_paths, label_map, findings=findings)
        if output_dir:
            result["figure"].savefig(os.path.join(output_dir, f"summary_{patient_id}.png"),
                                     dpi=150, bbox_inches="tight")
    return result
