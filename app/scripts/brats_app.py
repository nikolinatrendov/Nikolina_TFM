"""
brats_app.py

PyQt5 interface for the brain-tumor inference pipeline.

  1: Select patient MRI folder (the 4 modalities).
  2:  Run segmentation and analysis.

Inference runs on a background QThread. The result page shows the predicted segmentation, the 3-panel summary figure, the structured Phase-4 findings with
provenance, a "Learn more" explainer (Phase 5) that runs ON DEMAND on a local model and a "Pipeline Config" window.
"""

# Imports
import os
import re
import sys
import json
import numpy as np
import nibabel as nib
import brats_inference as engine
import traceback
import requests
from io import BytesIO

from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QLineEdit, QPushButton, QVBoxLayout, QHeaderView,
    QHBoxLayout, QFrame, QStackedWidget, QTextEdit, QScrollArea, QFileDialog, QStyle, QStyleOptionSlider,
    QMessageBox, QTableView, QTabWidget, QProgressBar, QCheckBox, QSlider, QComboBox, 
)
from PyQt5.QtGui import (
    QPixmap, QFont, QStandardItemModel, QStandardItem, QImage, QDesktopServices,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSettings, QUrl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))   # app/scripts/
APP_DIR = os.path.dirname(SCRIPT_DIR)                      # app/
REFERENCES_DIR = os.path.join(APP_DIR, "references")       # app/references/
if not os.path.exists(REFERENCES_DIR):
    print(f"WARNING: references folder not found at {REFERENCES_DIR}")
_RAG_READY = False

# Configuration
# RAG settings
try:
    import explainer_core as ec        
except Exception:
    ec = None
try:
    import rag_explainer as rag        
except Exception:
    rag = None

# Units per feature (matched by the feature name)
FEATURE_UNITS = {
    "VoxelVolume": "mm³", "Voxel Volume": "mm³",
    "Flatness": "0–1", "Elongation": "0–1", "Sphericity": "0–1",
    "SurfaceVolumeRatio": "mm⁻¹", "Surface Volume Ratio": "mm⁻¹",
    "Skewness": "unitless",
    "TC_WT_ratio": "fraction", "ET_WT_ratio": "fraction",
    "ET_TC_ratio": "fraction", "necrosis_fraction": "fraction",
    "Core Fraction": "fraction", "Enhancing Fraction": "fraction",
    "Necrotic Fraction": "fraction", "Active Core Fraction": "fraction",
}

LOCAL_MODEL = "llama3:8b"

def _unit_for(feature_name):
    name = str(feature_name)
    for key, unit in FEATURE_UNITS.items():
        if key.lower() in name.lower():
            return unit
    return ""


# Colours & image helpers
CLASS_COLORS = {engine.LABEL_NCR: (0, 102, 255), engine.LABEL_ED:  (255, 229, 0), engine.LABEL_ET:  (255, 0, 0)}
CLASS_NAMES = {engine.LABEL_NCR: "Necrotic core (NCR)", engine.LABEL_ED:  "Edema (ED)", engine.LABEL_ET:  "Enhancing (ET)"}
QUALITY = {"Fast (overlap 0.25)": 0.25, "Balanced (overlap 0.5)": 0.5, "Accurate, slower (overlap 0.7)": 0.7}


def normalize_u8(vol):
    v = vol.astype(np.float32)
    lo, hi = np.percentile(v, 1), np.percentile(v, 99)
    if hi <= lo:
        lo, hi = float(v.min()), float(v.max())
    if hi <= lo:
        return np.zeros(vol.shape, np.uint8)
    v = np.clip((v - lo) / (hi - lo), 0, 1)
    return (v * 255).astype(np.uint8)


def composite_rgb(base_u8_2d, label_2d, show, opacity):
    rgb = np.stack([base_u8_2d] * 3, axis=-1).astype(np.float32)
    for cls, (r, g, b) in CLASS_COLORS.items():
        if not show.get(cls, True):
            continue
        m = label_2d == cls
        if not m.any():
            continue
        rgb[m] = (1 - opacity) * rgb[m] + opacity * np.array([r, g, b], np.float32)
    return np.clip(rgb, 0, 255).astype(np.uint8)

def rgb_to_qpixmap(rgb):
    rgb = np.ascontiguousarray(rgb)
    h, w, _ = rgb.shape
    return QPixmap.fromImage(QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy())

def dataframe_to_table(df):
    if 'ICC' in df.columns and df['ICC'].isna().all():
        df = df.drop(columns=['ICC'])

    df = df.copy()
    # add a Unit column based on the feature column
    feat_col = "feature" if "feature" in df.columns else df.columns[2]
    df["Unit"] = df[feat_col].apply(_unit_for)

    model = QStandardItemModel()
    model.setColumnCount(len(df.columns))
    model.setHorizontalHeaderLabels([str(c).title() for c in df.columns])

    for row in df.itertuples(index=False):
        items = [QStandardItem("" if v is None else str(v)) for v in row]
        for it in items:
            it.setEditable(False)
            it.setTextAlignment(Qt.AlignCenter)
        model.appendRow(items)

    t = QTableView()
    t.setModel(model)
    t.setAlternatingRowColors(True)
    t.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
    t.verticalHeader().setVisible(False)
    t.setFrameShape(QFrame.NoFrame)
    t.setStyleSheet("QTableView{font-family:'Segoe UI';font-size:13px;background:#ffffff;color:#1a2733;"
                    "alternate-background-color:#f8faff;gridline-color:#cfe0f5;border:1px solid #cfe0f5;border-radius:8px;}"
                    "QHeaderView::section{background:#dce8f6;color:#1f3b57;padding:8px;border:none;font-weight:700;}")
    return t

def scan_modalities(folder):
    found = {}
    try:
        for fn in sorted(os.listdir(folder)):
            if fn.lower().endswith((".nii", ".nii.gz")):
                role = engine.modality_token(fn)
                if role and role not in found:
                    found[role] = fn
    except OSError:
        pass
    return found

def detect_format(folder):
    toks = set()
    try:
        for fn in os.listdir(folder):
            fl = fn.lower()
            if not fl.endswith((".nii", ".nii.gz")):
                continue
            base = re.sub(r"\.nii(\.gz)?$", "", fl)
            toks.add(re.split(r"[_\-]", base)[-1])
    except OSError:
        return "unknown"
    if toks & {"t1ce", "flair"}:
        return "BraTS2021"
    return "unknown"

def humanize_error(message):
    m = (message or "").lower()
    if "05_report_features_final" in m or "feature" in m:
        return "Feature list not found in assets/. Make sure 05_report_features_final.json is there."
    if "fine_tuning_best_model" in m or "weights" in m or ".pt" in m:
        return "Model weights not found in assets/. Make sure fine_tuning_best_model.pt is there."
    if "modalit" in m:
        return "One or more MRI modalities are missing or mis-named in the patient folder."
    if "cuda" in m or "memory" in m:
        return "GPU memory error. Try again, or run on CPU."
    return "Inference failed. See the details below."


# Workers
class InferenceWorker(QThread):
    progress = pyqtSignal(int, int, str)
    finished_ok = pyqtSignal(object)
    failed = pyqtSignal(str, str)

    def __init__(self, input_dir, output_dir, overlap):
        super().__init__()
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.overlap = overlap

    def run(self):
        def cb(i, total, msg):
            self.progress.emit(i, total, msg)
        try:
            kw = dict(output_dir=self.output_dir, make_figure=True,
                      overlap=self.overlap, progress_cb=cb)
            res = engine.run_inference(self.input_dir, **kw)
            res.setdefault("dice", None); res.setdefault("gt_label_map", None)
            try:
                res["base_volume"] = nib.load(res["modality_paths"]["flair"]).get_fdata()
            except Exception:
                res["base_volume"] = None
            res["output_dir"] = self.output_dir
            res["overlap"] = self.overlap
            self.finished_ok.emit(res)
        except Exception as e:
            self.failed.emit(humanize_error(str(e)), f"{e}\n\n{traceback.format_exc()}")

class PreviewWorker(QThread):
    ready = pyqtSignal(object)

    def __init__(self, flair_path):
        super().__init__()
        self.flair_path = flair_path

    def run(self):
        try:
            vol = nib.load(self.flair_path).get_fdata()
            u8 = normalize_u8(vol)
            self.ready.emit(np.rot90(u8[:, :, u8.shape[2] // 2]))
        except Exception:
            self.ready.emit(None)

class ExplainWorker(QThread):
    """RAG explainer for retrieval-augmented explanation grounded in local references."""
    done = pyqtSignal(str)

    def __init__(self, findings, term_key=None):
        super().__init__()
        self.findings = findings
        self.term_key = term_key

    def _term_value(self):
        """Build a descriptive query for retrieval from the clicked term."""
        f = self.findings or {}
        if self.term_key == "necrosis_fraction":
            return f"necrosis level: {f.get('composition', {}).get('necrosis_category', '')}"
        if self.term_key == "enhancing_fraction_of_wt":
            return f"enhancing fraction: {f.get('composition', {}).get('enhancing_category', '')}"
        if self.term_key == "margin_descriptor":
            return f"margins: {f.get('morphology', {}).get('margin_descriptor', '')}"
        if self.term_key == "size_cm3":
            return f"tumor size: {f.get('size_cm3', {}).get('whole_tumor_cm3', '')} cm3"
        return "overall tumor composition and characteristics"

    def run(self):
        def generate_fn(system, user):
            r = requests.post("http://localhost:11434/api/chat", json={
                "model": LOCAL_MODEL, "stream": False,
                "options": {"temperature": 0.0},
                "messages": [{"role": "system", "content": system},
                             {"role": "user", "content": user}],
            }, timeout=120)
            r.raise_for_status()
            return r.json()["message"]["content"].strip()

        query = self._term_value()

        # RAG
        if rag is None or not os.path.isdir(REFERENCES_DIR):
            self.done.emit("Could not retrieve explanation: the references folder "
                           "or RAG module is unavailable.")
            return

        try:
            global _RAG_READY
            if not _RAG_READY:
                rag.build_index(REFERENCES_DIR)
                _RAG_READY = True
            context = ec._strip_identifiers(self.findings) if ec else self.findings
            text = rag.explain_with_rag(query, context, generate_fn, k=3)
            self.done.emit(text)
        except Exception as e:
            self.done.emit(f"Could not retrieve the requested explanation.\n\n"
                           f"(Reason: {e})\n\n"
                           f"Check that Ollama is running and the references folder has .txt files.")

# Small UI helpers
def step_header(text):
    lab = QLabel(text)
    lab.setStyleSheet("color:#2f6fb2; font-weight:700; font-size:12px; background:transparent;")
    return lab


# PAGE 1: input page
class WelcomePage(QWidget):
    """Single-step input. The user picks the patient MRI folder."""
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.settings = QSettings("NikolinaTFM", "NeuroSeg")
        self._preview_path = None
        self.setObjectName("welcomeRoot")
        self.setAcceptDrops(True)
        self.setStyleSheet("""
            #welcomeRoot { background: qlineargradient(x1:0,y1:0,x2:1,y2:1,
                           stop:0 #d9ecff, stop:1 #f4f9ff); }
            QLabel { background: transparent; }
        """)

        # The main title
        title = QLabel("Brain Tumor Segmentation and Analysis")
        title.setFont(QFont("Segoe UI", 20, QFont.Bold)) 
        title.setStyleSheet("color: #1a3a5a; background: transparent;") 
        title.setAlignment(Qt.AlignCenter)

        # Optional: Add a version or subtitle for a "Clinical App" look
        howto = QLabel("Select a patient MRI folder (the 4 modalities), then run.")
        howto.setStyleSheet("color:#5a6b7a; background:transparent; font-size:12px;")
        howto.setAlignment(Qt.AlignCenter); howto.setWordWrap(True)

        # Patient folder is the only input
        self.patient_input = QLineEdit()
        self.patient_input.setPlaceholderText("Patient folder  (one patient's 4 .nii.gz files)")
        self.patient_input.setMinimumWidth(380)
        pat_btn = QPushButton("Browse"); pat_btn.clicked.connect(self.pick_patient)
        pat_btn.setStyleSheet("QPushButton{background:#92d4f6;color:#1f5b86;border-radius:8px;padding:9px 14px;}"
                              "QPushButton:hover{background:#7cc6ef;}")
        self.patient_status = QLabel(""); self.patient_status.setStyleSheet("font-size:11px;background:transparent;")
        self.format_label = QLabel(""); self.format_label.setStyleSheet("font-size:11px;color:#3a6ea5;background:transparent;")
        self.mapping = QLabel("")
        self.mapping.setTextFormat(Qt.RichText)
        self.mapping.setStyleSheet("""
                    QLabel {
                        background: #ffffff; 
                        border: 2px solid #dce8f6; 
                        border-radius: 12px;
                        padding: 15px; 
                        font-size: 13px; 
                        color: #1f3b57;  /* Professional Deep Navy Blue */
                    }
                """)
        self.mapping.setMinimumWidth(300)
        self.preview = QLabel("preview"); self.preview.setFixedSize(140, 140)
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setStyleSheet("background:#0d1b2a;border-radius:8px;color:#789;")
        map_row = QHBoxLayout(); map_row.addWidget(self.mapping, 1); map_row.addWidget(self.preview, 0)
        self._set_border(self.patient_input, "neutral")
        self.patient_input.textChanged.connect(self._on_patient_changed)
        self.patient_input.returnPressed.connect(self.go_run)

        # Assets status
        self.assets_label = QLabel(""); self.assets_label.setWordWrap(True)
        self.assets_label.setStyleSheet("font-size:11px;background:transparent;")
        self._check_assets()

        # Quality & run
        self.quality = QComboBox(); self.quality.addItems(list(QUALITY.keys()))
        self.quality.setCurrentText("Fast (overlap 0.25)")
        qrow = QHBoxLayout()
        qrow.addWidget(QLabel("Inference quality:")); qrow.addWidget(self.quality, 1)

        self.run_btn = QPushButton("Run segmentation and analysis")
        self.run_btn.setFixedSize(260, 44); self.run_btn.setEnabled(False); self._style_run(False)
        self.run_btn.clicked.connect(self.go_run)

        self.progress = QProgressBar(); self.progress.setFixedHeight(10); self.progress.hide()
        self.status = QLabel(""); self.status.setAlignment(Qt.AlignCenter); self.status.setWordWrap(True)
        self.status.setStyleSheet("color:#2f6fb2;font-size:12px;background:transparent;")

        self.log_btn = QPushButton("Show log")
        self.log_btn.setStyleSheet("QPushButton{background:transparent;color:#3a6ea5;border:none;font-size:11px;}")
        self.log_btn.clicked.connect(self._toggle_log)
        self.log = QTextEdit(); self.log.setReadOnly(True); self.log.setFixedHeight(110); self.log.hide()
        self.log.setStyleSheet("QTextEdit{background:#0d1b2a;color:#cfe3ff;border-radius:6px;"
                               "font-family:Consolas;font-size:11px;padding:6px;}")

        def field_row(line, btn):
            h = QHBoxLayout(); h.addWidget(line); h.addWidget(btn); return h

        card = QFrame()
        card.setObjectName("card")
        card.setStyleSheet("#card{background:white;border-radius:20px; border: 1px solid #dce8f6;}")
        card.setFixedWidth(650) # Updated to fit your new longer title
        
        cl = QVBoxLayout(card)
        cl.setContentsMargins(40, 30, 40, 30) # More breathing room on sides
        cl.setSpacing(18) # More vertical space between elements
        cl.addWidget(title); cl.addWidget(howto); cl.addSpacing(6)
        cl.addWidget(step_header("PATIENT MRI FOLDER"))
        cl.addLayout(field_row(self.patient_input, pat_btn))
        cl.addWidget(self.patient_status); cl.addWidget(self.format_label)
        cl.addLayout(map_row)
        cl.addWidget(self.assets_label)
        cl.addSpacing(6)
        cl.addLayout(qrow)
        cl.addWidget(self.run_btn, alignment=Qt.AlignCenter)
        cl.addWidget(self.progress); cl.addWidget(self.status)
        cl.addWidget(self.log_btn, alignment=Qt.AlignLeft); cl.addWidget(self.log)

        scroll = QScrollArea(); scroll.setWidgetResizable(True); scroll.setFrameShape(QFrame.NoFrame)
        holder = QWidget(); hl = QVBoxLayout(holder)
        hl.addStretch(); hl.addWidget(card, alignment=Qt.AlignCenter); hl.addStretch()
        holder.setStyleSheet("background:transparent;")
        scroll.setWidget(holder)
        outer = QVBoxLayout(self); outer.setContentsMargins(0, 0, 0, 0); outer.addWidget(scroll)

        self._on_patient_changed()

    def _check_assets(self):
        """Inform the user if a bundled asset is missing (developer-facing)."""
        adir = engine.ASSETS_DIR
        need = {"fine_tuning_best_model.pt": "model weights",
                "05_report_features_final.json": "feature list"}
        missing = [name for name in need if not os.path.exists(os.path.join(adir, name))]
        if missing:
            self.assets_label.setText(
                "<span style='color:#c0392b'>&#9888; Missing app assets in assets/: "
                + ", ".join(missing) + ". The app cannot run until these are added.</span>")
        else:
            self.assets_label.setText("<span style='color:#2e8b57'>&#10003; App assets found.</span>")

    def _style_run(self, enabled):
            if enabled:
                # Blue when ready
                self.run_btn.setStyleSheet("""
                    QPushButton {
                        background: #2f6fb2; color: white; border-radius: 10px; 
                        font-size: 15px; font-weight: bold;
                    }
                    QPushButton:hover { background: #1f5b86; }
                """)
            else:
                # Gray when disabled
                self.run_btn.setStyleSheet("""
                    QPushButton {
                        background: #e0e6ed; color: #a0acba; border-radius: 10px; 
                        font-size: 15px;
                    }
                """)

    def _set_border(self, widget, state):
        col = {"ok": "#3ac06a", "bad": "#e06a6a", "neutral": "#aacdf2"}[state]
        widget.setStyleSheet(f"QLineEdit{{padding:9px;border-radius:8px;border:2px solid {col};"
                             f"font-size:13px;background:#fff;color:#222;}}QLineEdit:focus{{border:2px solid #5aa3e8;}}")

    def pick_patient(self):
        d = QFileDialog.getExistingDirectory(self, "Select patient folder",
                                             self.settings.value("last_patient_parent", "", str))
        if d: self.patient_input.setText(d)

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls(): e.acceptProposedAction()

    def dropEvent(self, e):
        for url in e.mimeData().urls():
            p = url.toLocalFile()
            if os.path.isdir(p):
                self.patient_input.setText(p); break

    def _on_patient_changed(self):
        folder = self.patient_input.text().strip()
        if folder and os.path.isdir(folder):
            found = scan_modalities(folder)
            self.format_label.setText(f"Detected format: {detect_format(folder)}")
            rows = []
            for role, lbl in [("t1ce", "T1ce"), ("t1", "T1"), ("t2", "T2"), ("flair", "FLAIR")]:
                if role in found:
                    rows.append(f"<span style='color:#2e8b57'>&#10003; {lbl}: {found[role]}</span>")
                else:
                    rows.append(f"<span style='color:#c0392b'>&#10007; {lbl}: missing</span>")
            self.mapping.setText("<b>Modality &rarr; file</b><br>" + "<br>".join(rows))
            missing = [r for r in ("t1", "t1ce", "t2", "flair") if r not in found]
            if missing:
                self._set_border(self.patient_input, "bad")
                self.patient_status.setText(f"<span style='color:#c0392b'>&#10007; Expected 4 modalities, found "
                                            f"{4 - len(missing)}: missing {', '.join(missing)}</span>")
            else:
                self._set_border(self.patient_input, "ok")
                self.patient_status.setText("<span style='color:#2e8b57'>&#10003; Valid patient folder</span>")
                fpath = os.path.join(folder, found["flair"])
                if fpath != self._preview_path:
                    self._preview_path = fpath; self._load_preview(fpath)
        else:
            self.mapping.setText(""); self.format_label.setText(""); self.patient_status.setText("")
            self.preview.setText("preview"); self.preview.setPixmap(QPixmap()); self._preview_path = None
            self._set_border(self.patient_input, "neutral" if not folder else "bad")
        self._refresh_run()

    def _refresh_run(self):
        folder = self.patient_input.text().strip()
        assets_ok = all(os.path.exists(os.path.join(engine.ASSETS_DIR, n))
                        for n in ("fine_tuning_best_model.pt", "05_report_features_final.json"))
        ok = (bool(folder) and os.path.isdir(folder)
              and all(r in scan_modalities(folder) for r in ("t1", "t1ce", "t2", "flair"))
              and assets_ok)
        self.run_btn.setEnabled(ok); self._style_run(ok)

    def _load_preview(self, path):
        self.preview.setText("loading...")
        self._pw = PreviewWorker(path); self._pw.ready.connect(self._preview_ready); self._pw.start()

    def _preview_ready(self, slice2d):
        if slice2d is None:
            self.preview.setText("preview\n(n/a)"); return
        pix = rgb_to_qpixmap(np.stack([slice2d] * 3, axis=-1)).scaled(140, 140, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.preview.setPixmap(pix)

    def _toggle_log(self):
        self.log.setVisible(not self.log.isVisible())
        self.log_btn.setText("Hide log" if self.log.isVisible() else "Show log")

    def go_run(self):
        if not self.run_btn.isEnabled():
            return
        patient = self.patient_input.text().strip()
        self.settings.setValue("last_patient_parent", os.path.dirname(os.path.normpath(patient)))

        self.run_btn.setEnabled(False); self._style_run(False)
        self.progress.show(); self.progress.setRange(0, 8); self.progress.setValue(0)
        self.status.setText("Starting..."); self.log.clear()
        QApplication.setOverrideCursor(Qt.WaitCursor)

        out_dir = os.path.join(APP_DIR, "outputs", os.path.basename(os.path.normpath(patient)))
        overlap = QUALITY[self.quality.currentText()]
        self.worker = InferenceWorker(patient, out_dir, overlap)
        self.worker.progress.connect(self._on_progress)
        self.worker.finished_ok.connect(self._done)
        self.worker.failed.connect(self._fail)
        self.worker.start()

    def _on_progress(self, i, total, msg):
        self.progress.setRange(0, total); self.progress.setValue(i)
        self.status.setText(f"Step {i}/{total}: {msg}")
        self.log.append(f"[{i}/{total}] {msg}")

    def _done(self, result):
        QApplication.restoreOverrideCursor()
        self.progress.setValue(self.progress.maximum())
        self.status.setText("Done."); self.log.append("Done.")
        self._refresh_run()
        if not result.get("channel_order_detected", True):
            QMessageBox.warning(self, "Verify channel order",
                f"Fallback modality order {result['channel_order']} was used (no folds JSON). "
                "Verify it matches training, or predictions may be wrong.")
        self.parent.result_page.show_result(result)
        self.parent.stack.setCurrentIndex(1)

    def _fail(self, friendly, details):
        QApplication.restoreOverrideCursor()
        self.progress.hide(); self.status.setText(""); self._refresh_run()
        self.log.append("ERROR:\n" + details)
        if not self.log.isVisible():
            self._toggle_log()
        box = QMessageBox(self); box.setIcon(QMessageBox.Critical)
        box.setWindowTitle("Inference failed"); box.setText(friendly); box.setDetailedText(details)
        copy_btn = box.addButton("Copy error details", QMessageBox.ActionRole)
        box.addButton(QMessageBox.Close)
        box.exec_()
        if box.clickedButton() is copy_btn:
            QApplication.clipboard().setText(details)


# PAGE 2: output page
class ResultPage(QWidget):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.setObjectName("resultRoot")
        self.setStyleSheet("#resultRoot{background:#1a1a1a;} QLabel{background:transparent;}")
        
        self._base_u8 = None
        self._label = None
        self._out_dir = None
        self._disp_w = 720
        self._findings = None
        self._rendering = False
        self.wheel_accumulator = 0 

        self.title = QLabel("Result")
        self.title.setFont(QFont("Segoe UI", 18, QFont.Bold))
        self.title.setStyleSheet("color:#ffffff; padding: 10px;")
        self.title.setAlignment(Qt.AlignCenter)

        # Tab widget setup
        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_viewer_tab(), "Prediction")
        self.tabs.addTab(self._build_findings_tab(), "Description")
        
        self.table_holder = QVBoxLayout()
        feat_tab = QWidget(); feat_tab.setLayout(self.table_holder)
        self.tabs.addTab(feat_tab, "Features")
        self.tabs.addTab(self._build_explain_tab(), "Learn more")

        # Config tab setup
        self.config_box = QTextEdit()
        self.config_box.setReadOnly(True)
        self.config_box.setFixedWidth(850)     
        self.config_box.setMinimumHeight(600)   
        
        cfg_tab = QWidget()
        cfgl = QVBoxLayout(cfg_tab)
        
        # Horizontal centering
        hbox = QHBoxLayout()
        hbox.addStretch(1)
        hbox.addWidget(self.config_box)
        hbox.addStretch(1)
        
        # Vertical centering
        cfgl.addStretch(1)
        cfgl.addLayout(hbox)
        cfgl.addStretch(1)
        
        self.tabs.addTab(cfg_tab, "Pipeline config")

        # Footer
        self.out_label = QLabel("")
        self.out_label.setStyleSheet("color: #888;")
        open_btn = QPushButton("Open folder"); open_btn.clicked.connect(self._open_output)
        back_btn = QPushButton("Back"); back_btn.clicked.connect(lambda: self.parent.stack.setCurrentIndex(0))
        
        footer = QHBoxLayout()
        footer.addWidget(self.out_label, 1)
        footer.addWidget(open_btn); footer.addWidget(back_btn)

        layout = QVBoxLayout(self)
        layout.addWidget(self.title)
        layout.addWidget(self.tabs)
        layout.addLayout(footer)

    def slider_click_jump(self, event):
        """Only allows movement if the user grabs the actual handle."""
        
        opt = QStyleOptionSlider()
        self.slice_slider.initStyleOption(opt)
        
        handle_rect = self.slice_slider.style().subControlRect(
            QStyle.CC_Slider, opt, QStyle.SC_SliderHandle, self.slice_slider
        )
        
        if handle_rect.contains(event.pos()):
            type(self.slice_slider).mousePressEvent(self.slice_slider, event)
        else:
            event.accept()

    def wheelEvent(self, event):
        if self.tabs.currentIndex() == 0 and hasattr(self, 'slice_slider'):
            event.accept()
            self.wheel_accumulator += event.angleDelta().y()
            threshold = 40 
            if abs(self.wheel_accumulator) >= threshold:
                steps = int(self.wheel_accumulator / threshold)
                self.wheel_accumulator -= (steps * threshold)
                current = self.slice_slider.value()
                new_val = max(self.slice_slider.minimum(), min(current + steps, self.slice_slider.maximum()))
                if new_val != current:
                    self.slice_slider.setValue(new_val)

    def _build_viewer_tab(self):
        tab = QWidget()
        tab.setStyleSheet("background-color: #000000; color: #ffffff;")
        v = QVBoxLayout(tab)

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background-color: #000000; border: none;")
        
        sc = QScrollArea()
        sc.setWidgetResizable(True)
        sc.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        sc.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        sc.setStyleSheet("background-color: #000000; border: none;") 
        sc.setWidget(self.image_label)
        v.addWidget(sc, 1)

        control_panel = QFrame()
        control_panel.setStyleSheet("background: #1e1e1e; border-radius: 12px; padding: 10px;")
        cp_layout = QVBoxLayout(control_panel)

        srow = QHBoxLayout()
        srow.setContentsMargins(10, 0, 10, 0)
        self.slice_slider = QSlider(Qt.Horizontal)
        self.slice_slider.setTracking(True)
        self.slice_slider.mousePressEvent = self.slider_click_jump
        self.slice_slider.valueChanged.connect(self._render)

        self.slice_label = QLabel("Slice: - / -")
        self.slice_label.setFixedWidth(140)
        self.slice_label.setStyleSheet("color: #ffffff; font-weight: bold;")
        self.slice_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        
        slice_icon = QLabel("SLICE")
        slice_icon.setStyleSheet("font-weight: bold; color: #5aa3e8;")
        
        srow.addWidget(slice_icon)
        srow.addWidget(self.slice_slider, 1)
        srow.addWidget(self.slice_label)
        cp_layout.addLayout(srow)

        crow = QHBoxLayout()
        self.cb = {} 
        for cls in [engine.LABEL_NCR, engine.LABEL_ED, engine.LABEL_ET]:
            box = QCheckBox(CLASS_NAMES[cls])
            box.setChecked(True)
            box.stateChanged.connect(self._render)
            r, g, b = CLASS_COLORS[cls]
            box.setStyleSheet(f"QCheckBox{{color: #bbb;}} QCheckBox::indicator{{background-color: rgb({r},{g},{b}); width: 14px; height: 14px;}}")
            self.cb[cls] = box
            crow.addWidget(box)
        
        crow.addStretch(1)
        crow.addWidget(QLabel("OPACITY"))
        self.opacity = QSlider(Qt.Horizontal)
        self.opacity.setRange(0, 100); self.opacity.setValue(50); self.opacity.setFixedWidth(120)
        self.opacity.valueChanged.connect(self._render)
        crow.addWidget(self.opacity)
        
        cp_layout.addLayout(crow)
        v.addWidget(control_panel)
        return tab

    def _build_findings_tab(self):
        """Structured Phase-4 findings & the summary figure & provenance."""
        tab = QWidget(); v = QVBoxLayout(tab)
        
        # Top half: The Figure
        self.fig_label = QLabel(); self.fig_label.setAlignment(Qt.AlignCenter)
        self.fig_label.setStyleSheet("background:white; border:1px solid #dce8f6; border-radius:8px;")
        sc = QScrollArea(); sc.setWidgetResizable(True); sc.setWidget(self.fig_label)
        sc.setMinimumHeight(450)
        v.addWidget(sc, 3)
        
        # Bottom half: The Description Box
        self.findings_box = QTextEdit(); self.findings_box.setReadOnly(True)
        
        self.findings_box.setAcceptRichText(True)
        self.findings_box.setStyleSheet("""
            QTextEdit {
                background-color: #ffffff;
                color: #1a2733;
                border: 1px solid #dce8f6;
                border-radius: 15px;
                padding: 25px;
            }
        """)
        v.addWidget(self.findings_box, 2)
        return tab

    def _build_explain_tab(self):
        tab = QWidget(); v = QVBoxLayout(tab)
        v.addWidget(QLabel("<b>Research-Based Explanation (RAG)</b>"))
        self.rag_btn = QPushButton("Generate Research Analysis")
        self.rag_btn.clicked.connect(self._run_rag_explanation)
        self.rag_btn.setStyleSheet("background: #2f6fb2; color: white; padding: 10px; border-radius: 8px;")
        v.addWidget(self.rag_btn)
        self.rag_output_box = QTextEdit(); self.rag_output_box.setReadOnly(True)
        self.rag_output_box.setStyleSheet("background: white; border-radius:15px; padding:25px;")
        v.addWidget(self.rag_output_box)
        return tab

    def _run_rag_explanation(self):
        if self._findings is None:
            self.rag_output_box.setHtml("<p style='color:#c0392b;'>No findings loaded — run inference on a patient first.</p>")
            return
        self.rag_output_box.setHtml(
            "<div style='color:#2f6fb2; font-size:14px;'>"
            "<i>Searching local references and generating explanation… "
            "this may take around 10 seconds.</i></div>"
        )
        self.rag_btn.setEnabled(False)
        self.rag_btn.setText("Generating…")
        QApplication.processEvents()  
        self._rag_worker = ExplainWorker(self._findings, None)
        self._rag_worker.done.connect(self._on_rag_finished)
        self._rag_worker.start()

    def _on_rag_finished(self, text):
        safe = text.replace(chr(10), "<br>")
        self.rag_output_box.setHtml(f"<div style='color:#1a2733; line-height:1.6;'>{safe}</div>")
        self.rag_btn.setEnabled(True)
        self.rag_btn.setText("Generate Research Analysis")  

    def _render(self):
        if getattr(self, '_rendering', False) or self._base_u8 is None: return
        self._rendering = True
        try:
            z = self.slice_slider.value()
            self.slice_label.setText(f"Slice: {z:03} / {self._base_u8.shape[2]-1:03}")
            base2d = np.rot90(self._base_u8[:, :, z])
            lab2d = np.rot90(self._label[:, :, z])
            show = {cls: self.cb[cls].isChecked() for cls in self.cb}
            rgb = composite_rgb(base2d, lab2d, show, self.opacity.value() / 100.0)
            pix = rgb_to_qpixmap(rgb)
            # Scaling logic
            parent = self.image_label.parent()
            self.image_label.setPixmap(pix.scaled(parent.width()-20, parent.height()-20, Qt.KeepAspectRatio, Qt.FastTransformation))
        finally:
            self._rendering = False

    def show_result(self, result):
        self.title.setText(f"Result - {result['patient_id']}")
        self._label = result.get("label_map"); base = result.get("base_volume")
        self._findings = result.get("findings")

        # 1. Interactive Viewer Setup
        if base is not None and self._label is not None:
            self._base_u8 = normalize_u8(base)
            D = self._base_u8.shape[2]
            
            # Focus on slice with largest tumor area
            tumor = (self._label > 0).sum(axis=(0, 1))
            z0 = int(tumor.argmax()) if tumor.sum() > 0 else D // 2
            
            self.slice_slider.blockSignals(True)
            self.slice_slider.setRange(0, D - 1)
            self.slice_slider.setValue(z0)
            self.slice_slider.blockSignals(False)
            self._render()

        # 2. Summary Figure (Description Tab)
        fig = result.get("figure")
        if fig is not None:
            buf = BytesIO(); fig.savefig(buf, format="png", dpi=150, bbox_inches="tight"); buf.seek(0)
            pix = QPixmap(); pix.loadFromData(buf.getvalue(), "PNG")
            self.fig_label.setPixmap(pix.scaledToWidth(1100, Qt.SmoothTransformation))

        # 3. Structured Findings (Description Tab)
        self.findings_box.setHtml(self._format_findings(self._findings))

        # 4. Features Table (Centered Layout)
        while self.table_holder.count():
            item = self.table_holder.takeAt(0)
            if item.widget(): item.widget().deleteLater()

        if result.get("features_table") is not None:
            tbl = dataframe_to_table(result["features_table"])
            tbl.setMinimumWidth(1100)
            tbl.setMinimumHeight(550)
            
            row = QHBoxLayout()
            row.addStretch(1); row.addWidget(tbl); row.addStretch(1)
            
            wrap = QWidget(); wrap.setLayout(row)
            self.table_holder.addStretch(1) 
            self.table_holder.addWidget(wrap)
            self.table_holder.addStretch(1)

        # 5. Reset RAG Box
        if hasattr(self, 'rag_output_box'):
            self.rag_output_box.clear()

        # 6. Pipeline Configuration
        cfg = engine.pipeline_config(overlap=result.get("overlap"))
        cfg["channel_order"] = ", ".join(result.get("channel_order", []))
        
        cfg_html = """
        <div style='font-family: "Segoe UI", sans-serif; color: #1a2733;'>
            <h2 style='color: #1f3b57; border-bottom: 2px solid #dce8f6; padding-bottom: 10px; margin-bottom: 25px; font-size: 20px;'>
                System Configuration & Parameters
            </h2>
        """
        for k, v in cfg.items():
            title = k.replace('_', ' ').title()
            cfg_html += f"""
            <div style='margin-bottom: 20px;'>
                <span style='color: #2f6fb2; font-weight: bold; font-size: 15px;'>{title}</span><br>
                <span style='color: #4a5a6a; font-size: 14px; line-height: 1.5;'>{v}</span>
            </div>
            """
        cfg_html += "</div>"
        
        self.config_box.setHtml(cfg_html)
        self.config_box.setStyleSheet("""
            QTextEdit {
                background: #ffffff;
                border: 1px solid #cfe0f5;
                border-radius: 12px;
                padding: 35px;
            }
        """)

        # 7. Footer Path
        self._out_dir = result.get("output_dir")
        if self._out_dir:
            self.out_label.setText(f"Saved to: {self._out_dir}")

    def _format_findings(self, f):
        if not f or "dynamic_report" not in f:
            return "<p style='color:red;'>No report data available.</p>"

        report = f["dynamic_report"]
        
        # 1. Start Table
        html = """
        <div style='font-family: "Segoe UI", sans-serif; color: #1a2733;'>
            <h3 style='color: #2f6fb2; border-bottom: 2px solid #dce8f6; padding-bottom: 8px; margin-bottom: 15px;'>
                STRUCTURED RADIOLOGICAL DESCRIPTION
            </h3>
            <table style='width: 100%; border-collapse: collapse; border: 1px solid #dce8f6;'>
                <thead>
                    <tr style='background-color: #f8faff;'>
                        <th style='padding: 10px; border: 1px solid #dce8f6; text-align: left; width: 15%;'>Section</th>
                        <th style='padding: 10px; border: 1px solid #dce8f6; text-align: left; width: 40%;'>Findings</th>
                        <th style='padding: 10px; border: 1px solid #dce8f6; text-align: left; width: 45%;'>Data Provenance & Logic</th>
                    </tr>
                </thead>
                <tbody>
        """

        # 2. Add Rows
        for i, sec in enumerate(report):
            bg = "#ffffff" if i % 2 == 0 else "#fcfdfe"
            # Join findings with a SINGLE bullet point
            findings_html = "<br>".join([f"&bull; {item}" for item in sec["findings"]])
            
            html += f"""
                <tr style='background-color: {bg};'>
                    <td style='padding: 12px; border: 1px solid #dce8f6; vertical-align: top;'><b>{sec['section']}</b></td>
                    <td style='padding: 12px; border: 1px solid #dce8f6; vertical-align: top;'>{findings_html}</td>
                    <td style='padding: 12px; border: 1px solid #dce8f6; font-size: 11px; color: #5a6b7a; vertical-align: top;'>
                        <b>{sec['prov_h']}</b><br>{sec['prov_b']}
                    </td>
                </tr>
            """

        # 3. Close Table and Add Disclaimer
        disclaimer = f.get("disclaimer", "Descriptive only; not a prediction of grade or outcome.")
        html += f"""
                </tbody>
            </table>
            <div style='margin-top: 20px; padding: 12px; background-color: #fbfcfd; border-left: 4px solid #dce8f6;'>
                <p style='font-size: 11px; color: #7f8c8d; margin: 0; font-style: italic;'>
                    <b>Disclaimer:</b> {disclaimer}
                </p>
            </div>
        </div>
        """
        return html

    def _open_output(self):
        """Opens the folder containing the saved results in Mac Finder."""
        if hasattr(self, '_out_dir') and self._out_dir and os.path.isdir(self._out_dir):
            QDesktopServices.openUrl(QUrl.fromLocalFile(self._out_dir))
        else:
            QMessageBox.information(self, "Folder Not Found", "The output folder has not been created yet.")

    def _save_view(self):
        """Saves the current MRI slice view as a PNG image."""
        if not hasattr(self, 'image_label') or self.image_label.pixmap() is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save View", "brain_slice.png", "PNG (*.png)")
        if path:
            self.image_label.pixmap().save(path, "PNG")


# MAIN
class MainApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Brain Tumor Segmentation and Analysis")
        self.setGeometry(150, 70, 1100, 800)
        
        self.stack = QStackedWidget()
        self.welcome = WelcomePage(self)
        self.result_page = ResultPage(self)
        
        self.stack.addWidget(self.welcome)
        self.stack.addWidget(self.result_page)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.stack)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainApp(); win.show()
    sys.exit(app.exec_())