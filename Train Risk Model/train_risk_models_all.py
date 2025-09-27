# train_risk_models_all.py
# Models cohorts for Grades 1–3 and writes risk CSVs + metrics (UTF-8 BOM).

import os, re, json, warnings, inspect
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

warnings.filterwarnings("ignore")

# ---- Import your DOMAIN_CONFIG from the data-prep script
DOMAIN_CONFIG = {
    # ===================== Grade 1 =====================
    # -------- English --------
    ("Grade_1.xlsx", "Eng - Oral"): {
        "subject": "English",
        "modality": "Oral",
        "domains_counts": {
            "Letter Recognition": {"cols": ["Question 1: Letter Recognition"]},
            "Sound Recognition":  {"cols": ["Question 2: Sound Recognition"]},
            "Word Decoding":      {"cols": ["Question 3: Word Decoding"]},
            "Reading":            {"cols": ["Reading"]},
        },
    },
    ("Grade_1.xlsx", "En - Written"): {
        "subject": "English",
        "modality": "Written",
        "domains": {
            "Reading Comprehension": [
                "Section 1: Reading Comprehension",
                "Unnamed: 8","Unnamed: 9","Unnamed: 10","Unnamed: 11","Unnamed: 12","Unnamed: 13",
            ],
            "Writing": [
                "Section 2: Writing",
                "Unnamed: 15","Unnamed: 16","Unnamed: 17","Unnamed: 18",
            ],
        },
    },

    # -------- French --------
    ("Grade_1.xlsx", "Fr - Oral"): {
        "subject": "French",
        "modality": "Oral",
        "domains_counts": {
            "Nom de la lettre":                {"cols": ["Activité 1 : Nom de la lettre"]},
            "Son du graphème":                 {"cols": ["Activité 2 : Son du graphème"]},
            "Lecture de mots":                 {"cols": ["Activité 3 : Lecture de mots"]},
            "Lecture de mots dans une phrase": {"cols": ["Activité 4 : Lecture de mots dans une phrase"]},
        },
    },
    ("Grade_1.xlsx", "Fr - Written"): {
        "subject": "French",
        "modality": "Written",
        "domains": {
            "Compréhension écrite": [
                "Section 1 - Compréhension écrite",
                "Unnamed: 8","Unnamed: 9","Unnamed: 10","Unnamed: 11","Unnamed: 12","Unnamed: 13",
            ],
            "Écriture et Production d’écrits": [
                "Section 2 - Écriture et Production d’écrits",
                "Unnamed: 15","Unnamed: 16","Unnamed: 17","Unnamed: 18",
            ],
        },
    },

    # -------- Arabic --------
   
    ("Grade_1.xlsx", "Ar - Oral"): {
    "subject": "Arabic",
    "modality": "Oral",
    "domains_counts": {
        "معرفة اسم الحرف": {"cols": ["النشاط الأول - معرفةّ اسمّ الحرف"]},
        "معرفة صوت الحرف": {"cols": ["النشاط الثاني - معرفةّ صوت  الحرف"]},
        "قراءة كلمات":      {"cols": ["النشاط الثالث - قراءةّ كلمات"]},
        "قراءة جملة":       {"cols": ["النشاط الرابع - قراءةّ جملة"]},
    },
},

    ("Grade_1.xlsx", "Ar - Written"): {
        "subject": "Arabic",
        "modality": "Written",
        "domains_counts": {
            "فهم المقروء": {
                "cols": ["الْقِسْمُ اّلْأَوَّلُّ","Unnamed: 8","Unnamed: 9","Unnamed: 10","Unnamed: 11","Unnamed: 12","Unnamed: 13"],
            },
            "الكتابة والتعبير الكتابي": {
                "cols": ["الّْقِسْمُّ اّلثاني","Unnamed: 15","Unnamed: 16","Unnamed: 17","Unnamed: 18"],
            },
        },
    },

    # ===================== Grade 2 =====================
    # -------- English --------
    ("Grade_2.xlsx", "En - oral"): {  # note: lower-case 'oral' in your file
        "subject": "English",
        "modality": "Oral",
        "domains_counts": {
            "Letter Recognition": {"cols": ["Question 1: Letter Recognition"]},
            "Sound Recognition":  {"cols": ["Question 2: Sound Recognition"]},
            "Word Decoding":      {"cols": ["Question 3: Word Decoding","Word Decoding"]},
            "Text Reading":       {"cols": ["Text Reading"]},
        },
    },
    ("Grade_2.xlsx", "En - written"): {
        "subject": "English",
        "modality": "Written",
        "domains": {
            "Reading Comprehension": [
                "Section 1: Reading Comprehension",
                "Unnamed: 8","Unnamed: 9","Unnamed: 10","Unnamed: 11","Unnamed: 12","Unnamed: 13",
            ],
            "Writing": [
                "Section 2: Writing",
                "Unnamed: 15","Unnamed: 16","Unnamed: 17","Unnamed: 18",
            ],
        },
    },

    # -------- French --------
    ("Grade_2.xlsx", "Fr - oral"): {
        "subject": "French",
        "modality": "Oral",
        "domains_counts": {
            "Nom de la lettre": {"cols": ["Activité 1 : Nom de la lettre"]},
            "Son du graphème":  {"cols": ["Activité 2 : Son du graphème"]},
            "Lecture de mots":  {"cols": ["Activité 3 : Lecture de mots","Activité 4 : Lecture de mots"]},
            "Lecture de texte": {"cols": ["Activité 5 : Lecture de texte"]},
        },
    },
    ("Grade_2.xlsx", "Fr - written"): {
        "subject": "French",
        "modality": "Written",
        "domains": {
            "Compréhension écrite": [
                "Section 1 - Compréhension écrite",
                "Unnamed: 8","Unnamed: 9","Unnamed: 10","Unnamed: 11","Unnamed: 12","Unnamed: 13",
            ],
            "Écriture et Production d’écrits": [
                "Section 2 - Écriture et Production d’écrits",
                "Unnamed: 15","Unnamed: 16","Unnamed: 17","Unnamed: 18",
            ],
        },
    },

    # -------- Arabic --------
    ("Grade_2.xlsx", "Ar - oral"): {
    "subject": "Arabic",
    "modality": "Oral",
    "domains_counts": {
        "معرفة اسم الحرف": {
            "cols": [
                "النشاط 1 - معرفة اسم الحرف",          # clean
                "النّشاط 1 - معرفةّ اسمّ الحر  رف",      # fallback noisy
            ],
            "max": [12]
        },
        "معرفة صوت الحرف": {
            "cols": [
                "النشاط 2 - معرفة صوت الحرف",
                "النّشاط 2 - معرفةّصوت الحرف",
            ],
            "max": [12]
        },
        "قراءة كلمات 1": {                              # split col 3
            "cols": [
                "النشاط 3 - قراءة كلمات",
                "النّشاط 3 - قراءةّ كلمات",
            ],
            "max": [4]
        },
        "قراءة كلمات 2": {                              # split col 4
            "cols": [
                "النشاط 4 - قراءة كلمات",
                "النّشاط 4 - قراءةّ كلمات",
                "   'النّشاط 4 - قراءةّ كلمات",
            ],
            "max": [6]
        },
        "قراءة نص": {
            "cols": [
                "النشاط 5 - قراءة نص",
                "النّشاط 5 - قراءةّ نصّ",
            ],
            "max": [50]
        },
    },
},

    ("Grade_2.xlsx", "Ar - written"): {
        "subject": "Arabic",
        "modality": "Written",
        "domains_counts": {
            "فهم المقروء": {
                "cols": ["الْقِسْمُ اّلْأَوَّلُّ","Unnamed: 8","Unnamed: 9","Unnamed: 10","Unnamed: 11","Unnamed: 12","Unnamed: 13"],
            },
            "الكتابة والتعبير الكتابي": {
                "cols": ["الّْقِسْمُّ اّلثاني","Unnamed: 15","Unnamed: 16","Unnamed: 17","Unnamed: 18"],
            },
        },
    },

    # ===================== Grade 3 =====================
    # -------- English --------
    ("Grade_3.xlsx", "En - oral"): {
        "subject": "English",
        "modality": "Oral",
        "domains_counts": {
            "Letter Recognition": {"cols": ["Question 1: Letter Recognition"]},
            "Sound Recognition":  {"cols": ["Question 2: Sound Recognition"]},
            "Word Decoding":      {"cols": ["Question 3: Word Decoding","Word Decoding"]},
            "Text Reading":       {"cols": ["Text Reading"]},
        },
    },
    ("Grade_3.xlsx", "En - written"): {
        "subject": "English",
        "modality": "Written",
        "domains": {
            "Reading Comprehension": [
                "Section 1: Reading Comprehension",
                "Unnamed: 8","Unnamed: 9","Unnamed: 10","Unnamed: 11","Unnamed: 12","Unnamed: 13",
            ],
            "Writing": [
                "Section 2: Writing",
                "Unnamed: 15","Unnamed: 16","Unnamed: 17","Unnamed: 18",
            ],
        },
    },

    # -------- French --------
    ("Grade_3.xlsx", "Fr - oral"): {
        "subject": "French",
        "modality": "Oral",
        "domains_counts": {
            "Nom de la lettre": {"cols": ["Activité 1 : Nom de la lettre"]},
            "Son du graphème":  {"cols": ["Activité 2 : Son du graphème"]},
            "Lecture de mots":  {"cols": ["Activité 3 : Lecture de mots","Activité 4 : Lecture de mots"]},
            "Lecture de texte": {"cols": ["Activité 5 : Lecture de texte"]},
        },
    },
    ("Grade_3.xlsx", "Fr -written"): {  # note: your sheet is 'Fr -written' (no space)
        "subject": "French",
        "modality": "Written",
        "domains": {
            "Compréhension écrite": [
                "Section 1 - Compréhension écrite",
                "Unnamed: 8","Unnamed: 9","Unnamed: 10","Unnamed: 11","Unnamed: 12","Unnamed: 13",
            ],
            "Écriture et Production d’écrits": [
                "Section 2 - Écriture et Production d’écrits",
                "Unnamed: 15","Unnamed: 16","Unnamed: 17","Unnamed: 18",
            ],
        },
    },

    # -------- Arabic --------
    ("Grade_3.xlsx", "Ar - oral"): {
        "subject": "Arabic",
        "modality": "Oral",
        "domains_counts": {
            "معرفة اسم الحرف": {"cols": ["النّشاط 1 - معرفةّ اسمّ الحر  رف","النشاط 1 - معرفة اسم الحرف"]},
            "معرفة صوت الحرف": {"cols": ["النّشاط 2 - معرفةّ صوت  الحرف","النشاط 2 - معرفة صوت الحرف"]},
            "قراءة كلمات":     {"cols": ["النّشاط 3 - قراءةّ كلمات","النشاط 3 - قراءة كلمات","النّشاط 4 - قراءةّ كلمات","النشاط 4 - قراءة كلمات"]},
            "قراءة نص":        {"cols": ["النّشاط 5 - قراءةّ نصّ","النشاط 5 - قراءة نص"]},
        },
    },
    ("Grade_3.xlsx", "Ar - written"): {
        "subject": "Arabic",
        "modality": "Written",
        "domains_counts": {
            "فهم المقروء": {
                "cols": ["الْقِسْمُ اّلْأَوَّلُّ","Unnamed: 8","Unnamed: 9","Unnamed: 10","Unnamed: 11","Unnamed: 12","Unnamed: 13"],
            },
            "الكتابة والتعبير الكتابي": {
                "cols": ["الّْقِسْمُّ اّلثاني","Unnamed: 15","Unnamed: 16","Unnamed: 17","Unnamed: 18"],
            },
        },
    },

    # ===================== Grade 4 =====================
    # -------- English --------
    ("Grade_4.xlsx", "En - oral"): {
        "subject": "English",
        "modality": "Oral",
        "domains_counts": {
            "Letter Recognition": {"cols": ["Question 1: Letter Recognition"]},
            "Sound Recognition":  {"cols": ["Question 2: Sound Recognition"]},
            "Word Decoding":      {"cols": ["Question 3: Word Decoding","Word Decoding"]},
            "Text Reading":       {"cols": ["Text Reading"]},
        },
    },
    ("Grade_4.xlsx", "En - written"): {
        "subject": "English",
        "modality": "Written",
        "domains": {
            "Reading Comprehension": [
                "Section 1: Reading Comprehension",
                "Unnamed: 8","Unnamed: 9","Unnamed: 10","Unnamed: 11","Unnamed: 12","Unnamed: 13",
            ],
            "Writing": [
                "Section 2: Writing",
                "Unnamed: 15","Unnamed: 16","Unnamed: 17","Unnamed: 18",
            ],
        },
    },

    # -------- French --------
    ("Grade_4.xlsx", "Fr - oral"): {
        "subject": "French",
        "modality": "Oral",
        "domains_counts": {
            "Nom de la lettre": {"cols": ["Activité 1 : Nom de la lettre"]},
            "Son du graphème":  {"cols": ["Activité 2 : Son du graphème"]},
            "Lecture de mots":  {"cols": ["Activité 3 : Lecture de mots","Activité 4 : Lecture de mots"]},
            "Lecture de texte": {"cols": ["Activité 5 : Lecture de texte"]},
        },
    },
    ("Grade_4.xlsx", "Fr - written"): {
        "subject": "French",
        "modality": "Written",
        "domains": {
            "Compréhension écrite": [
                "Section 1 - Compréhension écrite",
                "Unnamed: 8","Unnamed: 9","Unnamed: 10","Unnamed: 11","Unnamed: 12","Unnamed: 13",
            ],
            "Écriture et Production d’écrits": [
                "Section 2 - Écriture et Production d’écrits",
                "Unnamed: 15","Unnamed: 16","Unnamed: 17","Unnamed: 18",
            ],
        },
    },

    # -------- Arabic --------
    ("Grade_4.xlsx", "Ar - Oral"): {
        "subject": "Arabic",
        "modality": "Oral",
        "domains_counts": {
            "معرفة اسم الحرف": {"cols": ["النّشاط 1 -- معرفةّ اسمّ الحرف","النشاط 1 - معرفة اسم الحرف"]},
            "معرفة صوت الحرف": {"cols": ["النّشاط 2 - معرفةّ صوت  الحرف","النشاط 2 - معرفة صوت الحرف"]},
            "قراءة كلمات":     {"cols": ["النّشاط 3    3 - قراءةّ كلمات","النّشاط 4 - قراءةّ كلمات","النشاط 3 - قراءة كلمات","النشاط 4 - قراءة كلمات"]},
            "قراءة نص":        {"cols": ["النّشاط 5 - قراءة   ةّ نصّ","النشاط 5 - قراءة نص"]},
        },
    },
    ("Grade_4.xlsx", "Ar - written"): {
        "subject": "Arabic",
        "modality": "Written",
        "domains_counts": {
            "فهم المقروء": {
                "cols": ["الْقِسْمُ اّلْأَوَّلُّ","Unnamed: 8","Unnamed: 9","Unnamed: 10","Unnamed: 11","Unnamed: 12","Unnamed: 13"],
            },
            "الكتابة والتعبير الكتابي": {
                "cols": ["الّْقِسْمُّ اّلثاني","Unnamed: 15","Unnamed: 16","Unnamed: 17","Unnamed: 18"],
            },
        },
    },

    # ===================== Grade 5 =====================
    # -------- English --------
    ("Grade_5.xlsx", "En - oral"): {
        "subject": "English",
        "modality": "Oral",
        "domains_counts": {
            "Letter Recognition": {"cols": ["Question 1: Letter Recognition"]},
            "Sound Recognition":  {"cols": ["Question 2: Sound Recognition"]},
            "Word Decoding":      {"cols": ["Question 3: Word Decoding","Word Decoding"]},
            "Text Reading":       {"cols": ["Text Reading"]},
        },
    },
    ("Grade_5.xlsx", "En - written"): {
        "subject": "English",
        "modality": "Written",
        "domains": {
            "Reading Comprehension": [
                "Section 1: Reading Comprehension",
                "Unnamed: 8","Unnamed: 9","Unnamed: 10","Unnamed: 11","Unnamed: 12","Unnamed: 13",
            ],
            "Writing": [
                "Section 2: Writing",
                "Unnamed: 15","Unnamed: 16","Unnamed: 17","Unnamed: 18",
            ],
        },
    },

    # -------- French --------
    ("Grade_5.xlsx", "Fr - oral"): {
        "subject": "French",
        "modality": "Oral",
        "domains_counts": {
            "Nom de la lettre": {"cols": ["Activité 1 : Nom de la lettre"]},
            "Son du graphème":  {"cols": ["Activité 2 : Son du graphème"]},
            "Lecture de mots":  {"cols": ["Activité 3 : Lecture de mots","Activité 4 : Lecture de mots"]},
            "Lecture de texte": {"cols": ["Activité 5 : Lecture de texte"]},
        },
    },
    ("Grade_5.xlsx", "Fr - written"): {
        "subject": "French",
        "modality": "Written",
        "domains": {
            "Compréhension écrite": [
                "Section 1 - Compréhension écrite",
                "Unnamed: 8","Unnamed: 9","Unnamed: 10","Unnamed: 11","Unnamed: 12","Unnamed: 13",
            ],
            "Écriture et Production d’écrits": [
                "Section 2 - Écriture et Production d’écrits",
                "Unnamed: 15","Unnamed: 16","Unnamed: 17","Unnamed: 18",
            ],
        },
    },

    # -------- Arabic --------
    ("Grade_5.xlsx", "Ar - Oral"): {
        "subject": "Arabic",
        "modality": "Oral",
        "domains_counts": {
            "معرفة اسم الحرف": {"cols": ["النّشاط 1 - معرفةّ اسمّ الحر  رف","النشاط 1 - معرفة اسم الحرف"]},
            "معرفة صوت الحرف": {"cols": ["النّشاط 2 - معرفةّ صوت  الحرف","النشاط 2 - معرفة صوت الحرف"]},
            "قراءة كلمات":     {"cols": ["النّشاط 3 - قراءةّ كلمات","النشاط 3 - قراءة كلمات","النّشاط 4 - قراءةّ كلمات","النشاط 4 - قراءة كلمات"]},
            "قراءة نص":        {"cols": ["النّشاط 5 - قراءةّ نصّ","النشاط 5 - قراءة نص"]},
        },
    },
    ("Grade_5.xlsx", "Ar - written"): {
        "subject": "Arabic",
        "modality": "Written",
        "domains_counts": {
            "فهم المقروء": {
                "cols": ["الْقِسْمُ اّلْأَوَّلُّ","Unnamed: 8","Unnamed: 9","Unnamed: 10","Unnamed: 11","Unnamed: 12","Unnamed: 13"],
            },
            "الكتابة والتعبير الكتابي": {
                "cols": ["الّْقِسْمُّ اّلثاني","Unnamed: 15","Unnamed: 16","Unnamed: 17","Unnamed: 18"],
            },
        },
    },

    # ===================== Grade 6 =====================
    # -------- English --------
    ("Grade_6.xlsx", "En - oral"): {
        "subject": "English",
        "modality": "Oral",
        "domains_counts": {
            "Letter Recognition": {"cols": ["Question 1: Letter Recognition"]},
            "Sound Recognition":  {"cols": ["Question 2: Sound Recognition"]},
            "Word Decoding":      {"cols": ["Question 3: Word Decoding","Word Decoding"]},
            "Text Reading":       {"cols": ["Text Reading"]},
        },
    },
    ("Grade_6.xlsx", "En - written"): {
        "subject": "English",
        "modality": "Written",
        "domains": {
            "Reading Comprehension": [
                "Section 1: Reading Comprehension",
                "Unnamed: 8","Unnamed: 9","Unnamed: 10","Unnamed: 11","Unnamed: 12","Unnamed: 13",
            ],
            "Writing": [
                "Section 2: Writing"," Section 2: Writing",  # note: your dump shows a leading space variant once
                "Unnamed: 15","Unnamed: 16","Unnamed: 17","Unnamed: 18",
            ],
        },
    },

    # -------- French --------
    ("Grade_6.xlsx", "Fr - oral"): {
        "subject": "French",
        "modality": "Oral",
        "domains_counts": {
            "Nom de la lettre": {"cols": ["Activité 1 : Nom de la lettre"]},
            "Son du graphème":  {"cols": ["Activité 2 : Son du graphème"]},
            "Lecture de mots":  {"cols": ["Activité 3 : Lecture de mots","Activité 4 : Lecture de mots"]},
            "Lecture de texte": {"cols": ["Activité 5 : Lecture de texte"]},
        },
    },
    ("Grade_6.xlsx", "Fr - written"): {
        "subject": "French",
        "modality": "Written",
        "domains": {
            "Compréhension écrite": [
                "Section 1 - Compréhension écrite",
                "Unnamed: 8","Unnamed: 9","Unnamed: 10","Unnamed: 11","Unnamed: 12","Unnamed: 13",
            ],
            "Écriture et Production d’écrits": [
                "Section 2 - Écriture et Production d’écrits"," Section 2 - Écriture et Production d’écrits",
                "Unnamed: 15","Unnamed: 16","Unnamed: 17","Unnamed: 18",
            ],
        },
    },

    # -------- Arabic --------
    ("Grade_6.xlsx", "Ar - oral"): {
        "subject": "Arabic",
        "modality": "Oral",
        "domains_counts": {
            "معرفة اسم الحرف": {"cols": ["النّشاط 1 -  معرفةّ اسمّ الحرف","النشاط 1 - معرفة اسم الحرف"]},
            "معرفة صوت الحرف": {"cols": ["النّشاط 2 - معرفةّ صوت  الحرف","النشاط 2 - معرفة صوت الحرف"]},
            "قراءة كلمات":     {"cols": ["النّشاط 3      - قراءةّ كلمات","النّشاط 4 - قراءةّ كلمات","النشاط 3 - قراءة كلمات","النشاط 4 - قراءة كلمات"]},
            "قراءة نص":        {"cols": ["النّشاط 5 - قراءةّ     نصّ","النشاط 5 - قراءة نص"]},
        },
    },
    ("Grade_6.xlsx", "Ar - written"): {
        "subject": "Arabic",
        "modality": "Written",
        "domains_counts": {
            "فهم المقروء": {
                "cols": ["الْقِسْمُ اّلْأَوَّلُّ","Unnamed: 8","Unnamed: 9","Unnamed: 10","Unnamed: 11","Unnamed: 12","Unnamed: 13"],
            },
            "الكتابة والتعبير الكتابي": {
                "cols": ["الّْقِسْمُّ اّلثاني","Unnamed: 15","Unnamed: 16","Unnamed: 17","Unnamed: 18"],
            },
        },
    },
}

GRADE_WHITELIST = {1, 2, 3,4,5,6}
GRADE_FILES = [f"Grade_{i}.xlsx" for i in range(1,7) if os.path.exists(f"Grade_{i}.xlsx")]
OUT_DIR = Path("data_proc"); OUT_DIR.mkdir(exist_ok=True)

def normalize_region(g):
    if pd.isna(g): return "Unknown"
    m = str(g).strip()
    mp = {
        "بيروت":"Beirut","جبل لبنان":"Mount Lebanon","البقاع":"Bekaa","بعلبك الهرمل":"Baalbek-Hermel",
        "النبطية":"Nabatieh","الجنوب":"South","الشمال":"North","عكار":"Akkar",
        "كسروان-جبيل":"Keserwan-Jbeil","زحلة":"Zahleh",
    }
    return mp.get(m, m)

def sheet_subject_modality(sheet):
    s = sheet.strip().lower()
    if s.startswith("en"): subj="English"
    elif s.startswith("fr"): subj="French"
    elif s.startswith("ar"): subj="Arabic"
    elif "math" in s: subj="Math"
    else: subj="Unknown"
    mod = "Written" if "written" in s else ("Oral" if "oral" in s else "Unknown")
    return subj, mod

def find_first_row(df):
    if "ScRN" in df.columns and "StRN" in df.columns:
        idx = df.index[df["ScRN"].notna() & df["StRN"].notna()]
        if len(idx): return int(idx[0])
        idx2 = df.index[df["ScRN"].notna()]
        if len(idx2): return int(idx2[0])
    return 0

def make_ohe():
    """Create a OneHotEncoder that works across sklearn versions."""
    try:
        return OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
    except TypeError:
        return OneHotEncoder(drop="first", sparse=False, handle_unknown="ignore")

def get_feature_names(ohe, input_features):
    """Handle get_feature_names_out vs get_feature_names."""
    if hasattr(ohe, "get_feature_names_out"):
        return list(ohe.get_feature_names_out(input_features))
    return list(ohe.get_feature_names(input_features))

def make_calibrator(base):
    """Create CalibratedClassifierCV across sklearn API changes."""
    sig = inspect.signature(CalibratedClassifierCV.__init__)
    if "estimator" in sig.parameters:
        return CalibratedClassifierCV(estimator=base, cv=3, method="sigmoid")
    else:
        return CalibratedClassifierCV(base_estimator=base, cv=3, method="sigmoid")

def extract_features(df_raw, file_name, sheet):
    """Return X (np.array), y (np.array), meta (DataFrame), feat_names (list) or None."""
    subj, mod = sheet_subject_modality(sheet)
    m = re.search(r"Grade_(\d+)\.xlsx", file_name)
    grade = int(m.group(1)) if m else None
    if (GRADE_WHITELIST is not None) and (grade not in GRADE_WHITELIST):
        return None

    start = find_first_row(df_raw)
    df = df_raw.iloc[start:].reset_index(drop=True).copy()

    # ids + demos
    df = df.rename(columns={"ScRN":"school_id","StRN":"student_id","Governate":"region","Gender":"gender"})
    df["region"] = df["region"].apply(normalize_region)
    df["gender"] = df["gender"].map({"ذكر":"Male","انثى":"Female"}).fillna(df["gender"])

    # ----- build feature blocks -----
    cfg = DOMAIN_CONFIG.get((file_name, sheet))
    feat_blocks = []

    # 1) Binary-item domains
    if cfg and "domains" in cfg:
        for dom, cols in cfg["domains"].items():
            keep = [c for c in cols if c in df.columns]
            if keep:
                block = (df[keep].replace("-", 0)
                         .apply(pd.to_numeric, errors="coerce")
                         .fillna(0).clip(0,1))
                block.columns = [f"{dom}__{c}" for c in block.columns]
                feat_blocks.append(block)

    # 2) Count/max domains → percent features
    if cfg and "domains_counts" in cfg:
        for dom, spec in cfg["domains_counts"].items():
            cols = [c for c in spec["cols"] if c in df.columns]
            if cols:
                vals = df[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
                given = spec.get("max", [])
                maxima = []
                for i, c in enumerate(cols):
                    mmax = float(given[i]) if i < len(given) and given[i] else float(max(vals[c].max(), 1.0))
                    maxima.append(mmax)
                for c, mmax in zip(cols, maxima):
                    feat_blocks.append(pd.DataFrame({f"{dom}__{c}_pct": (vals[c]/mmax).clip(0,1)}))

    # 3) Auto-detect additional binary 0/1 columns (helps when config is partial)
    demo_cols = {"school_id","student_id","region","gender","Class Name","Course Name","Exam Type"}
    for c in [c for c in df.columns if c not in demo_cols]:
        s = pd.to_numeric(df[c].replace("-", 0), errors="coerce")
        if s.notna().sum() >= max(20, int(0.2*len(s))):
            sb = s.fillna(0).clip(0,1)
            if np.all(np.isin(sb.unique(), [0,1])):
                feat_blocks.append(pd.DataFrame({f"AUTO__{c}": sb}))

    if not feat_blocks:
        return None

    X_items_df = pd.concat(feat_blocks, axis=1)
    # drop near-constant cols
    keep_cols = X_items_df.nunique(dropna=False)
    keep_cols = keep_cols[keep_cols > 1].index.tolist()
    X_items_df = X_items_df[keep_cols]

    # target: below proficiency from observed items
    # prefer true item-like columns; if none, average all features
    item_like = X_items_df.filter(regex="AUTO__|Section|Unnamed:|Reading Comprehension|Writing")
    if item_like.shape[1] >= 4:
        pct = item_like.to_numpy(dtype=float).mean(axis=1) * 100.0
    else:
        pct = X_items_df.to_numpy(dtype=float).mean(axis=1) * 100.0
    y = (pct < 50.0).astype(int)

    # encode gender + region
    cats = df[["gender","region"]].fillna("Unknown")
    ohe = make_ohe()
    Z = ohe.fit_transform(cats)   # ndarray
    z_names = get_feature_names(ohe, ["gender","region"])

    # final feature matrix
    X_items = X_items_df.to_numpy(dtype=float)
    X = np.hstack([X_items, Z])
    feat_names = list(X_items_df.columns) + z_names

    meta = pd.DataFrame({
        "student_id": df["student_id"],
        "school_id": df["school_id"],
        "region": df["region"],
        "grade": grade,
        "subject": subj,
        "modality": mod,
        "sheet": sheet,
        "file": file_name
    })

    return X, y, meta, feat_names

def train_one_cohort(X, y, feat_names, cv_splits=5):
    # require enough size and class balance
    if len(y) < 300 or y.mean() in (0.0, 1.0):
        return None

    skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
    oof = np.zeros(len(y))
    coefs = np.zeros((cv_splits, len(feat_names)))
    mets = []

    for fold, (tr, te) in enumerate(skf.split(X, y)):
        base = LogisticRegression(max_iter=1000, class_weight="balanced", solver="lbfgs")
        # Create calibrator with correct kw for your sklearn version
        try:
            clf = make_calibrator(base)
            clf.fit(X[tr], y[tr])
            p = clf.predict_proba(X[te])[:, 1]
        except Exception:
            # fallback: uncalibrated logistic
            base.fit(X[tr], y[tr])
            p = base.predict_proba(X[te])[:, 1]

        oof[te] = p

        # coefficients from base (refit on train for interpretability)
        base.fit(X[tr], y[tr])
        coef_vec = getattr(base, "coef_", np.zeros((1, len(feat_names))))[0]
        coefs[fold, :] = coef_vec

        mets.append({
            "fold": int(fold),
            "auc": float(roc_auc_score(y[te], p)),
            "avg_precision": float(average_precision_score(y[te], p)),
            "brier": float(brier_score_loss(y[te], p)),
        })

    metrics = {
        "n": int(len(y)),
        "pos_rate": float(y.mean()),
        "auc_mean": float(np.mean([m["auc"] for m in mets])),
        "ap_mean": float(np.mean([m["avg_precision"] for m in mets])),
        "brier_mean": float(np.mean([m["brier"] for m in mets])),
        "folds": mets,
    }
    coef_mean = dict(zip(feat_names, coefs.mean(axis=0).tolist()))
    return oof, metrics, coef_mean

# ----------------- run over all usable sheets -----------------
all_student_risks = []
all_metrics = []
all_coefs = []

for file in GRADE_FILES:
    try:
        xls = pd.ExcelFile(file)
    except Exception as e:
        print(f"Skip {file}: {e}")
        continue

    for sheet in xls.sheet_names:
        try:
            df_raw = pd.read_excel(file, sheet_name=sheet)
            parsed = extract_features(df_raw, file, sheet)
            if parsed is None:
                continue
            X, y, meta, feat_names = parsed

            res = train_one_cohort(X, y, feat_names, cv_splits=5)
            if res is None:
                # fallback: wrong-rate ranking (transparent)
                wrong = 1.0 - X.mean(axis=1)
                meta2 = meta.copy()
                meta2["risk_prob_below"] = np.clip(wrong, 0, 1)
                meta2["is_below"] = y
                all_student_risks.append(meta2)
                all_metrics.append({
                    "file": file, "sheet": sheet,
                    "grade": int(meta["grade"].iloc[0]),
                    "subject": meta["subject"].iloc[0],
                    "modality": meta["modality"].iloc[0],
                    "n": int(len(y)), "pos_rate": float(y.mean()),
                    "note": "heuristic risk (insufficient data for calibrated model)"
                })
                continue

            oof, metrics, coef_mean = res
            meta2 = meta.copy()
            meta2["risk_prob_below"] = np.round(oof, 4)
            meta2["is_below"] = y
            all_student_risks.append(meta2)

            metrics.update({
                "file": file, "sheet": sheet,
                "grade": int(meta["grade"].iloc[0]),
                "subject": meta["subject"].iloc[0],
                "modality": meta["modality"].iloc[0],
            })
            all_metrics.append(metrics)

            all_coefs.append(pd.DataFrame({
                "file": file, "sheet": sheet,
                "grade": int(meta["grade"].iloc[0]),
                "subject": meta["subject"].iloc[0],
                "modality": meta["modality"].iloc[0],
                "feature": list(coef_mean.keys()),
                "coef": list(coef_mean.values())
            }))
        except Exception as e:
            print(f"Skip {file} / {sheet}: {e}")

# ----------------- write outputs -----------------
if all_student_risks:
    risks = pd.concat(all_student_risks, ignore_index=True)
    risks.to_csv(OUT_DIR / "student_risk_scores.csv", index=False, encoding="utf-8-sig")

    cohort = (risks.groupby(["region","grade","subject"])
                    .agg(avg_risk=("risk_prob_below","mean"),
                         pct_students_above80=("risk_prob_below", lambda s: (s>=0.8).mean()),
                         n_students=("student_id","count"))
                    .reset_index())
    cohort["avg_risk"] = (100*cohort["avg_risk"]).round(1)
    cohort["pct_students_above80"] = (100*cohort["pct_students_above80"]).round(1)
    cohort.to_csv(OUT_DIR / "risk_by_region_grade_subject.csv", index=False, encoding="utf-8-sig")

if all_metrics:
    with open(OUT_DIR / "risk_model_metrics.json", "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, ensure_ascii=False, indent=2)

if all_coefs:
    coef_df = pd.concat(all_coefs, ignore_index=True)
    coef_df = coef_df.sort_values(["grade","subject","modality","coef"], ascending=[True,True,True,False])
    coef_df.to_csv(OUT_DIR / "risk_model_coefficients.csv", index=False, encoding="utf-8-sig")

print("✓ Risk models complete. Files written to data_proc/")
