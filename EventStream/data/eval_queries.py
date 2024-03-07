# import os, rootutils, sys
# root = rootutils.setup_root(os.path.abspath(""), dotenv=True, pythonpath=True, cwd=False)
# sys.path.append(os.environ["EVENT_STREAM_PATH"])
# from pathlib import Path
# from EventStream.data.config import PytorchDatasetConfig

# COHORT_NAME = "ESD_09-01-23-1"
# TASK_NAME = "readmission_30d_all"
# PROJECT_DIR = Path(os.environ["PROJECT_DIR"])
# dataset_dir = f"/storage/shared/mgh-hf-dataset/processed/{COHORT_NAME}" # PROJECT_DIR / "data" / COHORT_NAME

EVAL_QUERIES = [
    {
        'name': 'High NT-proBNP',
        'code':'N-terminal pro-brain natriuretic peptide',
        'range':(125, 10000),
    },
    {
        'name': 'High Trop-T',
        'code':'Troponin T cardiac',
        'range':(0.05, 10),
    },
    {
        'name': 'High Creatinine',
        'code':'Creatinine',
        'range':(1.0, 10),
    },
    {
        'name': 'Low Potassium',
        'code':'Potassium',
        'range':(0, 3.5),
    },
    {
        'name': 'Normal Potassium',
        'code':'Potassium',
        'range':(3.5, 5),
    },
    {
        'name': 'High Potassium',
        'code':'Potassium',
        'range':(5, 10),
    },
    {
        'name': 'Low Sodium',
        'code':'Sodium',
        'range':(0,136),
    },
    {
        'name': 'Normal Sodium',
        'code':'Sodium',
        'range':(136,145),
    },
    {
        'name': 'High Sodium',
        'code':'Sodium',
        'range':(145,200),
    },
    {
        'name': 'High BUN',
        'code':'Urea nitrogen',
        'range':(20, 200),
    },
    {
        'name': 'Low GFR',
        'code':'Glomerular filtration rate',
        'range':(0, 60),
    },
    {
        'name': 'High HbA1c',
        'code':'Hemoglobin A1c/Hemoglobin total',
        'range':(8.5, 30),
    },
    {
        'name': 'LVEF < 40',
        'code':'lv_ef_value',
        'range':(0, 40),
    },
    {
        'name': 'LVEF > 40',
        'code':'lv_ef_value',
        'range':(41, 100),
    },
    {
        'name': 'High PCWP',
        'code':'mean_wedge_pressure',
        'range':(15, 200),
    },
    {
        'name': 'Stage 4 CKD',
        'code':'Chronic kidney disease, Stage IV (severe)',
        'range':(.0,.0),
    },
    {
        'name': 'Cardiogenic shock',
        'code':'Cardiogenic shock',
        'range':(.0,.0),
    },
    {
        'name': 'Hypertension',
        'code':'Essential (primary) hypertension',
        'range':(.0,.0),
    },
    {
        'name': 'Sepsis after procedure',
        'code':'Sepsis following a procedure, initial encounter',
        'range':(.0,.0),
    },
    {
        'name': 'Thrombus or embolism',
        'code':'Thrombus/Embolism',
        'range':(.0,.0),
    },
    {
        'name': 'Ventricular tachycardia',
        'code':'Ventricular tachycardia',
        'range':(.0,.0),
    },
    {
        'name': 'Endocarditis',
        'code':'Endocarditis, valve unspecified, unspecified cause',
        'range':(.0,.0),
    },
    {
        'name': 'Aortic dissection',
        'code':'Dissection of unspecified site of aorta',
        'range':(.0,.0),
    },
    {
        'name': 'Cardiac tamponade',
        'code':'Cardiac tamponade',
        'range':(.0,.0),
    },
    {
        'name': 'Atrial fibrillation',
        'code':'Atrial fibrillation',
        'range':(.0,.0),
    },
    {
        'name': 'Acute pulmonary edema',
        'code':'Acute pulmonary edema',
        'range':(.0,.0),
    },
    {
        'name':'Congestive heart failure',
        'code':'Congestive heart failure',
        'range':(.0,.0),
    },
    {
        'name': 'Shortness of breath',
        'code':'Shortness of breath',
        'range':(.0,.0),
    },
    {
        'name': 'Loss of weight',
        'code':'Loss of weight',
        'range':(.0,.0),
    },
    {
        'name': 'Syncope and collapse',
        'code':'Syncope and collapse',
        'range':(.0,.0),
    },
    {
        'name': 'High Systolic BP',
        'code':'Systolic-Epic',
        'range':(120.,250.),
    },
    {
        'name': 'Tachycardic Pulse',
        'code':'Pulse',
        'range':(100.,250.),
    },
    {
        'name': 'Bradycardic Pulse',
        'code':'Pulse',
        'range':(0.,60.),
    },
    {
        'name': 'Endometriosis',
        'code':'Endometriosis of ovary',
        'range':(.0,.0),
    },
    {
        'name': 'Pregnancy',
        'code':'Pregnancy examination or test, positive result',
        'range':(.0,.0),
    },
    {
        'name': 'Diabetes in pregnancy',
        'code':'Diabetes mellitus of mother, complicating pregnancy, childbirth, or the puerperium, unspecified as to episode of care',
        'range':(.0,.0),
    },
    {
        'name': 'Prostate cancer',
        'code':'Malignant neoplasm of prostate',
        'range':(.0,.0),
    },
    {
        'name': 'Androgenic alopecia',
        'code':'Androgenic alopecia, unspecified',
        'range':(.0,.0),
    },
    {
        'name': 'Senile osteoporosis',
        'code':'Senile osteoporosis',
        'range':(.0,.0),
    },
    {
        'name': 'Senile dementia',
        'code':'Senile dementia with delirium',
        'range':(.0,.0),
    },
]


# for q in EVAL_QUERIES: 
#     pyd_config = PytorchDatasetConfig(
#     save_dir=dataset_dir,
#     max_seq_len=256,
#     train_subset_size=0.001,
#     train_subset_seed=79163,
#     do_include_start_time_min=True,
#     static_query_mode=True,
#     static_query_name=q['name'],
#     static_query_code=q['code'],
#     static_query_range=q['range'],
# )
#     code = pyd_config.sample_code()
#     if code['has_value']:
#         print(f"{q['name']} with range {q['range']}")
#     else: 
#         print(f"{q['name']}")
#     print(f"observation freq ~ {code['obs_freq']:.0e}\n")


'''
relative to the this patients last number, will this change? 
    - trends? a lot patients are consistently outside the normal
    - some pt s have baseline elevated BNP... 

'''