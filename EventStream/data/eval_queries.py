EVAL_TIMES = [
    {'offset':0, 'duration':365},
    {'offset':0, 'duration':180},
    {'offset':0, 'duration':90},
    {'offset':0, 'duration':30},
    {'offset':0, 'duration':7},
    {'offset':7, 'duration':30},
    {'offset':30, 'duration':30},
    {'offset':90, 'duration':30},
    {'offset':180, 'duration':30},
    {'offset':365, 'duration':30},
]
EVAL_CODES = [
    {
        'name': 'High NT-proBNP',
        'code':'N-terminal pro-brain natriuretic peptide',
        'range_min':125,
        'range_max':10000,
    },
    {
        'name': 'High Trop-T',
        'code':'Troponin T cardiac',
        'range_min':0.05,
        'range_max':10,
    },
    {
        'name': 'High Creatinine',
        'code':'Creatinine',
        'range_min':1.0,
        'range_max':10,
    },
    {
        'name': 'Low Potassium',
        'code':'Potassium',
        'range_min':0,
        'range_max':3.5,
    },
    {
        'name': 'Normal Potassium',
        'code':'Potassium',
        'range_min':3.5,
        'range_max':5,
    },
    {
        'name': 'High Potassium',
        'code':'Potassium',
        'range_min':5,
        'range_max':10,
    },
    {
        'name': 'Low Sodium',
        'code':'Sodium',
        'range_min':0,
        'range_max':136,
    },
    {
        'name': 'Normal Sodium',
        'code':'Sodium',
        'range_min':136,
        'range_max':145,
    },
    {
        'name': 'High Sodium',
        'code':'Sodium',
        'range_min':145,
        'range_max':200,
    },
    {
        'name': 'High BUN',
        'code':'Urea nitrogen',
        'range_min':20,
        'range_max':200,
    },
    {
        'name': 'Low GFR',
        'code':'Glomerular filtration rate',
        'range_min':0,
        'range_max':60,
    },
    {
        'name': 'High HbA1c',
        'code':'Hemoglobin A1c/Hemoglobin total',
        'range_min':8.5,
        'range_max':30,
    },
    {
        'name': 'LVEF < 40',
        'code':'lv_ef_value',
        'range_min':0,
        'range_max':40,
    },
    {
        'name': 'LVEF > 40',
        'code':'lv_ef_value',
        'range_min':41,
        'range_max':100,
    },
    {
        'name': 'High PCWP',
        'code':'mean_wedge_pressure',
        'range_min':15,
        'range_max':200,
    },
    {
        'name': 'High Systolic BP',
        'code':'Systolic-Epic',
        'range_min':120.,
        'range_max':250.,
    },
    {
        'name': 'Tachycardic Pulse',
        'code':'Pulse',
        'range_min':100.,
        'range_max':250.,
    },
    {
        'name': 'Bradycardic Pulse',
        'code':'Pulse',
        'range_min':0.,
        'range_max':60.,
    },
    {
        'name': 'Stage 4 CKD',
        'code':'Chronic kidney disease, Stage IV (severe)',
    },
    {
        'name': 'Cardiogenic shock',
        'code':'Cardiogenic shock',
    },
    {
        'name': 'Hypertension',
        'code':'Essential (primary) hypertension',
    },
    {
        'name': 'Sepsis after procedure',
        'code':'Sepsis following a procedure, initial encounter',
    },
    {
        'name': 'Thrombus or embolism',
        'code':'Thrombus/Embolism',
    },
    {
        'name': 'Ventricular tachycardia',
        'code':'Ventricular tachycardia',
    },
    {
        'name': 'Endocarditis',
        'code':'Endocarditis, valve unspecified, unspecified cause',
    },
    {
        'name': 'Aortic dissection',
        'code':'Dissection of unspecified site of aorta',
    },
    {
        'name': 'Cardiac tamponade',
        'code':'Cardiac tamponade',
    },
    {
        'name': 'Atrial fibrillation',
        'code':'Atrial fibrillation',
    },
    {
        'name': 'Acute pulmonary edema',
        'code':'Acute pulmonary edema',
    },
    {
        'name':'Congestive heart failure',
        'code':'Congestive heart failure',
    },
    {
        'name': 'Shortness of breath',
        'code':'Shortness of breath',
    },
    {
        'name': 'Loss of weight',
        'code':'Loss of weight',
    },
    {
        'name': 'Syncope and collapse',
        'code':'Syncope and collapse',
    },
    {
        'name': 'Endometriosis',
        'code':'Endometriosis of ovary',
    },
    {
        'name': 'Pregnancy',
        'code':'Pregnancy examination or test, positive result',
    },
    {
        'name': 'Diabetes in pregnancy',
        'code':'Diabetes mellitus of mother, complicating pregnancy, childbirth, or the puerperium, unspecified as to episode of care',
    },
    {
        'name': 'Prostate cancer',
        'code':'Malignant neoplasm of prostate',
    },
    {
        'name': 'Androgenic alopecia',
        'code':'Androgenic alopecia, unspecified',
    },
    {
        'name': 'Senile osteoporosis',
        'code':'Senile osteoporosis',
    },
    {
        'name': 'Senile dementia',
        'code':'Senile dementia with delirium',
    },
]