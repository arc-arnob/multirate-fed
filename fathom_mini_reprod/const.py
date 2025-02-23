AIR_QUALITY_PATH = '../Datasets/AirQuality/'
ELECTRICITY_PATH = '../Datasets/Electricity/'
SOLAR_PATH = '../Datasets/Solar/'
ROSSMAN_PATH = '../Datasets/Rossman-Sales'
CRYPTO_PATH = '../Datasets/Crypto'

AIR_QUALITY_FILES = ['PRSA_Data_Tiantan_20130301-20170228.csv', 'PRSA_Data_Gucheng_20130301-20170228.csv',
                     'PRSA_Data_Aotizhongxin_20130301-20170228.csv', 'PRSA_Data_Huairou_20130301-20170228.csv',
                     'PRSA_Data_Dingling_20130301-20170228.csv', 'PRSA_Data_Wanshouxigong_20130301-20170228.csv']

ELECTRICITY_FILES = 'LD2011_2014.txt'
ELECTRICITY_PRO_FILES = ['LD2011_2014.txt_MT_003_concatenated.csv', 'LD2011_2014.txt_MT_004_concatenated.csv',
                         'LD2011_2014.txt_MT_001_concatenated.csv', 'LD2011_2014.txt_MT_005_concatenated.csv',
                         'LD2011_2014.txt_MT_006_concatenated.csv', 'LD2011_2014.txt_MT_002_concatenated.csv']

SOLAR_FILES = ['al-pv-2006/Actual_30.55_-88.15_2006_DPV_38MW_5_Min.csv',
               'al-pv-2006/Actual_30.55_-87.75_2006_DPV_36MW_5_Min.csv',
               'al-pv-2006/Actual_30.55_-87.55_2006_UPV_80MW_5_Min.csv',
               'al-pv-2006/Actual_30.45_-88.25_2006_UPV_70MW_5_Min.csv',
               'al-pv-2006/Actual_30.55_-88.25_2006_DPV_38MW_5_Min.csv',
               'al-pv-2006/Actual_30.65_-87.55_2006_UPV_50MW_5_Min.csv'
               ]

ROSSMAN_FILES = 'train.csv'
ROSSMAN_PRO_FILES = 'train_processed.csv'

CRYPTO_FILES = 'train.csv'
CRYPTO_PRO_FILES = [f'train{i}.csv' for i in range(6)]

SOLAR_FILES_MV = [['al-pv-2006/Actual_30.55_-87.75_2006_DPV_36MW_5_Min.csv',
                   'al-pv-2006/Actual_30.55_-88.15_2006_DPV_38MW_5_Min.csv',
                   'al-pv-2006/Actual_30.55_-88.25_2006_DPV_38MW_5_Min.csv',
                   # 'al-pv-2006/Actual_30.65_-87.65_2006_DPV_36MW_5_Min.csv',
                   # 'al-pv-2006/Actual_30.65_-87.75_2006_DPV_36MW_5_Min.csv',
                   'al-pv-2006/Actual_30.65_-87.85_2006_DPV_36MW_5_Min.csv'],
                  ['fl-pv-2006/Actual_25.55_-80.55_2006_DPV_62MW_5_Min.csv',
                   'fl-pv-2006/Actual_25.55_-80.45_2006_DPV_62MW_5_Min.csv',
                   'fl-pv-2006/Actual_25.55_-80.35_2006_DPV_62MW_5_Min.csv',
                   # 'fl-pv-2006/Actual_25.45_-80.65_2006_DPV_62MW_5_Min.csv',
                   # 'fl-pv-2006/Actual_25.45_-80.55_2006_DPV_62MW_5_Min.csv',
                   'fl-pv-2006/Actual_25.45_-80.45_2006_DPV_62MW_5_Min.csv'],
                  ['il-pv-2006/Actual_37.75_-88.95_2006_DPV_27MW_5_Min.csv',
                   'il-pv-2006/Actual_38.45_-89.95_2006_DPV_36MW_5_Min.csv',
                   'il-pv-2006/Actual_38.45_-89.85_2006_DPV_36MW_5_Min.csv',
                   # 'il-pv-2006/Actual_38.85_-89.85_2006_DPV_37MW_5_Min.csv',
                   # 'il-pv-2006/Actual_38.55_-89.95_2006_DPV_36MW_5_Min.csv',
                   'il-pv-2006/Actual_38.85_-89.95_2006_DPV_37MW_5_Min.csv'],
                  ['ks-pv-2006/Actual_37.55_-97.45_2006_DPV_35MW_5_Min.csv',
                   'ks-pv-2006/Actual_37.65_-97.45_2006_DPV_35MW_5_Min.csv',
                   'ks-pv-2006/Actual_37.65_-97.35_2006_DPV_35MW_5_Min.csv',
                   # 'ks-pv-2006/Actual_37.75_-97.45_2006_DPV_35MW_5_Min.csv',
                   # 'ks-pv-2006/Actual_38.75_-94.85_2006_DPV_31MW_5_Min.csv',
                   'ks-pv-2006/Actual_38.85_-94.75_2006_DPV_31MW_5_Min.csv'],
                  ['ma-pv-2006/Actual_42.05_-70.75_2006_DPV_28MW_5_Min.csv',
                   'ma-pv-2006/Actual_42.05_-71.15_2006_DPV_28MW_5_Min.csv',
                   'ma-pv-2006/Actual_41.95_-70.75_2006_DPV_28MW_5_Min.csv',
                   # 'ma-pv-2006/Actual_42.15_-71.15_2006_DPV_28MW_5_Min.csv',
                   # 'ma-pv-2006/Actual_42.15_-72.55_2006_DPV_26MW_5_Min.csv',
                   'ma-pv-2006/Actual_42.15_-72.65_2006_DPV_26MW_5_Min.csv'],
                  ['me-pv-2006/Actual_43.85_-70.25_2006_DPV_23MW_5_Min.csv',
                   'me-pv-2006/Actual_43.85_-70.35_2006_DPV_23MW_5_Min.csv',
                   'me-pv-2006/Actual_43.55_-70.65_2006_DPV_26MW_5_Min.csv',
                   # 'me-pv-2006/Actual_43.95_-69.85_2006_DPV_9MW_5_Min.csv',
                   # 'me-pv-2006/Actual_43.95_-70.35_2006_DPV_23MW_5_Min.csv',
                   'me-pv-2006/Actual_43.45_-70.65_2006_DPV_26MW_5_Min.csv']]

ELECTRICITY_CLUSTER_1 = [[2, 4, 5], [1], [3], [6]]
ELECTRICITY_CLUSTER_2 = [[2, 3, 4, 5, 6], [1]]

INDUSTRY_CLUSTER_1 = [[1, 2, 4], [3], [5], [6]]

# ELECTRICITY_FILES = ['LD2011_2014.txt_MT_003_concatenated.csv', 'LD2011_2014.txt_MT_004_concatenated.csv',
#                      'LD2011_2014.txt_MT_006_concatenated.csv', 'LD2011_2014.txt_MT_001_concatenated.csv',
#                      'LD2011_2014.txt_MT_005_concatenated.csv', 'LD2011_2014.txt_MT_002_concatenated.csv']
#
# SOLAR_FILES = ['Actual_30.55_-88.15_2006_DPV_38MW_5_Min.csv', 'Actual_30.65_-87.55_2006_UPV_50MW_5_Min.csv',
#                'Actual_30.55_-88.25_2006_DPV_38MW_5_Min.csv', 'Actual_30.55_-87.75_2006_DPV_36MW_5_Min.csv',
#                'Actual_30.55_-87.55_2006_UPV_80MW_5_Min.csv', 'Actual_30.45_-88.25_2006_UPV_70MW_5_Min.csv']
