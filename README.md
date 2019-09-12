# ESUN-HousePricePrediction
之前參加了玉山-人工智慧夏季挑戰賽房價預測
在1333隊中位列290名(前21%)

底下為程式碼
    
    #匯入相關套件
    import pandas as pd
    import numpy as np
    from scipy.special import boxcox1p
    from sklearn.metrics import explained_variance_score
    from sklearn.metrics import mean_squared_error
    from sklearn.metrics import r2_score
    import seaborn as sns
    from sklearn.model_selection import train_test_split
    from xgboost import XGBRegressor
    import matplotlib.pyplot as plt
    from sklearn import preprocessing
    from sklearn.model_selection import cross_val_score
    import warnings
    warnings.filterwarnings('ignore')
    
    # numpy 不顯示科學記號
    np.set_printoptions(suppress=False)

    raw_data = pd.read_csv('Desktop/dataset-0510/train.csv')
    raw_data = raw_data[['N_50','N_500','N_1000','N_5000','N_10000',
    'I_10','I_50','I_index_50','I_100','I_250','I_500','I_index_500','I_1000','I_index_1000','I_5000','I_index_5000','I_10000','I_index_10000','I_MIN'
    ,'II_10','II_50','II_index_50','II_100','II_250','II_500','II_index_500','II_1000','II_index_1000','II_5000','II_index_5000','II_10000','II_index_10000','II_MIN'
    ,'III_10','III_50','III_index_50','III_100','III_250','III_500','III_index_500','III_1000','III_index_1000','III_5000','III_index_5000','III_10000','III_index_10000','III_MIN'
    ,'IV_10','IV_50','IV_index_50','IV_100','IV_250','IV_500','IV_index_500','IV_1000','IV_index_1000','IV_5000','IV_index_5000','IV_10000','IV_index_10000','IV_MIN'
    ,'V_10','V_50','V_index_50','V_100','V_250','V_500','V_index_500','V_1000','V_index_1000','V_5000','V_index_5000','V_10000','V_index_10000','V_MIN'
    ,'VI_10','VI_50','VI_index_50','VI_100','VI_250','VI_500','VI_index_500','VI_1000','VI_index_1000','VI_5000','VI_index_5000','VI_10000','VI_index_10000','VI_MIN'
    ,'VII_10','VII_50','VII_index_50','VII_100','VII_250','VII_500','VII_index_500','VII_1000','VII_5000','VII_index_5000','VII_10000','VII_index_10000','VII_MIN'
    ,'VIII_10','VIII_50','VIII_index_50','VIII_100','VIII_250','VIII_500','VIII_index_500','VIII_1000','VIII_index_1000','VIII_5000','VIII_index_5000','VIII_10000','VIII_index_10000','VIII_MIN'
    ,'IX_10','IX_50','IX_index_50','IX_100','IX_250','IX_500','IX_index_500','IX_1000','IX_index_1000','IX_5000','IX_index_5000','IX_10000','IX_index_10000','IX_MIN'
    ,'X_10','X_50','X_index_50','X_100','X_250','X_500','X_index_500','X_1000','X_index_1000','X_5000','X_index_5000','X_10000','X_index_10000','X_MIN'
    ,'XI_10','XI_50','XI_index_50','XI_100','XI_250','XI_500','XI_index_500','XI_1000','XI_index_1000','XI_5000','XI_index_5000','XI_10000','XI_index_10000','XI_MIN'
    ,'XII_10','XII_50','XII_index_50','XII_100','XII_250','XII_500','XII_index_500','XII_1000','XII_index_1000','XII_5000','XII_index_5000','XII_10000','XII_index_10000','XII_MIN'
    ,'XIII_10','XIII_50','XIII_index_50','XIII_100','XIII_250','XIII_500','XIII_index_500','XIII_1000','XIII_index_1000','XIII_5000','XIII_index_5000','XIII_10000','XIII_index_10000','XIII_MIN'
    ,'XIV_10','XIV_50','XIV_index_50','XIV_100','XIV_250','XIV_500','XIV_index_500','XIV_1000','XIV_index_1000','XIV_5000','XIV_index_5000','XIV_10000','XIV_index_10000','XIV_MIN'
    ,'building_type','building_use','town','city'
    ,'total_floor','txn_floor','building_area','parking_price'
    ,'txn_dt','building_complete_dt','village_income_median','town_population_density'
    ,'total_price']]

    # raw_data columns into x_columns for sub data use
    # x_columns drop (total_price) because sub data doesn't have it
    x_columns = raw_data.columns
    x_columns = x_columns.drop(['total_price'])

    # new_f1 為 txn_dt 與 building_complete_dt 差異
    # raw_data['new_f1'] = raw_data['txn_dt'] - raw_data['building_complete_dt']

    # Fill NaN with 0
    raw_data = raw_data.fillna(0)

    # # Y 透過 log 轉換
    raw_data['total_price_log'] = np.log(raw_data['total_price'])

    # city 變為 dummies
    city_dummies = ['city']
    raw_data = pd.get_dummies(raw_data , columns=city_dummies)

    # building_use 變為 dummies
    building_use_dummies = ['building_use']
    raw_data = pd.get_dummies(raw_data , columns=building_use_dummies)

    # building_type 變為 dummies
    building_type_dummies = ['building_type']
    raw_data = pd.get_dummies(raw_data , columns=building_type_dummies)

    # 若 txn_floor 為 0，將其改為總樓層(total_floor)
    raw_data.txn_floor[raw_data.txn_floor == 0] = raw_data['total_floor']

    # 變數透過 log 轉換並存成新的欄位
    raw_data['ba_log'] = raw_data['building_area']
    raw_data.ba_log[raw_data.ba_log > 0] = np.log(raw_data['ba_log'])

    raw_data['tpd_log'] = raw_data['town_population_density']
    raw_data.tpd_log[raw_data.tpd_log > 0] = np.log(raw_data['tpd_log'])

    raw_data['vim_log'] = raw_data['village_income_median']
    raw_data.vim_log[raw_data.vim_log > 0] = np.log(raw_data['vim_log'])

    # 為了方便，將欄位名稱存入變數
    x_features = raw_data.columns
    x_features = x_features.drop(['village_income_median','town_population_density','building_area','total_price','total_price_log'])
    x = raw_data[x_features]
    y = raw_data['total_price_log'].values

    # ### 機器學習 Linear regression ###
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 100)
    xgb = XGBRegressor(max_depth=15)
    xgb.fit(x_train,y_train) #訓練資料丟入線性回歸的機器學習

    # 預測資料(test)
    y_predict = xgb.predict(x_test)
    plt.scatter(y_test,y_predict)

    ### 機器學習 Linear regression ###
    scores = cross_val_score(xgb,x,y,cv=10)

    print('XGBRegressor MSE:',mean_squared_error(y_test, y_predict))
    print('XGBRegressor RMSE:',np.sqrt(mean_squared_error(y_test, y_predict)))
    print('XGBRegressor R2 score',r2_score(y_test,y_predict))
    print('XGBRegressor Explained_variance_score',explained_variance_score(y_test,y_predict))
    print('Mean of XGBRegressor',np.mean(scores))

    ###################################################### 讀取比賽資料 ######################################################
    sub = pd.read_csv('Desktop/dataset-0510/test.csv')
    test_id = sub['building_id']

    # 取特定欄位
    sub = sub[x_columns]

    # new_f1 為 txn_dt 與 building_complete_dt 差異
    # sub['new_f1'] = sub['txn_dt'] - sub['building_complete_dt']

    # 將所有空值補 0
    sub = sub.fillna(0)

    # city 變為 dummies
    sub = pd.get_dummies(sub , columns=city_dummies)

    # building_use 變為 dummies
    sub = pd.get_dummies(sub , columns=building_use_dummies)

    # building_type 變為 dummies
    sub = pd.get_dummies(sub , columns=building_type_dummies)

    # 若 txn_floor 為 0，將其改為總樓層(total_floor)
    sub.txn_floor[sub.txn_floor == 0] = sub['total_floor']

    # 將欄位經過 log 轉換
    sub['ba_log'] = sub['building_area']
    sub.ba_log[sub.ba_log > 0] = np.log(sub['ba_log'])

    sub['tpd_log'] = sub['town_population_density']
    sub.tpd_log[sub.tpd_log > 0] = np.log(sub['tpd_log'])

    sub['vim_log'] = sub['village_income_median']
    sub.vim_log[sub.vim_log > 0] = np.log(sub['vim_log'])
    # 變數透過 log 轉換 ##結束##

    x = sub[x_features]

    # 將資料丟入線性回歸
    y_predict = xgb.predict(x)
    expo = np.exp(y_predict)
    # np.savetxt("Desktop/dataset-0510/predict0710.csv", expo, delimiter=",",fmt='%f')
    output = pd.DataFrame({'building_id': test_id,
                           'total_price': expo})
    output.to_csv('Desktop/dataset-0510/submission_test0715.csv', index=False,float_format ='%f')
    print(expo)
