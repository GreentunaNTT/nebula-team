import pandas as pd
import numpy as np
import lightgbm as lgb
from datetime import timedelta
import sys
import warnings

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

warnings.filterwarnings('ignore')

# ==========================================
# 1. TẢI DỮ LIỆU & XỬ LÝ SỐ THẬP PHÂN CHUẨN XÁC
# ==========================================
print("--- Bước 1: Khởi tạo dữ liệu và làm sạch định dạng ---")
train = pd.read_csv('train.csv')
train['Date'] = pd.to_datetime(train['Date'])

num_cols = ['Unit Cost', 'Cost Amount', 'Quantity', 'UnitPrice', 'SalesAmount']
for col in num_cols:
    if col in train.columns:
        normalized = (
            train[col]
            .astype(str)
            .str.replace('"', '', regex=False)
            .str.replace(',', '.', regex=False)
            .str.strip()
            .replace({'nan': np.nan, 'None': np.nan, '': np.nan})
        )
        train[col] = pd.to_numeric(normalized, errors='coerce')
        if col == 'Quantity':
            train[col] = train[col].fillna(0)

daily_data = train.groupby(['ItemCode', 'Date']).agg({
    'Quantity': 'sum',
    'UnitPrice': 'mean',
    'Unit Cost': 'mean'
}).reset_index()
daily_data['Quantity'] = daily_data['Quantity'].clip(lower=0)
# ==========================================
# 2. TẠO LƯỚI THỜI GIAN & KHỬ NHIỄU LEADING ZEROS
# ==========================================
print("--- Bước 2: Khử nhiễu chuỗi số 0 và tạo lưới thời gian đầy đủ ---")
min_date = daily_data['Date'].min()
max_train_date = daily_data['Date'].max()
all_dates = pd.date_range(start=min_date, end=max_train_date, freq='D')
all_items = daily_data['ItemCode'].unique()

grid = pd.MultiIndex.from_product([all_items, all_dates], names=['ItemCode', 'Date']).to_frame().reset_index(drop=True)
grid = grid.merge(daily_data, on=['ItemCode', 'Date'], how='left')
grid['Quantity'] = grid['Quantity'].fillna(0)

first_sales = daily_data[daily_data['Quantity'] > 0].groupby('ItemCode')['Date'].min().reset_index()
first_sales.columns = ['ItemCode', 'FirstSaleDate']
grid = grid.merge(first_sales, on='ItemCode', how='left')
grid = grid[grid['Date'] >= grid['FirstSaleDate']].drop(columns=['FirstSaleDate'])

grid = grid.sort_values(['ItemCode', 'Date']).reset_index(drop=True)
grid['UnitPrice'] = grid.groupby('ItemCode')['UnitPrice'].ffill().bfill()
grid['Unit Cost'] = grid.groupby('ItemCode')['Unit Cost'].ffill().bfill()

# ==========================================
# 3. KỸ THUẬT ĐẶC TRƯNG BẬC CAO (KHÔNG GÂY LỆCH PHA)
# ==========================================
print("--- Bước 3: Triển khai kỹ thuật đặc trưng chuyên sâu ---")

max_prices = grid.groupby('ItemCode')['UnitPrice'].max().reset_index().rename(columns={'UnitPrice': 'max_price'})
grid = grid.merge(max_prices, on='ItemCode', how='left')
grid['discount_ratio'] = grid['UnitPrice'] / (grid['max_price'] + 1e-5)
grid['margin'] = (grid['UnitPrice'] - grid['Unit Cost']) / (grid['UnitPrice'] + 1e-5)

# Tính toán bảng tra cứu doanh số theo Thứ (DOW) độc lập để map chính xác khi Test
grid['dayofweek_temp'] = grid['Date'].dt.dayofweek
item_dow_mean = grid.groupby(['ItemCode', 'dayofweek_temp'])['Quantity'].mean().reset_index().rename(columns={'Quantity': 'item_dow_mean'})
grid = grid.merge(item_dow_mean, on=['ItemCode', 'dayofweek_temp'], how='left').drop(columns=['dayofweek_temp'])

# Tính toán các cửa sổ trượt nền tảng
grid['rmean_7'] = grid.groupby('ItemCode')['Quantity'].transform(lambda x: x.rolling(7).mean())
grid['rmean_14'] = grid.groupby('ItemCode')['Quantity'].transform(lambda x: x.rolling(14).mean())
grid['rmean_30'] = grid.groupby('ItemCode')['Quantity'].transform(lambda x: x.rolling(30).mean())
grid['rmean_60'] = grid.groupby('ItemCode')['Quantity'].transform(lambda x: x.rolling(60).mean())
grid['rstd_7'] = grid.groupby('ItemCode')['Quantity'].transform(lambda x: x.rolling(7).std())
grid['rstd_30'] = grid.groupby('ItemCode')['Quantity'].transform(lambda x: x.rolling(30).std())

test_base = grid[grid['Date'] == max_train_date].copy()

# ==========================================
# 4. TRỰC TIẾP HUẤN LUYỆN 56 MÔ HÌNH TOÀN DIỆN (ĐÃ SỬA LỖI)
# ==========================================
print("--- Bước 4: Khởi chạy luồng huấn luyện tối ưu hóa sâu ---")

submission_results = []

# Đổi lại Objective chuẩn Tweedie cho retail data
lgb_params = {
    'objective': 'tweedie',
    'tweedie_variance_power': 1.2,
    'metric': 'rmse',
    'learning_rate': 0.04,
    'num_leaves': 63,
    'min_data_in_leaf': 30,
    'feature_fraction': 0.7,
    'bagging_fraction': 0.7,
    'bagging_freq': 1,
    'extra_trees': True,
    'seed': 2026,
    'verbose': -1,
    'n_jobs': -1
}

features = [
    'day', 'month', 'dayofweek', 'is_weekend', 'item_dow_mean',
    'feat_discount', 'feat_margin',
    'feat_lag_0', 'feat_rmean_7', 'feat_rmean_14', 'feat_rmean_30', 'feat_rmean_60', 
    'feat_rstd_7', 'feat_rstd_30'
]

for h in range(1, 57):
    target_date = max_train_date + timedelta(days=h)
    
    df_h = grid[['ItemCode', 'Date', 'Quantity', 'item_dow_mean']].copy()
    
    # SỬA LỖI 3A: Dịch chuyển cả tính năng Giá theo đúng tầm nhìn lịch sử h
    df_h['feat_discount'] = grid.groupby('ItemCode')['discount_ratio'].shift(h)
    df_h['feat_margin'] = grid.groupby('ItemCode')['margin'].shift(h)
    
    # Dịch chuyển chuỗi thời gian
    df_h['feat_lag_0'] = df_h.groupby('ItemCode')['Quantity'].shift(h)
    df_h['feat_rmean_7'] = grid.groupby('ItemCode')['rmean_7'].shift(h)
    df_h['feat_rmean_14'] = grid.groupby('ItemCode')['rmean_14'].shift(h)
    df_h['feat_rmean_30'] = grid.groupby('ItemCode')['rmean_30'].shift(h)
    df_h['feat_rmean_60'] = grid.groupby('ItemCode')['rmean_60'].shift(h)
    df_h['feat_rstd_7'] = grid.groupby('ItemCode')['rstd_7'].shift(h)
    df_h['feat_rstd_30'] = grid.groupby('ItemCode')['rstd_30'].shift(h)
    
    df_h['day'] = df_h['Date'].dt.day
    df_h['month'] = df_h['Date'].dt.month
    df_h['dayofweek'] = df_h['Date'].dt.dayofweek
    df_h['is_weekend'] = df_h['dayofweek'].isin([5, 6]).astype(int)
    
    # SỬA LỖI 2: KHÔNG .dropna() toàn bộ nữa, chỉ loại bỏ dòng khuyết lag_0 gốc
    df_h = df_h[df_h['feat_lag_0'].notna()]
    
    train_mask = df_h['Date'] <= (max_train_date - timedelta(days=28))
    val_mask = (df_h['Date'] > (max_train_date - timedelta(days=28))) & (df_h['Date'] <= max_train_date)
    
    X_train, y_train = df_h[train_mask][features], df_h[train_mask]['Quantity'].clip(lower=0)
    X_val, y_val = df_h[val_mask][features], df_h[val_mask]['Quantity'].clip(lower=0)
    
    train_set = lgb.Dataset(X_train, label=y_train)
    val_set = lgb.Dataset(X_val, label=y_val, reference=train_set)
    
    model = lgb.train(
        lgb_params,
        train_set,
        valid_sets=[train_set, val_set],
        num_boost_round=600,
        callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)]
    )
    
    # Xây dựng tập test đầu vào sạch cho Horizon h
    X_test = test_base.copy()
    X_test['feat_lag_0'] = X_test['Quantity']
    X_test['feat_rmean_7'] = X_test['rmean_7']
    X_test['feat_rmean_14'] = X_test['rmean_14']
    X_test['feat_rmean_30'] = X_test['rmean_30']
    X_test['feat_rmean_60'] = X_test['rmean_60']
    X_test['feat_rstd_7'] = X_test['rstd_7']
    X_test['feat_rstd_30'] = X_test['rstd_30']
    X_test['feat_discount'] = X_test['discount_ratio']
    X_test['feat_margin'] = X_test['margin']
    
    X_test['day'] = target_date.day
    X_test['month'] = target_date.month
    X_test['dayofweek'] = target_date.dayofweek
    X_test['is_weekend'] = int(target_date.dayofweek in [5, 6])
    
    # SỬA LỖI 3B: Map lại chính xác đặc trưng Thứ của ngày tương lai (target_date)
    X_test = X_test.drop(columns=['item_dow_mean']).merge(
        item_dow_mean.rename(columns={'dayofweek_temp': 'dayofweek'}),
        on=['ItemCode', 'dayofweek'],
        how='left'
    )
    
    preds = model.predict(X_test[features])
    preds_optimized = np.clip(preds, 0, None) * 0.95  # Ép chặt hệ số nhân giảm overforecast cho Tweedie
    
    pred_df = pd.DataFrame({
        'ItemCode': X_test['ItemCode'],
        'Horizon': h,
        'Prediction': preds_optimized
    })
    submission_results.append(pred_df)
    
    if h % 7 == 0 or h == 56:
        print(f"   -> Hoàn thành xuất sắc mô hình cho Ngày thứ {h}/56")
# ==========================================
# 5. CHUYỂN ĐỔI ĐỊNH DẠNG SUBMISSION TỐI ƯU
# ==========================================
print("--- Bước 5: Tạo file nộp bài chuẩn cấu trúc cuộc thi ---")
all_preds = pd.concat(submission_results, ignore_index=True)

all_preds['Phase'] = np.where(all_preds['Horizon'] <= 28, 'validation', 'evaluation')
all_preds['F_col'] = np.where(
    all_preds['Horizon'] <= 28, 
    'F' + all_preds['Horizon'].astype(str),
    'F' + (all_preds['Horizon'] - 28).astype(str)
)
all_preds['id'] = all_preds['ItemCode'] + '_' + all_preds['Phase']

sub_pivot = all_preds.pivot(index='id', columns='F_col', values='Prediction').reset_index()
f_cols = [f'F{i}' for i in range(1, 29)]
sub_pivot = sub_pivot[['id'] + f_cols]

sample_sub = pd.read_csv('sample_submission.csv')
final_submission = sample_sub[['id']].merge(sub_pivot, on='id', how='left').fillna(0)

final_submission.to_csv('submission_ultra_grandmaster.csv', index=False)
print("--- KHỞI TẠO THÀNH CÔNG: File 'submission_ultra_grandmaster.csv'---")
