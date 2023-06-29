from google.cloud import bigquery
import pandas as pd
import os
import numpy as np
import shap
from sklearn.metrics import mean_squared_error, mean_absolute_error
import evaluation
from flask import Flask, request, jsonify
from flask_cors import CORS
from pandas.api.types import CategoricalDtype
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import concurrent.futures
import time
from google.cloud import storage
from tensorflow.keras.models import load_model
import os
from sklearn.model_selection import train_test_split
#os.environ['GOOGLE_APPLICATION_CREDENTIALS'] =  'tcloud-ga.json'
bq_client = bigquery.Client()

#get model
sql_query1 ="""
with orderinfo as(
SELECT sme_ban, sum(sol_price) as total_pay, sum(sol_point) as pointsused , sum(sol_selfpay) as selfpay, sum(solution_duration) servicelen, count(order_num) as ordernums, solution_uuid FROM `tcloud-data-analysis.highly_use_data.orders` 
group by sme_ban,solution_uuid 
),
sme as (
  select * from tcloud-data-analysis.ml_data.sme_basic_numeric
),
page as (
  select * from `tcloud-data-analysis.ga3.solution_pv`
),
ind as (
  select sme_ban, ind_large from `tcloud-data-analysis.tcloud_analytics_iii.sme_basic`
)

select orderinfo.* , sme.* EXCEPT(sme_ban), page.* EXCEPT(clean_path2), ind_large
from orderinfo
join sme on orderinfo.sme_ban = sme.sme_ban
join page on orderinfo.solution_uuid = page.clean_path2
join ind on orderinfo.sme_ban= ind.sme_ban
"""
query_job1 = bq_client.query(sql_query1)
recommend = query_job1.to_dataframe()

query_indnm = """
SELECT * FROM `tcloud-data-analysis.tcloud_analytics_iii.industry_large`
"""

# 查詢資料並將結果存為 DataFrame
query_job = bq_client.query(query_indnm)
industry_df = query_job.to_dataframe()

# 提取所有可能的 ind_large 選項
ind_large_values = industry_df['ind_large'].unique()


query_subcate = """
SELECT * FROM `tcloud-data-analysis.tcloud_analytics_iii.solution_subcategory_encoding`
"""

# 查詢資料並將結果存為 DataFrame
query_job = bq_client.query(query_subcate)
solution_sub = query_job.to_dataframe()



# 將 ind_large 轉換為 CategoricalDtype 並指定所有可能的類別
ind_large_type = CategoricalDtype(categories=ind_large_values, ordered=False)
recommend['ind_large'] = recommend['ind_large'].astype(ind_large_type)

# 進行 one-hot encoding
ind_large_dummies = pd.get_dummies(recommend['ind_large'], prefix='ind_large')

# 將所有編碼列轉換為 'Int64' 數據類型
ind_large_dummies = ind_large_dummies.astype('Int64')

# 合併原始 DataFrame 和編碼後的 DataFrame
recommend = pd.concat([recommend.drop('ind_large', axis=1), ind_large_dummies], axis=1)
recommend = recommend.merge(solution_sub, on='solution_uuid', how='left')



# ...其他程式碼(資料讀取等)...

def create_mappings(df, user_col, item_col):
    user_mapping = {user: idx for idx, user in enumerate(df[user_col].unique())}
    item_mapping = {item: idx for idx, item in enumerate(df[item_col].unique())}
    return user_mapping, item_mapping

def encode_data(df, user_col, item_col, user_mapping, item_mapping):
    df[user_col] = df[user_col].map(user_mapping)
    df[item_col] = df[item_col].map(item_mapping)
    return df

def reverse_mappings(mapping):
    return {idx: key for key, idx in mapping.items()}

def save_mappings(user_reverse_mapping, item_reverse_mapping, user_mapping_filename, item_mapping_filename):
    user_reverse_mapping_df = pd.DataFrame(list(user_reverse_mapping.items()), columns=['encoded', 'original'])
    item_reverse_mapping_df = pd.DataFrame(list(item_reverse_mapping.items()), columns=['encoded', 'original'])
    user_reverse_mapping_df.to_csv(user_mapping_filename, index=False)
    item_reverse_mapping_df.to_csv(item_mapping_filename, index=False)

sme_ban_mapping, solution_uuid_mapping = create_mappings(recommend, 'sme_ban', 'solution_uuid')

recommend_encoded = encode_data(recommend.copy(), 'sme_ban', 'solution_uuid', sme_ban_mapping, solution_uuid_mapping)

sme_ban_reverse_mapping = reverse_mappings(sme_ban_mapping)
solution_uuid_reverse_mapping = reverse_mappings(solution_uuid_mapping)

save_mappings(sme_ban_reverse_mapping, solution_uuid_reverse_mapping, 'sme_ban_reverse_mapping.csv', 'solution_uuid_reverse_mapping.csv')

recommend_encoded = recommend_encoded.dropna(axis=0)
# 數據分割
train_data, test_data = train_test_split(recommend_encoded, test_size=0.2, random_state=42)


# 欄位分割
sme_ban_columns = [
    'q_organizationsize_level', 'q_planningtime_level', 'q_budget_level',
    'opscore1', 'opscore2', 'marscore1', 'marscore2', 'salescore1', 'salescore2',
    'securscore1', 'securscore2', 'remotescore1', 'remotescore2', 'schedscore1',
    'schedscore2', 'sme_age', 'capital', 'employee_count',
    'ind_large_A', 'ind_large_B', 'ind_large_C', 'ind_large_D',
    'ind_large_E', 'ind_large_F', 'ind_large_G', 'ind_large_H',
    'ind_large_I', 'ind_large_J', 'ind_large_K', 'ind_large_L',
    'ind_large_M', 'ind_large_N', 'ind_large_P', 'ind_large_Q',
    'ind_large_R', 'ind_large_S'
]

solution_uuid_columns = [
    'pageview', 'bound', 'in_site', 'crm_system', 'erp_system', 'pos_integration', 'seo',
    'hr_management', 'credit_card_ticketing', 'survey_analysis',
    'big_data_analysis', 'customer_interaction', 'market_research',
    'digital_advertising', 'document_processing_software',
    'membership_point_system', 'production_logistics_management',
    'carbon_emission_calculation_analysis',
    'community_content_management_operation', 'sms_system',
    'online_customer_service', 'online_meeting', 'online_reservation',
    'energy_management_system', 'mobile_payment',
    'marketing_matchmaking_kol', 'financial_management',
    'information_security', 'public_opinion_analysis',
    'inventory_management_system', 'remote_collaboration',
    'antivirus_software', 'ecommerce_online_shopping_platform',
    'enewsletter_edm', 'electronic_invoice'
]
interaction_columns = ['total_pay']

# 將訓練集和測試集拆分為用戶編碼、物品編碼和交互作用
train_sme_ban = train_data['sme_ban'].astype('int32')
train_solution_uuid = train_data['solution_uuid'].astype('int32')
train_interactions = train_data[interaction_columns].astype('int32')

test_sme_ban = test_data['sme_ban'].astype('int32')
test_solution_uuid = test_data['solution_uuid'].astype('int32')
test_interactions = test_data[interaction_columns].astype('int32')

# 分別獲取訓練集和測試集中的用戶和物品特徵
train_sme_ban_features = train_data[sme_ban_columns].astype('int32')
train_solution_uuid_features = train_data[solution_uuid_columns].astype('int32')

test_sme_ban_features = test_data[sme_ban_columns].astype('int32')
test_solution_uuid_features = test_data[solution_uuid_columns].astype('int32')


start_time = time.time()

# 超参数设置
embedding_dim = 16
dense_layer_sizes = (128, 64)
learning_rate = 0.0005
epochs = 30
batch_size = 32

# 找到可能的最大 SME_BAN 和 Solution_UUID
max_sme_ban = max(train_sme_ban.max(), 100000)
max_solution_uuid = max(train_solution_uuid.max(), 10000)

# SME_BAN 输入
sme_ban_input = Input(shape=(1,), name='sme_ban_input')
sme_ban_embedding = Embedding(input_dim=max_sme_ban + 1, output_dim=embedding_dim, mask_zero=True, name='sme_ban_embedding')(sme_ban_input)
sme_ban_vec = Flatten()(sme_ban_embedding)

sme_ban_features_input = Input(shape=(len(sme_ban_columns),), name='sme_ban_features_input')
sme_ban_combined = Concatenate()([sme_ban_vec, sme_ban_features_input])

# Solution_UUID 输入
solution_uuid_input = Input(shape=(1,), name='solution_uuid_input')
solution_uuid_embedding = Embedding(input_dim=max_solution_uuid + 1, output_dim=embedding_dim, mask_zero=True, name='solution_uuid_embedding')(solution_uuid_input)
solution_uuid_vec = Flatten()(solution_uuid_embedding)

solution_uuid_features_input = Input(shape=(len(solution_uuid_columns),), name='solution_uuid_features_input')
solution_uuid_combined = Concatenate()([solution_uuid_vec, solution_uuid_features_input])

# 将用户和物品组合在一起
combined = Concatenate()([sme_ban_combined, solution_uuid_combined])

# 添加全连接层
dense = Dense(dense_layer_sizes[0], activation='relu')(combined)
dense = Dense(dense_layer_sizes[1], activation='relu')(dense)

# 输出层：预测 total_pay
total_pay_output = Dense(1, activation='linear', name='total_pay_output')(dense)

# 创建模型
model = Model(inputs=[sme_ban_input, sme_ban_features_input, solution_uuid_input, solution_uuid_features_input], outputs=[total_pay_output])

# 编译模型
optimizer = Adam(lr=learning_rate)
model.compile(optimizer=optimizer, loss='mse')



# 设置 Google Cloud 的认证环境变量
#os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'tcloud-ga.json'

# 初始化 GCS 客户端
client = storage.Client()

# 从 GCS 下载模型文件
bucket_name = 'model_of_ncf'
blob_name = 'model.h5'
bucket = client.get_bucket(bucket_name)
blob = bucket.blob(blob_name)
blob.download_to_filename('model.h5')

# 加载模型
model = load_model('model.h5')


# 從訓練集中選擇一組輸入
example_index = 0
example_sme_ban = np.array([train_sme_ban.iloc[example_index]])
example_sme_ban_features = train_sme_ban_features.iloc[[example_index]]
example_solution_uuid = np.array([train_solution_uuid.iloc[example_index]])
example_solution_uuid_features = train_solution_uuid_features.iloc[[example_index]]

# 使用模型進行預測
example_output = model.predict([example_sme_ban, example_sme_ban_features, example_solution_uuid, example_solution_uuid_features])
print(f"Example output: {example_output}")

def recommend_items_for_user(user_id, model, sme_ban_features, solution_uuid_features, top_n=5):
    # 從映射中找到用戶的編碼
    encoded_user_id = sme_ban_mapping[user_id]

    # 獲取已互動過的物品
    interacted_items_encoded = recommend_encoded[recommend_encoded['sme_ban'] == encoded_user_id]['solution_uuid'].unique()
    interacted_items = [solution_uuid_reverse_mapping[item] for item in interacted_items_encoded]

    # 找到未互動過的物品
    all_items = set(solution_uuid_mapping.keys())
    not_interacted_items = list(all_items - set(interacted_items))

    # 準備用於預測的輸入數據
    user_input = np.array([encoded_user_id] * len(not_interacted_items))
    not_interacted_items_encoded = [solution_uuid_mapping[item] for item in not_interacted_items]
    item_input = np.array(not_interacted_items_encoded)
    user_features_input = np.repeat(sme_ban_features.loc[encoded_user_id].values.reshape(1, -1), len(not_interacted_items), axis=0)
    item_features_input = solution_uuid_features.loc[not_interacted_items_encoded].values

    # 進行預測
    predictions = model.predict([user_input, user_features_input, item_input, item_features_input])

    # 獲取前 N 個物品的索引
    top_n_indices = predictions[:, 0].argsort()[-top_n:][::-1]

    # 獲取前 N 個物品的編碼
    top_n_items_encoded = item_input[top_n_indices]

    # 將物品編碼轉換為原始 UUID
    top_n_items = [solution_uuid_reverse_mapping[item] for item in top_n_items_encoded]

    return top_n_items

def prepare_new_user_input(user_features_df, num_items):
    # 檢查新用戶特徵是否具有正確的列順序
    if not all(user_features_df.columns == sme_ban_columns):
        raise ValueError("The columns of the new user features DataFrame must match the order of sme_ban_columns.")
    
    # 為新用戶分配一個編碼（可以選擇大於所有現有編碼的最大值的數字）
    new_user_encoded = max(sme_ban_mapping.values()) + 1

    # 準備輸入數據
    user_input = np.array([new_user_encoded] * num_items)
    user_features_input = np.repeat(user_features_df.values.reshape(1, -1), num_items, axis=0)

    return user_input, user_features_input

# 假設以下是新用戶的特徵
new_user_features = pd.DataFrame([{
    'q_organizationsize_level': 3, 'q_planningtime_level': 2, 'q_budget_level': 1,
    'opscore1': 4, 'opscore2': 3, 'marscore1': 3, 'marscore2': 2, 'salescore1': 1, 'salescore2': 2,
    'securscore1': 2, 'securscore2': 3, 'remotescore1': 1, 'remotescore2': 2, 'schedscore1': 2,
    'schedscore2': 3, 'sme_age': 5, 'capital': 10000, 'employee_count': 50,
    'ind_large_A': 0, 'ind_large_B': 1, 'ind_large_C': 0, 'ind_large_D': 0,
    'ind_large_E': 0, 'ind_large_F': 0, 'ind_large_G': 0, 'ind_large_H': 0,
    'ind_large_I': 0, 'ind_large_J': 0, 'ind_large_K': 0, 'ind_large_L': 0,
    'ind_large_M': 0, 'ind_large_N': 0, 'ind_large_P': 0, 'ind_large_Q': 0,
    'ind_large_R': 0, 'ind_large_S': 0
}], columns=sme_ban_columns)
# 在這裡，我們使用所有物品進行預測
num_items = len(solution_uuid_mapping)

# 調用函數，為新用戶準備輸入數據
new_user_input, new_user_features_input = prepare_new_user_input(new_user_features, num_items)

# 選擇一個物品
example_item_index = 2
example_solution_uuid = np.array([train_solution_uuid.iloc[example_item_index]])
example_solution_uuid_features = train_solution_uuid_features.iloc[[example_item_index]]

# 使用函數為新用戶準備輸入數據
new_user_input, new_user_features_input = prepare_new_user_input(new_user_features, 1)

# 使用模型進行預測
example_output = model.predict([new_user_input, new_user_features_input, example_solution_uuid, example_solution_uuid_features])
print(f"Example output: {example_output}")

# 获取所有上架产品的数据
sql_query2 = """
SELECT solution_uuid FROM `tcloud-data-analysis.tcloud_analytics_iii.solution_info` 
WHERE solution_status ='上架' AND solution_uuid IN (SELECT DISTINCT(solution_uuid) FROM tcloud-data-analysis.tcloud_analytics_iii.order_basic)
"""

query_job2 = bq_client.query(sql_query2)
on_shelf_solutions = query_job2.to_dataframe()
on_shelf_item_ids = on_shelf_solutions['solution_uuid'].tolist()
# 合并 train_solution_uuid_features 和 test_solution_uuid_features
all_solution_uuid_features = pd.concat([train_solution_uuid_features, test_solution_uuid_features], axis=0)


# 从所有数据集中提取已上架物品的特征
on_shelf_item_encoded = []
on_shelf_solution_uuid_features = all_solution_uuid_features.loc[on_shelf_item_encoded]

# 使用 solution_uuid_mapping 将 item id 映射回所有数据集中的编码，并忽略没有映射到的数据

for item_id in on_shelf_item_ids:
    try:
        encoded_id = solution_uuid_mapping[item_id]
        on_shelf_item_encoded.append(encoded_id)
    except KeyError:
        continue
# 使用 solution_uuid_mapping 将 item id 映射回所有数据集中的编码，并忽略没有映射到的数据
on_shelf_item_encoded = []
for item_id in on_shelf_item_ids:
    try:
        encoded_id = solution_uuid_mapping[item_id]
        if encoded_id in all_solution_uuid_features.index:   # 新增的检查
            on_shelf_item_encoded.append(encoded_id)
    except KeyError:
        continue

# 从所有数据集中提取已上架物品的特征
on_shelf_solution_uuid_features = all_solution_uuid_features.loc[on_shelf_item_encoded]

# 确认映射后的列表长度
print(f"Length of the encoded on-shelf items list: {len(on_shelf_item_encoded)}")
"""對新用戶進行預測"""
import time

start_time = time.time()

# 预测结果的 DataFrame
predictions = []

# 对每个上架产品进行预测
for item_encoded in on_shelf_item_encoded:
    encoded_solution_uuid = np.array([item_encoded])
    encoded_solution_uuid_features = on_shelf_solution_uuid_features.loc[[item_encoded]]

    # 为新用户准备输入数据
    new_user_input, new_user_features_input = prepare_new_user_input(new_user_features, 1)

    # 使用模型进行预测
    output = model.predict([new_user_input, new_user_features_input, encoded_solution_uuid, encoded_solution_uuid_features])

    # 将预测结果添加到列表中
    predictions.append({
        'solution_uuid_encoded': item_encoded,
        'total_pay': output[0][0]
    })

# 将预测结果列表转换为 DataFrame
predictions_df = pd.DataFrame(predictions)

# 显示预测结果
predictions_df.head(5)

# 挑选 total_pay 最高的前五个 solution_uuid_encoded
top5_encoded = predictions_df.nlargest(5, 'total_pay')['solution_uuid_encoded']

# 使用 solution_uuid_reverse_mapping 将 solution_uuid_encoded 解码回原始的 item id
top5_item_ids = [solution_uuid_reverse_mapping[encoded] for encoded in top5_encoded]


end_time = time.time()

print(f"Time taken: {end_time - start_time} seconds")

def make_prediction(item_encoded):
    encoded_solution_uuid = np.array([item_encoded])
    encoded_solution_uuid_features = on_shelf_solution_uuid_features.loc[[item_encoded]]

    # 为新用户准备输入数据
    new_user_input, new_user_features_input = prepare_new_user_input(new_user_features, 1)

    # 使用模型进行预测
    output = model.predict([new_user_input, new_user_features_input, encoded_solution_uuid, encoded_solution_uuid_features])

    return {
        'solution_uuid_encoded': item_encoded,
        'total_pay': output[0][0]
    }

start_time = time.time()

# 预测结果的 DataFrame
predictions = []

# 使用线程池进行并行处理
with concurrent.futures.ThreadPoolExecutor() as executor:
    predictions = list(executor.map(make_prediction, on_shelf_item_encoded))

# 将预测结果列表转换为 DataFrame
predictions_df = pd.DataFrame(predictions)

# 显示预测结果
predictions_df.head(5)

# 挑选 total_pay 最高的前五个 solution_uuid_encoded
top5_encoded = predictions_df.nlargest(5, 'total_pay')['solution_uuid_encoded']

# 使用 solution_uuid_reverse_mapping 将 solution_uuid_encoded 解码回原始的 item id
top5_item_ids = [solution_uuid_reverse_mapping[encoded] for encoded in top5_encoded]

end_time = time.time()

print(f"Time taken: {end_time - start_time} seconds")
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    sme_ban = data['sme_ban']
    user_features_dict = data['features']
    
    # Convert the features to a DataFrame
    user_features_df = pd.DataFrame([user_features_dict], columns=sme_ban_columns)
    
    # Prepare the input data for the new user
    new_user_input, new_user_features_input = prepare_new_user_input(user_features_df, 1)
    
    # Predict total_pay for all on-shelf items and keep the results in a list
    predictions = []
    for item_encoded in on_shelf_item_encoded:
        encoded_solution_uuid = np.array([item_encoded])
        encoded_solution_uuid_features = on_shelf_solution_uuid_features.loc[[item_encoded]]

        # Use the model to make a prediction
        output = model.predict([new_user_input, new_user_features_input, encoded_solution_uuid, encoded_solution_uuid_features])

        # Add the prediction result to the list
        predictions.append({
            'solution_uuid_encoded': item_encoded,
            'total_pay': output[0][0]
        })

    # Convert the prediction result list into a DataFrame
    predictions_df = pd.DataFrame(predictions)

    # Select the top 5 items with highest predicted total_pay
    top5_encoded = predictions_df.nlargest(5, 'total_pay')['solution_uuid_encoded']

    # Use the reverse mapping to decode the encoded item ids back to the original item ids
    top5_item_ids = [solution_uuid_reverse_mapping[encoded] for encoded in top5_encoded]

    return jsonify({'top_5_item_ids': top5_item_ids})

if __name__ == '__main__':
    import os
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
