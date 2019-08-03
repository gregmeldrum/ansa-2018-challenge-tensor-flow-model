from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pandas as pd

import tensorflow as tf

from tensorflow import feature_column
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

# A utility method to create a tf.data dataset from a Pandas Dataframe
def df_to_dataset(dataframe, shuffle=True, batch_size=32):
  dataframe = dataframe.copy()
  labels = dataframe.pop('ClaimClass')
  ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
  if shuffle:
    ds = ds.shuffle(buffer_size=len(dataframe))
  ds = ds.batch(batch_size)
  return ds

URL = 'file:///tensor_flow/training_data.csv'
dataframe = pd.read_csv(URL)
dataframe.head()

train, test = train_test_split(dataframe, test_size=0.2)
train, val = train_test_split(train, test_size=0.2)
print(len(train), 'train examples')
print(len(val), 'validation examples')
print(len(test), 'test examples')

# Setup feature columns
feature_columns = []

# numeric cols
for header in ['Income', 'MonthlyPremiumAuto' , 'MonthsSinceLastClaim', 'MonthsSincePolicyInception', 'NumberofOpenComplaints' , 'NumberofPolicies']:
    feature_columns.append(feature_column.numeric_column(header))

# indicator cols
education = feature_column.categorical_column_with_vocabulary_list(
      'Education', ['High School or Below', 'College', 'Doctor', 'Bachelor', 'Master'])
education_one_hot = feature_column.indicator_column(education)
feature_columns.append(education_one_hot)

employment_status = feature_column.categorical_column_with_vocabulary_list(
      'EmploymentStatus', ['Employed', 'Unemployed', 'Disabled', 'Medical Leave', 'Retired'])
employment_status_one_hot = feature_column.indicator_column(employment_status)
feature_columns.append(employment_status_one_hot)

marital_status = feature_column.categorical_column_with_vocabulary_list(
      'MaritalStatus', ['Married', 'Single', 'Divorced'])
marital_status_one_hot = feature_column.indicator_column(marital_status)
feature_columns.append(marital_status_one_hot)

state_code = feature_column.categorical_column_with_vocabulary_list(
      'StateCode', ['MO', 'IA', 'NE', 'OK', 'KS'])
state_code_one_hot = feature_column.indicator_column(state_code)
feature_columns.append(state_code_one_hot)

coverage = feature_column.categorical_column_with_vocabulary_list(
      'Coverage', ['Premium', 'Basic', 'Extended'])
coverage_one_hot = feature_column.indicator_column(coverage)
feature_columns.append(coverage_one_hot)

gender = feature_column.categorical_column_with_vocabulary_list(
      'Gender', ['M', 'F'])
gender_one_hot = feature_column.indicator_column(gender)
feature_columns.append(gender_one_hot)

location_code = feature_column.categorical_column_with_vocabulary_list(
      'LocationCode', ['Suburban', 'Urban', 'Rural'])
location_code_one_hot = feature_column.indicator_column(location_code)
feature_columns.append(location_code_one_hot)

claim_reason = feature_column.categorical_column_with_vocabulary_list(
      'ClaimReason', ['Hail', 'Collision', 'Scratch/Dent', 'Other'])
claim_reason_one_hot = feature_column.indicator_column(claim_reason)
feature_columns.append(claim_reason_one_hot)

sales_channel = feature_column.categorical_column_with_vocabulary_list(
      'SalesChannel', ['Agent', 'Call Center', 'Branch', 'Web'])
sales_channel_one_hot = feature_column.indicator_column(sales_channel)
feature_columns.append(sales_channel_one_hot)

vehicle_class = feature_column.categorical_column_with_vocabulary_list(
      'VehicleClass', ['Two-Door Car', 'Luxury Car', 'Luxury SUV', 'Four-Door Car', 'SUV', 'Sports Car'])
vehicle_class_one_hot = feature_column.indicator_column(vehicle_class)
feature_columns.append(vehicle_class_one_hot)

vehicle_size = feature_column.categorical_column_with_vocabulary_list(
      'VehicleSize', ['Medsize', 'Small', 'Large'])
vehicle_size_one_hot = feature_column.indicator_column(vehicle_size)
feature_columns.append(vehicle_size_one_hot)

feature_layer = tf.keras.layers.DenseFeatures(feature_columns)

train_ds = df_to_dataset(train, batch_size=28800)
val_ds = df_to_dataset(val, shuffle=False, batch_size=7200)
test_ds = df_to_dataset(test, shuffle=False, batch_size=9000)

model = tf.keras.Sequential([
  feature_layer,
  layers.Dense(1024, activation='relu'),
  layers.Dense(1024, activation='relu'),
  layers.Dense(512, activation='relu'),
  layers.Dense(4, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'],
              run_eagerly=True)

model.fit(train_ds,
          validation_data=val_ds,
          epochs=5)

loss, accuracy = model.evaluate(test_ds)
print("Accuracy", accuracy)

model.summary()

model.save('actuary_model.h5')

# Predict
predict_url = 'file:///tensor_flow/test_data.csv'
predict_dataframe = pd.read_csv(predict_url)
predict_dataframe.head()

predict_dataset = df_to_dataset(predict_dataframe, batch_size=10)

print(predict_dataset)

#predict_dataset = tf.convert_to_tensor(predict_dataframe)

predictions = model.predict(tf.convert_to_tensor(predict_dataframe))

for i, logits in enumerate(predictions):
      class_idx = tf.argmax(logits).numpy()
      print("prediction :", class_idx)
