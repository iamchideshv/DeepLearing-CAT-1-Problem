# Weather Forecasting Using LSTM

## Description
This project predicts daily temperatures using historical weather data from the Daily Delhi Climate dataset. It uses an LSTM deep learning model built with TensorFlow and Keras to learn patterns from temperature, humidity, wind speed, and atmospheric pressure.

## Technologies Used
- Python
- Google Colab
- TensorFlow / Keras
- Pandas, NumPy
- Scikit-learn
- Matplotlib

## Methodology
1. Load and clean the weather dataset.
2. Normalize the meteorological features.
3. Generate sequences using the previous 30 days.
4. Train an LSTM model to predict temperature.
5. Evaluate performance using MAE and RMSE.
6. Visualize actual and predicted temperatures.
7. Demonstrate a 7-day temperature forecast.

## Google Colab Notebook
**[Open Weather Forecasting Notebook](https://colab.research.google.com/drive/1hqu5vHawHUsaETz1es-NPHWR6yJcxKB5?usp=sharing)**

## Expected Output
- Daily temperature predictions
- MAE and RMSE evaluation metrics
- Actual vs. predicted temperature graph
- 7-day demonstration forecast

## Author
Chidesh V
732224AI014

## OUTPUT

<img width="1189" height="490" alt="downloadimg" src="https://github.com/user-attachments/assets/6c2ac104-3679-4d1a-a828-ca1393cd88ef" />

<img width="989" height="390" alt="download2img" src="https://github.com/user-attachments/assets/4a0c0b08-5c00-447f-9a50-c26d3d6c1661" />

## Dataset Sequential

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ lstm (LSTM)                     │ (None, 30, 64)         │        17,664 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout (Dropout)               │ (None, 30, 64)         │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ lstm_1 (LSTM)                   │ (None, 32)             │        12,416 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense (Dense)                   │ (None, 16)             │           528 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_1 (Dense)                 │ (None, 1)              │            17 │
└─────────────────────────────────┴────────────────────────┴───────────────┘


## Dataset

MAE:  7.379 °C
RMSE: 7.758 °C

Date	Actual Temperature	Predicted Temperature
0	2017-04-02	29.750000	23.251823
1	2017-04-03	30.500000	23.281888
2	2017-04-04	30.933333	23.306993
3	2017-04-05	29.230769	23.343293
4	2017-04-06	31.222222	23.363396
5	2017-04-07	27.000000	23.381643
6	2017-04-08	25.625000	23.377365
7	2017-04-09	27.125000	23.377186
8	2017-04-10	27.857143	23.386649
9	2017-04-11	29.250000	23.399888

