1. make_splits.ipynb  ->  make_aug.ipynb  ->  python make_aug3.py
2. python -u lstm_gru_train.py --folds 0 --out runs_aug3                       (BiGRU)
3. python -u lstm_gru_train.py --rnn LSTM --folds all --epochs 12 --out runs_lstm_full
4. python -c "import lstm_gru_train as t; t.ensemble('gru_plus_lstm', ['runs_aug3/bigru/fold0.pkl', 'runs_lstm_full/bilstm/foldall.pkl'])"
5. python apply_rules.py submissions/ens_gru_plus_lstm.csv
