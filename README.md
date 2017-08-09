# 深層学習とささやきフィルタによる音声変換
 * 入力された音声を
 * ディープラーニングってやつと
 * ささやきフィルタで
 * ささやき声に変換する

# 依存モジュール
 * numpy
 * scipy
 * pyworld
 * pysptk
 * h5py
 * chainer (v2.0)
 * cupy (v1.0.1)
 * Cython

# 動作環境
Ubuntu 14.04 (64 bit) w/ NVIDIA Quadro K620

# 実行方法
 1. 音声データを用意する (同一発話内容・同名の wav ファイルを data/wav 以下に配置)
 2. F0分析パラメータを調整する → python scripts/pyworld_test.py {minF0} {maxF0} {name_of_spearker} で頑張る
 3. 良さ気なパラメータを train.csh の minf0_s, maxf0_s, minf0_t, maxf0_t にセットする (s: 変換元話者, t: 変換先話者)
 4. その他のパラメータも適当に設定する
 5. csh train.csh {input_wav} {output_wav}

# モデル学習後に変換だけしたい場合
python scripts/synthesis.py {dirname_to_load_model} {input_wav} {minF0_s} {maxF0_s} {output_wav}
