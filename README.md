# 深層学習とささやきフィルタによる音声変換
 * 入力された音声を
 * ディープラーニングってやつと
 * ささやきフィルタで
 * ささやき声に変換する

# 依存モジュール
 * numpy (v1.13.0)
 * scipy (v0.17.1)
 * pyworld (v0.2.1b0)
 * pysptk (v0.1.7)
 * h5py (v2.6.0)
 * chainer (v2.0.1)
 * cupy (v1.0.1)
 * Cython (v0.26)

# 動作環境
Ubuntu 14.04 (64 bit) w/ NVIDIA Quadro K620

# 実行方法
 1. 音声データを用意する (同一発話内容・同名の wav ファイルを data/wav 以下に配置, ディレクトリ名とかは適宜変更)
 2. F0分析パラメータを調整する → python scripts/pyworld_test.py {minF0} {maxF0} {name_of_spearker} で頑張って探す
 3. 良さ気なパラメータを train.csh の minf0_s, maxf0_s, minf0_t, maxf0_t にセットする (s: 変換元話者, t: 変換先話者)
 4. その他のパラメータも適当に設定する (特に気にしなければデフォルトでOK)
 5. csh train.csh {input_wav} {output_wav}

# その他
 * train.csh の flags をいじると，学習だけできるようになったりする (TRGENだけ1にする)

