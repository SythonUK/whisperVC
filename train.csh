#!/bin/csh
#################################################################
#                          train.csh                            #
#################################################################
#               Copyright (c) 2017 Yuki Saito                   #
#      This software is released under the MIT License.         #
#       http://opensource.org/licenses/mit-license.php          #
#################################################################

# output directory
set out_gen="out"

# names of source/target speaker
set src="hoge"
set tgt="fuga"

# if negative, use cpu
set gpu=0

# order of mel-cepstral coefficients
set omc=59

# frame shift [ms]
set shiftl=5

# sampling rate [Hz]
set fs=16000

# F0 range (src)
set minf0_s=45
set maxf0_s=200

# F0 range (tgt)
set minf0_t=180
set maxf0_t=420

# batchsize
set batch=200

# number of iterations
set epoch=25

# number of hidden unit
set nhid=512

# learning late
set lr=0.01

# if 1, use input-to-output highway networks
set hw=1

# VUF for whisper filtering
set VUF=1

# ---------------------------------------

# tools
set EXT_FEATURES="python scripts/extract_features.py"
# -> extract features from wav files
set MAKE_DATA="python scripts/make_paradata.py"
# -> make train and test data
set TRAIN_GEN="python scripts/train_gen.py"
# -> train generator model
set SYNTHESIS="python scripts/synthesis.py"
# -> synthesis waveform

# ---------------------------------------

# flags
set COMPL=1
set EXTFT=1
set MKDAT=1
set TRGEN=1
set SYNTH=1

# ---------------------------------------

# COMPILE .pyx FILES
if (${COMPL} == 1) then
  cd scripts/; python setup.py build_ext --inplace; cd ../
endif

# EXTRACT FEATURES
set opt=" --omc ${omc} --shiftl ${shiftl} --fs ${fs} "
if (${EXTFT} == 1) then
  ${EXT_FEATURES} ${opt} ${src} ${minf0_s} ${maxf0_s} ${tgt} ${minf0_t} ${maxf0_t}
endif

# MAKE DATA
set opt=" --omc ${omc} "
if (${MKDAT} == 1) then
  ${MAKE_DATA} ${opt} ${src} ${tgt}
endif

# TRAIN GENERATOR
set opt=" --batch ${batch} --epoch ${epoch} --gpu ${gpu} --lr ${lr} --nhid ${nhid} --hw ${hw} --omc ${omc} "
if (${TRGEN} == 1) then
  ${TRAIN_GEN} ${opt} ${out_gen} ${src} ${tgt}
endif

# SYNTHESIS WAVEFORM
set opt=" --gpu ${gpu} --nhid ${nhid} --hw ${hw} --omc ${omc} --shiftl ${shiftl} --fs ${fs} --VUF ${VUF} "
if (${SYNTH} == 1) then
  ${SYNTHESIS} ${opt} ${out_gen} $1 ${minf0_s} ${maxf0_s} $2
endif
