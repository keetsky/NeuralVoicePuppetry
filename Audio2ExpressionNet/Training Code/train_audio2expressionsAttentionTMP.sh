set -ex
# . train_audio2expressionsAttentionTMP.sh &
GPUID=0
DATASETS_DIR=./datasets
DATASET_MODE=multi_face_audio_eq_tmp_cached
OBJECT=ARD_ZDF

# neural texture, not used here
TEX_DIM=128
TEX_FEATURES=16

# loss
#LOSS=VGG
#LOSS=L1
LOSS=RMS
#LOSS=L4

# models
MODEL=audio2ExpressionsAttentionTMP4
RENDERER_TYPE=estimatorAttention


# optimizer parameters
#LR=0.00001
LR=0.0001

#N_ITER=150 #50 #N_ITER=150
#N_ITER_LR_DECAY=50

N_ITER=20 #50 #N_ITER=150
N_ITER_LR_DECAY=30

BATCH_SIZE=16
SEQ_LEN=8


RENDERER=$OBJECT
EROSION=1.0

################################################################################
################################################################################
################################################################################
DATE_WITH_TIME=`date "+%Y%m%d-%H%M%S"`
NAME=$MODEL-$RENDERER_TYPE-SL$SEQ_LEN-BS$BATCH_SIZE-$OBJECT-$DATASET_MODE-$LOSS-$DATE_WITH_TIME-look_ahead
DISPLAY_NAME=${MODEL}-$DATASET_MODE_${OBJECT}-${RENDERER_TYPE}-SL$SEQ_LEN-BS$BATCH_SIZE-${LOSS}-look_ahead


# training
# --input_noise_augmentation
python train.py --look_ahead --seq_len $SEQ_LEN  --save_latest_freq 100000 --no_augmentation --compute_val --name $NAME --erosionFactor $EROSION --tex_dim $TEX_DIM --tex_features $TEX_FEATURES --rendererType $RENDERER_TYPE --lossType $LOSS --display_env $DISPLAY_NAME --niter $N_ITER --niter_decay $N_ITER_LR_DECAY --dataroot $DATASETS_DIR/$OBJECT --model $MODEL --netG unet_256 --lambda_L1 100 --dataset_mode $DATASET_MODE --no_lsgan --norm instance --pool_size 0  --gpu_ids $GPUID --lr $LR --batch_size $BATCH_SIZE

# # testing
#EPOCH=latest
#python test.py --seq_len $SEQ_LEN --write_no_images --name $NAME --erosionFactor $EROSION --epoch $EPOCH --display_winsize 512 --tex_dim $TEX_DIM --tex_features $TEX_FEATURES --rendererType $RENDERER_TYPE --lossType $LOSS --dataroot $DATASETS_DIR/$OBJECT --model $MODEL --netG unet_256 --dataset_mode $DATASET_MODE --norm instance  --gpu_ids $GPUID

################################################################################
################################################################################
################################################################################