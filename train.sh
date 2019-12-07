DATADIR=/search/odin/zhaifeifei/exp//wmt16_en_de_bpe32k/
EXPDIR=/search/odin/zhaifeifei/exp//wmt16_en_de_bpe32k/
MODELDIR=$EXPDIR/model
TENSORBOARDDIR=$EXPDIR/log
python -u train.py $DATADIR/train_data --tensorboard-logdir $TENSORBOARDDIR --log-interval 20 \
                --arch transformer_vaswani_wmt_en_de_big --share-all-embeddings --update-freq 16 \
                --ddp-backend=no_c10d \
                --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
                --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 4000 \
                --lr 0.001 --min-lr 1e-09 \
                --dropout 0.3 --weight-decay 0.0 \
                --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
                --max-tokens 3584 --distributed-world-size 8 --save-dir $MODELDIR
