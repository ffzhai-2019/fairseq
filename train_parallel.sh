#DATADIR=/search/odin/zhaifeifei/exp/bigdata/current_and_sogouca/
#EXPDIR=/search/odin/zhaifeifei/exp/bigdata/current_and_sogouca/
DATADIR=/search/odin/zhaifeifei/exp//wmt16_en_de_bpe32k/
EXPDIR=/search/odin/zhaifeifei/exp//wmt16_en_de_bpe32k/
MODELDIR=$EXPDIR/model
TENSORBOARDDIR=$EXPDIR/log
#python -m torch.distributed.launch --nproc_per_node=8 \
#    --nnodes=2 --node_rank 0 --master_addr="10.141.202.59" \
#    train.py $DATADIR/train_data --tensorboard-logdir $TENSORBOARDDIR \
#                --arch transformer_vaswani_wmt_en_de_big --no-progress-bar --log-interval 1 \
#                --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
#                --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 4000 \
#                --lr 0.001 --min-lr 1e-09 \
#                --dropout 0.1 --weight-decay 0.0 \
#                --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
#                --max-tokens 3584 --update-freq 6 --distributed-world-size 8 --save-dir $MODELDIR --max-source-positions 300 --max-target-positions 300

python -m torch.distributed.launch --nproc_per_node=8 \
    --nnodes=2 --node_rank=0 --master_addr="10.141.202.59" \
    --master_port=1234 \
    train.py $DATADIR/train_data --tensorboard-logdir $TENSORBOARDDIR \
                --arch transformer_vaswani_wmt_en_de_big --share-all-embeddings --no-progress-bar --log-interval 1 \
                --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
                --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 4000 \
                --lr 0.0005 --min-lr 1e-09 \
                --dropout 0.3 --weight-decay 0.0 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
                --max-tokens 3584 --update-freq 6 \
                --save-dir $MODELDIR --max-source-positions 300 --max-target-positions 300 \
                --distributed-no-spawn
