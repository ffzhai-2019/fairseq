EXPDIR=/search/odin/zhaifeifei/exp/wmt16_en_de_bpe32k/
TEXT=$EXPDIR/train_data
CORPUS=$EXPDIR/corpus
python preprocess.py --source-lang en --target-lang de \
                     --trainpref $CORPUS/train.tok.clean.bpe.32000 \
                     --validpref $CORPUS/newstest2013.tok.bpe.32000 \
                     --testpref $CORPUS/newstest2014.tok.bpe.32000 \
                     --destdir $EXPDIR/train_data \
                     --nwordssrc 32768 --nwordstgt 32768 \
                     --joined-dictionary --workers 30
