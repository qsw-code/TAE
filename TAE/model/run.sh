MODEL=LstmTAE

DATA=EUR-Lex

DATA_PATH=XXXX

GLOVE_PATH=XXXX

python preprocess.py \
--text-path $DATA_PATH/$DATA/train_raw_texts.txt \
--tokenized-path $DATA_PATH/$DATA/train_texts.txt \
--label-path $DATA_PATH/$DATA/train_labels.txt \
--vocab-path $DATA_PATH/$DATA/vocab.npy \
--emb-path $DATA_PATH/$DATA/emb_init.npy \
--w2v-model $GLOVE_PATH/glove.840B.300d.gensim

python preprocess.py \
--text-path $DATA_PATH/$DATA/test_raw_texts.txt \
--tokenized-path $DATA_PATH/$DATA/test_texts.txt \
--label-path $DATA_PATH/$DATA/test_labels.txt \
--vocab-path $DATA_PATH/$DATA/vocab.npy

python main.py --data-cnf ../configure/datasets/$DATA.yaml --model-cnf ../configure/models/$MODEL-$DATA.yaml --mode 'train'

python main.py --data-cnf ../configure/datasets/$DATA.yaml --model-cnf ../configure/models/$MODEL-$DATA.yaml --mode 'eval'

python main_evaluation.py --results ../data/$DATA/results/$MODEL-$DATA-labels.npy --targets ../data/$DATA/test_labels.npy --train-labels ../data/$DATA/train_labels.npy --modelname $MODEL --dataname $DATA
