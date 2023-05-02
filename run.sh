GPU=7
method='tipi'
batchsize=64
dataset='imagenet'
severity='5'
back_bone='resnet50'
selective_entropy='False'
dataset_folder='/netssd/tuan/data'

for TEST_ENV in {0..14}
do
    CUDA_VISIBLE_DEVICES=$GPU python main.py --dataset_folder $dataset_folder --method $method --batchsize $batchsize --test_env $TEST_ENV --dataset $dataset --epsilon 0.00784313725 --severity $severity --back_bone $back_bone --selective_entropy $selective_entropy
done
