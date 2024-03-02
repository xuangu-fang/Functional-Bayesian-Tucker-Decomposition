for i in  7;
do
    # # Beijing PM25
    # python tucker_continues.py --dataset=beijing_PM25 --R_U=$i --num_fold=5  --machine=$USER 

    # # Beijing PM10
    python tucker_continues.py --dataset=beijing_PM10 --R_U=$i --num_fold=5  --machine=$USER

    # # Beijing SO2
    # python tucker_continues.py --dataset=beijing_SO2 --R_U=$i --num_fold=5  --machine=$USER

    # # server
    # python dynamic_streaming_CP.py --dataset=server --R_U=$i --num_fold=5  --machine=$USER
done
