for i in  2 3 5 7;
do
    # Beijing PM25
    python CP_continues.py --dataset=beijing_PM25 --R_U=$i --num_fold=5  --machine=$USER 

    # Beijing PM10
    python CP_continues.py --dataset=beijing_PM10 --R_U=$i --num_fold=5  --machine=$USER

    # Beijing SO2
    python CP_continues.py --dataset=beijing_SO2 --R_U=$i --num_fold=5  --machine=$USER

    # US-temperature
    python CP_continues.py --dataset=US_temp --R_U=$i --num_fold=5  --machine=$USER


done

