#!/bin/bash

for g in {0.0,}
do
        echo $g
        for e in {0.0,0.1}
        do
                echo $e
                for s in {0..50}
                do
                        for a in {0,}
                        do
                                echo $s
                                python generate_synth_data.py --N 100 --K 2 --nbad 0 --d 10 --gamma $g --epsilon $e --seed $s --action_seed $a 
                        done
                done
        done
done
