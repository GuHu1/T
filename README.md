## train&test
CUDA_VISIBLE_DEVICES=4,5,6,7 nohup bash tools/dist_train.sh projects/configs/resworld/resworld_config.py 4 > resworld_train.log 2>&1 &

CUDA_VISIBLE_DEVICES=4,5,6,7 nohup bash tools/dist_test.sh projects/configs/resworld/resworld_config.py work_dirs/resworld_config/epoch_12_ema.pth 4 --eval bbox > resworld_test.log 2>&1 &
## Main Results

### UniAD-style metrics
| Method | L2<sub>MAX</sub> (m) 1s | L2<sub>MAX</sub> (m) 2s | L2<sub>MAX</sub> (m) 3s | L2<sub>MAX</sub> (m) Avg. | CR<sub>MAX</sub> (%) 1s | CR<sub>MAX</sub> (%) 2s | CR<sub>MAX</sub> (%) 3s | CR<sub>MAX</sub> (%) Avg. | 
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ResWorld | 0.19 | 0.50 | 1.08 | 0.59 | 0.02 | 0.06 | 0.43 | 0.17 | 

### VAD-style metrics
| Method | L2<sub>AVG</sub> (m) 1s | L2<sub>AVG</sub> (m) 2s | L2<sub>AVG</sub> (m) 3s | L2<sub>AVG</sub> (m) Avg. | CR<sub>AVG</sub> (%) 1s | CR<sub>AVG</sub> (%) 2s | CR<sub>AVG</sub> (%) 3s | CR<sub>AVG</sub> (%) Avg. |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ResWorld | 0.14 | 0.27 | 0.49 | 0.30 | 0.01 | 0.03 | 0.14 | 0.06 |