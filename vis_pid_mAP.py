import json
import matplotlib.pyplot as plt
import numpy as np

with open('/mnt/data4/soowon/skt-intern/aug_all_pid/baseline/epoch24/pid_mAP.json', 'r') as f:
    pid_mAP = json.load(f)


pids = pid_mAP['pid']
mAPs = pid_mAP['mAP']
    
with open('pid_sample_num.json', 'r') as f:
    pid_sample_num = json.load(f)
    
mAP_50 = len([i for i in mAPs if i < 0.5])
mAP_70 = len([i for i in mAPs if i < 0.7])
mAP_95 = len([i for i in mAPs if i < 0.95])

xaxis = range(len(pids))
yaxis = [pid_sample_num[str(pids[i])] for i in xaxis]
mean_sample = np.mean(yaxis)

plt.figure(figsize=(30, 30))
bars = plt.bar(xaxis, yaxis)

plt.axvline(x=mAP_50, color='red', linestyle='-', linewidth=2)
plt.axvline(x=mAP_70, color='green', linestyle='-', linewidth=2)
plt.axvline(x=mAP_95, color='purple', linestyle='-', linewidth=2)
plt.axhline(y=mean_sample, color='black', linestyle='-', linewidth=2)


# for bar, alpha in zip(bars, mAPs):
#     bar.set_alpha(1 - alpha)
    
    
plt.savefig('pid_sample_map.png')