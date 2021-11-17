from glob import glob
import numpy as np

data_lists = glob("/home/data/jinyoung/classification/output_voxceleb_VAD/*/*/*.npy")
data_lists.sort()
max_lists = []
min_lists = []
i = 0
for data in data_lists:
    n_ = np.load(data, allow_pickle=True)
    max_ = float(np.max(n_))
    min_ = float(np.min(n_))
    print("max :", str(max_), "min", str(min_))
    max_lists.append(max_)
    min_lists.append(min_)
np_max = np.array(max_lists)
np_min = np.array(min_lists)
print("overall max's average / max / min:", str(np.max(np_max)), str(np.min(np_max)), str(np.average(np_max)))
print("overall min's average / max / min:", str(np.max(np_min)), str(np.min(np_min)), str(np.average(np_min)))
np.save("voxceleb_max.npy", np_max)
np.save("voxceleb_min.npy", np_min)

