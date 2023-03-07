import os, numpy as np, itertools, math
import random
import warnings
from math import factorial

import librosa 
import librosa.display
import os, numpy as np, itertools, pickle, math
from random import random
import random

frames_train = []
y_frames_train = []
frame_len = 22050
fls = []

for i,curr in enumerate(os.listdir("/home/mukesh/Desktop/infiniteremixer/songs/gen_audio/")):
    y, sr = librosa.load("/home/mukesh/Desktop/infiniteremixer/songs/gen_audio/"+curr, 22050)
    label = int(curr.split('.')[0])
    y = ((y-np.amin(y))*2)/(np.amax(y) - np.amin(y)) - 1

    org_len = len(y)
    intervals = librosa.effects.split(y, top_db= 15, ref= np.max)
    intervals = intervals.tolist()
    y = (y.flatten()).tolist()
    nonsilent_y = []

    for p,q in intervals :
        nonsilent_y = nonsilent_y + y[p:q+1] 

    y = np.array(nonsilent_y)
    final_len = len(y)
    sil = org_len - final_len

    if len(y)!=0:
        fls.append("/home/mukesh/Desktop/infiniteremixer/songs/gen_audio/"+curr)

    start = 0
    end = frame_len
    k = 0
    for j in range(0, len(y), int(frame_len)) :
        k+=1
        if(k>1):
            break
        frame = y[j:j+frame_len]
        if len(frame) < frame_len :
            frame = frame.tolist() + [0]* (frame_len-len(frame))
        frame = np.array(frame)
        S = np.abs(librosa.stft(frame, n_fft=512))
        frames_train.append(S)
        y_frames_train.append(label)

print(len(frames_train),len(fls))
y_frames_train = np.array(y_frames_train)

r,c = frames_train[0].shape
frames_train = np.array(frames_train)
frames_train = frames_train.reshape((len(frames_train), r, c))

f = open(os.getcwd() + "/training_frames.pkl", 'wb')
pickle.dump([frames_train], f)
f.close()

f = open("/home/mukesh/Desktop/infiniteremixer/songs/gen_data" + "/mapping_snn.pkl", 'wb')
pickle.dump(fls, f)
f.close()

# Standardize the data
mu = frames_train.mean()
std = frames_train.std()
frames_train = (frames_train-mu)/std

# There are imbalanced classes.
u, f = np.unique(y_frames_train, return_counts=True)
frames_train1 = []
y_frames_train1 = []
maximum = max(f)
count = 0
for i in u :
    ind, = np.where(y_frames_train == i)
    ind = ind.tolist()
    while len(ind) < maximum :
        ind = ind + ind
    ind = ind[:maximum]
    temp = frames_train[ind]
    if count == 0 :
        frames_train1 = temp
        count += 1
    else :
        frames_train1 = np.concatenate((frames_train1, temp), axis= 0)

    y_frames_train1 += [i] * maximum

y_frames_train1 = np.array(y_frames_train1)


def generate_positive_pairs(X, y, rand_samples, pair_len) :
    row, col = X.shape[0], X.shape[1]
    uniq, freq = np.unique(y, return_counts=True)

    anchor = []
    pos = []
    count = 0

    for x, f in zip(uniq, freq) :
        ind, = np.where(y == x)
        ind = list(ind)

        if rand_samples <= f and rand_samples != -1:
            random_indices = random.sample(ind, rand_samples)
        elif rand_samples == -1 :
            random_indices = ind
        else :
            warnings.warn("ValueError ! 'samples' required are more than number of elements in class. So, all elements are selected.")
            random_indices = random.sample(ind, f)

        pairs = list(itertools.combinations(random_indices, 2))

        if len(pairs) >= pair_len :
            pairs = list(pairs)[:pair_len]

        anchor += [X[i] for i, _ in pairs]
        pos += [X[i] for _, i in pairs]    

    anchor = np.array(anchor)
    a = (len(anchor),)
    b = tuple(X[0].shape)
    anchor = anchor.reshape(a+b)

    a = (len(anchor),)

    pos = np.array(pos)
    pos = pos.reshape(a+b)
    return anchor, pos, uniq, freq

def generate_negative_pairs(X, y, uniq, freq, rand_samples, pair_len) :
    row, col = X.shape[0], X.shape[1]
    indices = []

    for x, f in zip(uniq, freq) :
        ind, = np.where(y == x)
        ind = list(ind)

        if rand_samples <= f and rand_samples != -1:
            random_indices = random.sample(ind, rand_samples)
        elif rand_samples == -1 :
            random_indices = ind
        else :
            warnings.warn("ValueError ! 'samples' required are more than number of elements in class. So, all elements are selected.")
            random_indices = random.sample(ind, f)
        indices.append(random_indices)   

    neg = []

    for i in range(len(uniq)) :
        for j in range(len(uniq)) : 
            if i == j :
                continue
            curr = indices[j]
            num = math.ceil(pair_len/(len(uniq)-1))
            while len(curr) < num :
                curr += indices[j]

            indices1 = random.sample(curr, num)
            indices1 = indices1[:pair_len]
            neg += [X[p] for p in indices1]

    neg = np.array(neg)
    a = (len(neg),)
    b = tuple(X[0].shape)

    neg = neg.reshape(a+b)


    return neg

# Computed number of combinations nCr
def calculate_combinations(n, r):
    return factorial(n) // factorial(r) // factorial(n-r)

def generate_pairs(X, y, rand_samples, pos_pair_size=-1, extra_data=[]) :
    uniq, f = np.unique(y, return_counts= True)

    N = len(uniq)
    pair_neg = calculate_combinations(N, 2)
    pair_pos = calculate_combinations(min(f), 2)

    pos_pair_size = min(pos_pair_size, pair_pos )

    if pos_pair_size == -1 :
        if rand_samples == -1 :
            # pos_pair_size = calculate_combinations(int(len(y)/N), 2)
            pos_pair_size = calculate_combinations(int(min(f)), 2)
        else :
            # pos_pair_size = min(calculate_combinations(rand_samples, 2), int(len(y)/N) )
            pos_pair_size = min(calculate_combinations(rand_samples, 2), pair_pos )

    pos_pair_size = int(pos_pair_size)
    # print("pos pair size ", pos_pair_size)
    for i in range(pos_pair_size, 0, -1):
        if ((N/ pair_neg) * i).is_integer() :
            break

    neg_samples = int((N/ pair_neg) * i)
    pos_samples = i

    if ((N/ pair_neg) * i).is_integer() is False :
        warnings.warn("Number of samples per class are less than total combinations of all classes. 1 sample will be selected from each negative pair. ")
        neg_samples = 1

    anchor, pos, uniq, freq = generate_positive_pairs(X, y, rand_samples= rand_samples, pair_len=pos_samples)
    neg = generate_negative_pairs(X, y, uniq, freq, rand_samples= rand_samples, pair_len=pos_samples)
    
    return anchor, pos, neg

# Make pairs for siamese network
anchor, pos, neg = generate_pairs(frames_train1, y_frames_train1, rand_samples= -1, pos_pair_size=300)

anchor = anchor.astype(np.float16)
pos = pos.astype(np.float16)
neg = neg.astype(np.float16)

print("Siamese pairs ", anchor.shape, pos.shape, neg.shape)

np.savez_compressed(os.getcwd()+'/training_siamese_frames', a=anchor, b=pos, c=neg)