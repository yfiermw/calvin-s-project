# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


#encoding:utf-8
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import  tensorflow as tf
from pandas.core.frame import DataFrame
 
#获取文件夹下所有的WAV文件
def get_wav_files(wav_path):
    wav_files = []
    for (dirpath, dirnames, filenames) in os.walk(wav_path):
        for filename in filenames:
            if filename.endswith('.wav') or filename.endswith('.WAV'):
                # print(filename)
                filename_path = os.path.join(dirpath, filename)
                # print(filename_path)
                wav_files.append(filename_path)

    return wav_files



#获取wav文件对应的翻译文字
def get_tran_texts(wav_files, tran_path):
    tran_texts = []
    for wav_file in wav_files:
        (wav_path, wav_filename) = os.path.split(wav_file)
        tran_file = os.path.join(tran_path, wav_filename + '.trn')
        # print(tran_file)
        if os.path.exists(tran_file) is False:
            return None
 
        fd = open(tran_file, 'r')
        text = fd.readline()
        tran_texts.append(text.split('\n')[0])
        fd.close()

    return tran_texts


#获取wav和对应的翻译文字
def get_wav_files_and_tran_texts(wav_path, tran_path):
    wav_files = get_wav_files(wav_path)
    tran_texts = get_tran_texts(wav_files, tran_path)

    return  wav_files,tran_texts


# testing for the dataset
# =============================================================================
wav_files, tran_texts = get_wav_files_and_tran_texts('/Users/calvin/python/audio_recognition/LSTM/data_thchs30/train', '/Users/calvin/python/audio_recognition/LSTM/data_thchs30/data')
print(wav_files[0])
print("\n")
print(tran_texts[0])
print(len(wav_files), len(tran_texts))
 
# =============================================================================

import python_speech_features


#将音频信息转成MFCC特征
#参数说明---audio_filename：音频文件   numcep：梅尔倒谱系数个数
#       numcontext：对于每个时间段，要包含的上下文样本个数
def audiofile_to_input_vector(audio_filename, numcep, numcontext):
    # 加载音频文件
    fs, audio = wav.read(audio_filename)
    # 获取MFCC系数
    orig_inputs = mfcc(audio, samplerate=fs, numcep=numcep)
    #打印MFCC系数的形状，得到比如(955, 26)的形状
    #955表示时间序列，26表示每个序列的MFCC的特征值为26个
    #这个形状因文件而异，不同文件可能有不同长度的时间序列，但是，每个序列的特征值数量都是一样的
    print(np.shape(orig_inputs))
 
    # 因为我们使用双向循环神经网络来训练,它的输出包含正、反向的结
    # 果,相当于每一个时间序列都扩大了一倍,所以
    # 为了保证总时序不变,使用orig_inputs =
    # orig_inputs[::2]对orig_inputs每隔一行进行一次
    # 取样。这样被忽略的那个序列可以用后文中反向
    # RNN生成的输出来代替,维持了总的序列长度。
    orig_inputs = orig_inputs[::2]#(478, 26)
    print(np.shape(orig_inputs))
    #因为我们讲解和实际使用的numcontext=9，所以下面的备注我都以numcontext=9来讲解
    #这里装的就是我们要返回的数据，因为同时要考虑前9个和后9个时间序列，
    #所以每个时间序列组合了19*26=494个MFCC特征数
    train_inputs = np.array([], np.float32)
    train_inputs.resize((orig_inputs.shape[0], numcep + 2 * numcep * numcontext))
    print(np.shape(train_inputs))#)(478, 494)
 
    # Prepare pre-fix post fix context
    empty_mfcc = np.array([])
    empty_mfcc.resize((numcep))
 
    # Prepare train_inputs with past and future contexts
    #time_slices保存的是时间切片，也就是有多少个时间序列
    time_slices = range(train_inputs.shape[0])
 
    #context_past_min和context_future_max用来计算哪些序列需要补零
    context_past_min = time_slices[0] + numcontext
    context_future_max = time_slices[-1] - numcontext
 
    #开始遍历所有序列
    for time_slice in time_slices:
        #对前9个时间序列的MFCC特征补0，不需要补零的，则直接获取前9个时间序列的特征
        need_empty_past = max(0, (context_past_min - time_slice))
        empty_source_past = list(empty_mfcc for empty_slots in range(need_empty_past))
        data_source_past = orig_inputs[max(0, time_slice - numcontext):time_slice]
        assert(len(empty_source_past) + len(data_source_past) == numcontext)
 
        #对后9个时间序列的MFCC特征补0，不需要补零的，则直接获取后9个时间序列的特征
        need_empty_future = max(0, (time_slice - context_future_max))
        empty_source_future = list(empty_mfcc for empty_slots in range(need_empty_future))
        data_source_future = orig_inputs[time_slice + 1:time_slice + numcontext + 1]
        assert(len(empty_source_future) + len(data_source_future) == numcontext)
 
        #前9个时间序列的特征
        if need_empty_past:
            past = np.concatenate((empty_source_past, data_source_past))
        else:
            past = data_source_past
 
        #后9个时间序列的特征
        if need_empty_future:
            future = np.concatenate((data_source_future, empty_source_future))
        else:
            future = data_source_future
 
        #将前9个时间序列和当前时间序列以及后9个时间序列组合
        past = np.reshape(past, numcontext * numcep)
        now = orig_inputs[time_slice]
        future = np.reshape(future, numcontext * numcep)
 
        train_inputs[time_slice] = np.concatenate((past, now, future))
        assert(len(train_inputs[time_slice]) == numcep + 2 * numcep * numcontext)
 
    # 将数据使用正太分布标准化，减去均值然后再除以方差
    train_inputs = (train_inputs - np.mean(train_inputs)) / np.std(train_inputs)
 
    return train_inputs



#将字符转成向量，其实就是根据字找到字在word_num_map中所应对的下标
def get_ch_lable_v(txt_file,word_num_map,txt_label=None):
    words_size = len(word_num_map)
 
    to_num = lambda word: word_num_map.get(word, words_size) 
 
    if txt_file!= None:
        txt_label = get_ch_lable(txt_file)
 
    print(txt_label)
    labels_vector = list(map(to_num, txt_label))
    print(labels_vector)
    return labels_vector


# 字表 
#all_words = []  
#for label in labels:  
#    #print(label)    
#    all_words += [word for word in label]
# 
##Counter，返回一个Counter对象集合，以元素为key，元素出现的个数为value
#counter = Counter(all_words)
##排序
#words = sorted(counter)
#words_size= len(words)
#word_num_map = dict(zip(words, range(words_size)))
# 
#print(word_num_map)
#get_ch_lable_v(None, word_num_map, labels[0])
#exit()



#将音频数据转为时间序列（列）和MFCC（行）的矩阵，将对应的译文转成字向量    
def get_audio_and_transcriptch(txt_files, wav_files, n_input, n_context,word_num_map,txt_labels=None):
    
    audio = []
    audio_len = []
    transcript = []
    transcript_len = []
    if txt_files!=None:
        txt_labels = txt_files
 
    for txt_obj, wav_file in zip(txt_labels, wav_files):
        # load audio and convert to features
        audio_data = audiofile_to_input_vector(wav_file, n_input, n_context)
        audio_data = audio_data.astype('float32')
        # print(word_num_map)
        audio.append(audio_data)
        audio_len.append(np.int32(len(audio_data)))
 
        # load text transcription and convert to numerical array
        target = []
        if txt_files!=None:#txt_obj是文件
            target = get_ch_lable_v(txt_obj,word_num_map)
        else:
            target = get_ch_lable_v(None,word_num_map,txt_obj)#txt_obj是labels
        #target = text_to_char_array(target)
        transcript.append(target)
        transcript_len.append(len(target))
 
    audio = np.asarray(audio)
    audio_len = np.asarray(audio_len)
    transcript = np.asarray(transcript)
    transcript_len = np.asarray(transcript_len)
    return audio, audio_len, transcript, transcript_len



#对齐处理
def pad_sequences(sequences, maxlen=None, dtype=np.float32,
                  padding='post', truncating='post', value=0.):
    #[478 512 503 406 481 509 422 465]
    lengths = np.asarray([len(s) for s in sequences], dtype=np.int64)
 
    nb_samples = len(sequences)
 
    #maxlen，该批次中，最长的序列长度
    if maxlen is None:
        maxlen = np.max(lengths)
 
    # 在下面的主循环中，从第一个非空序列中获取样本形状以检查一致性
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break
 
    x = (np.ones((nb_samples, maxlen) + sample_shape) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if len(s) == 0:
            continue  # 序列为空，跳过
 
        #post表示后补零，pre表示前补零
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError('Truncating type "%s" not understood' % truncating)
 
        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))
 
        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
 
    return x, lengths



#创建序列的稀疏表示
def sparse_tuple_from(sequences, dtype=np.int32):
    indices = []
    values = []
 
    for n, seq in enumerate(sequences):
        indices.extend(zip([n] * len(seq), range(len(seq))))
        values.extend(seq)
 
    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), indices.max(0)[1] + 1], dtype=np.int64)
    # return tf.SparseTensor(indices=indices, values=values, shape=shape)
    return indices, values, shape

sq = [[0,1,2,3,4], [5,6,7,8,]]
indices, values, shape = sparse_tuple_from(sq)
print(indices)
print(values)
print(shape)