from vmdpy import VMD
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import scipy.signal
import pandas as pd
import librosa

def vmd_my(a):
    '''vmd算法进行去噪'''
    sig = pd.read_excel('acoustic resonance_'+str(a)+'.xlsx')#读取信号数据
    sig = np.array(sig['电压'])
    max_num=np.argmax(sig)
    sig=sig[max_num-500:max_num+15000]#寻找有效的信号片段较少计算量
    alpha=10000
    tau=0
    K=3
    DC=0
    init=1
    tol=1e-7
    imfs,u_hat,omega=VMD(sig,alpha,tau,K,DC,init,tol)#VMD函数进行分解
    sig1=imfs[1,:]
    sig_len1=len(sig1)
    time1= np.arange(sig_len1)
    half_sig1=sig_len1//2    
    ff=40000*np.arange(half_sig1)/sig_len1
    fft1=np.fft.fft(sig1)
    fft1=abs(fft1)/20000
    max_fft1=np.max(fft1[:half_sig1])
    num1=np.argmax(fft1[:half_sig1])
    new_p1 = np.abs(fft1[:half_sig1]-max_fft1/2)
    pidx_f1=np.argmin(new_p1[:num1])
    pidx_b1=np.argmin(new_p1[num1:])
    damping_half1 = (ff[pidx_b1+num1]-ff[pidx_f1])/ff[num1]
    sig2=imfs[2,:]
    sig_len2=len(sig2)
    time2= np.arange(sig_len2)
    half_sig2=sig_len2//2    
    ff2=40000*np.arange(half_sig2)/sig_len2
    fft2=np.fft.fft(sig2)
    fft2=abs(fft2)/20000
    max_fft2=np.max(fft2[:half_sig2])
    num2=np.argmax(fft2[:half_sig2])
    new_p2 = np.abs(fft2[:half_sig2]-max_fft2/2)
    pidx_f2=np.argmin(new_p2[:num2])
    pidx_b2=np.argmin(new_p2[num2:])
    damping_half2 = (ff[pidx_b2+num2]-ff[pidx_f2])/ff[num2]
    return max_fft1,ff[num1],max_fft2,ff[num2],imfs,sig,damping_half1,damping_half2
def data_time(imf):
    '''利用分段函数得到衰减时间'''
    #分段
    N=len(imf)
    segL = 50
    overlap = 0
    delta = segL-overlap
    segNum = np.int32(np.ceil((N-overlap)/delta));
    #扩展信号
    padNum = segNum*delta+overlap-N
    if padNum==0:
        sigEx = imf
    elif padNum>0:
        sigEx = np.hstack((imf,np.zeros(padNum)))
    #分段标签
    segIdx = np.arange(0,segNum)*delta
    #分段矩阵
    from numpy.lib.stride_tricks import as_strided
    segMat = as_strided(sigEx,shape=(segNum,segL),strides=(sigEx.strides[0]*delta,sigEx.strides[0]))
    segMat=abs(segMat)
    Mat_len=len(segMat)
    num_sta=[]
    num_end=[]
    for i in range(Mat_len):  
        seg_max=np.max(segMat[i,:])
        if seg_max > 0.5:
            num_sta.append(i*50)
            for a in range(i+1,Mat_len):
                seg_min=np.max(segMat[a,:])
                if seg_min < 0.2:
                    num_end.append(a*50)
                    break
            break
    data_time=num_end[0]-num_sta[0]
    sig_real=imf[num_sta[0]:num_end[0]]
    return data_time,sig_real
Dt1=[]
zero_crossing_lis1=[]
spec_cent=[]
FF_MAX1= []
PIDX1=[]
Dt2=[]
zero_crossing_lis2=[]
FF_MAX2= []
PIDX2=[]
DAMPING_HALF1=[]
DAMPING_HALF2=[]
for i in range(1,44):
    max_fft1,ff1,max_fft2,ff2,imfs,sig,damping_half1,damping_half2=vmd_my(i)#vmd处理信号
    FF_MAX1.append(max_fft1)
    PIDX1.append(ff1)
    data_t1,sig_real1=data_time(sig-imfs[0,:])#获取衰减时间
    zero1=librosa.feature.zero_crossing_rate(sig_real1, frame_length = 2048, hop_length = 100, center = True)#获取过零率 
    zero_mean1=np.mean(zero1)#获取平均过零率
    Dt1.append(data_t1)#储存衰减时间
    zero_crossing_lis1.append(zero_mean1)#储存过零率
    DAMPING_HALF1.append(damping_half1)
    FF_MAX2.append(max_fft2)
    PIDX2.append(ff2)
    # data_t2,sig_real2=data_time(imfs[2,:])#获取衰减时间
    # zero2=librosa.feature.zero_crossing_rate(sig_real2, frame_length = 2048, hop_length = 100, center = True)#获取过零率 
    # zero_mean2=np.mean(zero2)#获取平均过零率
    # Dt2.append(data_t2)#储存衰减时间
    # zero_crossing_lis2.append(zero_mean2)#储存过零率 
    DAMPING_HALF2.append(damping_half2)
    a = librosa.feature.spectral_centroid(sig-imfs[0,:],sr=40000)[0]
    b = np.mean(a)
    spec_cent.append(b)
    print(i)
'''下面是将各数据规范格式，转换成列数据，并合并起来'''
FF_MAX1 = np.array(FF_MAX1).reshape(-1,1)
PIDX1 = np.array(PIDX1).reshape(-1,1)
spectral_cent=np.array(spec_cent).reshape(-1,1)
data_time1=np.array(Dt1).reshape(-1,1)/40000
zero_crossing_rate1=np.array(zero_crossing_lis1).reshape(-1,1)
FF_MAX2 = np.array(FF_MAX2).reshape(-1,1)
PIDX2 = np.array(PIDX2).reshape(-1,1)
DAMPING_HALF1=np.array(DAMPING_HALF1).reshape(-1,1)
DAMPING_HALF2=np.array(DAMPING_HALF2).reshape(-1,1)
# data_time2=np.array(Dt2).reshape(-1,1)/40000
zero_crossing_rate2=np.array(zero_crossing_lis2).reshape(-1,1)
# A=np.concatenate((zero_crossing_rate1,data_time1,FF_MAX1,PIDX1,zero_crossing_rate2,data_time2,FF_MAX2,PIDX2,spectral_cent),axis=1)
A=np.concatenate((zero_crossing_rate1,data_time1,FF_MAX1,PIDX1,FF_MAX2,PIDX2,spectral_cent,DAMPING_HALF1,DAMPING_HALF2),axis=1)

A=pd.DataFrame(A)
'''保存数据以.xlsx格式'''
with pd.ExcelWriter('Thick_tile_GRADUATION.xlsx') as writer:
        A.to_excel(writer,sheet_name=str(0))
