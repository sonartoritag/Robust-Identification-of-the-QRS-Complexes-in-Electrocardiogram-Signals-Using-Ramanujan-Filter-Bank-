import os
import time
import easygui
import numpy as np
import pandas as pd
from RFB import RFB
import ECG_denoising
import scipy.io as sio
from plotly.offline import plot
import plotly.graph_objects as go
from plotly.subplots import make_subplots



if __name__ == '__main__':
    path = easygui.fileopenbox(multiple=True)
    fs = float(input("\nEnter the sampling frequency in Hz = "))
    #path = easygui.fileopenbox()
    for i in range(len(path)):
        print('\nProcessing ' + str(i+1) + '/' + str(len(path)) + ' files')
        file = os.path.basename(path[i])
        print("\nProcessing: ", file)
        m = sio.loadmat(path[i])
        for lead in range(0, 1):
            print("\nLead: ", lead+1)
            ecg_signal = m['val'][lead, :]
            start_time = time.time()
            ecg_signal = ECG_denoising.ECG_deno(ecg_signal, fs)
            Pmax = int(fs * 0.06)  # 0.06 #Calculation of Maximum-period of ECG
            Rpeaks, Period = RFB(fs, ecg_signal, Pmax, 2, 6, 0)
            end_time = time.time()
            runtime = end_time - start_time
            print('Time / Denoising+QRS-beat detection = ', runtime / len(Rpeaks))
            fig = make_subplots(rows=3, cols=1, vertical_spacing=0.1, horizontal_spacing=0.07, shared_xaxes=True, shared_yaxes=False)
            fig.add_trace(go.Scatter(y=ecg_signal, name='Filtered ECG Signal', legendgroup="Filtered ECG"), row=1, col=1)
            fig.add_trace(go.Scatter(mode='markers', x=Rpeaks, y=ecg_signal[Rpeaks], name='R-peaks--Ramanujan', marker=dict(color='Red',size=10)), row=1, col=1)
            fig.add_trace(go.Scatter(y=Period, name='Period', legendgroup="Period"), row=2, col=1)
            plot(fig, filename=file+"Fuducial_Points.html")

