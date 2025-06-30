import os
import numpy as np
import pandas as pd
from typing import Optional, List
from scipy.signal import find_peaks, butter, filtfilt, sosfilt
import heartpy as hp
import matplotlib.pyplot as plt
import signal

def read_log_to_csv(
    data_path: str,
    person: str,
    color: Optional[str] = None,
    date: Optional[str] = None,
    both: bool = False
) -> pd.DataFrame:
    """
    Read log files and convert to CSV format
    
    Args:
        data_path (str)
        person (str): Name of the person (duc, duong, or dung)
        color (str, optional): Color filter ('blue' or 'green'). If None, read both colors
        date (str, optional): Date filter (e.g., '25_6'). If None, read all dates
        both (bool): If True, read all dates for the specified person
    
    Returns:
        pd.DataFrame: DataFrame containing the combined log data with columns [pcg, ppg_red, ppg_ir]
    """
    # List all files in the directory
    all_files = os.listdir(data_path)
    
    # Filter files based on parameters
    filtered_files = []
    for file in all_files:
        if not file.endswith('.log'):
            continue
            
        parts = file.split('_')
        if len(parts) != 5:
            continue
            
        file_person = parts[0]
        file_date = f"{parts[1]}_{parts[2]}"
        file_color = parts[3]
        
        # Check if file matches the criteria
        if file_person != person:
            continue
            
        if color is not None and file_color != color:
            continue
            
        if not both and date is not None and file_date != date:
            continue
            
        filtered_files.append(file)
    
    if not filtered_files:
        raise ValueError(f"No matching files found for person={person}, color={color}, date={date}")
    
    # Read and combine all matching files
    dfs = []
    for file in filtered_files:
        file_path = os.path.join(data_path, file)
        # Read the log file and convert to DataFrame
        df = pd.read_csv(file_path, header=None, names=['pcg', 'ppg_red', 'ppg_ir'])
        dfs.append(df)
    
    # Combine all DataFrames
    result = pd.concat(dfs, ignore_index=True)
    return result

def read_clean_pcg_txt(file_path):
    clean_lines = []
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            # Chỉ giữ dòng là số nguyên hoặc số thực, bỏ dòng có ký tự lạ
            try:
                value = float(line)
                clean_lines.append(value)
            except ValueError:
                continue  # Bỏ dòng lỗi
    return pd.DataFrame({'pcg': clean_lines})

def plot_signal(signal, title = 'Raw'):
    plt.figure(figsize=(12, 4))
    plt.plot(signal)
    plt.title(title)
    plt.xlabel('Sample Index')
    plt.ylabel('Value')
    plt.grid(True)
    plt.show()

def add_timestamp_column(df: pd.DataFrame, fs: float) -> pd.DataFrame:
    n = len(df)
    df = df.copy()
    df['timestamp'] = [i / fs for i in range(n)]
    return df

def get_ac_component(ppg_data, fs):
    window_median = int(fs * 0.1)
    window_mean = fs

    median_data = movmedian_data(ppg_data, window_median)
    baseline = movmean_data(median_data, window_mean)
    return median_data - baseline

# ====== TÍNH BPM (NHỊP TIM) ======
def calculate_bpm(signal, fs, min_distance_sec=0.4, height_factor=0.5):
    """
    Tính nhịp tim (BPM) từ tín hiệu.
    Args:
        signal (array-like): Tín hiệu đầu vào
        fs (float): Tần số lấy mẫu (Hz)
        min_distance_sec (float): Khoảng cách tối thiểu giữa 2 đỉnh (giây)
        height_factor (float): Hệ số nhân với std để làm ngưỡng phát hiện đỉnh
    Returns:
        float hoặc None: BPM nếu đủ đỉnh, None nếu không đủ
    """
    min_distance = int(min_distance_sec * fs)
    height = np.std(signal) * height_factor
    peaks, _ = find_peaks(signal, distance=min_distance, height=height)
    if len(peaks) < 2:
        return None
    rr_intervals = np.diff(peaks)
    bpm = 60 * fs / np.mean(rr_intervals)
    return bpm

def calculate_bpm_per_window(pcg_signal, fs, bpm_window_sec=60):
    """
    Tính BPM theo từng cửa sổ thời gian từ tín hiệu PCG.

    Args:
        pcg_signal: np.array, tín hiệu PCG đầu vào
        fs: int, tần số lấy mẫu (Hz)
        bpm_window_sec: int, độ dài mỗi cửa sổ tính BPM (giây)

    Returns:
        bpm_list: danh sách BPM tính được cho mỗi cửa sổ
        time_stamps: danh sách thời điểm trung tâm của từng cửa sổ (tính bằng giây)
    """
    window_size = int(fs * bpm_window_sec)
    num_windows = len(pcg_signal) // window_size
    bpm_list = []
    time_stamps = []

    for i in range(num_windows):
        start = i * window_size
        end = start + window_size
        segment = pcg_signal[start:end]

        # Bandpass filter (giả sử tần số PCG 20–50Hz)
        sos = butter(4, [20, min(50, fs / 2 - 1)], btype='bandpass', fs=fs, output='sos')
        filtered = sosfilt(sos, segment)

        # Peak detection
        peaks, _ = find_peaks(filtered, distance=fs*0.3, height=np.max(filtered)*0.3)
        rr_intervals = np.diff(peaks) / fs

        # Tính BPM
        if len(rr_intervals) > 0:
            avg_rr = np.mean(rr_intervals)
            bpm = 60 / avg_rr
        else:
            bpm = 0

        bpm_list.append(bpm)
        time_stamps.append((start + end) / 2 / fs)  # Thời điểm giữa cửa sổ

    return bpm_list, time_stamps

# ========== PREPROCESSING ==========

def movmedian_data(signal, k):
    return pd.Series(signal).rolling(k, min_periods=1, center=True).median().to_numpy()

def movmean_data(signal, k):
    return pd.Series(signal).rolling(k, min_periods=1, center=True).mean().to_numpy()

def bandpass_filter(signal, fs=1000, fL=20, fH=120, order=3):
    nyq = 0.5 * fs  # Tần số Nyquist
    low = fL / nyq
    high = fH / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

def preprocess_ppg(signal, fs):
    # Step 1: Median filter to remove spikes
    win_median = int(fs * 0.1)
    signal_median = movmedian_data(signal, win_median)

    # Step 2: Moving average for baseline
    win_mean = fs
    baseline = movmean_data(signal_median, win_mean)

    # Step 3: Get AC & DC
    ac = signal_median - baseline
    dc = baseline

    # Step 4: Bandpass filter (optional but useful)
    ac_filtered = bandpass_filter(ac, fs)

    return ac_filtered, dc

def calculate_spo2(ac_red, dc_red, ac_ir, dc_ir):
    # Avoid division by zero
    eps = 1e-6
    ac_red_mean = np.mean(np.abs(ac_red))
    ac_ir_mean = np.mean(np.abs(ac_ir))
    dc_red_mean = np.mean(dc_red + eps)
    dc_ir_mean = np.mean(dc_ir + eps)

    R = (ac_red_mean / dc_red_mean) / (ac_ir_mean / dc_ir_mean)
    spo2 = 110 - 25 * R
    return np.clip(spo2, 0, 100)
