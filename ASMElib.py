import pandas as pd
import numpy as np
from scipy.interpolate import interpn
from scipy.interpolate import interp1d

import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def read_t_im(filename, input_temp, input_si):

    # Read St table for t_im 
    # For given temperature and input_si
    # Read conservative allowable time 
    
    df = pd.read_csv(filename, encoding='utf-8-sig', header=None)
    x_list = df.iloc[0, 1:].astype(float).to_numpy()
    y_list = df.iloc[1:, 0].astype(float).to_numpy()
    si_matrix = df.iloc[1:, 1:].astype(float).to_numpy()

    if input_temp in y_list:
        temp_idx = list(y_list).index(input_temp)
        si_row = si_matrix[temp_idx]
    else:
        if input_temp < y_list.min() or input_temp > y_list.max():
            return f"Error: 입력한 온도 {input_temp}는 보간 가능한 범위를 벗어났습니다."

        lower_idx = np.where(y_list < input_temp)[0].max()
        upper_idx = np.where(y_list > input_temp)[0].min()

        y1, y2 = y_list[lower_idx], y_list[upper_idx]
        si1, si2 = si_matrix[lower_idx], si_matrix[upper_idx]
        ratio = (input_temp - y1) / (y2 - y1)
        si_row = si1 + ratio * (si2 - si1)

    # 최소 Si보다 작으면 무한대로 간주
    if input_si < min(si_row):
        return 9999999

    # 정확히 일치하는 Si 값이 있으면 가장 작은 시간 반환
    for i in range(len(si_row)):
        if si_row[i] == input_si:
            return int(x_list[i])

    # Si > input_si인 경우, 오른쪽에서 왼쪽으로 검사 (보수적 시간)
    for i in reversed(range(len(si_row))):
        if si_row[i] > input_si:
            return int(x_list[i])

    # default return 은 9999999라는 큰 시간으로 잡음. 
    return 9999999

def determine_K_and_Kt():
    
    img = mpimg.imread('./images/SmReductionRatio.png')
    plt.imshow(img)
    plt.axis('off')  # 축 없애기
    plt.show()

    print("섹션 형상에 맞는 K값을 입력하세요. (참고: Shell-Type 구조의 벽면 Stress를 평가하는 경우 K=1.5를 입력)")

    K= float(input())
    Kt= (K+1.0)/2.0

    return K,Kt

from typing import List, Tuple, Dict
import pandas as pd

Segments = List[Tuple[float, float, float]]  # (T_start, T_end, duration_h)

def extract_Tt_list_for_Smt(segments: Segments, mode: str = "mid") -> pd.DataFrame:
    """
    Convert segments (Ts, Te, duration) into a list of (Temperature, cumulative time).
    
    Parameters
    ----------
    segments : list of (T_start [°C], T_end [°C], duration_h)
    mode     : "mid" → ramp is lumped at midpoint temperature
               "two" → ramp is split into two halves (optional refinement)
    
    Returns
    -------
    DataFrame with Temperature [°C], Time [h]
    (same Temperature values are aggregated)
    """
    rows = []
    for (Ts, Te, dur) in segments:
        if Ts == Te:
            # plateau
            rows.append((Ts, dur))
        else:
            if mode == "mid":
                Tmid = 0.5 * (Ts + Te)
                rows.append((Tmid, dur))
            elif mode == "two":
                # split into two equal halves
                T1 = Ts + 0.25 * (Te - Ts)
                T2 = Ts + 0.75 * (Te - Ts)
                rows.append((T1, dur/2.0))
                rows.append((T2, dur/2.0))
            else:
                raise ValueError("mode must be 'mid' or 'two'")

    # 합산: 같은 T끼리 시간 누적
    df = pd.DataFrame(rows, columns=["Temperature [°C]", "Time [h]"])
    df = df.groupby("Temperature [°C]", as_index=False).sum()
    
    return df

def determine_So(filename):
    # CSV 파일 읽기
    df = pd.read_csv(filename)
    # 처음 두 열 보여주기
    print(df.iloc[:, :2])  
    # 온도 값: 열 이름에서 앞의 두 개 제외
    temperatureValues = df.columns[2:].astype(float).to_numpy()
    # 열 이름 재지정
    df = df.rename(columns={df.columns[0]: 'Case', df.columns[1]: 'Label'})
    # 사용자 입력: Case Number
    case = int(input("Case Number를 입력하세요 (예: 33~44): "))   
    # 해당 케이스 행 찾기
    matched_row = df[df['Case'] == case]
    if matched_row.empty:
        print(f"Error: Case {case} not found.")
        return None
    # stress 값만 추출 (앞 2열 제외)
    SoValues = matched_row.iloc[0, 2:].to_numpy(dtype=float)
    # 사용자 입력: Target Temperature
    targetT = float(input("온도를 입력하세요 [ xx °C] (예: 350.0  °C): "))
    # 선형 보간 수행
    So = np.interp(targetT, temperatureValues, SoValues)
    return So



def read_2D_data_xy(filename,x,y):
    
    df_raw = pd.read_csv(filename, header=None)

    # Extract time and temperature grids
    x_grid = df_raw.iloc[0, 1:].astype(float).to_numpy()       # shape: (11,)
    y_grid = df_raw.iloc[1:, 0].astype(float).to_numpy()       # shape: (16,)
    value_grid = df_raw.iloc[1:, 1:].astype(float).to_numpy()     # shape: (16, 11)

    # Optional: sanity check
    # print(f"X grid: {x_grid.shape}, Y grid: {y_grid.shape}, Values: {value_grid.shape}")
    
    point = np.array([[y, x]])  # Must be 2D
    grid = (y_grid, x_grid)     # (y, x)

    try:
        result = interpn(grid, value_grid, point, method='linear', bounds_error=True)
        return result[0]
    except ValueError as e:
        return f"Interpolation error: {e}"
    
def interpolate_with_case_and_x(filename, x, case):

    try:
        # Load CSV
        df = pd.read_csv(filename)

        # Rename for clarity
        df = df.rename(columns={df.columns[0]: 'Case', df.columns[1]: 'Label'})

        # Locate row for the specified case
        matched_row = df[df['Case'] == case]
        if matched_row.empty:
            return f"Error: Case {case} not found."

        # ✅ Print label when case is matched
        label = matched_row.iloc[0]['Label']
        print(f"Matched case-{case} has Label: {label}")

        # Convert x-axis headers to float, starting from column 3 onward
        try:
            x_grid = np.array([float(col) for col in df.columns[2:]])
        except ValueError:
            return "Error: All column headers after the second must be numeric x-values."

        # Extract y-values from the matched row (excluding Case and Label)
        y_values = matched_row.iloc[0, 2:].to_numpy()

        # Filter valid values
        valid_mask = ~pd.isna(y_values)
        x_valid = x_grid[valid_mask]
        y_valid = y_values[valid_mask]

        if len(x_valid) < 2:
            return f"Error: Not enough valid data points to interpolate for case {case}."

        # Perform interpolation
        f = interp1d(x_valid, y_valid, kind='linear', bounds_error=True)
        return float(f(x))

    except Exception as e:
        return f"Unexpected error during interpolation: {e}"

