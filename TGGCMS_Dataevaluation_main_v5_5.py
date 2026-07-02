# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 16:09:21 2022
requires:
python 3.12
pandas: 2.2.2
numpy: 1.26.4
matplotlib: 3.9.2
scipy: 1.13.1
or compatible
@author: poschmann

Overview

    The code is a data analysis tool for GC-MS (Gas Chromatography-Mass Spectrometry) data. It provides various functions to process and visualize the data, including:
    
    Extracting single mass scans from the Total Ion Chromatogram (TIC)
    Finding maxima in the chromatogram and extracting mass spectra at each maxima
    Creating heatmaps and tables of the GC-MS run
    Denoising the data using a Butterworth filter
    Normalizing the intensity of mass spectra
    Important: The program operates on files stemming from PerkinElmer TG-GC-MS system. Naming of the files must be **.TXT for TG-data and **_GCMS.TXT for MS data. ** must be in both cases exactly the same. Both files must be in the same folder. 
Functions

    get_temperature_column(MSdf, file): This function reads the temperature data from a .TXT file and creates a pandas DataFrame with the temperature data.
    add_temp_column(pivRng1, TG_list, TGdataframe): This function adds a temperature column to the pivoted data and returns the updated pivoted data.
    getlistoffloat(MSdf, text): This function prompts the user to input a list of float values and returns the list.
    getlistofint(MSdf, text): This function prompts the user to input a list of integer values and returns the list.
    getlimits(MSdf, limitinput): This function prompts the user to input a range of values and returns the start and end values.
    getvalue(MSdf, valueinputtext, valuetype): This function prompts the user to input a value of a specific type and returns the value.
    getyninput(MSdf, inputquestion): This function prompts the user to input a yes/no question and returns the answer.
    normalizeIntensity(table): This function normalizes the intensity of a table by dividing by the maximum intensity and rounding to the nearest integer.
    makegraph(X, Y): This function creates a simple line graph with the given X and Y values.
    massspecplot(X, Y, Label, title, show): This function creates a bar chart with the given X and Y values and saves it to a file.
    makegraphtofile(X, Y, LABEL, X_axis, Y_axis, X_rangemin, X_rangemax, Y_rangemin, Y_rangemax, title, show): This function creates a line graph with the given X and Y values and saves it to a file.
    makegraphplusscattertofile(X, Y, LABEL, X_scatter, Y_scatter, scatter_label, X_axis, Y_axis, X_rangemin, X_rangemax, Y_rangemin, Y_rangemax, title, show): This function creates a line graph with the given X and Y values and adds a scatter plot to the graph.
    massestolookat(MSdf): This function rearranges the data of each TIC, calculates the Signal-to-Noise ratio (S/N) for each mass, and saves the results to a file.
    noisefilter(MSdf): This function applies a Butterworth filter to the data to remove noise.
    getsinglemassspectra(MSdf): This function extracts single mass spectra from the TIC at a specified retention time.
    maximamassscan(MSdf, TG_list, TGdataframe): This function finds maxima in the chromatogram and extracts mass spectra at each maxima.
    maximamassscan_oTG(MSdf): This function finds maxima in the chromatogram and extracts mass spectra at each maxima without using the temperature data.
    getchromandspectraA(MSdf): This function finds maxima in the chromatogram and extracts mass spectra at each maxima using a linear background subtraction method.
    getchromandspectraB(MSdf): This function finds maxima in the chromatogram and extracts mass spectra at each maxima using a non-linear background subtraction method.
    makingheatmapandfile(MSdf): This function creates a heatmap and table of the GC-MS run.
    extractmassscan(MSdf): This function extracts single mass scans from the TIC.
    getmassspectrabyRettime(RSdf, Range, Nr): This function extracts mass spectra at a specified retention time.
    Main Function
    
    The main function main(MSdf) is the entry point of the program. It prompts the user to select an option from a menu and calls the corresponding function based on the user's choice.

Variables

    The code uses several variables to store data, including:
    
    MSdf: a pandas DataFrame containing the GC-MS data
    file: a list of file names
    subdir: a string containing the path to the dataset
    RetTime: a float value containing the retention time
    ISDT: a float value containing the inter-scan delay time
    Scan: a float value containing the scan number
    MSlist: a list of lists containing the GC-MS data
    valid: a string containing the user's answer to a yes/no question
Notes

    The code assumes that the GC-MS data is stored in a .TXT file with a specific format.
    The code uses several external libraries, including pandas, numpy, and matplotlib.
    The code has several options for customizing the output, including the ability to select the type of background subtraction method to use.
    The code has several options for visualizing the data, including the ability to create heatmaps and tables.

"""

# Overall programm handling TGGCMS files
import os
import pandas as pd
from scipy.signal import argrelmin
from scipy.signal import argrelextrema
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import seaborn as sns
from scipy.signal import butter, filtfilt
from matplotlib.pyplot import pause 
import glob
import sys
import json
import copy
import tkinter as tk
from tkinter import filedialog, messagebox

def remember_path():
    """
    Reads the last used path from a text file and returns it.
    
    Returns:
        str: The last used path as a string. If the file does not exist, returns an empty string.
    """
    try:
        path = os.path.realpath(__file__).rsplit("\\",1)[0]+'\\'
        with open(f'{path}last_used_paths.json', 'r') as f:
            json_dump = json.load(f)
            return json_dump.get(__file__, 'C:\\')  # Return the last used path or default if not found
    except (FileNotFoundError, json.JSONDecodeError):
        return ''  # Return empty string if file does not exist or is invalid

def save_path(pathdata_folder):
    """
    Saves the given path to a text file for future reference.
    
    Args:
        pathdata_folder (str): The path to save.
    """
    path = os.path.realpath(__file__).rsplit("\\",1)[0]+'\\'
    try:
        with open(f'{path}last_used_paths.json', 'r') as f:
            program_list = json.load(f)
            program_list[__file__] = pathdata_folder
        with open(f'{path}last_used_paths.json', 'w') as f:
            json.dump(program_list, f)
    except (FileNotFoundError, json.JSONDecodeError):
        with open(f'{path}last_used_paths.json', 'w') as f:
            json.dump({__file__: pathdata_folder}, f)


def select_folder(Titletext, initial_path):
    """Opens a folder selection dialog and allows the user to choose a folder.

    This function creates a hidden Tkinter root window and displays a native folder dialog.
    The selected folder path can then be used for further processing.

    Features:
    - Native folder dialog (operating system-specific).
    - Returns the folder path as a string or `None` if canceled.

    Args:
        None

    Returns:
        str:
            The absolute path of the selected folder as a string.
            Returns `None` if the user cancels the dialog.
            Returns seperate info if a folder is selected.

    Raises:
        None

    Examples:
        >>> selected_folder = select_folder()
        >>> if selected_folder:
        ...     print(f"Selected folder: {selected_folder}")
        Selected folder: /home/user/data.csv

    Notes:
        - The dialog does not support custom descriptions (e.g., as a label).
          Use the `title` parameter in the dialog for user instructions.
        - For a more detailed UI with descriptions, create a custom Tkinter window.
    """
    initial_path = remember_path()
    # Tkinter-Hauptfenster erstellen (wird für filedialog benötigt)
    root = tk.Tk()
    root.withdraw()  # Fenster verstecken, da wir nur den Dialog brauchen

    # Dateiauswahldialog anzeigen
    file_path = filedialog.askdirectory(
        title=Titletext,
        initialdir=initial_path,  # Standardmäßig das Wurzelverzeichnis
        mustexist=True
    )

    # Überprüfen, ob eine Datei ausgewählt wurde
    if file_path:
        print(f"Selected Folder: {file_path}")  # Optional für die Konsole
        file_true = True
        save_path(file_path)
    else:
        messagebox.showinfo("Break", "No folder selected.")
        file_true = False
    return file_path, file_true



def get_temperature_column(MSdf, file):
    """
    Extract temperature program data and sample temperature information from TGA (Thermogravimetric Analysis) 
    log files and associate it with mass spectrometry (MS) data.
    
    This function reads a TGA control file (`.TXT`) to extract the thermal program (heating, cooling, holding 
    steps) and maps the sample temperature to specific time points in the MS data. It returns:
    - A DataFrame with time-resolved sample temperature data,
    - A DataFrame describing the thermal program steps,
    - A list of time-temperature pairs at key program milestones.
    
    The function handles complex TGA events such as:
    - Heating, cooling, and holding phases
    - Channel 2 switching (On/Off)
    - Temperature- and time-triggered actions
    - Isothermal steps
    
    Args:
        MSdf (pd.DataFrame): The mass spectrometry data DataFrame, expected to have a 'Time' column.
        file (list): A list containing the file path or name (e.g., [filename]) used to locate the TGA log.
    
    Returns:
        tuple:
            - TGdataframe (pd.DataFrame): DataFrame with time-series data from TGA, including:
                - Time (s)
                - Unsubstracted_Weight (% of initial weight)
                - Baseline_Weight (g)
                - Program_Temperature (°C)
                - Sample_Temperature (°C)
                - Sample_Purge_Flow (mL/min)
                - Balance_Purge_Flow (mL/min)
            - TG_list (pd.DataFrame): DataFrame describing the thermal program steps:
                - Operation (Heat, Cool, Hold, Off)
                - Start_Temperature (°C)
                - End_Temperature (°C)
                - Rate (°C/min)
                - Time (min)
                - Channel_2 (On/Off)
                - Runtime (cumulative time in min)
            - TimeTemplist (list): List of [time, temperature] pairs at key milestones (e.g., start, end, 
              and any time points where temperature is recorded).
    
    Raises:
        FileNotFoundError: If the corresponding .TXT file is not found.
        ValueError: If the TGA file format is invalid or parsing fails.
        Exception: For any unexpected errors during file reading or processing.
    
    Notes:
        - The function assumes the TGA log file is named after the sample (e.g., "SampleName.TXT").
        - Temperature values are rounded to the nearest integer.
        - The sample weight is normalized to 100% at the start.
        - If no TGA data is found, an empty DataFrame is returned with default column names.
    
    Example:
        >>> TGdata, TGsteps, milestones = get_temperature_column(MSdf, ["Sample1_01.txt"])
        >>> print(TGsteps)
          Operation  Start_Temperature  End_Temperature  Rate  Time Channel_2  Runtime
        0      Heat              25.0            100.0   10.0   7.5       On      7.5
        1      Hold             100.0            100.0    0.0  30.0       On     37.5
    """
    samplename = file[0].split("_")[0]
    TG_list = []
    TGdataframe = []
    TGdata_in_following_line = False
    Channel_2 = "Off"
    read_next_line = False
    try:
        with open(samplename + ".TXT") as TGdata:
            for line in nonblank_lines(TGdata):
                if "Start the Run" in line:
                    TG_list.append(["Off", None, None, None, None, Channel_2])
                if "Hold" in line:
                    TG_list.append(
                        [
                            "Hold",
                            line.split(" ")[5].split("°")[0],
                            line.split(" ")[5].split("°")[0],
                            0,
                            line.split(" ")[2],
                            Channel_2,
                        ]
                    )
                if "Heat" in line:
                    TG_list.append(
                        [
                            "Heat",
                            line.split(" ")[2].split("°")[0],
                            line.split(" ")[4].split("°")[0],
                            line.split(" ")[6].split("°")[0],
                            (
                                float(line.split(" ")[4].split("°")[0])
                                - float(line.split(" ")[2].split("°")[0])
                            )
                            / float(line.split(" ")[6].split("°")[0]),
                            Channel_2,
                        ]
                    )
                if "Cool" in line:
                    TG_list.append(
                        [
                            "Cool",
                            line.split(" ")[2].split("°")[0],
                            line.split(" ")[4].split("°")[0],
                            -float(line.split(" ")[6].split("°")[0]),
                            (
                                float(line.split(" ")[2].split("°")[0])
                                - float(line.split(" ")[4].split("°")[0])
                            )
                            / float(line.split(" ")[6].split("°")[0]),
                            Channel_2,
                        ]
                    )
                if "Switch External Channel" in line:
                    if "to On" in line:
                        # TG_list[-1][-1]='On'
                        Channel_2 = "On"
                    if "to Off" in line:
                        # TG_list[-1][-1]='Off'
                        Channel_2 = "Off"
                    read_next_line = True
                if (
                    "Action occurs Immediately" in line
                    and Channel_2 == "On"
                    and read_next_line == True
                ):
                    TG_list[-1][-1] = "On"
                    read_next_line = False
                    continue
                if (
                    "if the Temperature" in line
                    and Channel_2 == "On"
                    and read_next_line == True
                ):
                    trigger = line.rsplit(" ")[-1]
                    TG_list[-1][-1] = "On @ " + str(trigger)
                    read_next_line = False
                    continue
                if (
                    "if the Time" in line
                    and Channel_2 == "On"
                    and read_next_line == True
                ):
                    trigger = line.rsplit(" ")[-2]
                    TG_list[-1][-1] = "On @ " + str(trigger) + " min"
                    read_next_line = False
                    continue
                if (
                    "Action occurs Immediately" in line
                    and Channel_2 == "Off"
                    and read_next_line == True
                ):
                    TG_list[-1][-1] = "Off"
                    read_next_line = False
                    continue
                if (
                    "if the Temperature" in line
                    and Channel_2 == "Off"
                    and read_next_line == True
                ):
                    trigger = line.rsplit(" ")[-1]
                    TG_list.append(copy.deepcopy(TG_list[-1]))
                    TG_list[-1][-2] = 0
                    TG_list[-1][-1] = "Off " + str(trigger)
                    read_next_line = False
                    continue
                if (
                    "if the Time" in line
                    and Channel_2 == "Off"
                    and read_next_line == True
                ):
                    trigger = line.rsplit(" ")[-2]
                    TG_list.append(copy.deepcopy(TG_list[-1]))
                    TG_list[-1][-2] = 0
                    TG_list[-1][-1] = "Off " + str(trigger) + " min"
                    read_next_line = False
                    continue
                if " TGA Isothermal" in line:
                    TGdata_in_following_line = True
                if TGdata_in_following_line == True:
                    if ("TGA" in line) or ("Balance" in line) or ("Purge Flow" in line):
                        TGdata_in_following_line = True
                    else:
                        TGdataframe.append(line.split("\t"))

        TGdataframe = pd.DataFrame(
            TGdataframe,
            columns=[
                "Index",
                "Time",
                "Unsubstracted_Weight",
                "Basline_Weight",
                "Program_Temperature",
                "Sample_Temperature",
                "Sample_Purge_Flow",
                "Balance_Purge_Flow",
            ],
        )
        TG_list = pd.DataFrame(
            TG_list,
            columns=[
                "Operation",
                "Start_Temperature",
                "End_Temperature",
                "Rate",
                "Time",
                "Channel_2",
            ],
        )
        TG_list.iloc[:, 1:-1] = TG_list.iloc[:, 1:-1].astype(float)
        TG_list = TG_list.loc[TG_list["Operation"] != "Off"].reset_index(drop=True)
        runtime = 0
        for i in TG_list.index:
            element = TG_list.iloc[i, 4]
            runtime += element
            TG_list.loc[TG_list.index == i, "Runtime"] = runtime
    except:
        print("No matching TG-data in folder")
        TGdataframe = pd.DataFrame(
            TGdataframe,
            columns=[
                "Index",
                "Time",
                "Unsubstracted_Weight",
                "Basline_Weight",
                "Program_Temperature",
                "Sample_Temperature",
                "Sample_Purge_Flow",
                "Balance_Purge_Flow",
            ],
        )

    TGdataframe = TGdataframe.drop(["Index"], axis=1).astype(float)
    TGdataframe["Unsubstracted_Weight"] = (
        TGdataframe["Unsubstracted_Weight"]
        / TGdataframe["Unsubstracted_Weight"][0]
        * 100
    )

    Runtimelist = [TG_list["Runtime"].min(), TG_list["Runtime"].max()]

    TimeTemplist = []
    last = False
    if len(Runtimelist) == 0:
        TimeTemplist = []
    else:
        for i in Runtimelist:
            if i <= TGdataframe["Time"].max():
                temperature = np.round(
                    TGdataframe.loc[
                        (TGdataframe["Time"].round(1) >= np.round(i, 1) - 0.05)
                        & (TGdataframe["Time"].round(1) <= np.round(i, 1) + 0.05)
                    ]
                    .loc[:, "Sample_Temperature"]
                    .max(),
                    0,
                )
                TimeTemplist.append([i, temperature])
            elif last == False:
                temperature = np.round(
                    TGdataframe.loc[
                        TGdataframe["Time"] == TGdataframe["Time"].max()
                    ].loc[:, "Sample_Temperature"],
                    0,
                ).values[-1]
                last = True
                TimeTemplist.append([i, temperature])
    return TGdataframe, TG_list, TimeTemplist


def add_temp_column(pivRng1, TG_list, TGdataframe):
    """
    Add a sample temperature column to the pivoted mass spectrometry (MS) data based on TGA (Thermogravimetric Analysis) 
    temperature program and timing information.

    This function maps the sample temperature from the TGA data to each time point in the MS data, accounting for:
    - The start time offset from the TGA program (including "Off" periods)
    - Channel 2 activation (On/Off) and triggered events
    - Gas chromatograph (GC) integration (if used)
    - Multi-shot experiments

    The temperature is interpolated from the TGA data using time proximity (±0.1 min) and averaged if multiple points exist.

    Args:
        pivRng1 (pd.DataFrame): Pivoted MS data DataFrame, where each row corresponds to a time point (index = time).
        TG_list (pd.DataFrame): DataFrame describing the thermal program steps from TGA, including:
            - Operation (Heat, Cool, Hold, Off)
            - Start_Temperature (°C)
            - End_Temperature (°C)
            - Rate (°C/min)
            - Time (min)
            - Channel_2 (On/Off)
            - Runtime (cumulative time in min)
        TGdataframe (pd.DataFrame): DataFrame with high-resolution TGA time-series data, including:
            - Time (min)
            - Sample_Temperature (°C)
            - Unsubstracted_Weight (%)
            - Baseline_Weight (g)
            - Program_Temperature (°C)
            - Sample_Purge_Flow (mL/min)
            - Balance_Purge_Flow (mL/min)

    Returns:
        pd.DataFrame: The input `pivRng1` DataFrame with an added 'Sample_Temperature' column (or 'Temperature' if GC is used),
        where each value corresponds to the sample temperature at the time of measurement.

    Raises:
        ValueError: If the start time cannot be determined or if temperature data is missing.
        KeyError: If required columns are missing in `TGdataframe` or `TG_list`.
        Exception: For any unexpected errors during processing.

    Notes:
        - The function uses `getyninput()` and `getlistoffloat()` for user input (assumes these are defined elsewhere).
        - If GC is used, the temperature is set to the initial activation temperature (`starttemp`) for all entries.
        - For multi-shot experiments, the user must provide shot times to align temperature data.
        - Temperature values are rounded to 1 decimal place.
        - The function assumes `pivRng1` index represents time in minutes.

    Example:
        >>> pivRng1 = add_temp_column(pivRng1, TG_list, TGdataframe)
        >>> print(pivRng1.head())
           m/z_18  m/z_28  Sample_Temperature
        0    100     200              25.0
        1    105     210              26.1
    """
    GC_MS_Data = getyninput(
        MSdf,
        "Is gaschromatograph used? (y/n): ",
    )
    pivRng1 = pivRng1.transpose()
    starttime = 0
    no_on = True
    for i in TG_list.index:
        status = TG_list.iloc[i, 5]
        if status == "Off" and no_on == True:
            starttime += TG_list.iloc[i, 6]
        if status == "On":
            starttime += 0
            starttemp = TGdataframe.loc[TGdataframe["Time"] == starttime][
                "Sample_Temperature"
            ].values
            no_on = False
        if "On " in status:
            if "min" in status:
                start = str(status.split(" ")[2])
                starttime += start
                starttemp = TGdataframe.loc[TGdataframe["Time"] == starttime][
                    "Sample_Temperature"
                ]
                no_on = False
            if "°C" in status:
                starttemp = float(status.split("=")[-1].split("°")[0])
                starttime += 0
                no_on = False
    temperature_column = []
    if GC_MS_Data == "n":
        for entry in pivRng1.index:
            temperature = np.round(
                TGdataframe.loc[
                    (TGdataframe["Time"] <= entry + 0.1 + starttime)
                    & (TGdataframe["Time"] >= entry - 0.1 + starttime)
                ]["Sample_Temperature"].mean(),
                1,
            )
            temperature_column.append(temperature)
        pivRng1["Sample_Temperature"] = temperature_column
    if GC_MS_Data == "y":
        multishot = getyninput(MSdf, "Is it a multishot experiment? :")
        if multishot == "n":
            pivRng1.loc[:, "Temperature"] = starttemp
    pivRng1 = pivRng1.transpose()
    return pivRng1


def getlistoffloat(MSdf, text):
    """
    Prompt the user to enter a list of floating-point numbers, with input validation and error handling.

    This function repeatedly asks for input until all entries are valid floats or the user requests to return to the main menu.
    It supports space-separated values and includes a 'b' shortcut to return to the main menu.

    Args:
        MSdf (pd.DataFrame): The mass spectrometry data DataFrame (used to call the main menu).
        text (str): The prompt message to display to the user (e.g., "Enter values: ").

    Returns:
        list: A list of float values extracted from the user input.

    Raises:
        SystemExit: If the user enters 'b' and the main menu is called (this exits the current flow).

    Notes:
        - The function uses `input()` and `split()` to parse space-separated values.
        - If any entry is not a valid float and is not 'b', the user is prompted again.
        - Entering 'b' triggers the `main(MSdf)` function, which typically returns to the main menu.
        - The function assumes `main()` is defined elsewhere and handles the menu logic.

    Example:
        >>> values = getlistoffloat(MSdf, "Enter shot times (min): ")
        Enter shot times (min): 1.5 2.3 3.7 4.1
        >>> print(values)
        ['1.5', '2.3', '3.7', '4.1']
    """
    print("'b' for back to main menu")
    floatlist = input(text).split()
    correctentry = "n"
    while correctentry == "n":
        for entry in floatlist:
            try:
                float(entry)
                correctentry = "y"
            except:
                if entry == "b":
                    main(MSdf)
                else:
                    print("Wrong input")
                    floatlist = input(text).split()

    return floatlist


def getlistofint(MSdf, text):
    """
    Prompt the user to enter a list of integers, with input validation and error handling.   
    This function repeatedly asks for input until all entries are valid integers or the user requests to return to the main menu.
    It supports space-separated values and includes a 'b' shortcut to return to the main menu.
       
    Args:
    MSdf (pd.DataFrame): The mass spectrometry data DataFrame (used to call the main menu).
    text (str): The prompt message to display to the user (e.g., "Enter values: ").
       
    Returns:
    list: A list of integer values extracted from the user input.
       
    Raises:
    SystemExit: If the user enters 'b' and the main menu is called (this exits the current flow).
       
    Notes:
    - The function uses `input()` and `split()` to parse space-separated values.
    - If any entry is not a valid integer and is not 'b', the user is prompted again.
    - Entering 'b' triggers the `main(MSdf)` function, which typically returns to the main menu.
    - The function assumes `main()` is defined elsewhere and handles the menu logic.
       
    Example:
    >>> values = getlistofint(MSdf, "Enter shot numbers: ")
    Enter shot numbers: 1 2 3 4 5
    >>> print(values)
    ['1', '2', '3', '4', '5']
    """
    print("'b' for back to main menu")
    intlist = input(text).split()
    correctentry = "n"
    while correctentry == "n":
        for entry in intlist:
            try:
                int(entry)
                correctentry = "y"
            except:
                if entry == "b":
                    main(MSdf)
                else:
                    print("Wrong input")
                    intlist = input(text).split()

    return intlist


def getlimits(MSdf, limitinput):
    """
    Prompt the user to enter a range of values (start and end) with input validation and error handling.

    This function reads a space-separated string of two values (start and end) and converts them to floats.
    If the start and end values are equal, it automatically expands the range by 10% on both sides.
    The user can also return to the main menu by entering 'b'.

    Args:
        MSdf (pd.DataFrame): The mass spectrometry data DataFrame (used to call the main menu).
        limitinput (str): The prompt message to display to the user (e.g., "Enter range (start end): ").

    Returns:
        tuple: A tuple containing:
            - start (float): The start of the range (expanded if equal to end).
            - end (float): The end of the range (expanded if equal to start).

    Raises:
        SystemExit: If the user enters 'b' and the main menu is called (this exits the current flow).

    Notes:
        - The function expects two space-separated values (start and end).
        - If the values are equal, the range is expanded to [0.9*start, 1.1*end] to avoid zero-width intervals.
        - If input is invalid or 'b' is entered, the function either returns to the main menu or re-prompts.
        - The function assumes `main()` is defined elsewhere and handles the menu logic.

    Example:
        >>> start, end = getlimits(MSdf, "Enter m/z range: ")
        Enter m/z range: 18 18
        >>> print(start, end)
        16.2 19.8
    """
    print("'b' for back to main menu")
    limits = input(limitinput).split(" ")
    start = limits[0]
    try:
        start = float(start)
        end = float(limits[-1])
        if start == end:
            start = 0.9 * start
            end = 1.1 * end
    except:
        if start == "b":
            main(MSdf)
        else:
            print("wrong input")
            getlimits(MSdf, limitinput)
    start = float(start)
    end = float(end)
    return start, end


def getvalue(MSdf, valueinputtext, valuetype):
    """
    Prompt the user to enter a single value of a specified type, with input validation and error handling.

    This function repeatedly asks for input until a valid value of the specified type (int or float) is entered.
    It supports a 'b' shortcut to return to the main menu.

    Args:
        MSdf (pd.DataFrame): The mass spectrometry data DataFrame (used to call the main menu).
        valueinputtext (str): The prompt message to display to the user (e.g., "Enter temperature: ").
        valuetype (type): The expected type of the input (int or float).

    Returns:
        int or float: The validated input value converted to the specified type.

    Raises:
        SystemExit: If the user enters 'b' and the main menu is called (this exits the current flow).

    Notes:
        - The function uses `input()` and attempts to convert the input to the specified type.
        - If conversion fails and the input is not 'b', the user is prompted again.
        - Entering 'b' triggers the `main(MSdf)` function, which typically returns to the main menu.
        - The function assumes `main()` is defined elsewhere and handles the menu logic.
        - The `valuetype` parameter must be `int` or `float`.

    Example:
        >>> temp = getvalue(MSdf, "Enter temperature (°C): ", float)
        Enter temperature (°C): 25.5
        >>> print(temp)
        25.5
    """
    valueinput = ""
    while (type(valueinput) != int) and (type(valueinput) != float):
        print("'b' for back to main menu")
        valueinput = input(valueinputtext)
        try:
            valueinput = valuetype(valueinput)
        except:
            if valueinput == "b":
                main(MSdf)
            else:
                print("wrong input")
    return valuetype(valueinput)


def getyninput(MSdf, inputquestion):
    """
    Prompt the user to answer a yes/no question with input validation and error handling.

    This function repeatedly asks for input until the user enters 'y' (yes) or 'n' (no).
    It also supports a 'b' shortcut to return to the main menu.

    Args:
        MSdf (pd.DataFrame): The mass spectrometry data DataFrame (used to call the main menu).
        inputquestion (str): The prompt message to display to the user (e.g., "Is gaschromatograph used? (y/n): ").

    Returns:
        str: The user's response, either 'y' or 'n'.

    Raises:
        SystemExit: If the user enters 'b' and the main menu is called (this exits the current flow).

    Notes:
        - The function expects the user to input 'y' for yes or 'n' for no.
        - If the input is invalid, the user is prompted again.
        - Entering 'b' triggers the `main(MSdf)` function, which typically returns to the main menu.
        - The function assumes `main()` is defined elsewhere and handles the menu logic.
        - The returned value is always lowercase ('y' or 'n').

    Example:
        >>> response = getyninput(MSdf, "Is gaschromatograph used? (y/n): ")
        Is gaschromatograph used? (y/n): yes
        wrong input
        Is gaschromatograph used? (y/n): y
        >>> print(response)
        y
    """
    print("'b' for back to main menu")
    ok = input(inputquestion)
    while (ok != "n") and (ok != "y"):
        if ok == "b":
            main(MSdf)
        else:
            print("wrong input")
            ok = input(inputquestion)
    return ok


def normalizeIntensity(table):
    """
    Normalize the 'Intensity' column of a DataFrame to a range of [0, 999] based on the maximum absolute intensity.

    This function scales the intensity values so that the maximum absolute value becomes 999, and all other values are
    proportionally scaled. The result is rounded to the nearest integer.

    Args:
        table (pd.DataFrame): The input DataFrame containing an 'Intensity' column to normalize.

    Returns:
        pd.DataFrame: The input DataFrame with the 'Intensity' column normalized to integers in the range [0, 999].

    Raises:
        KeyError: If the 'Intensity' column is not present in the DataFrame.
        ValueError: If the 'Intensity' column is empty or contains only NaN values.

    Notes:
        - The normalization uses the maximum absolute value to preserve the sign of the original data.
        - The result is scaled to 999 (not 1000) to avoid potential overflow in downstream processing.
        - The function modifies the DataFrame in-place and returns it.
        - The original data type of 'Intensity' is preserved after rounding.

    Example:
        >>> df = pd.DataFrame({'m/z': [18, 28, 32], 'Intensity': [100, 500, 200]})
        >>> normalized_df = normalizeIntensity(df)
        >>> print(normalized_df['Intensity'])
        0    199
        1    999
        2    399
        Name: Intensity, dtype: int64
    """
    for item in table["Intensity"]:
        table["Intensity"] = table["Intensity"] / table["Intensity"].abs().max() * 999
        table["Intensity"] = table["Intensity"].round(0)
    return table


def makegraph(X, Y):
    """
    Create and display a line plot of Y versus X with minimal styling.

    This function generates a simple line plot using matplotlib, clears any existing figure,
    plots the data with a line width of 1, displays the plot non-blocking, and pauses briefly
    to allow rendering before continuing execution.

    Args:
        X (array-like): The x-axis data (e.g., time, m/z, temperature).
        Y (array-like): The y-axis data (e.g., intensity, signal, voltage).

    Returns:
        None: This function does not return any value.

    Raises:
        ValueError: If X and Y have different lengths or are empty.
        TypeError: If X or Y cannot be converted to numpy arrays.

    Notes:
        - The function uses `plt.close()` to prevent figure accumulation.
        - `plt.show(block=False)` ensures the plot appears without blocking execution.
        - `pause(0.1)` gives time for the plot to render before continuing.
        - This is useful in interactive scripts where multiple plots are generated in sequence.
        - The figure size is fixed at 10x8 inches for consistent visualization.

    Example:
        >>> x = np.linspace(0, 10, 100)
        >>> y = np.sin(x)
        >>> makegraph(x, y)
    """
    plt.close()
    plt.figure(figsize=(10, 8))
    plt.plot(X, Y, lw=1)
    plt.show(block=False)
    pause(0.1)
    
    return


def massspecplot(X, Y, Label, title, show):
    """
    Create and save a bar plot of mass spectrometry data (m/z vs. intensity).

    This function generates a high-resolution bar plot for mass spectrometry data, with proper labeling,
    legend, and optional interactive display. The plot is saved to a file and can be shown optionally.

    Args:
        X (array-like): The x-axis data (e.g., m/z values).
        Y (array-like): The y-axis data (e.g., intensity counts).
        Label (str): The label for the legend (e.g., sample name or condition).
        title (str): The filename (including path) to save the plot as a PNG file.
        show (int or bool): If non-zero or True, display the plot interactively after saving.
            If 0 or False, only save the file without showing.

    Returns:
        None: This function does not return any value.

    Raises:
        ValueError: If X and Y have different lengths or are empty.
        TypeError: If X or Y cannot be converted to numpy arrays.
        OSError: If the file cannot be saved (e.g., due to permission or path issues).

    Notes:
        - The function uses `plt.close()` to prevent figure accumulation.
        - The figure size is fixed at 10x8 inches for consistent visualization.
        - The plot is saved at 300 DPI for high-quality output.
        - The x-axis is labeled "m/z", and the y-axis is labeled "Intensity /counts".
        - The legend is displayed using the provided Label.
        - The plot is saved before being shown (if requested), ensuring the file is complete.

    Example:
        >>> mzs = [18, 28, 32, 44]
        >>> intensities = [100, 500, 200, 80]
        >>> massspecplot(mzs, intensities, "Sample_A", "output/mass_spectrum.png", show=1)
    """
    plt.close()
    # start Chromatogramm plot
    plt.figure(figsize=(10, 8))
    # plot Chromatogramm
    plt.bar(X, Y, label=Label)
    # axis labels
    plt.xlabel("m/z")
    plt.ylabel("Intensity /counts")
    # legend
    plt.legend()
    # plotsaving
    plt.savefig(title, dpi=300)
    if show != 0:
        plt.show(block=False)
        pause(0.1)

    return


def makegraphtofile(
    X,
    Y,
    LABEL,
    X_axis,
    Y_axis,
    X_rangemin,
    X_rangemax,
    Y_rangemin,
    Y_rangemax,
    title,
    show):
    """
    Create and save a line plot with customizable axes, ranges, and optional interactive display.
    
    This function generates a high-resolution line plot using matplotlib, with support for:
    - Custom axis labels
    - Axis range limits
    - Legend
    - File saving at 300 DPI
    - Optional non-blocking display
    
    Args:
        X (array-like): The x-axis data (e.g., time, m/z, temperature).
        Y (array-like): The y-axis data (e.g., intensity, voltage, signal).
        LABEL (str): The label for the legend (e.g., sample name, condition).
        X_axis (str): The label for the x-axis (e.g., "Time (min)", "m/z").
        Y_axis (str): The label for the y-axis (e.g., "Intensity", "Voltage").
        X_rangemin (float or None): Minimum x-axis limit. Use False to skip setting range.
        X_rangemax (float or None): Maximum x-axis limit. Use False to skip setting range.
        Y_rangemin (float or None): Minimum y-axis limit. Use False to skip setting range.
        Y_rangemax (float or None): Maximum y-axis limit. Use False to skip setting range.
        title (str): The filename (including path) to save the plot as a PNG file.
        show (int or bool): If non-zero or True, display the plot interactively after saving.
            If 0 or False, only save the file without showing.
    
    Returns:
        None: This function does not return any value.
    
    Raises:
        ValueError: If X and Y have different lengths or are empty.
        TypeError: If X or Y cannot be converted to numpy arrays.
        OSError: If the file cannot be saved (e.g., due to permission or path issues).
    
    Notes:
        - The function uses `plt.close()` to prevent figure accumulation.
        - The figure size is fixed at 10x8 inches for consistent visualization.
        - The plot is saved at 300 DPI for high-quality output.
        - The line width is set to 1 for clean visualization.
        - Axis ranges are only applied if the corresponding min/max values are not False.
        - The plot is saved before being shown (if requested), ensuring the file is complete.
        - This function is ideal for batch processing and automated data visualization.
    
    Example:
        >>> time = [0, 1, 2, 3, 4]
        >>> signal = [1.2, 1.5, 1.3, 1.7, 1.6]
        >>> makegraphtofile(time, signal, "Sample A", "Time (min)", "Signal", 0, 4, 1.0, 1.8, "output/plot.png", show=1)
    """
    plt.close()
    # start  plot
    plt.figure(figsize=(10, 8))
    # plot
    plt.plot(X, Y, label=LABEL, lw=1)
    # axis labels
    plt.xlabel(X_axis)
    plt.ylabel(Y_axis)
    # legend
    plt.legend()
    # plot range if value given
    if X_rangemin != False:
        plt.xlim(X_rangemin, X_rangemax)
    if Y_rangemin != False:
        plt.ylim(Y_rangemin, Y_rangemax)
    # save figure
    plt.savefig(title, dpi=300)
    if show != 0:
        plt.show(block=False)

    return


def makegraphplusscattertofile(
    X,
    Y,
    LABEL,
    X_scatter,
    Y_scatter,
    scatter_label,
    X_axis,
    Y_axis,
    X_rangemin,
    X_rangemax,
    Y_rangemin,
    Y_rangemax,
    title,
    show):
    """
    Create and save a combined line and scatter plot with customizable axes, ranges, and optional interactive display.

    This function generates a high-resolution plot combining a line graph and scatter points, ideal for
    visualizing trends with highlighted data points. The plot is saved to a file and can be shown optionally.

    Args:
        X (array-like): The x-axis data for the line plot (e.g., time, m/z, temperature).
        Y (array-like): The y-axis data for the line plot (e.g., intensity, voltage, signal).
        LABEL (str): The label for the line plot in the legend (e.g., "Baseline", "Control").
        X_scatter (array-like): The x-axis data for the scatter points.
        Y_scatter (array-like): The y-axis data for the scatter points.
        scatter_label (str): The label for the scatter points in the legend (e.g., "Anomalies", "Peaks").
        X_axis (str): The label for the x-axis (e.g., "Time (min)", "m/z").
        Y_axis (str): The label for the y-axis (e.g., "Intensity", "Voltage").
        X_rangemin (float or None): Minimum x-axis limit. Use False to skip setting range.
        X_rangemax (float or None): Maximum x-axis limit. Use False to skip setting range.
        Y_rangemin (float or None): Minimum y-axis limit. Use False to skip setting range.
        Y_rangemax (float or None): Maximum y-axis limit. Use False to skip setting range.
        title (str or int): The filename (including path) to save the plot as a PNG file.
            If 0, the plot is not saved to file.
        show (int or bool): If non-zero or True, display the plot interactively after saving.
            If 0 or False, only save the file (if requested) without showing.

    Returns:
        None: This function does not return any value.

    Raises:
        ValueError: If X and Y have different lengths, or X_scatter and Y_scatter have different lengths.
        TypeError: If X, Y, X_scatter, or Y_scatter cannot be converted to numpy arrays.
        OSError: If the file cannot be saved (e.g., due to permission or path issues).

    Notes:
        - The function uses `plt.close()` to prevent figure accumulation.
        - The figure size is fixed at 10x8 inches for consistent visualization.
        - The line plot uses a line width of 1 for clean visualization.
        - Scatter points are colored red (`c="r"`) for high contrast.
        - Axis ranges are only applied if the corresponding min/max values are not False.
        - The plot is saved at 300 DPI for high-quality output.
        - The plot is saved before being shown (if requested), ensuring the file is complete.
        - This function is ideal for highlighting specific data points (e.g., peaks, anomalies) on a trend line.

    Example:
        >>> time = [0, 1, 2, 3, 4]
        >>> signal = [1.2, 1.5, 1.3, 1.7, 1.6]
        >>> peaks = [1.5, 3.0]
        >>> peak_values = [1.5, 1.7]
        >>> makegraphplusscattertofile(
        ...     time, signal, "Signal", peaks, peak_values, "Peaks",
        ...     "Time (min)", "Signal", 0, 4, 1.0, 1.8, "output/peaks.png", show=1
        ... )
    """
    plt.close()
    # start  plot
    plt.figure(figsize=(10, 8))
    # plot
    plt.plot(X, Y, label=LABEL, lw=1)
    plt.scatter(X_scatter, Y_scatter, label=scatter_label, c="r")
    # axis labels
    plt.xlabel(X_axis)
    plt.ylabel(Y_axis)
    # legend
    plt.legend()
    # plot range if value given
    if X_rangemin != False:
        plt.xlim(X_rangemin, X_rangemax)
    if Y_rangemin != False:
        plt.ylim(Y_rangemin, Y_rangemax)
    # save figure
    if title != 0:
        plt.savefig(title, dpi=300)
    if show != 0:
        plt.show(block=False)
        pause(0.1)
    return


def massestolookat(MSdf):
    """
    Identify and export mass-to-charge (m/z) values with high signal-to-noise (S/N) ratios from each scan range in mass spectrometry data.

    This function processes MS data by:
    - Grouping data by scan range (ScanRange)
    - Creating a pivot table of intensity vs. retention time for each scan range
    - Calculating the signal-to-noise ratio (S/N) for each m/z value
    - Filtering and ranking m/z values with S/N ≥ 20
    - Saving the results to a CSV file for each scan range

    The S/N ratio is calculated as:
        S/N = (max_intensity - mean_background) / std_background
    where background is defined as the first 70 data points (early retention times).

    Args:
        MSdf (pd.DataFrame): The mass spectrometry data DataFrame containing at least:
            - "ScanRange": The scan range identifier
            - "Mass": The m/z value (float)
            - "RetentionTime": The retention time (float)
            - "Intensity": The signal intensity (float)

    Returns:
        None: This function does not return any value.

    Raises:
        KeyError: If required columns ("ScanRange", "Mass", "RetentionTime", "Intensity") are missing.
        ValueError: If the data is empty or contains invalid values.
        OSError: If the output file cannot be saved (e.g., due to permission or path issues).

    Notes:
        - The function rounds m/z values to the nearest integer and retention times to 2 decimal places.
        - Background noise is calculated from retention times 0.02 to 0.72 (first 70 points).
        - Only m/z values > 11 are considered (to avoid low-mass noise).
        - Results are saved as "MassesSoverNratio_in_Scanrange_X.xy" for each scan range X.
        - The output file contains two columns: "Mass" and "S/R_ratio" (S/N ratio).
        - The results are sorted in descending order by S/N ratio.

    Example:
        >>> massestolookat(MSdf)
        # Creates files like:
        # MassesSoverNratio_in_Scanrange_1.xy
        # MassesSoverNratio_in_Scanrange_2.xy
    """
    # rearrange data of each TIC
    for Nr in pd.unique(MSdf["ScanRange"]):
        Range = MSdf[MSdf["ScanRange"] == Nr].drop("ScanRange", axis=1)
        Range["Mass"] = Range["Mass"].round(0).astype(int)
        Range["RetentionTime"] = Range["RetentionTime"].round(2)
        pivRng1 = Range.pivot_table(
            index="Mass", columns="RetentionTime", values="Intensity", aggfunc="sum"
        )
        pivRng1.fillna(0, inplace=True)
        df = pivRng1.reset_index()
        SoverNlist = []
        # iterate through every detected SIR
        for index, row in df.iterrows():
            # ignore masses below 11
            if row.iloc[0] <= 11:
                continue
            else:
                # calculate mean Intensity and stddeviation in first datapoints to determine Noise
                meanbase = row.iloc[2:72].sum() / 70
                stddevbase = row.iloc[2:72].std()
                # find maximum of SIR
                maxint = row.max()
                # devide maximum of background substracted SIR by standard deviation as value for Signal over noise ratio
                maxoverstddev = (maxint - meanbase) / stddevbase
                # generate list of masses with high S/N ratio with values above 20
                if maxoverstddev >= 20:
                    listentry = [row.iloc[0], maxoverstddev.round(2)]
                    SoverNlist.append(listentry)
        # generate file with interessting masses
        SoverNdf = pd.DataFrame(SoverNlist, columns=["Mass", "S/R_ratio"]).set_index(
            "Mass"
        )
        SoverNdf.sort_values("S/R_ratio", axis=0, ascending=False, inplace=True)
        SoverNdf.to_csv("MassesSoverNratio_in_Scanrange_" + str(Nr) + ".xy")
    return


def noisefilter(MSdf):
    """
    Apply a digital Butterworth low-pass filter to reduce noise in mass spectrometry chromatographic data.

    This function processes mass spectrometry data by:
    - Applying a zero-phase Butterworth low-pass filter to each mass signal
    - Preserving the temporal structure of peaks while smoothing out high-frequency noise
    - Generating filtered data with improved signal-to-noise ratio
    - Returning a cleaned DataFrame suitable for downstream analysis

    The filtering is performed on a per-mass basis using a second-order Butterworth filter with user-defined parameters.

    Args:
        MSdf (pd.DataFrame): The mass spectrometry data DataFrame containing at least:
            - "ScanRange": The scan range identifier
            - "Mass": The m/z value (float)
            - "RetentionTime": The retention time (float)
            - "Intensity": The signal intensity (float)

    Returns:
        pd.DataFrame: The filtered data with the same structure as the input, but with:
            - "ScanRange": Scan range identifier
            - "Scannumber": Sequential scan number
            - "Mass": m/z value
            - "RetentionTime": Retention time (float)
            - "Intensity": Filtered intensity (int)

    Raises:
        ValueError: If required columns are missing or if filter parameters are invalid.
        TypeError: If input data cannot be processed.
        OSError: If the output cannot be saved (though this function doesn't save files).
        Exception: For any unexpected errors during filtering.

    Notes:
        - The function uses `butter` and `filtfilt` from scipy.signal for zero-phase filtering.
        - The filter parameters are:
            - `fs`: Sampling frequency (derived from minimum peak width)
            - `cutoff`: Desired cutoff frequency (in Hz)
            - `order`: Filter order (fixed at 2)
        - The function creates and displays comparison plots of the original and filtered chromatograms.
        - Only positive intensity values are retained in the output.
        - The output DataFrame is sorted by "Scannumber" and "ScanRange" to match the input order.
        - The function uses `getvalue()` for interactive parameter input (assumes it's defined elsewhere).

    Example:
        >>> filtered_data = noisefilter(MSdf)
        >>> print(filtered_data.head())
           ScanRange  Scannumber  Mass  RetentionTime  Intensity
        0          1           1   18          0.01        120
        1          1           2   18          0.02        115
    """
    filter_ok = "n"
    while filter_ok == "n":
        # generate workingcopy of datafile
        MSdf_operate = MSdf.copy(deep=True)
        # ask for inputvalues for the Butterworth filter
        fs = 0.7 * getvalue(
            MSdf, "Minimum Peak Width in Seconds (~1.0): ", float
        )  #  int(input('Minimum Peak Width in Seconds (~1.0): '))
        cutoff = (
            getvalue(MSdf, "Noise Frequency in Hz (~0.2): 0.", int) / 10
        )  # int(input ('Noise Frequency in Hz (~0.2): 0.')) /10      # desired cutoff frequency of the filter, Hz ,      slightly higher than actual 1.2 Hz
        # Filter requirements.
        nyq = 0.5 * fs  # Nyquist Frequency
        order = 2  # sin wave can be approx represented as quadratic
        normal_cutoff = cutoff / nyq
        # generate dataframe to put noisefiltered data in
        datalist = []
        RetTimelist = []
        # scan through each TIC
        for Nr in pd.unique(MSdf_operate["ScanRange"]):
            # rearange data to work on
            Range = MSdf_operate.loc[MSdf_operate["ScanRange"] == Nr].copy(deep=True)
            pivRng1 = Range.pivot_table(
                index="Mass", columns="RetentionTime", values="Intensity", aggfunc="sum"
            )
            pivRng1.fillna(0, inplace=True)
            df = pivRng1.transpose()
            # make Chromatogram plot to compare later
            makegraph(df.sum(axis=1).index, df.sum(axis=1))
            # operate the noise filter on each single mass
            Counter = 0
            for mass in df.columns:
                data = df[mass]
                # if (Counter % 10 ==0):
                #    print(str(int(Counter/len(df.columns)*100)) + ' %')
                Counter += 1
                # Filter requirements.
                nyq = 0.5 * fs  # Nyquist Frequency
                order = 2  # sin wave can be approx represented as quadratic
                normal_cutoff = cutoff / nyq
                # Get the filter coefficients
                b, a = butter(order, normal_cutoff, btype="low", analog=False)
                # filter the noise of mass scan
                df[mass] = filtfilt(b, a, data).astype(int)
                # rearrange noise filtered data
                dfnew = df[mass].reset_index()
                dfnew.columns = ["RetentionTime", "Intensity"]
                dfnew.insert(loc=1, column="Mass", value=mass)
                dfnew.insert(
                    loc=0,
                    column="Scannumber",
                    value=range(1, len(dfnew["RetentionTime"]) + 1),
                )
                dfnew.insert(loc=0, column="ScanRange", value=Nr)
                # collect all noise filtered mass data in new dataframes
                # to save memory Retention Time is put into extra list as float numbers, while other list contains integers
                rettime = dfnew.pop("RetentionTime")
                RetTimelist.extend(rettime.values.tolist())
                datalist.extend(dfnew.values.tolist())
            # make graph of noisefiltered Chromatogram to compare with unfiltered data
            makegraph(df.sum(axis=1).index, df.sum(axis=1))
        # rearrange datalist so it fits into other programs
        df = pd.DataFrame(
            datalist, columns=["ScanRange", "Scannumber", "Mass", "Intensity"]
        )
        # reinsert Retention times into dataframe
        df.insert(loc=2, column="RetentionTime", value=pd.DataFrame(RetTimelist))
        # sort dataframe so it matches input MSdf
        df = df.sort_values(by=["Scannumber", "ScanRange"])
        df = df[df["Intensity"] > 0]
        filter_ok = "y"  # input('Filter factor level ok ? (y/n): ')
    return df


def getsinglemassspectra(MSdf):
    """
    Extract and analyze a single mass spectrum from a specific retention time in mass spectrometry data.

    This function allows the user to:
    - Select a scan range (ScanRange) from the data
    - Identify the retention time of the maximum intensity peak
    - Extract a mass spectrum by integrating data points around the peak
    - Subtract a baseline (from a user-defined noise-free region)
    - Normalize and save the resulting mass spectrum as a .xy file

    The function provides interactive feedback through plots and input validation.

    Args:
        MSdf (pd.DataFrame): The mass spectrometry data DataFrame containing at least:
            - "ScanRange": The scan range identifier
            - "Scannumber": The scan number
            - "RetentionTime": The retention time (float)
            - "Mass": The m/z value (float)
            - "Intensity": The signal intensity (float)

    Returns:
        None: This function does not return any value.

    Raises:
        ValueError: If required columns are missing or if user input is invalid.
        KeyError: If the selected scan range or retention time is not found.
        OSError: If the output file cannot be saved (e.g., due to permission or path issues).
        Exception: For any unexpected errors during processing.

    Notes:
        - The function uses `getvalue()`, `getlimits()`, and `getyninput()` for interactive input (assumes they're defined elsewhere).
        - The user selects a scan range and then identifies the peak retention time.
        - A baseline is calculated from a user-defined noise-free region.
        - The mass spectrum is generated by summing data points within a specified integration range around the peak.
        - The baseline is subtracted from the raw intensities.
        - The final spectrum is normalized using `normalizeIntensity()` and filtered for intensities ≥ 10.
        - The output is saved as a .xy file with the format:
          `ScanRange_X_Massspectrum at RetTime YZ min.xy`
        - A plot of the mass spectrum is also saved as a PNG file.
        - The function displays comparison plots of the chromatogram and the selected region.

    Example:
        >>> getsinglemassspectra(MSdf)
        # Creates files like:
        # ScanRange_1_Massspectrum at RetTime 1234 min.xy
        # ScanRange_1_Mass spectrum at RetTime 1234 min.png
    """
    scannumberlist = pd.unique(MSdf["ScanRange"])
    if len(scannumberlist) >= 1:
        scannumberlist = [int(x) for x in scannumberlist]
        print("'b' for back to main menu")
        scannumber = input(
            "On which scan do you want to operate? (" + str(scannumberlist) + ") :"
        )
        scannumber=int(scannumber)
        while scannumber not in scannumberlist:
            if scannumber == "b":
                main(MSdf)
            else:
                print("wrong input")
                print("'b' for back to main menu")
                scannumber = input(
                    "On which scan do you want to operate? ("
                    + str(scannumberlist)
                    + ") :"
                )
    else:
        scannumber = 1
    Range = MSdf[MSdf["ScanRange"] == scannumber].drop("ScanRange", axis=1)
    MSChromatogramm1 = Range.groupby(["RetentionTime"], sort=False).sum("Intensity")
    # generating Scannumber for operation
    MSChromatogramm1["Scannumber"] = range(len(MSChromatogramm1))
    MSChromatogramm1.reset_index(inplace=True)
    # starting loop to limit data range to relevant region
    range_ok = "n"
    while range_ok != "y":
        # making graph so operator sees what data he is working on
        makegraph(MSChromatogramm1['RetentionTime'], MSChromatogramm1["Intensity"])
        # asking for left and right data limit
        xstartB, xendB = getlimits(
            MSdf,
            "Range you want to have a closer look (0 - "
            + str(MSChromatogramm1["RetentionTime"].max())
            + ") (min max): ",
        )
        MSChromatogrammcut = MSChromatogramm1[
            MSChromatogramm1["RetentionTime"] >= float(xstartB)
        ]
        MSChromatogrammcut = MSChromatogrammcut[
            MSChromatogrammcut["RetentionTime"] <= float(xendB)
        ]
        # show graph of data range maxima finding routine works on
        makegraph(MSChromatogrammcut["RetentionTime"], MSChromatogrammcut["Intensity"])
        # checking if range is correct
        range_ok = getyninput(MSdf, "Is the range of data points correct?(y/n): ")
    print('\n\nMaximum Intensity at '+ str(MSChromatogrammcut.loc[MSChromatogrammcut['Intensity']==MSChromatogrammcut['Intensity'].max()].iloc[0,0]) + ' min\n\n')
    
    Rettime = getvalue(
        MSdf,
        "What Retention time do you want the mass spectra extracted? (X.XX): ",
        float,
    )
    # cutting dataframe to limits
    MSChromatogrammcut = MSChromatogramm1[
        MSChromatogramm1["RetentionTime"] >= (Rettime - 0.1)
    ]
    MSChromatogrammcut = MSChromatogrammcut[
        MSChromatogrammcut["RetentionTime"] <= (Rettime + 0.1)
    ]
    Rettimedet = MSChromatogrammcut["RetentionTime"].iloc[
        int(len(MSChromatogrammcut) / 2)
    ]
    # just an empty list for operation
    massspeclist = []
    # asking for range of mass spectra summation
    integrationrange = getvalue(
        MSdf,
        "Datapoints to left and right of maximum at retention time of "
        + str(Rettimedet)
        + " min to integrate mass spectra: ",
        int,
    )
    zero_range_min, zero_range_max = getlimits(MSdf, "Range where no peak is visible that can be used as baseline (min max): ")
    zero_range=Range.loc[(Range['RetentionTime']>=zero_range_min)& (Range['RetentionTime']<=zero_range_max)].groupby(['Mass']).mean()
    zero_range= zero_range.reset_index().round(0).groupby(['Mass']).mean().reset_index()

    # integrationrange = int(integrationrange)
    # starting loop for collecting all single massspectra in integrationrange in positive and negative directionto sum up
    for item in range(-integrationrange, integrationrange, 1):
        # finding mass spectra in inputfile with scannumber equals to maxima(entry) + integrationrange
        singlemassspec = MSdf[MSdf["ScanRange"] == scannumber]
        Massspectrum_at_Rettime = MSdf[MSdf["RetentionTime"] == Rettimedet]
        entry = Massspectrum_at_Rettime["Scannumber"].iat[1]
        msscan = entry + item
        singlemassspec = singlemassspec[singlemassspec["Scannumber"] == msscan].drop(
            ["Scannumber", "ScanRange", "RetentionTime"], axis=1
        )  # .reset_index(drop = True)
        # round the masses to integer values
        singlemassspec["Mass"] = np.round(singlemassspec["Mass"], decimals=0)
        # putting mass spectrum to a list and adding to dataframe end
        massspeclist.append(singlemassspec)
        massspec = pd.concat(massspeclist, ignore_index=True)
    # sorting mass sigmnals by entries of mass integer values
    massspec.sort_values(by=["Mass"])
    # summing up the Intensity values in massspectra having the same integer mass value
    massspecsum = massspec.groupby(["Mass"]).sum().reset_index()
    mass_spec=[]
    for mass in massspecsum['Mass'].to_numpy().tolist():
        value = massspecsum.loc[massspecsum['Mass']==mass].iloc[0,1]
        base = zero_range.loc[zero_range['Mass']==mass].iloc[0,3]
        Intensity = value - base
        mass_spec.append([mass,Intensity])
        #massspecsum.loc[massspecsum['Mass']==mass, 'Corr']=massspecsum.loc[massspecsum['Mass']==mass]-zero_range.loc[zero_range['Mass']==mass]['Intensity']
    mass_spec =pd.DataFrame(mass_spec, columns=['Mass','Intensity'])
    mass_spec=mass_spec.loc[mass_spec['Intensity']>=0]
    # normalizing mass spectra
    massspecsum = normalizeIntensity(mass_spec)
    massspecsum = massspecsum.loc[massspecsum['Intensity']>=10]
    # exporting the mass spectra as .xy-file
    timestamp = (
        str(str(Rettimedet).split(".")[0]) + "_" + str(str(Rettimedet).split(".")[1])
    )
    massspecsum.set_index('Mass').to_csv(
        "ScanRange_"
        + str(scannumber)
        + "_Massspectrum at RetTime "
        + str(timestamp)
        + " min.xy",
        sep=" ",
    )
    # Chromatogramm plot
    massspecplot(
        massspecsum['Mass'],
        massspecsum["Intensity"],
        "Mass Spectrum at Retention Time of " + str(Rettimedet) + " min",
        "ScanRange_"
        + str(scannumber)
        + "_Mass spectrum at RetTime "
        + str(timestamp)
        + " min.png",
        0,
    )
    return


def maximamassscan(MSdf, TGdataframe, TG_list):
    """
    Extract and analyze mass-specific signal intensity profiles (SIR) from mass spectrometry data,
    detect local maxima, and optionally extract mass spectra at peak retention times.

    This function processes data by:
    - Grouping by scan range (ScanRange)
    - Creating signal intensity profiles (SIR) for user-specified m/z values
    - Allowing interactive selection of retention time ranges
    - Detecting local maxima using peak detection algorithms
    - Saving peak data and generating visualizations
    - Optionally extracting mass spectra at each detected maximum

    The function integrates temperature data (from TG_list and TGdataframe) to enhance context.

    Args:
        MSdf (pd.DataFrame): The mass spectrometry data DataFrame containing at least:
            - "ScanRange": The scan range identifier
            - "Scannumber": The scan number
            - "RetentionTime": The retention time (float)
            - "Mass": The m/z value (float)
            - "Intensity": The signal intensity (float)
        TGdataframe (pd.DataFrame): The TGA (Thermogravimetric Analysis) data DataFrame with:
            - "Time": Retention time (float)
            - "Sample_Temperature": Sample temperature (float)
            - Other TGA parameters
        TG_list (pd.DataFrame): The thermal program steps DataFrame with:
            - "Operation": Heating, cooling, holding, etc.
            - "Start_Temperature", "End_Temperature": Temperature values (float)
            - "Runtime": Cumulative time (float)

    Returns:
        None: This function does not return any value.

    Raises:
        ValueError: If required columns are missing or if user input is invalid.
        KeyError: If the selected scan range or retention time is not found.
        OSError: If the output file cannot be saved (e.g., due to permission or path issues).
        Exception: For any unexpected errors during processing.

    Notes:
        - The function uses `getlistofint()`, `getvalue()`, `getlimits()`, `getyninput()`, and `find_peaks()` 
          (assumes they're defined elsewhere).
        - The user selects specific m/z values to analyze.
        - For each m/z, the function:
            - Creates a signal intensity profile (SIR)
            - Allows interactive selection of retention time range
            - Detects local maxima using `find_peaks()` with user-defined noise threshold and peak width
            - Saves peak data and generates visualizations
            - Optionally extracts mass spectra at each peak using `getmassspectrabyRettime()`
        - The function integrates temperature data via `add_temp_column()` to provide thermal context.
        - Output files include:
            - SIR data: `SIR_Mass_X.xy`
            - Peak data: `Maxima in Scanrange X of mass Y.xy`
            - Plots: `Maxima in Scanrange X of mass Y.png`
        - The function supports interactive feedback through plots and input validation.

    Example:
        >>> maximamassscan(MSdf, TGdataframe, TG_list)
        # Creates files like:
        # SIR_Mass_18.xy
        # Maxima in Scanrange 1 of mass 18.xy
        # Maxima in Scanrange 1 of mass 18.png
    """
    for Nr in pd.unique(MSdf["ScanRange"]):
        Range = MSdf[MSdf["ScanRange"] == Nr].drop("ScanRange", axis=1)
        Range["Mass"] = Range["Mass"].round(0).astype(int)
        Range["RetentionTime"] = Range["RetentionTime"]
        pivRng1 = Range.pivot_table(
            index="Mass", columns="RetentionTime", values="Intensity", aggfunc="sum"
        )
        pivRng1.fillna(0, inplace=True)
        df = pivRng1.reset_index()
        pivRng1 = add_temp_column(pivRng1, TGdataframe, TG_list)
        massmin = Range["Mass"].min()
        massmax = Range["Mass"].max()
        masses = getlistofint(
            MSdf,
            "What Masses ("
            + str(massmin)
            + "-"
            + str(massmax)
            + ") do you want extracted? (seperated by spaces): ",
        )  # input('What Masses ('+str(massmin)+'-' + str(massmax) + ') do you want extracted? (seperated by spaces): ').split()
        if len(masses) != 0:
            figures = getyninput(
                MSdf,
                "Do you want later a graph of each mass with marked maxima?(y/n): ",
            )  # input('Do you want later a graph of each mass with marked maxima?(y/n): ')
        else:
            figures = "n"
        counter = 0
        for mass in masses:
            SIR = df.loc[df["Mass"] == int(mass)]
            SIR = SIR.transpose().reset_index()
            SIR.columns = ["Time", "Mass " + str(mass)]
            SIR = SIR.iloc[2:]
            SIR.to_csv("SIR_Mass_" + str(mass) + ".xy")
            range_ok = "n"
            while range_ok == "n":
                # making graph so operator sees what data he is working on
                if counter == 0:
                    makegraph(SIR['Time'], SIR["Mass " + str(mass)])
                # asking for left and right data limit
                if counter == 0:
                    xstart, xend = getlimits(
                        MSdf, "Retention time to start and end at (min max): "
                    )
                SIRcut = SIR.loc[(SIR['Time']>=float(xstart))&(SIR['Time']<=float(xend))]
                # make graph
                makegraph(SIRcut['Time'], SIRcut["Mass " + str(mass)])
                # checking if range is correct only first loop
                if counter == 0:
                    range_ok = getyninput(
                        MSdf, "Is the range of data correct?(y/n): "
                    )  # input('Is the range of data correct?(y/n): ')
                else:
                    range_ok = "y"
                if range_ok == "y":
                    counter += 1
                # starting loop of maxima detection
            maxima_ok = "n"
            while maxima_ok == "n":
                # noise level and peakwidth input
                noise = getvalue(
                    MSdf, "Noise level in % of Maximum (typically = 3): ", int
                )/100  # int(input('Noise level in % of Maximum (typically = 3): ')) 
                delta = getvalue(
                    MSdf, "Peakwidth in Number of Datapoints (typically = 6): ", int
                )  # int(input('Peakwidth in Number of Datapoints (typically = 6): '))
                # search for local maxima
                maxima_numbers1 = find_peaks(
                    SIRcut["Mass " + str(mass)],
                    height=(
                        SIRcut["Mass " + str(mass)].max() * noise,
                        SIRcut["Mass " + str(mass)].max(),
                    ),
                    distance=delta,
                )
                # rearrange maxima data found by find_peaks
                maximadict = maxima_numbers1[-1]
                maxarray = maximadict["peak_heights"]
                maxnum = pd.DataFrame(maxima_numbers1[0], columns=["Scan"])
                # numbers only as integers
                [int(num) for num in maxarray]
                maxdf = pd.DataFrame(maxarray)
                maxnum["Intensity"] = maxdf
                RSlist = list()
                # loop to find retention time of each maxima
                if len(maxnum["Intensity"]) != 0:
                    for entry in maxnum["Intensity"]:
                        RSlist.append(SIRcut.loc[SIRcut["Mass " + str(mass)] == entry])
                        RSdf = pd.concat(RSlist)
                    makegraphplusscattertofile(
                        SIRcut['Time'],
                        SIRcut["Mass " + str(mass)],
                        "Mass " + str(mass),
                        RSdf['Time'],
                        RSdf["Mass " + str(mass)],
                        "Mass " + str(mass),
                        "Time /min",
                        "Intensity /counts",
                        False,
                        False,
                        False,
                        False,
                        0,
                        1,
                    )
                    maxima_ok = getyninput(
                        MSdf, "Are the maxima found correct?(y/n): "
                    )  # input('Are the maxima found correct?(y/n): ')
                    if maxima_ok == "y":
                        RSdf.to_csv(
                            "Maxima in Scanrange "
                            + str(Nr)
                            + "of mass "
                            + str(mass)
                            + ".xy"
                        ,index=False)
                        if figures == "y":
                            makegraphplusscattertofile(
                                SIRcut['Time'],
                                SIRcut["Mass " + str(mass)],
                                "Mass " + str(mass),
                                RSdf['Time'],
                                RSdf["Mass " + str(mass)],
                                "Mass " + str(mass),
                                "Time /min",
                                "Intensity /counts",
                                SIRcut['Time'].min(),
                                SIRcut['Time'].max(),
                                SIRcut["Mass " + str(mass)].min(),
                                SIRcut["Mass " + str(mass)].max(),
                                "Maxima in Scanrange "
                                + str(Nr)
                                + "of mass "
                                + str(mass)
                                + ".png",
                                1,
                            )
                massatret = getyninput(
                    MSdf, "Do you need the mass spectra at each maxima? (y/n): "
                )  # input('Do you need the mass spectra at each maxima? (y/n): ')
                if massatret == "y":
                    getmassspectrabyRettime(RSdf, Range, Nr)
    return

def maximamassscan_oTG(MSdf):
    for Nr in pd.unique(MSdf['ScanRange']):
        Range = MSdf[MSdf['ScanRange'] == Nr].drop('ScanRange', axis = 1)
        Range['Mass'] = Range['Mass'].round(0).astype(int)
        Range['RetentionTime'] = Range['RetentionTime']
        pivRng1 = Range.pivot_table(index = 'Mass', columns = 'RetentionTime', values = 'Intensity', aggfunc='sum')
        pivRng1.fillna(0, inplace = True)
        df = pivRng1.reset_index()
        massmin = Range['Mass'].min()
        massmax = Range['Mass'].max()
        masses = getlistofint(MSdf, 'What Masses ('+str(massmin)+'-' + str(massmax) + ') do you want extracted? (seperated by spaces): ') #input('What Masses ('+str(massmin)+'-' + str(massmax) + ') do you want extracted? (seperated by spaces): ').split()
        if len(masses) != 0:
            figures = getyninput(MSdf, 'Do you want later a graph of each mass with marked maxima?(y/n): ') #input('Do you want later a graph of each mass with marked maxima?(y/n): ')
        else:
            figures = 'n'
        counter = 0
        for mass in masses:
            SIR = df[df['Mass'] == int(mass)]
            SIR = SIR.transpose().reset_index()
            SIR.columns = ['Time', 'Mass '+ str(mass)]
            SIR = SIR.iloc[2:]
            SIR.set_index('Time').to_csv('SIR_Mass_'+ str(mass)+ '.xy') 
            range_ok = 'n'             
            while range_ok == 'n':
                #making graph so operator sees what data he is working on
                if counter == 0:
                    makegraph(SIR['Time'], SIR['Mass '+ str(mass)])
                #asking for left and right data limit
                if counter == 0:
                    xstart, xend = getlimits(MSdf, 'Retention time to start and end at (min max): ') 
                SIRcut = SIR.loc[(SIR['Time']>=float(xstart))&(SIR['Time']<=float(xend))]
                SIRcut.loc[:,'Mass '+str(mass)]=SIRcut.loc[:,'Mass '+str(mass)]-SIRcut.loc[:,'Mass '+str(mass)].min()
                #make graph
                makegraph(SIRcut['Time'], SIRcut['Mass '+ str(mass)])
                #checking if range is correct only first loop
                if counter == 0:    
                    range_ok = getyninput(MSdf, 'Is the range of data correct?(y/n): ') #input('Is the range of data correct?(y/n): ')
                else:
                    range_ok = 'y'
                if range_ok == 'y':
                    counter += 1
                #starting loop of maxima detection
            maxima_ok = 'n'
            while maxima_ok == 'n':
                #noise level and peakwidth input
                noise = getvalue(MSdf,'Noise level in % of Maximum (typically = 3): ', int)/100 #int(input('Noise level in % of Maximum (typically = 3): ')) 
                delta = getvalue(MSdf, 'Peakwidth in Number of Datapoints (typically = 6): ', int) #int(input('Peakwidth in Number of Datapoints (typically = 6): '))     
                #search for local maxima
                maxima_numbers1 = find_peaks(SIRcut['Mass '+ str(mass)], height = (SIRcut['Mass '+ str(mass)].max()*noise, SIRcut['Mass '+ str(mass)].max()), distance = delta)   
                #rearrange maxima data found by find_peaks
                maximadict = maxima_numbers1[-1]
                maxarray = maximadict['peak_heights']
                maxnum = pd.DataFrame(maxima_numbers1[0], columns = ['Scan'])
                #numbers only as integers
                [int(num) for num in maxarray]
                maxdf = pd.DataFrame(maxarray)
                maxnum['Intensity']= maxdf
                RSlist = list()
                print(maxnum)
                #loop to find retention time of each maxima
                for entry in maxnum['Intensity']:
                    RSlist.append(SIRcut.loc[SIRcut['Mass '+ str(mass)] == entry])
                    RSdf = pd.concat(RSlist) 
                makegraphplusscattertofile(SIRcut.index, SIRcut['Mass '+ str(mass)], 'Mass '+ str(mass), RSdf.index, RSdf['Mass '+ str(mass)], 'Mass '+ str(mass), 'Time /min', 'Intensity /counts', False, False, False, False, 0, 1)
                maxima_ok = getyninput(MSdf, 'Are the maxima found correct?(y/n): ')#input('Are the maxima found correct?(y/n): ')
                if maxima_ok == 'y':
                    RSdf.set_index('Time').to_csv('Maxima in Scanrange ' + str(Nr) + 'of mass ' + str(mass)+ '.xy')
                    if figures == 'y':
                        makegraphplusscattertofile(SIRcut.index, SIRcut['Mass '+ str(mass)], 'Mass '+ str(mass), RSdf.index, RSdf['Mass '+ str(mass)], 'Mass '+ str(mass), 'Time /min', 'Intensity /counts', SIRcut.index.min(), SIRcut.index.max(), SIRcut['Mass '+ str(mass)].min(), SIRcut['Mass '+ str(mass)].max(), 'Maxima in Scanrange ' + str(Nr) + 'of mass ' + str(mass)+ '.png', 1)
        massatret = getyninput(MSdf, 'Do you need the mass spectra at each maxima? (y/n): ')#input('Do you need the mass spectra at each maxima? (y/n): ')
        if massatret == 'y':
            getmassspectrabyRettime(RSdf, Range, Nr) 
    return


def getmassspectrabyRettime(RSdf, Range, Nr):
    """
    Extract and analyze mass-specific signal intensity profiles (SIR) from mass spectrometry data,
    detect local maxima, and optionally extract mass spectra at peak retention times — **without TGA integration**.

    This function processes data by:
    - Grouping by scan range (ScanRange)
    - Creating signal intensity profiles (SIR) for user-specified m/z values
    - Allowing interactive selection of retention time ranges
    - Detecting local maxima using peak detection algorithms
    - Saving peak data and generating visualizations
    - Optionally extracting mass spectra at each detected maximum

    The function is designed for cases where TGA (Thermogravimetric Analysis) data is not available or not needed.

    Args:
        MSdf (pd.DataFrame): The mass spectrometry data DataFrame containing at least:
            - "ScanRange": The scan range identifier
            - "Scannumber": The scan number
            - "RetentionTime": The retention time (float)
            - "Mass": The m/z value (float)
            - "Intensity": The signal intensity (float)

    Returns:
        None: This function does not return any value.

    Raises:
        ValueError: If required columns are missing or if user input is invalid.
        KeyError: If the selected scan range or retention time is not found.
        OSError: If the output file cannot be saved (e.g., due to permission or path issues).
        Exception: For any unexpected errors during processing.

    Notes:
        - The function uses `getlistofint()`, `getvalue()`, `getlimits()`, `getyninput()`, and `find_peaks()`
          (assumes they're defined elsewhere).
        - The user selects specific m/z values to analyze.
        - For each m/z, the function:
            - Creates a signal intensity profile (SIR)
            - Allows interactive selection of retention time range
            - Detects local maxima using `find_peaks()` with user-defined noise threshold and peak width
            - Saves peak data and generates visualizations
            - Optionally extracts mass spectra at each peak using `getmassspectrabyRettime()`
        - The function performs baseline correction by subtracting the minimum intensity in the selected range.
        - Output files include:
            - SIR data: `SIR_Mass_X.xy`
            - Peak data: `Maxima in Scanrange X of mass Y.xy`
            - Plots: `Maxima in Scanrange X of mass Y.png`
        - The function supports interactive feedback through plots and input validation.
        - This version does **not** integrate temperature data (unlike `maximamassscan`).

    Example:
        >>> maximamassscan_oTG(MSdf)
        # Creates files like:
        # SIR_Mass_18.xy
        # Maxima in Scanrange 1 of mass 18.xy
        # Maxima in Scanrange 1 of mass 18.png
    """
    integrationrange = getvalue(
        MSdf,
        "Datapoints to left and right of each maxima to integrate mass spectra?: ",
        int,
    )  # int(input('Datapoints to left and right of each maxima to integrate mass spectra?: '))
    # for every entry in list of found maxima....
    for time in RSdf['Time']:
        # get scannumber of each maxima
        Scan = Range.loc[Range["RetentionTime"] == time]["Scannumber"].iloc[1]
        # starting loop for collecting all single massspectra in integrationrange in positive and negative directionto sum up
        massspectra = []
        # get massspectra left end right of range and make a list
        for item in range(-integrationrange, integrationrange, 1):
            entry = Scan + item
            massspec = Range.loc[Range["Scannumber"] == entry].drop(
                ["Scannumber", "RetentionTime"], axis=1
            )
            massspectra.extend(massspec.values.astype(int).tolist())
        # take list of collected mass spectra and group by mass and sum up intensity
        massspectra = (
            pd.DataFrame(massspectra, columns=["Mass", "Intensity"])
            .groupby(["Mass"])
            .sum()
        )
        # normalize intensity and limit to 999 for Massbank.eu
        massspectra["Intensity"] = (
            massspectra["Intensity"] / massspectra["Intensity"].max() * 999
        )         
        # get timestamp and create string        
        timestamp = str(str(time).split(".")[0]) + "_" + str(str(time).split(".")[1])       
        #makegraph and save
        massspecplot(massspectra.index, massspectra['Intensity'], 'Massspectrum at ' + str(time) + ' min', "Massspectrum in Scanrange" + str(Nr) + " at " + str(timestamp) + ".png", 0)
        # sort data so highes intensity is at beginning of list
        massspectra = (
            massspectra.sort_values(by=["Intensity"], ascending=False)
            .round(0)
            .astype(int)
        )
        # output of integrated massspectra
        massspectra.to_csv(
            "Massspectrum in Scanrange" + str(Nr) + " at " + str(timestamp) + ".xy",
            sep=" ",
        )

    return


def extractmassscan(MSdf):
    """
    Extract and save signal intensity profiles (SIR) for user-specified m/z values from mass spectrometry data.

    This function processes mass spectrometry data by:
    - Grouping by scan range (ScanRange)
    - Creating pivot tables of intensity vs. retention time for each m/z value
    - Allowing interactive selection of specific m/z values to extract
    - Saving each SIR as a .xy file
    - Optionally generating and saving high-resolution plots of the chromatograms

    The function provides interactive feedback through plots and input validation.

    Args:
        MSdf (pd.DataFrame): The mass spectrometry data DataFrame containing at least:
            - "ScanRange": The scan range identifier
            - "Mass": The m/z value (float)
            - "RetentionTime": The retention time (float)
            - "Intensity": The signal intensity (float)

    Returns:
        None: This function does not return any value.

    Raises:
        ValueError: If required columns are missing or if user input is invalid.
        KeyError: If the selected scan range or retention time is not found.
        OSError: If the output file cannot be saved (e.g., due to permission or path issues).
        Exception: For any unexpected errors during processing.

    Notes:
        - The function uses `getyninput()` and `getlimits()` for interactive input (assumes they're defined elsewhere).
        - The user selects specific m/z values to analyze.
        - For each m/z, the function:
            - Creates a signal intensity profile (SIR) by pivoting the data
            - Saves the SIR as a .xy file: `SIR_Mass_X.xy`
            - Allows interactive selection of retention time range
            - Generates and saves a high-resolution plot: `Chromatogram of Mass X.png`
        - The function rounds m/z values to integers and retention times to 5 decimal places.
        - The output files are saved in the current working directory.
        - The function supports interactive feedback through plots and input validation.
        - This function is ideal for quick exploration of specific m/z signals.

    Example:
        >>> extractmassscan(MSdf)
        # Creates files like:
        # SIR_Mass_18.xy
        # Chromatogram of Mass 18.png
    """    
    for Nr in pd.unique(MSdf["ScanRange"]):
        Range = MSdf[MSdf["ScanRange"] == Nr].drop("ScanRange", axis=1)
        Range["Mass"] = Range["Mass"].round(0).astype(int)
        Range["RetentionTime"] = Range["RetentionTime"].round(5)
        pivRng1 = Range.pivot_table(
            index="Mass", columns="RetentionTime", values="Intensity", aggfunc="sum"
        )
        pivRng1.fillna(0, inplace=True)
        df = pivRng1.reset_index()
        # get min and max mass in scanrange
        massmin = Range["Mass"].min()
        massmax = Range["Mass"].max()
        # ask for masses oyu want to look at
        masses = input(
            "What Masses ("
            + str(massmin)
            + "-"
            + str(massmax)
            + ") do you want extracted? (seperated by spaces): "
        ).split()
        # ask if png files are needed
        if len(masses) != 0:
            figures = getyninput(
                MSdf, "Do you want a graph of each mass?(y/n): "
            )  # input('Do you want a graph of each mass?(y/n): ')
        else:
            figures = "n"
        counter = 0
        for mass in masses:
            SIR = df[df["Mass"] == int(mass)]
            SIR = SIR.transpose().reset_index()
            SIR.columns = ["Time", "Mass " + str(mass)]
            SIR = SIR.iloc[2:].set_index("Time")
            SIR.to_csv("SIR_Mass_" + str(mass) + ".xy")
            if str(figures) == "y":
                range_ok = "n"
                while range_ok == "n":
                    # making graph so operator sees what data he is working on
                    if counter == 0:
                        makegraph(SIR.index, SIR["Mass " + str(mass)])
                    # asking for left and right data limit
                    if counter == 0:
                        xstart, xend = getlimits(
                            MSdf, "Retention time to start and end at (min max): "
                        )  # input('Retention time to start and end at (min max): ').split()
                    SIRcut = SIR.loc[float(xstart) : float(xend)]
                    # make graph
                    makegraph(SIRcut.index, SIRcut["Mass " + str(mass)])
                    # checking if range is correct only first loop
                    if counter == 0:
                        range_ok = getyninput(
                            MSdf, "Is the range of data correct?(y/n): "
                        )  # input('Is the range of data correct?(y/n): ')
                    else:
                        range_ok = "y"
                    if range_ok == "y":
                        counter += 1
                        makegraphtofile(
                            SIRcut.index,
                            SIRcut["Mass " + str(mass)],
                            "Mass " + str(mass),
                            "Time /min",
                            "Intensity /counts",
                            SIRcut.index.min(),
                            SIRcut.index.max(),
                            False,
                            False,
                            "Chromatogram of Mass " + str(mass) + ".png",
                            counter,
                        )
    return


def getchromandspectraB(MSdf):
    """
    Extract chromatograms and mass spectra from mass spectrometry data by:
    - Generating a total ion chromatogram (TIC) for each scan range
    - Identifying local maxima in the chromatogram
    - Extracting mass spectra at each maximum retention time
    - Saving chromatograms, maxima data, and mass spectra as files

    This function provides a complete workflow for analyzing scan range-specific data, including:
    - Chromatogram generation and export
    - Interactive peak detection with user-defined noise thresholds
    - Mass spectrum extraction around each peak
    - Visualization and file output

    Args:
        MSdf (pd.DataFrame): The mass spectrometry data DataFrame containing at least:
            - "ScanRange": The scan range identifier
            - "Scannumber": The scan number
            - "RetentionTime": The retention time (float)
            - "Mass": The m/z value (float)
            - "Intensity": The signal intensity (float)

    Returns:
        None: This function does not return any value.

    Raises:
        ValueError: If required columns are missing or if user input is invalid.
        KeyError: If the selected scan range or retention time is not found.
        OSError: If the output file cannot be saved (e.g., due to permission or path issues).
        Exception: For any unexpected errors during processing.

    Notes:
        - The function uses `getvalue()`, `getlimits()`, `getyninput()`, `argrelextrema()`, `normalizeIntensity()`, 
          `makegraphplusscattertofile()`, and `massspecplot()` (assumes they're defined elsewhere).
        - For each scan range:
            - Creates a total ion chromatogram (TIC) by summing intensities across all m/z values
            - Allows interactive selection of retention time range for peak detection
            - Uses `argrelextrema()` to find local maxima with user-defined noise threshold
            - Saves maxima data and generates visualizations
            - Extracts mass spectra by integrating data points around each peak
            - Normalizes and saves mass spectra as .xy files
        - Output files include:
            - Chromatogram: `Chromatogramm_ScanRange_X.xy`
            - Maxima data: `Maxima in Chromatogramm Scanrange X.xy`
            - Mass spectra: `ScanRange_X_Massspectrum at RetTime Y min.xy`
            - Plots: `Chromatogram of Scan Range X.png`, `ScanRange_X_Mass spectrum at RetTime Y min.png`
        - The function supports interactive feedback through plots and input validation.
        - This function is ideal for comprehensive analysis of scan range-specific data.

    Example:
        >>> getchromandspectraB(MSdf)
        # Creates files like:
        # Chromatogramm_ScanRange_1.xy
        # Maxima in Chromatogramm Scanrange 1.xy
        # ScanRange_1_Massspectrum at RetTime 123 min.xy
        # Chromatogram of Scan Range 1.png
    """    
    for Nr in pd.unique(MSdf["ScanRange"]):
        Range = MSdf[MSdf["ScanRange"] == Nr].drop("ScanRange", axis=1)
        MSChromatogramm1 = Range.groupby(["RetentionTime"], sort=False).sum()
        # removing Mass information
        MSChromatogramm1.pop("Mass")
        # generating Scannumber for operation
        MSChromatogramm1["Scannumber"] = range(len(MSChromatogramm1))
        MSChromatogramm1.reset_index(inplace=True)
        # starting loop to limit data range to relevant region
        # export chromatogramm to .xy file
        ChrExp1 = MSChromatogramm1.drop("Scannumber", axis=1).set_index("RetentionTime")
        ChrExp1.to_csv("Chromatogramm_ScanRange_" + str(Nr) + ".xy")
        range_ok = "n"
        while range_ok == "n":
            # making graph so operator sees what data he is working on
            makegraph(MSChromatogramm1.index, MSChromatogramm1["Intensity"])
            # asking for left and right data limit
            xstartB, xendB = getlimits(
                MSdf,
                "Datapoints to start and end at in ScanRange "
                + str(Nr)
                + " for determination of maxima (0 - "
                + str(len(MSChromatogramm1))
                + ")(min max): ",
            )  # input('Datapoints to start and end at for determination of maxima (0 - ' + str(len(MSChromatogramm1)) + ')(min max): ').split()
            xstartB = int(xstartB)
            xendB = int(xendB)
            # show graph of data range maxima finding routine works on
            makegraph(
                MSChromatogramm1["RetentionTime"].iloc[xstartB:xendB],
                MSChromatogramm1["Intensity"].iloc[xstartB:xendB],
            )
            # checking if range is correct
            range_ok = input("Is the range of data points correct?(y/n): ")
        # cutting dataframe to limits
        MSChromatogramm1x = MSChromatogramm1.iloc[xstartB:xendB]
        # starting loop to find maxima
        maxima_ok = "n"
        while maxima_ok == "n":
            # asking for noise level parameter given by operator
            noise = (
                getvalue(MSdf, "Noise level (typically = 3): ", int) * 1000
            )  # int(input('Noise level (typically = 3): ')) *1000
            # search for local maxima
            maxima_numbers1 = MSChromatogramm1x.iloc[
                (
                    argrelextrema(
                        MSChromatogramm1x.Intensity.values,
                        np.greater_equal,
                        order=noise,
                        mode="clip",
                    )
                )
            ]
            # plot Chromatogramm
            makegraphplusscattertofile(
                MSChromatogramm1x["RetentionTime"],
                MSChromatogramm1x["Intensity"],
                "Intensity",
                maxima_numbers1["RetentionTime"],
                maxima_numbers1["Intensity"],
                "Local Maxima",
                "Retention Time /min",
                "Intensity /counts",
                False,
                False,
                False,
                False,
                "Chromatogram of Scan Range_" + str(Nr) + ".png",
                1,
            )
            # output of found maxima data to file
            MaxExp = maxima_numbers1.loc[:, ("RetentionTime", "Intensity")].set_index(
                "RetentionTime"
            )
            MaxExp.to_csv("Maxima in Chromatogramm Scanrange " + str(Nr) + ".xy")
            maxima_ok = getyninput(
                MSdf, "Are the maxima found correct?(y/n): "
            )  # input('Are the maxima found correct?(y/n): ')
        # starting loop to find mass spectra of maxima in chromatogramm
        for entry in maxima_numbers1["Scannumber"]:
            # just an empty list for operation
            massspeclist = []
            # looking up retentiontime of maximum
            RT = round(maxima_numbers1.at[entry, "RetentionTime"], 3)
            # asking for range of mass spectra summation
            integrationrange = getvalue(
                MSdf,
                "Datapoints to left and right of maximum at retention time of "
                + str(RT)
                + " min to integrate mass spectra: ",
                int,
            )  # int(input('Datapoints to left and right of maximum at retention time of ' + str(RT) + ' min to integrate mass spectra: '))
            # starting loop for collecting all single massspectra in integrationrange in positive and negative directionto sum up
            for item in range(-integrationrange, integrationrange, 1):
                msscan = entry + item
                # finding mass spectra in inputfile with scannumber equals to maxima(entry) + integrationrange
                singlemassspec = MSdf[MSdf["ScanRange"] == Nr]
                singlemassspec = singlemassspec[
                    singlemassspec["Scannumber"] == msscan
                ].drop(
                    ["Scannumber", "ScanRange", "RetentionTime"], axis=1
                )  # .reset_index(drop = True)
                # round the masses to integer values
                singlemassspec["Mass"] = np.round(singlemassspec["Mass"], decimals=0)
                # putting mass spectrum to a list and adding to dataframe end
                massspeclist.append(singlemassspec)
                massspec = pd.concat(massspeclist, ignore_index=True)
            # sorting mass sigmnals by entries of mass integer values
            massspec.sort_values(by=["Mass"])
            # summing up the Intensity values in massspectra having the same integer mass value
            massspecsum = massspec.groupby(["Mass"]).sum()
            # normalizing mass spectra
            massspecsum = normalizeIntensity(massspecsum)
            # exporting the mass spectra as .xy-file
            massspecsum.to_csv(
                "ScanRange_"
                + str(Nr)
                + "_Massspectrum at RetTime "
                + str(int(RT))
                + " min.xy",
                sep=" ",
            )
            # Chromatogramm plot
            massspecplot(
                massspecsum.index,
                massspecsum["Intensity"],
                "Mass Spectrum at Retention Time of " + str(RT) + " min",
                "ScanRange_"
                + str(Nr)
                + "_Mass spectrum at RetTime "
                + str(int(RT))
                + " min.png",
                0,
            )
    return


def getchromandspectraA(MSdf):
    """
    Extract and analyze chromatograms with background correction and mass spectra from mass spectrometry data.

    This function provides a complete workflow for processing scan range-specific data by:
    - Generating a total ion chromatogram (TIC) with linear background subtraction
    - Identifying local maxima in the corrected chromatogram
    - Extracting mass spectra at each maximum retention time
    - Saving chromatograms, maxima data, and mass spectra as files

    The function includes advanced features such as:
    - Automatic baseline correction using local minima detection
    - Interactive peak detection with user-defined noise thresholds and peak width
    - Mass spectrum extraction with integration around each peak
    - Visualization and file output

    Args:
        MSdf (pd.DataFrame): The mass spectrometry data DataFrame containing at least:
            - "ScanRange": The scan range identifier
            - "Scannumber": The scan number
            - "RetentionTime": The retention time (float)
            - "Mass": The m/z value (float)
            - "Intensity": The signal intensity (float)

    Returns:
        None: This function does not return any value.

    Raises:
        ValueError: If required columns are missing or if user input is invalid.
        KeyError: If the selected scan range or retention time is not found.
        OSError: If the output file cannot be saved (e.g., due to permission or path issues).
        Exception: For any unexpected errors during processing.

    Notes:
        - The function uses `getvalue()`, `getlimits()`, `getyninput()`, `argrelmin()`, `find_peaks()`, 
          `normalizeIntensity()`, `makegraphplusscattertofile()`, and `massspecplot()` (assumes they're defined elsewhere).
        - For each scan range:
            - Creates a total ion chromatogram (TIC) by summing intensities across all m/z values
            - Detects local minima to establish a baseline
            - Performs linear background subtraction using `np.polyfit`
            - Uses `find_peaks()` to identify local maxima with user-defined noise threshold and peak width
            - Saves maxima data and generates visualizations
            - Extracts mass spectra by integrating data points around each peak
            - Normalizes and saves mass spectra as .xy files
        - Output files include:
            - Chromatogram: `Chromatogramm_ScanRange_X.xy`
            - Maxima data: `Maxima in Chromatogramm Scanrange X.xy`
            - Mass spectra: `ScanRange_X_maxima_Y_Mass spectrum at RetTime Z min.xy`
            - Plots: `Chromatogram of Scan Range X.png`, `ScanRange_X_Mass spectrum at RetTime Z min.png`
        - The function supports interactive feedback through plots and input validation.
        - This function is ideal for comprehensive analysis of scan range-specific data with background correction.

    Example:
        >>> getchromandspectraA(MSdf)
        # Creates files like:
        # Chromatogramm_ScanRange_1.xy
        # Maxima in Chromatogramm Scanrange 1.xy
        # ScanRange_1_maxima_1_Mass spectrum at RetTime 1234 min.xy
        # Chromatogram of Scan Range 1.png
    """    
    for Nr in pd.unique(MSdf["ScanRange"]):
        Range = MSdf[MSdf["ScanRange"] == Nr].drop("ScanRange", axis=1)
        MSChromatogramm1 = Range.groupby(["RetentionTime"], sort=False).sum()
        # removing Mass information
        MSChromatogramm1.pop("Mass")
        # generating Scannumber for operation
        MSChromatogramm1["Scannumber"] = range(len(MSChromatogramm1))
        MSChromatogramm1.reset_index(inplace=True)
        range_ok = "n"
        while range_ok == "n":
            # making graph so operator sees what data he is working on
            makegraph(MSChromatogramm1.index, MSChromatogramm1["Intensity"])
            # asking for left and right data limit
            xstartA, xendA = getlimits(
                MSdf,
                "Datapoints to start and end at in ScanRange "
                + str(Nr)
                + " for determination of maxima (0 - "
                + str(len(MSChromatogramm1))
                + ")(min max): ",
            )  # input('Datapoints to start and end at for determination of maxima (0 - ' + str(len(MSChromatogramm1)) + ')(min max): ').split()
            xstartA = int(xstartA)
            xendA = int(xendA)
            # show graph of data range maxima finding routine works on
            makegraph(
                MSChromatogramm1["RetentionTime"].iloc[xstartA:xendA],
                MSChromatogramm1["Intensity"].iloc[xstartA:xendA],
            )
            # checking if range is correct
            range_ok = getyninput(
                MSdf, "Is the range of data points correct?(y/n): "
            )  # input('Is the range of data points correct?(y/n): ')

        # cutting dataframe to limits
        MSChromatogramm1 = MSChromatogramm1.iloc[xstartA:xendA]
        # finding minima for linear background subtraction
        minima = MSChromatogramm1.iloc[
            (argrelmin(MSChromatogramm1["Intensity"].to_numpy(), order=10, mode="clip"))
        ]
        # linear fit of minima
        linear_model = np.poly1d(
            np.polyfit(minima["RetentionTime"], minima["Intensity"], 1)
        )
        # background subtraction loop
        for entry in MSChromatogramm1["RetentionTime"]:
            MSChromatogramm1["IntensityBLC"] = MSChromatogramm1[
                "Intensity"
            ] - linear_model(MSChromatogramm1["RetentionTime"])
        # plot subtracted data
        makegraph(MSChromatogramm1["RetentionTime"], MSChromatogramm1["IntensityBLC"])
        # starting loop of maxima detection
        maxima_ok = "n"
        while maxima_ok == "n":
            # noise level and peakwidth input
            noise = (
                getvalue(MSdf, "Noise level in % of Maximum (typically = 3): ", int)
                / 100
            )  # float(input('Noise level in % of Maximum (typically = 3): ')) /100
            delta = getvalue(
                MSdf, "Peakwidth in Number of Datapoints (typically = 6): ", int
            )  # int(input('Peakwidth in Number of Datapoints (typically = 6): '))
            # search for local maxima
            maxima_numbers1 = find_peaks(
                MSChromatogramm1["IntensityBLC"].to_numpy(),
                height=(
                    MSChromatogramm1["IntensityBLC"].max() * noise,
                    MSChromatogramm1["IntensityBLC"].max(),
                ),
                distance=delta,
            )
            maximadict = maxima_numbers1[-1]
            maxarray = maximadict["peak_heights"]
            maxnum = pd.DataFrame(maxima_numbers1[0], columns=["Scan"])
            [int(num) for num in maxarray]
            maxdf = pd.DataFrame(maxarray)
            maxnum["Intensity"] = maxdf
            RSlist = list()
            # loop to find retention time of each maxima
            for entry in maxnum["Scan"]:
                RSlist.append(
                    MSChromatogramm1["RetentionTime"].loc[
                        MSChromatogramm1["Scannumber"] == entry + xstartA
                    ]
                )
                RSdf = pd.concat(RSlist)
            maxdf = MSChromatogramm1.merge(
                RSdf,
                left_on="Scannumber",
                right_index=True,
                how="right",
                suffixes=["", "maximum"],
            )
            # start Chromatogramm plot
            makegraphplusscattertofile(
                MSChromatogramm1["RetentionTime"],
                MSChromatogramm1["Intensity"],
                "Intensity",
                maxdf["RetentionTime"],
                maxdf["Intensity"],
                "Local Maxima",
                "Retention Time /min",
                "Intensity /counts",
                MSChromatogramm1["RetentionTime"].min(),
                MSChromatogramm1["RetentionTime"].max(),
                MSChromatogramm1["Intensity"].min(),
                MSChromatogramm1["Intensity"].max(),
                0,
                1,
            )
            maxima_ok = getyninput(
                MSdf, "Are the maxima found correct?(y/n): "
            )  # input('Are the maxima found correct?(y/n): ')

        # start Chromatogramm plot
        makegraphplusscattertofile(
            MSChromatogramm1["RetentionTime"],
            MSChromatogramm1["Intensity"],
            "Intensity",
            maxdf["RetentionTime"],
            maxdf["Intensity"],
            "Local Maxima",
            "Retention Time /min",
            "Intensity /counts",
            MSChromatogramm1["RetentionTime"].min(),
            MSChromatogramm1["RetentionTime"].max(),
            MSChromatogramm1["Intensity"].min(),
            MSChromatogramm1["Intensity"].max(),
            "Chromatogram_of_Scan_Range_" + str(Nr) + ".png",
            0,
        )
        # output of found maxima data to file
        MaxExp = maxdf.loc[:, ("RetentionTime", "Intensity")].set_index("RetentionTime")
        MaxExp.to_csv("Maxima in Chromatogramm Scanrange 1.xy")
        # output of chromatogramm to file
        ChrExp = MSChromatogramm1.loc[:, ("RetentionTime", "Intensity")].set_index(
            "RetentionTime"
        )
        ChrExp.to_csv("Chromatogramm_ScanRange_" + str(Nr) + ".xy")
        # loop to get massspectra of each maxima in chromatogramm
        maxcount = 1
        for entry in maxdf["Scannumber"]:
            massspeclist = []
            RT = maxdf[maxdf["Scannumber"] == entry].iat[0, 0]
            timestamp = str(str(RT).split(".")[0]) + "_" + str(str(RT).split(".")[1])
            # asking for range of mass spectra summation
            if maxcount == 1:
                integrationrange = getvalue(
                    MSdf,
                    "Datapoints to left and right of each maximum to integrate mass spectra: ",
                    int,
                )  # int(input('Datapoints to left and right of each maximum to integrate mass spectra: '))
            for item in range(-integrationrange, integrationrange, 1):
                msscan = entry + item
                # finding mass spectra in inputfile with scannumber equals to maxima(entry) + integrationrange
                singlemassspec = MSdf[MSdf["ScanRange"] == Nr]
                singlemassspec = singlemassspec[
                    singlemassspec["Scannumber"] == msscan
                ].drop(
                    ["Scannumber", "ScanRange", "RetentionTime"], axis=1
                )  # .reset_index(drop = True)
                # round the masses to integer values
                singlemassspec["Mass"] = np.round(singlemassspec["Mass"], decimals=0)
                # putting mass spectrum to a list and adding to dataframe end
                massspeclist.append(singlemassspec)
                massspec = pd.concat(massspeclist, ignore_index=True)
            # sorting mass sigmnals by entries of mass integer values
            massspec.sort_values(by=["Mass"])
            # summing up the Intensity values in massspectra having the same integer mass value
            massspecsum = massspec.groupby(["Mass"]).sum()
            # normalizing mass spectra
            massspecsum = normalizeIntensity(massspecsum)
            massspecsum.to_csv(
                "ScanRange_"
                + str(Nr)
                + "_maxima_"
                + str(maxcount)
                + "_Mass spectrum at RetTime "
                + str(timestamp)
                + " min.xy",
                sep=" ",
            )
            # start plot
            massspecplot(
                massspecsum.index,
                massspecsum["Intensity"],
                "Mass Spectrum at Retention Time of " + str(RT) + " min",
                "ScanRange_"
                + str(Nr)
                + "_Mass spectrum at RetTime "
                + str(timestamp)
                + " min.png",
                0,
            )
            maxcount += 1
    return


def makingheatmapandfile(MSdf):
    """
    Generate and save heatmaps of mass spectrometry data with interactive filtering and export capabilities.

    This function creates heatmaps of intensity distributions across mass (m/z) and retention time for each scan range,
    with interactive user-defined limits and comprehensive file export.

    The workflow includes:
    - Data grouping by scan range (ScanRange)
    - Pivot table creation for intensity vs. mass and retention time
    - Interactive setting of visualization limits (retention time, mass, intensity)
    - Dynamic heatmap generation with customizable color scaling
    - Export of both raw data and visualizations

    Args:
        MSdf (pd.DataFrame): The mass spectrometry data DataFrame containing at least:
            - "ScanRange": The scan range identifier
            - "Mass": The m/z value (float)
            - "RetentionTime": The retention time (float)
            - "Intensity": The signal intensity (float)

    Returns:
        None: This function does not return any value.

    Raises:
        ValueError: If required columns are missing or if user input is invalid.
        KeyError: If the selected scan range or retention time is not found.
        OSError: If the output file cannot be saved (e.g., due to permission or path issues).
        Exception: For any unexpected errors during processing.

    Notes:
        - The function uses `getlimits()` and `getyninput()` for interactive input (assumes they're defined elsewhere).
        - For each scan range:
            - Creates a pivot table of intensity vs. mass and retention time
            - Allows interactive selection of retention time, mass, and intensity limits
            - Generates a heatmap using seaborn with customizable color scaling
            - Exports the filtered data as a CSV file: `heatmaptable_Scanrange_X.csv`
            - Saves the heatmap as a high-resolution PNG file: `heatmap_Scanrange_X.png`
        - The heatmap uses a "Greens" colormap with user-defined intensity range (vmin, vmax)
        - Tick labels are automatically adjusted based on the mass range
        - The function supports interactive feedback through plots and input validation.
        - This function is ideal for visualizing complex MS data patterns and exporting results.

    Example:
        >>> makingheatmapandfile(MSdf)
        # Creates files like:
        # heatmaptable_Scanrange_1.csv
        # heatmap_Scanrange_1.png
    """
    # splitting data into different Mass scan ranges
    for Nr in pd.unique(MSdf["ScanRange"]):
        Range = MSdf[MSdf["ScanRange"] == Nr].drop("ScanRange", axis=1)
        # reduce data depth
        Range["Mass"] = Range["Mass"].round(0).astype(int)
        Range["RetentionTime"] = Range["RetentionTime"].round(2)
        pivRng1 = Range.pivot_table(
            index="Mass", columns="RetentionTime", values="Intensity", aggfunc="sum"
        )
        pivRng1.fillna(0, inplace=True)
        # show Heatmap
        plt.figure(figsize=(16, 9))
        sns.heatmap(pivRng1, cmap="Greens", vmin=5000, vmax=50000, fmt="d")
        plt.show(block=False)
        pause(0.1)
        range_ok = "n"
        while range_ok == "n":
            # ask for limitations of graph
            xmin, xmax = getlimits(
                MSdf, "Retention time minimum and maximum (min max): "
            )  # input('Retention time minimum and maximum (min max): ').split()
            ymin, ymax = getlimits(
                MSdf, "Mass range from minimum to maximum (min max): "
            )  # input('Mass range from minimum to maximum (min max): ').split()
            Imin, Imax = getlimits(
                MSdf, "Intensity minimum and maximum value (min max): "
            )  # input('Intensity minimum and maximum value (min max): ').split()
            # cut the data to limitation
            wRange = Range[Range["Mass"] <= float(ymax)]
            wRange = wRange[wRange["Mass"] >= float(ymin)]
            wRange = wRange[wRange["RetentionTime"] <= float(xmax)]
            wRange = wRange[wRange["RetentionTime"] >= float(xmin)]
            # arrange data
            pivRng = wRange.pivot_table(
                index="Mass", columns="RetentionTime", values="Intensity", aggfunc="sum"
            )
            pivRng.fillna(0, inplace=True)
            # export datatable
            tabexport1 = pivRng
            tabexport1.to_csv("heatmaptable_Scanrange_" + str(Nr) + ".csv")
            # make new graph with limitations
            plt.close()
            plt.figure(figsize=(16, 9))
            heatmap = sns.heatmap(
                pivRng,
                cmap="Greens",
                yticklabels=int((float(ymax) - float(ymin)) / 10),
                xticklabels=100,
                vmin=float(Imin),
                vmax=float(Imax),
                fmt="d",
                cbar_kws={"label": "Intensity /counts"},
            )
            plt.xlabel("Retention Time /min")
            plt.ylabel("Mass m/z")
            plt.show(block=False)
            pause(0.1)
            # save heatmap to png
            fig = heatmap.get_figure()
            fig.savefig("heatmap_Scanrange_" + str(Nr) + ".png", dpi=600)
            range_ok = getyninput(
                MSdf, "Graphic limits ok?(y/n): "
            )  # input('Graphic limits ok?(y/n): ')
    return

def main(MSdf):
    """
    Main interactive menu for processing GC-MS and online-MS data with multiple analysis options.

    This function provides a command-line interface for selecting and executing various mass spectrometry data analysis workflows.
    It supports different types of MS measurements (narrow peaks in GC-MS vs. broad peaks in online-MS) and offers a range of
    data processing capabilities including chromatogram analysis, peak detection, mass spectrum extraction, and visualization.

    Args:
        MSdf (pd.DataFrame): The mass spectrometry data DataFrame containing at least:
            - "ScanRange": The scan range identifier
            - "Scannumber": The scan number
            - "RetentionTime": The retention time (float)
            - "Mass": The m/z value (float)
            - "Intensity": The signal intensity (float)

    Returns:
        None: This function does not return any value.

    Raises:
        SystemExit: If the user selects option 8 (Exit), the program terminates.
        ValueError: If required functions or variables are not defined.
        Exception: For any unexpected errors during execution.

    Notes:
        - The function uses `getvalue()` and `getyninput()` for interactive input (assumes they're defined elsewhere).
        - The available options are:
            1. `getchromandspectraA`: Analyze GC-MS data with background correction
            2. `getchromandspectraB`: Analyze online-MS data with broad peaks
            3. `makingheatmapandfile`: Generate heatmaps and export data tables
            4. `extractmassscan`: Extract single mass scans from TIC
            5. `maximamassscan` or `maximamassscan_oTG`: Extract maxima from single mass scans
            6. `getsinglemassspectra`: Analyze chromatogram and extract mass spectra at specific retention times
            7. `massestolookat`: Identify masses with high signal-to-noise ratio
            8. Exit: Terminate the program
        - The function supports TGA integration when `TG == True` (using `TG_list` and `TGdataframe`).
        - The user can continue processing multiple files until they choose to exit.
        - This function serves as the central control hub for the entire data analysis pipeline.

    Example:
        >>> main(MSdf)
        # Interactive menu appears:
        # 1. Get Maxima, Chromatogramm and Massspectra of each peak in GC-MS measurement (narrow peaks) = 1
        # ...
        # What do you want to do with the data: 1
        # [Processing begins...]
    """
    valid = "n"
    while valid == "n":
        print(
            "Get Maxima, Chromatogramm and Massspectra of each peack in GC-MS measurement (narrow peaks) = 1"
        )
        print(
            "Get Maxima, Chromatogramm and  Masspectra of each peak in Online-MS measurement (braod peaks) = 2"
        )
        print("Create Heatmap and table of GC-MS run = 3")
        print("Extract Single Mass Scans from TIC = 4")
        print("Extract Maxima of Single Mass Scans from TIC in GC-MS = 5")
        print(
            "Have a Closer Look on Chromatogramm and Extract Single Mass Scans from TIC = 6"
        )
        print("Generate List with Masses having high Signal/Noise ratio = 7")
        print("Exit = 8")
        prgchoice = getvalue(
            MSdf, "What do you want to do with the data: ", int
        )  # input('What do you want to do with the data: ')
        if prgchoice == 1:
            getchromandspectraA(MSdf)
            valid = getyninput(
                MSdf, "done with the file? (y/n): "
            )  # input('done with the file? (y/n): ')
        elif prgchoice == 2:
            getchromandspectraB(MSdf)
            valid = getyninput(
                MSdf, "done with the file? (y/n): "
            )  # input('done with the file? (y/n): ')
        elif prgchoice == 3:
            makingheatmapandfile(MSdf)
            valid = getyninput(
                MSdf, "done with the file? (y/n): "
            )  # input('done with the file? (y/n): ')
        elif prgchoice == 4:
            extractmassscan(MSdf)
            valid = getyninput(
                MSdf, "done with the file? (y/n): "
            )  # input('done with the file? (y/n): ')
        elif prgchoice == 5:
            if TG == False:
                maximamassscan_oTG(MSdf)
            if TG == True:
                maximamassscan(MSdf, TG_list, TGdataframe)
            valid = getyninput(
                MSdf, "done with the file? (y/n): "
            )  # input('done with the file? (y/n): ')
        elif prgchoice == 6:
            getsinglemassspectra(MSdf)
            valid = getyninput(
                MSdf, "done with the file? (y/n): "
            )  # input('done with the file? (y/n): ')
        elif prgchoice == 7:
            massestolookat(MSdf)
            valid = getyninput(
                MSdf, "done with the file? (y/n): "
            )  # input('done with the file? (y/n): ')
        elif prgchoice == 8:
            sys.exit()
        else:
            print("number not in program list")
            valid == "n"



# function to remove emptylines in datafile
def nonblank_lines(f):
    """
    Generator that yields non-empty lines from a file-like object after stripping whitespace.

    This function reads a file line by line and returns only lines that are not empty (after removing leading/trailing whitespace).
    It's useful for processing text files where blank lines, comments, or whitespace-only lines should be ignored.

    Args:
        f (file-like object): A file-like object (e.g., returned by `open()`) that supports iteration.

    Yields:
        str: A non-empty line from the input file, with leading and trailing whitespace removed.

    Notes:
        - The function uses `rstrip()` to remove trailing whitespace (including newlines).
        - Empty lines (after stripping) are filtered out.
        - The function is memory-efficient as it processes one line at a time.
        - This is ideal for parsing log files, configuration files, or any text data where blank lines should be ignored.
        - The generator stops when the file is exhausted.

    Example:
        >>> with open("data.txt", "r") as file:
        ...     for line in nonblank_lines(file):
        ...         print(line)
        # Output: only non-empty lines from the file
    """
    for l in f:
        line = l.rstrip()
        if line:
            yield line

#Enter full path of data folder
path, selected = select_folder('Select the folder containing the **.TXT for TG-data and **_GCMS.TXT for MS data', '/')
if selected == False:
    print('No folder selected. Exiting program.')
    exit()
# Get the path to the data folder from the user
os.chdir(path)
# looking for .TXT-file in working directory
file = glob.glob("*_GCMS.TXT")
# prints out working directory path
print("Used File is: " + str(file[0]))
# defining some variables for running the process of datafile import
RetTime = 0
ISDT = 0
Scan = 0
MSlist = []
# starting file opening routine
with open(str(file[0])) as MSdata:
    for line in nonblank_lines(MSdata):
        if "FUNCTION" in line:
            Function = int(line.split(" ")[-1])
            RetTime = 0
            continue
        if "CycleTime" in line:
            Cycletime = float(line.split(" ")[-1])
            continue
        if "InterScanDelayTime" in line:
            ISDT = float(line.split(" ")[-1])
            continue
        if "StartRetentionTime" in line:
            count = 0
            continue
        if "EndRetentionTime" in line:
            count = 0
            continue
        if "NumberofScans" in line:
            count = 0
            continue
        if "AcquisitionDataType" in line:
            count = 0
            continue
        if "Scan" in line:
            Scan = float(line.split("\t\t")[-1])
            continue
        if "RetentionTime\t" in line:
            RetTime = float(line.split("\t")[-1])
            continue
        if "$" in line:
            continue
        else:
            row = line.strip()
            Mass = float(row.split("\t")[0])
            Intensity = float(row.split("\t")[-1])
            MSline = [Function, Scan, RetTime, Mass, Intensity]
            MSlist.append(MSline)

# Making dataframe out of imported data
MSdf = pd.DataFrame(
    MSlist, columns=["ScanRange", "Scannumber", "RetentionTime", "Mass", "Intensity"]
)
denoise = getyninput(
    MSdf, "Do you want to work on denoised data (Memory Consuming)? (y/n): "
)  # input('Do you want to work on denoised data (Memory Consuming)? (y/n): ')

if denoise == "y":
    MSdf = noisefilter(MSdf)
try:
    TG=True
    TGdataframe, TG_list, TimeTemplist = get_temperature_column(MSdf, file)
except:
    TG=False
    print('No TG data found!')

main(MSdf)
