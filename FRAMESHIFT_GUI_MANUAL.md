# FrameShift GUI User Manual

## 1. Introduction

Welcome to FrameShift GUI! This application provides an easy-to-use graphical interface for the FrameShift tool, which allows you to automatically change the aspect ratio of videos (e.g., from horizontal to vertical) while keeping important subjects in the frame.

With this GUI, you can access the powerful reframing features of FrameShift without needing to use the command line.

## 2. Requirements

**If running from source code:**

* **Python:** Version 3.8 or later.
* **FrameShift Dependencies:** All Python libraries listed in the `requirements.txt` file of the FrameShift project must be installed (e.g., `opencv-python`, `mediapipe`, `ultralytics`, `scenedetect`, `numpy`, `tqdm`). You can install them with:
    ```bash
    pip install -r requirements.txt
    ```
* **FFmpeg (Strongly Recommended):** Required to process and include the audio from the original video into the reframed output.
    * Download FFmpeg from [ffmpeg.org](https://ffmpeg.org/download.html).
    * Install it and ensure the directory containing the `ffmpeg` (or `ffmpeg.exe`) executable is added to your system's PATH.
    * If FFmpeg is not found, videos will be processed without audio, and a warning will appear in the logs.

**If using a standalone executable (if available):**

* The executable should include all necessary Python dependencies.
* You may still need to install FFmpeg separately and add it to your system's PATH for audio handling, unless specified otherwise in the executable's documentation.

## 3. How to Launch the GUI

**From source code:**

1.  Ensure you have installed all the requirements.
2.  Navigate to the main directory of the FrameShift project using a terminal or command prompt.
3.  Run the following command:
    ```bash
    python frameshift_gui.py
    ```

**From an executable (if provided):**

* Simply double-click the executable file (e.g., `FrameShiftGUI.exe` on Windows).

## 4. User Interface Description

The GUI is organized into several sections:

* **Input & Output:**
    * **Input Video/Folder:** Displays the path to the source video or folder (for batch processing). Click "Browse..." to select a file or folder. When a single video is loaded, its resolution and aspect ratio are displayed below.
    * **Output Video/Folder:** Displays the destination path for the processed video or folder. Click "Browse..." to select.
    * **Batch Process:** Check this box to process all videos in an input folder and save them to an output folder. The labels will change from "Video" to "Folder".

* **Main Settings:**
    * **Aspect Ratio:** The desired format for the final video (e.g., `9:16` for vertical videos, `1:1` for square, `16:9` for horizontal). You can also select "Custom..." to enter a value manually.
    * **Output Height (pixels):** A dropdown menu to select the final video's height in pixels (e.g., `1080`, `720`). The options update dynamically based on the chosen Aspect Ratio to suggest standard resolutions. The width will be calculated automatically.

* **Padding Settings:**
    * **Enable Padding:** If the original video, once cropped, does not completely fill the new format, this option adds sidebars instead of cropping the image further.
    * **Padding Type:** (Active only if "Enable Padding" is checked)
        * `black`: Adds black bars. **This is the default.**
        * `blur`: Adds bars created by blurring the edges of the original video.
        * `color`: Adds bars of a solid color of your choice.
    * **Blur Amount:** (Active for `blur` padding) Adjusts how much the bars are blurred (0=minimum, 10=maximum).
    * **Padding Color:** (Active for `color` padding) Enter a color name (e.g., `red`, `green`) or an RGB code (e.g., `(255,0,0)`). You can also click "Choose..." to select a color from a palette.

* **Visual Quality:**
    * **Interpolation:** The algorithm used to resize video frames. `lanczos` and `cubic` offer good quality, `area` is useful for downscaling, and `linear` is faster.
    * **Content Opacity:** Controls the transparency of the main video. If less than 1.0, the video is blended with a blurred background of the original frame. 1.0 is fully opaque.

* **Object Detection Weights:**
    * This section allows you to control the importance of different objects when determining the crop.
    * Use the dropdown menu and "Add" button to add new objects to the list.
    * Adjust the weight (importance) of each object using the slider (from 0.0 to 1.0). Higher weights give more priority.
    * `default` applies to any object not specified in the list.
    * Click the "X" button to remove an object from the list.

* **Actions & Status:**
    * **Save detailed log to file:** If checked, prompts you to save a verbose log file of the process.
    * **START/CANCEL PROCESS:** The main button to start the processing. It changes to a "CANCEL" button while running, allowing you to stop the process (this is most effective between videos in batch mode).
    * **Progress Bar:** Shows the processing progress.
    * **Status Label:** Displays real-time progress information for the current file.
    * **Log Area:** Displays informational messages, warnings, and errors during the process.

## 5. Quick Start Guide

1.  **Launch FrameShift GUI.**
2.  **Select Input:** Click "Browse..." in the Input section and choose your video file. If you want to process multiple videos, check "Batch Process" and select a folder. The video's resolution and aspect ratio will appear if you select a single file.
3.  **Select Output:** Click "Browse..." in the Output section and choose where to save the processed video (or the output folder for batch mode).
4.  **Configure Settings:**
    * Set the desired **Aspect Ratio** (e.g., `9:16` for TikTok/Instagram Stories).
    * Select a suitable **Output Height** from the dropdown menu.
    * If you want the video to fill the frame by cropping the edges, uncheck "Enable Padding". If you prefer to see the entire cropped content with added bars, check "Enable Padding" and choose the padding type.
    * Fine-tune the **Object Detection Weights** if you have specific subjects to follow.
5.  **Review Tooltips:** Hover your mouse over the various options to view helpful hints.
6.  **Start:** Click the **START PROCESS** button.
7.  **Wait:** Processing may take time, especially for long videos or in batch mode. You can monitor the progress in the progress bar and read messages in the log area.
8.  **Done!** Once completed, you will find the reframed video at the specified output path.

## 6. Simple Troubleshooting / FAQ

* **"FFmpeg not found" (Warning in log):**
    * This means FFmpeg is not installed or not in your system's PATH. The video will be processed without audio. To include audio, install FFmpeg and ensure it's accessible from the terminal.

* **Import errors on startup (e.g., "No module named 'ultralytics'"):**
    * This happens if you're running the GUI from source code and the FrameShift dependencies are not installed correctly. Follow the instructions in the "Requirements" section to install them (usually `pip install -r requirements.txt`).

* **The application seems frozen or unresponsive:**
    * Video processing, especially for long files or with complex analysis, can be time-consuming. The GUI runs the heavy lifting in a separate thread to remain as responsive as possible, but intensive operations might still give this impression. Check the log area for activity messages or errors. If the progress bar is moving (or active in indeterminate mode), it is likely still working.

* **YOLO models not found / Download errors:**
    * FrameShift uses AI models (YOLO) to detect faces and objects. These models are usually downloaded automatically the first time they are used. Make sure you have an active internet connection on the first run. Persistent issues may indicate network problems or issues with the `ultralytics` model cache.
