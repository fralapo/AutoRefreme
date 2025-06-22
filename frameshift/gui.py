import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import subprocess
import threading
import os
from pathlib import Path
import webbrowser

# This is a placeholder for the main script's functionality
# In a real scenario, you would import and call functions from your main script (frameshift.py)

def get_script_path():
    """Determines the path to the main frameshift script."""
    # Try to find frameshift.py in the same directory as gui.py or one level up
    gui_dir = Path(__file__).parent
    main_script_path = gui_dir / "main.py"
    if main_script_path.exists():
        return str(main_script_path)

    # Fallback if structure is different (e.g., running from project root)
    main_script_path_alt = Path("frameshift") / "main.py"
    if main_script_path_alt.exists():
        return str(main_script_path_alt)

    # If not found, assume it's in the PATH or can be called directly
    return "frameshift.py"


class FrameShiftGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("FrameShift - Video Reframing Tool")
        self.geometry("1200x900") # Adjusted for more content

        # Theme and styling
        self.style = ttk.Style(self)
        self.style.theme_use('clam') # A modern theme

        # Configure styles for a more modern look
        self.style.configure("TFrame", background="#2E3B4E")
        self.style.configure("TLabel", background="#2E3B4E", foreground="white", font=('Inter', 10))
        self.style.configure("Header.TLabel", font=('Inter', 16, 'bold'))
        self.style.configure("TButton", font=('Inter', 10), padding=6)
        self.style.configure("Switch.TCheckbutton", indicatoron=False, padding=5) # Custom style for toggle
        self.style.map("Switch.TCheckbutton",
            foreground=[('disabled', '#B0B0B0'), ('selected', 'white'), ('!selected', 'white')],
            background=[('disabled', '#D0D0D0'), ('selected', '#0078D7'), ('!selected', '#707070')])
        self.style.configure("TEntry", font=('Inter', 10), padding=5)
        self.style.configure("TCombobox", font=('Inter', 10), padding=5)
        self.style.configure("TRadiobutton", background="#2E3B4E", foreground="white", font=('Inter', 10))
        self.style.configure("Help.TButton", font=('Inter', 8, 'italic'), padding=2)


        # Variables to store settings
        self.input_files = tk.StringVar(value="") # Stores paths of selected video files
        self.input_folder = tk.StringVar(value="")
        self.is_batch_mode = tk.BooleanVar(value=False)
        self.output_path_var = tk.StringVar(value="") # For output file/folder

        self.target_ratio = tk.StringVar(value="9:16")
        self.custom_ratio_w = tk.StringVar(value="9")
        self.custom_ratio_h = tk.StringVar(value="16")
        self.output_height = tk.StringVar(value="1080")

        self.enable_padding = tk.BooleanVar(value=False)
        self.padding_type = tk.StringVar(value="black")
        self.blur_amount = tk.IntVar(value=5)
        self.padding_color_preset = tk.StringVar(value="black")
        self.custom_padding_color = tk.StringVar(value="#000000")

        self.face_weight = tk.DoubleVar(value=8.0) # Stored as 0-10 for UI
        self.person_weight = tk.DoubleVar(value=5.0)
        self.object_weight = tk.DoubleVar(value=3.0)
        self.custom_weights_dict = {} # To store custom label:weight

        self.interpolation_method = tk.StringVar(value="lanczos")
        self.content_opacity = tk.DoubleVar(value=1.0)
        self.object_weights_string_var = tk.StringVar(value="") # Auto-generated

        self.test_mode = tk.BooleanVar(value=False)
        self.log_file_path = tk.StringVar(value="")

        self.selected_videos_list = [] # To store FileDialog results

        self._create_widgets()
        self.update_command_preview() # Initial preview

    def _create_widgets(self):
        main_frame = ttk.Frame(self, padding="20 20 20 20")
        main_frame.pack(expand=True, fill=tk.BOTH)

        # Header
        header_label = ttk.Label(main_frame, text="FrameShift - Intelligent Video Reframing Tool", style="Header.TLabel")
        header_label.pack(pady=(0, 20))

        # PanedWindow for resizable columns
        paned_window = ttk.PanedWindow(main_frame, orient=tk.HORIZONTAL)
        paned_window.pack(expand=True, fill=tk.BOTH)

        # Create frames for each column
        frame_input = ttk.Labelframe(paned_window, text="Input Settings", padding="10")
        frame_processing = ttk.Labelframe(paned_window, text="Processing Settings", padding="10")
        frame_preview_process = ttk.Labelframe(paned_window, text="Preview & Process", padding="10")

        paned_window.add(frame_input, weight=1)
        paned_window.add(frame_processing, weight=1)
        paned_window.add(frame_preview_process, weight=1)

        # --- Input Settings Column ---
        self._create_input_settings_ui(frame_input)

        # --- Processing Settings Column ---
        self._create_processing_settings_ui(frame_processing)

        # --- Preview & Process Column ---
        self._create_preview_process_ui(frame_preview_process)

        # Footer / How it works
        how_it_works_frame = ttk.Labelframe(main_frame, text="How FrameShift Works", padding="10")
        how_it_works_frame.pack(fill=tk.X, pady=(10,0))
        how_it_works_text = """
FrameShift intelligently reframes videos using a stationary (fixed) crop per scene approach:
1. Scene Detection: Divides the video into scenes.
2. Content Analysis: Detects faces and other objects using YOLO models.
   (Models are downloaded automatically if not present).
3. Optimal Stationary Crop: Calculates a fixed crop window for each scene.
4. Output Generation: Creates the reframed video, with optional padding.
Inspired by Google AutoFlip. For more details, see README.md.
        """
        ttk.Label(how_it_works_frame, text=how_it_works_text, wraplength=1100, justify=tk.LEFT).pack(anchor="w")

        # Bindings for updates
        self.is_batch_mode.trace_add("write", self.toggle_batch_mode_ui)
        self.is_batch_mode.trace_add("write", lambda *_: self.update_command_preview())
        self.target_ratio.trace_add("write", lambda *_: self.update_command_preview())
        self.custom_ratio_w.trace_add("write", lambda *_: self.update_command_preview())
        self.custom_ratio_h.trace_add("write", lambda *_: self.update_command_preview())
        self.output_height.trace_add("write", lambda *_: self.update_command_preview())
        self.enable_padding.trace_add("write", self._toggle_padding_options_ui)
        self.enable_padding.trace_add("write", lambda *_: self.update_command_preview())
        self.padding_type.trace_add("write", self._toggle_padding_color_blur_ui)
        self.padding_type.trace_add("write", lambda *_: self.update_command_preview())
        self.blur_amount.trace_add("write", lambda *_: self.update_command_preview())
        self.padding_color_preset.trace_add("write", self._update_padding_color_from_preset)
        self.padding_color_preset.trace_add("write", lambda *_: self.update_command_preview())
        self.custom_padding_color.trace_add("write", lambda *_: self.update_command_preview())
        self.face_weight.trace_add("write", lambda *_: self._update_object_weights_string())
        self.person_weight.trace_add("write", lambda *_: self._update_object_weights_string())
        self.object_weight.trace_add("write", lambda *_: self._update_object_weights_string())
        self.interpolation_method.trace_add("write", lambda *_: self.update_command_preview())
        self.content_opacity.trace_add("write", lambda *_: self.update_command_preview())
        self.test_mode.trace_add("write", lambda *_: self.update_command_preview())
        self.log_file_path.trace_add("write", lambda *_: self.update_command_preview())

        self.toggle_batch_mode_ui() # Initial UI state for batch mode
        self._toggle_padding_options_ui() # Initial UI state for padding
        self._update_object_weights_string() # Initial weights string

    def _create_input_settings_ui(self, parent_frame):
        ttk.Label(parent_frame, text="Input Video(s)").grid(row=0, column=0, columnspan=3, sticky="w", pady=(0,5))

        # Batch Processing Toggle
        batch_frame = ttk.Frame(parent_frame)
        batch_frame.grid(row=1, column=0, columnspan=3, sticky="ew", pady=5)
        ttk.Label(batch_frame, text="Batch Processing:").pack(side=tk.LEFT, padx=(0,10))
        self.batch_toggle_cb = ttk.Checkbutton(batch_frame, variable=self.is_batch_mode, style="Switch.TCheckbutton", command=self.toggle_batch_mode_ui)
        self.batch_toggle_cb.pack(side=tk.LEFT)
        self.batch_mode_text_label = ttk.Label(batch_frame, text=" (Process all videos in input directory)")
        self.batch_mode_text_label.pack(side=tk.LEFT, padx=5)


        # File/Folder Selection
        self.input_file_frame = ttk.Frame(parent_frame)
        self.input_file_frame.grid(row=2, column=0, columnspan=3, sticky="ew", pady=5)
        ttk.Label(self.input_file_frame, text="Video Files:").pack(side=tk.LEFT, padx=(0,5))
        self.input_files_entry = ttk.Entry(self.input_file_frame, textvariable=self.input_files, width=30, state="readonly")
        self.input_files_entry.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5)
        ttk.Button(self.input_file_frame, text="Browse Files", command=self._browse_files).pack(side=tk.LEFT)

        self.input_folder_frame = ttk.Frame(parent_frame)
        # self.input_folder_frame.grid(row=2, column=0, columnspan=3, sticky="ew", pady=5) # Position updated by toggle_batch_mode_ui
        ttk.Label(self.input_folder_frame, text="Video Folder:").pack(side=tk.LEFT, padx=(0,5))
        self.input_folder_entry = ttk.Entry(self.input_folder_frame, textvariable=self.input_folder, width=30, state="readonly")
        self.input_folder_entry.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5)
        ttk.Button(self.input_folder_frame, text="Browse Folder", command=self._browse_folder).pack(side=tk.LEFT)

        # File list for multi-select
        self.file_list_label = ttk.Label(parent_frame, text="Selected Files:")
        self.file_list_label.grid(row=3, column=0, columnspan=3, sticky="w", pady=(5,0))
        self.file_listbox = tk.Listbox(parent_frame, height=5, selectmode=tk.SINGLE)
        self.file_listbox.grid(row=4, column=0, columnspan=3, sticky="ew", pady=(0,5))
        self.file_listbox.bind("<<ListboxSelect>>", self._on_file_select_for_preview) # Basic preview concept

        # Video Preview (Placeholder)
        preview_label = ttk.Label(parent_frame, text="Video Preview (Not Implemented)")
        preview_label.grid(row=5, column=0, columnspan=3, pady=5)
        preview_placeholder = tk.Frame(parent_frame, width=300, height=169, background="black") # 16:9 aspect ratio
        preview_placeholder.grid(row=6, column=0, columnspan=3, pady=5)
        ttk.Label(preview_placeholder, text="Preview Here", foreground="grey", background="black").pack(expand=True)

        # Video Info (Placeholder)
        self.video_duration_label = ttk.Label(parent_frame, text="Duration: --:--")
        self.video_duration_label.grid(row=7, column=0, columnspan=3, sticky="w", pady=2)
        self.video_resolution_label = ttk.Label(parent_frame, text="Resolution: -- x --")
        self.video_resolution_label.grid(row=8, column=0, columnspan=3, sticky="w", pady=2)


        # Advanced Settings (Input related)
        adv_input_frame = ttk.Labelframe(parent_frame, text="Advanced Input", padding="5")
        adv_input_frame.grid(row=9, column=0, columnspan=3, sticky="ew", pady=(10,0))

        ttk.Checkbutton(adv_input_frame, text="Test Mode (Run predefined scenarios)", variable=self.test_mode).grid(row=0, column=0, columnspan=2, sticky="w", pady=2)

        ttk.Label(adv_input_frame, text="Log File Path:").grid(row=1, column=0, sticky="w", pady=2)
        ttk.Entry(adv_input_frame, textvariable=self.log_file_path, width=30).grid(row=1, column=1, sticky="ew", pady=2, padx=5)
        ttk.Button(adv_input_frame, text="Browse", command=self._browse_log_file).grid(row=1, column=2, pady=2)

        parent_frame.columnconfigure(1, weight=1)


    def _create_processing_settings_ui(self, parent_frame):
        notebook = ttk.Notebook(parent_frame)
        notebook.pack(expand=True, fill=tk.BOTH, pady=5)

        tab_basic = ttk.Frame(notebook, padding="10")
        tab_advanced_proc = ttk.Frame(notebook, padding="10")

        notebook.add(tab_basic, text="Basic Settings")
        notebook.add(tab_advanced_proc, text="Advanced Processing")

        # --- Basic Settings Tab ---
        ttk.Label(tab_basic, text="Target Aspect Ratio:", font=('Inter', 11, 'bold')).grid(row=0, column=0, columnspan=3, sticky="w", pady=(0,10))

        ratio_frame = ttk.Frame(tab_basic)
        ratio_frame.grid(row=1, column=0, columnspan=3, sticky="w", pady=5)
        ttk.Radiobutton(ratio_frame, text="9:16 (Portrait)", variable=self.target_ratio, value="9:16").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(ratio_frame, text="1:1 (Square)", variable=self.target_ratio, value="1:1").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(ratio_frame, text="16:9 (Landscape)", variable=self.target_ratio, value="16:9").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(ratio_frame, text="Custom:", variable=self.target_ratio, value="custom_ratio", command=self._enable_custom_ratio).pack(side=tk.LEFT, padx=5)

        self.custom_ratio_w_entry = ttk.Entry(ratio_frame, textvariable=self.custom_ratio_w, width=5, state="disabled")
        self.custom_ratio_w_entry.pack(side=tk.LEFT, padx=(5,0))
        ttk.Label(ratio_frame, text=":").pack(side=tk.LEFT)
        self.custom_ratio_h_entry = ttk.Entry(ratio_frame, textvariable=self.custom_ratio_h, width=5, state="disabled")
        self.custom_ratio_h_entry.pack(side=tk.LEFT)
        self.apply_custom_ratio_btn = ttk.Button(ratio_frame, text="Apply Custom", command=self._apply_custom_ratio, state="disabled")
        self.apply_custom_ratio_btn.pack(side=tk.LEFT, padx=5)

        ttk.Label(tab_basic, text="Output Resolution (Height):", font=('Inter', 11, 'bold')).grid(row=2, column=0, sticky="w", pady=(10,5))
        ttk.Combobox(tab_basic, textvariable=self.output_height, values=["720", "1080", "1440", "2160"], state="readonly").grid(row=2, column=1, columnspan=2, sticky="ew", pady=5, padx=5)

        # Padding
        padding_main_frame = ttk.Labelframe(tab_basic, text="Padding Options", padding="5")
        padding_main_frame.grid(row=3, column=0, columnspan=3, sticky="nsew", pady=(10,0))

        ttk.Checkbutton(padding_main_frame, text="Enable Padding", variable=self.enable_padding, command=self._toggle_padding_options_ui).grid(row=0, column=0, columnspan=2, sticky="w", pady=5)

        self.padding_options_frame = ttk.Frame(padding_main_frame) # This frame will be toggled
        self.padding_options_frame.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=10)

        ttk.Label(self.padding_options_frame, text="Padding Type:").grid(row=0, column=0, sticky="w", pady=2)
        ttk.Combobox(self.padding_options_frame, textvariable=self.padding_type, values=["black", "blur", "color"], state="readonly").grid(row=0, column=1, sticky="ew", pady=2, padx=5)

        # Blur Amount (conditional)
        self.blur_amount_frame = ttk.Frame(self.padding_options_frame)
        self.blur_amount_frame.grid(row=1, column=0, columnspan=2, sticky="ew") # Toggled
        ttk.Label(self.blur_amount_frame, text="Blur Amount (0-10):").grid(row=0, column=0, sticky="w", pady=2)
        ttk.Scale(self.blur_amount_frame, variable=self.blur_amount, from_=0, to=10, orient=tk.HORIZONTAL, command=lambda val: self.blur_amount.set(int(float(val)))).grid(row=0, column=1, sticky="ew", pady=2, padx=5)
        self.blur_value_label = ttk.Label(self.blur_amount_frame, text="5") # Placeholder, update with trace
        self.blur_value_label.grid(row=0, column=2, padx=5)
        self.blur_amount.trace_add("write", lambda *args: self.blur_value_label.config(text=str(self.blur_amount.get())))


        # Color Picker (conditional)
        self.color_picker_frame = ttk.Frame(self.padding_options_frame)
        self.color_picker_frame.grid(row=2, column=0, columnspan=2, sticky="ew") # Toggled
        ttk.Label(self.color_picker_frame, text="Padding Color:").grid(row=0, column=0, sticky="w", pady=2)
        ttk.Combobox(self.color_picker_frame, textvariable=self.padding_color_preset,
                     values=["black", "white", "red", "green", "blue", "yellow", "cyan", "magenta", "custom"],
                     state="readonly").grid(row=0, column=1, sticky="ew", pady=2, padx=5)
        self.custom_color_btn = ttk.Button(self.color_picker_frame, text="Choose Custom", command=self._choose_custom_color, state="disabled")
        self.custom_color_btn.grid(row=0, column=2, pady=2, padx=5)
        self.custom_color_display = tk.Frame(self.color_picker_frame, width=20, height=20, bg=self.custom_padding_color.get())
        self.custom_color_display.grid(row=0, column=3, padx=5)
        self.custom_padding_color.trace_add("write", lambda *args: self.custom_color_display.config(bg=self.custom_padding_color.get()))

        # Object Weights
        weights_frame = ttk.Labelframe(tab_basic, text="Object Weights (0-10)", padding="5")
        weights_frame.grid(row=4, column=0, columnspan=3, sticky="nsew", pady=(10,0))

        ttk.Label(weights_frame, text="Faces:").grid(row=0, column=0, sticky="w", pady=2)
        ttk.Scale(weights_frame, variable=self.face_weight, from_=0, to=10, orient=tk.HORIZONTAL, command=lambda val: self.face_weight.set(round(float(val),1))).grid(row=0, column=1, sticky="ew", pady=2, padx=5)
        self.face_weight_label = ttk.Label(weights_frame, text=f"{self.face_weight.get():.1f}")
        self.face_weight_label.grid(row=0, column=2, padx=5)
        self.face_weight.trace_add("write", lambda *args: self.face_weight_label.config(text=f"{self.face_weight.get():.1f}"))

        ttk.Label(weights_frame, text="People:").grid(row=1, column=0, sticky="w", pady=2)
        ttk.Scale(weights_frame, variable=self.person_weight, from_=0, to=10, orient=tk.HORIZONTAL, command=lambda val: self.person_weight.set(round(float(val),1))).grid(row=1, column=1, sticky="ew", pady=2, padx=5)
        self.person_weight_label = ttk.Label(weights_frame, text=f"{self.person_weight.get():.1f}")
        self.person_weight_label.grid(row=1, column=2, padx=5)
        self.person_weight.trace_add("write", lambda *args: self.person_weight_label.config(text=f"{self.person_weight.get():.1f}"))


        ttk.Label(weights_frame, text="Other Objects (Default):").grid(row=2, column=0, sticky="w", pady=2)
        ttk.Scale(weights_frame, variable=self.object_weight, from_=0, to=10, orient=tk.HORIZONTAL, command=lambda val: self.object_weight.set(round(float(val),1))).grid(row=2, column=1, sticky="ew", pady=2, padx=5)
        self.object_weight_label = ttk.Label(weights_frame, text=f"{self.object_weight.get():.1f}")
        self.object_weight_label.grid(row=2, column=2, padx=5)
        self.object_weight.trace_add("write", lambda *args: self.object_weight_label.config(text=f"{self.object_weight.get():.1f}"))

        ttk.Button(weights_frame, text="Add Custom Object Weight", command=self._add_custom_weight_dialog).grid(row=3, column=0, columnspan=3, pady=(5,0))
        self.custom_weights_display = ttk.Label(weights_frame, text="Custom: None", wraplength=300)
        self.custom_weights_display.grid(row=4, column=0, columnspan=3, sticky="w", pady=2)

        # --- Advanced Processing Tab ---
        ttk.Label(tab_advanced_proc, text="Interpolation Method:", font=('Inter', 11, 'bold')).grid(row=0, column=0, sticky="w", pady=(0,5))
        ttk.Combobox(tab_advanced_proc, textvariable=self.interpolation_method,
                     values=["lanczos", "cubic", "area", "linear", "nearest"], state="readonly").grid(row=0, column=1, sticky="ew", pady=5, padx=5)

        ttk.Label(tab_advanced_proc, text="Content Opacity (0.0-1.0):", font=('Inter', 11, 'bold')).grid(row=1, column=0, sticky="w", pady=(10,5))
        ttk.Scale(tab_advanced_proc, variable=self.content_opacity, from_=0.0, to=1.0, orient=tk.HORIZONTAL, command=lambda val: self.content_opacity.set(round(float(val),1))).grid(row=1, column=1, sticky="ew", pady=5, padx=5)
        self.opacity_value_label = ttk.Label(tab_advanced_proc, text=f"{self.content_opacity.get():.1f}")
        self.opacity_value_label.grid(row=1, column=2, padx=5)
        self.content_opacity.trace_add("write", lambda *args: self.opacity_value_label.config(text=f"{self.content_opacity.get():.1f}"))

        ttk.Label(tab_advanced_proc, text="Object Weights String (Auto-generated):", font=('Inter', 11, 'bold')).grid(row=2, column=0, columnspan=3, sticky="w", pady=(10,5))
        ttk.Entry(tab_advanced_proc, textvariable=self.object_weights_string_var, state="readonly", width=50).grid(row=3, column=0, columnspan=3, sticky="ew", pady=5)

        tab_basic.columnconfigure(1, weight=1)
        weights_frame.columnconfigure(1, weight=1)
        tab_advanced_proc.columnconfigure(1, weight=1)


    def _create_preview_process_ui(self, parent_frame):
        ttk.Label(parent_frame, text="Output File/Folder:", font=('Inter', 11, 'bold')).grid(row=0, column=0, sticky="w", pady=(0,5))
        self.output_path_entry = ttk.Entry(parent_frame, textvariable=self.output_path_var, width=35, state="readonly")
        self.output_path_entry.grid(row=0, column=1, sticky="ew", pady=5, padx=5)
        ttk.Button(parent_frame, text="Browse Output", command=self._browse_output).grid(row=0, column=2, pady=5)

        # Output Preview (Placeholder)
        ttk.Label(parent_frame, text="Output Preview (Conceptual)", font=('Inter', 11, 'bold')).grid(row=1, column=0, columnspan=3, pady=(10,5))
        self.output_preview_frame = tk.Frame(parent_frame, width=300, height=169, background="darkgrey") # Placeholder
        self.output_preview_frame.grid(row=2, column=0, columnspan=3, pady=5)
        self.output_preview_text = ttk.Label(self.output_preview_frame, text="9:16", foreground="black", background="darkgrey", font=('Inter', 20, 'bold'))
        self.output_preview_text.pack(expand=True)
        self.output_dimensions_label = ttk.Label(parent_frame, text="Output: --- x --- px")
        self.output_dimensions_label.grid(row=3, column=0, columnspan=3, pady=2)
        self.target_ratio.trace_add("write", self._update_output_preview_display)
        self.custom_ratio_w.trace_add("write", self._update_output_preview_display)
        self.custom_ratio_h.trace_add("write", self._update_output_preview_display)
        self.output_height.trace_add("write", self._update_output_preview_display)
        self._update_output_preview_display()


        # Process Button
        self.process_button = ttk.Button(parent_frame, text="Process Video(s)", command=self.start_processing_thread, style="Accent.TButton")
        self.style.configure("Accent.TButton", font=('Inter', 12, 'bold'), background="#0078D7", foreground="white")
        self.style.map("Accent.TButton", background=[('active', '#005A9E')])
        self.process_button.grid(row=4, column=0, columnspan=3, sticky="ew", pady=(20,10), ipady=10)

        # Processing Status
        self.status_label = ttk.Label(parent_frame, text="Status: Idle")
        self.status_label.grid(row=5, column=0, columnspan=3, sticky="w", pady=5)
        self.progress_bar = ttk.Progressbar(parent_frame, orient=tk.HORIZONTAL, length=300, mode='determinate')
        self.progress_bar.grid(row=6, column=0, columnspan=3, sticky="ew", pady=5)

        # Command Preview
        cmd_preview_frame = ttk.Labelframe(parent_frame, text="Command Preview", padding="5")
        cmd_preview_frame.grid(row=7, column=0, columnspan=3, sticky="nsew", pady=(10,0))

        self.command_preview_text = tk.Text(cmd_preview_frame, height=6, wrap=tk.WORD, state="disabled", font=('Courier', 9), relief=tk.FLAT, background="#F0F0F0")
        self.command_preview_text.pack(expand=True, fill=tk.BOTH)

        parent_frame.columnconfigure(1, weight=1)

    # --- UI Helper and Event Handler Methods ---
    def _on_file_select_for_preview(self, event):
        # Basic idea: if a single file is selected, update some info
        # In a real app, this would load metadata or a thumbnail
        selection = self.file_listbox.curselection()
        if selection:
            selected_file_path = self.file_listbox.get(selection[0])
            # Simulate getting info - replace with actual video metadata logic
            try:
                # Placeholder for actual video info fetching
                # For now, just display the name
                filename = Path(selected_file_path).name
                self.video_duration_label.config(text=f"Duration: (Info for {filename})")
                self.video_resolution_label.config(text=f"Resolution: (Info for {filename})")
            except Exception as e:
                self.video_duration_label.config(text="Duration: Error")
                self.video_resolution_label.config(text="Resolution: Error")
        else:
            self.video_duration_label.config(text="Duration: --:--")
            self.video_resolution_label.config(text="Resolution: -- x --")


    def toggle_batch_mode_ui(self, *args):
        if self.is_batch_mode.get():
            self.input_file_frame.grid_remove()
            self.file_list_label.grid_remove()
            self.file_listbox.grid_remove()
            self.input_folder_frame.grid(row=2, column=0, columnspan=3, sticky="ew", pady=5)
            self.batch_mode_text_label.config(font=('Inter', 10, 'italic'))
            self.output_path_var.set("") # Clear output path as it should be a folder
        else:
            self.input_folder_frame.grid_remove()
            self.input_file_frame.grid(row=2, column=0, columnspan=3, sticky="ew", pady=5)
            self.file_list_label.grid(row=3, column=0, columnspan=3, sticky="w", pady=(5,0))
            self.file_listbox.grid(row=4, column=0, columnspan=3, sticky="ew", pady=(0,5))
            self.batch_mode_text_label.config(font=('Inter', 10))
            self.output_path_var.set("") # Clear output path as it should be a file
        self.update_command_preview()

    def _browse_files(self):
        files = filedialog.askopenfilenames(
            title="Select Video Files",
            filetypes=(("Video files", "*.mp4 *.mov *.avi *.mkv"), ("All files", "*.*"))
        )
        if files:
            self.selected_videos_list = list(files)
            self.input_files.set("; ".join(Path(f).name for f in files)) # Display names
            self.file_listbox.delete(0, tk.END)
            for f_path in self.selected_videos_list:
                self.file_listbox.insert(tk.END, f_path)
            if self.selected_videos_list: # Select first for preview info
                self.file_listbox.selection_set(0)
                self._on_file_select_for_preview(None)
            self.output_path_var.set("") # Suggest new output
        self.update_command_preview()

    def _browse_folder(self):
        folder = filedialog.askdirectory(title="Select Input Folder")
        if folder:
            self.input_folder.set(folder)
            self.output_path_var.set("") # Suggest new output
        self.update_command_preview()

    def _browse_output(self):
        if self.is_batch_mode.get() or len(self.selected_videos_list) > 1:
            path = filedialog.askdirectory(title="Select Output Folder")
        else: # Single file output
            # Try to suggest a filename based on input
            suggested_name = ""
            if self.selected_videos_list:
                p = Path(self.selected_videos_list[0])
                suggested_name = f"{p.stem}_reframed{p.suffix}"

            path = filedialog.asksaveasfilename(
                title="Save Output Video As",
                defaultextension=".mp4",
                initialfile=suggested_name,
                filetypes=(("MP4 files", "*.mp4"), ("MOV files", "*.mov"), ("All files", "*.*"))
            )
        if path:
            self.output_path_var.set(path)
        self.update_command_preview()

    def _browse_log_file(self):
        logfile = filedialog.asksaveasfilename(
            title="Save Log File As",
            defaultextension=".log",
            filetypes=(("Log files", "*.log"), ("Text files", "*.txt"), ("All files", "*.*"))
        )
        if logfile:
            self.log_file_path.set(logfile)
        self.update_command_preview()

    def _enable_custom_ratio(self, *args):
        if self.target_ratio.get() == "custom_ratio":
            self.custom_ratio_w_entry.config(state="normal")
            self.custom_ratio_h_entry.config(state="normal")
            self.apply_custom_ratio_btn.config(state="normal")
        else: # Other radio buttons disable custom
            self.custom_ratio_w_entry.config(state="disabled")
            self.custom_ratio_h_entry.config(state="disabled")
            self.apply_custom_ratio_btn.config(state="disabled")
        self.update_command_preview()

    def _apply_custom_ratio(self):
        # This effectively makes "custom_ratio" active by ensuring the values are used
        # The actual value of target_ratio string remains "custom_ratio"
        self.update_command_preview()

    def _toggle_padding_options_ui(self, *args):
        if self.enable_padding.get():
            # Show the frame containing padding type, blur, color options
            for child in self.padding_options_frame.winfo_children():
                child.grid() # Or pack() depending on how they were added
            self._toggle_padding_color_blur_ui() # Further refine based on padding type
        else:
            # Hide the frame
            for child in self.padding_options_frame.winfo_children():
                child.grid_remove() # Or pack_forget()
        self.update_command_preview()

    def _toggle_padding_color_blur_ui(self, *args):
        if not self.enable_padding.get():
            self.blur_amount_frame.grid_remove()
            self.color_picker_frame.grid_remove()
            return

        pad_type = self.padding_type.get()
        if pad_type == "blur":
            self.blur_amount_frame.grid()
            self.color_picker_frame.grid_remove()
        elif pad_type == "color":
            self.blur_amount_frame.grid_remove()
            self.color_picker_frame.grid()
            self._update_padding_color_from_preset() # Enable/disable custom button
        else: # black or other
            self.blur_amount_frame.grid_remove()
            self.color_picker_frame.grid_remove()
        self.update_command_preview()

    def _update_padding_color_from_preset(self, *args):
        preset = self.padding_color_preset.get()
        if preset == "custom":
            self.custom_color_btn.config(state="normal")
            # Don't change custom_padding_color here, let _choose_custom_color do it
        else:
            self.custom_color_btn.config(state="disabled")
            color_map = {
                'black': '#000000', 'white': '#FFFFFF', 'red': '#FF0000',
                'green': '#00FF00', 'blue': '#0000FF', 'yellow': '#FFFF00',
                'cyan': '#00FFFF', 'magenta': '#FF00FF'
            }
            self.custom_padding_color.set(color_map.get(preset, "#000000"))
        self.update_command_preview()

    def _choose_custom_color(self):
        from tkinter import colorchooser
        color_code = colorchooser.askcolor(title="Choose Padding Color", initialcolor=self.custom_padding_color.get())
        if color_code and color_code[1]: # Check if a color was chosen (color_code[1] is hex)
            self.custom_padding_color.set(color_code[1])
        self.update_command_preview()

    def _add_custom_weight_dialog(self):
        dialog = CustomWeightDialog(self)
        self.wait_window(dialog) # Wait for dialog to close
        if dialog.result:
            label, weight = dialog.result
            self.custom_weights_dict[label] = weight
            self._update_object_weights_string()
            self._update_custom_weights_display()

    def _update_custom_weights_display(self):
        if not self.custom_weights_dict:
            self.custom_weights_display.config(text="Custom: None")
        else:
            display_str = "Custom: " + ", ".join([f"{lbl}:{wgt:.1f}" for lbl, wgt in self.custom_weights_dict.items()])
            self.custom_weights_display.config(text=display_str)

    def _update_object_weights_string(self, *args):
        # Convert UI scale 0-10 to script's 0.0-1.0 for default face/person/object
        face_w = self.face_weight.get() / 10.0
        person_w = self.person_weight.get() / 10.0
        object_w = self.object_weight.get() / 10.0 # This is 'default' in script

        weights = []
        weights.append(f"face:{face_w:.1f}")
        weights.append(f"person:{person_w:.1f}")

        # Add custom weights (already 0-10, convert to 0.0-1.0 for string)
        for label, weight_ui in self.custom_weights_dict.items():
            weights.append(f"{label}:{float(weight_ui)/10.0:.1f}")

        weights.append(f"default:{object_w:.1f}") # Add default for other objects

        self.object_weights_string_var.set(",".join(weights))
        self.update_command_preview()

    def _update_output_preview_display(self, *args):
        try:
            ratio_str = self.target_ratio.get()
            if ratio_str == "custom_ratio":
                w_str, h_str = self.custom_ratio_w.get(), self.custom_ratio_h.get()
            else:
                w_str, h_str = ratio_str.split(':')

            w = int(w_str)
            h = int(h_str)

            target_h_res = int(self.output_height.get())
            target_w_res = int(round(target_h_res * (w / h)))

            self.output_preview_text.config(text=f"{w}:{h}")
            self.output_dimensions_label.config(text=f"Output: {target_w_res} x {target_h_res} px")

            # Basic preview scaling (conceptual)
            preview_box_w = self.output_preview_frame.winfo_width()
            preview_box_h = self.output_preview_frame.winfo_height()
            if preview_box_w == 1 or preview_box_h == 1: # Not yet drawn
                self.after(100, self._update_output_preview_display) # Retry after a bit
                return

            # Scale font size based on aspect ratio for better visual representation
            if w > h: # Landscape
                font_size = max(10, min(20, int(preview_box_h / h * 0.8)))
            else: # Portrait or Square
                font_size = max(10, min(20, int(preview_box_w / w * 0.8)))
            self.output_preview_text.config(font=('Inter', font_size, 'bold'))

        except ValueError:
            self.output_dimensions_label.config(text="Output: Invalid ratio/height")
        except ZeroDivisionError:
            self.output_dimensions_label.config(text="Output: Ratio H cannot be 0")


    def update_command_preview(self):
        # This method constructs the command based on current GUI settings
        cmd_parts = ["python", get_script_path()]

        # Input and Output
        if self.is_batch_mode.get():
            input_val = self.input_folder.get() or "input_directory/"
            output_val = self.output_path_var.get() or "output_directory/"
            cmd_parts.append(f'"{input_val}"')
            cmd_parts.append(f'"{output_val}"')
            cmd_parts.append("--batch")
        elif self.selected_videos_list:
            if len(self.selected_videos_list) == 1:
                input_val = self.selected_videos_list[0]
                output_val = self.output_path_var.get() or f'"{Path(input_val).stem}_reframed{Path(input_val).suffix}"'
                cmd_parts.append(f'"{input_val}"')
                cmd_parts.append(f'{output_val}') # output_val might already be quoted if from dialog
            else: # Multiple files selected, treat as batch for command preview
                # Show first file and indicate more, output becomes a directory
                input_val = f'"{self.selected_videos_list[0]}" (and {len(self.selected_videos_list)-1} more)'
                output_val = self.output_path_var.get() or "output_directory/"
                cmd_parts.append(input_val) # Already quoted, special case
                cmd_parts.append(f'"{output_val}"')
                cmd_parts.append("--batch")

        else: # No files selected, single file mode placeholder
            cmd_parts.append("input.mp4")
            output_val = self.output_path_var.get() or "output.mp4"
            cmd_parts.append(f'"{output_val}"')

        # Ratio
        current_ratio = self.target_ratio.get()
        if current_ratio == "custom_ratio":
            cmd_parts.append(f"--ratio {self.custom_ratio_w.get()}:{self.custom_ratio_h.get()}")
        else:
            cmd_parts.append(f"--ratio {current_ratio}")

        cmd_parts.append(f"--output_height {self.output_height.get()}")

        # Padding
        if self.enable_padding.get():
            cmd_parts.append("--padding")
            pad_type = self.padding_type.get()
            cmd_parts.append(f"--padding_type {pad_type}")
            if pad_type == "blur":
                cmd_parts.append(f"--blur_amount {self.blur_amount.get()}")
            elif pad_type == "color":
                color_val_str = self.padding_color_preset.get()
                if color_val_str == "custom":
                    # Convert hex to RGB tuple string for command line
                    hex_color = self.custom_padding_color.get()
                    try:
                        r = int(hex_color[1:3], 16)
                        g = int(hex_color[3:5], 16)
                        b = int(hex_color[5:7], 16)
                        cmd_parts.append(f'--padding_color_value "({r},{g},{b})"')
                    except ValueError:
                         cmd_parts.append(f'--padding_color_value black') # Fallback
                else:
                    cmd_parts.append(f"--padding_color_value {color_val_str}")

        # Advanced
        cmd_parts.append(f"--interpolation {self.interpolation_method.get()}")
        if self.content_opacity.get() < 1.0:
             cmd_parts.append(f"--content_opacity {self.content_opacity.get():.1f}")

        cmd_parts.append(f'--object_weights "{self.object_weights_string_var.get()}"')

        if self.test_mode.get():
            cmd_parts.append("--test")
        if self.log_file_path.get():
            cmd_parts.append(f'--log_file "{self.log_file_path.get()}"')

        # Update the Text widget
        self.command_preview_text.config(state="normal")
        self.command_preview_text.delete("1.0", tk.END)
        self.command_preview_text.insert("1.0", " ".join(cmd_parts))
        self.command_preview_text.config(state="disabled")


    def start_processing_thread(self):
        # Validate inputs
        if self.is_batch_mode.get():
            if not self.input_folder.get():
                messagebox.showerror("Error", "Please select an input folder for batch mode.")
                return
            if not self.output_path_var.get(): # Output should be a folder in batch
                messagebox.showerror("Error", "Please select an output folder for batch mode.")
                return
        elif not self.selected_videos_list:
            messagebox.showerror("Error", "Please select at least one video file.")
            return
        elif not self.output_path_var.get(): # Output should be a file or folder if multiple inputs
             if len(self.selected_videos_list) > 1 and not os.path.isdir(self.output_path_var.get()):
                messagebox.showerror("Error", "Please select an output folder for multiple files.")
                return
             elif len(self.selected_videos_list) == 1 and os.path.isdir(self.output_path_var.get()):
                 messagebox.showerror("Error", "Please specify an output file name for a single video.")
                 return


        self.process_button.config(state="disabled")
        self.status_label.config(text="Status: Processing...")
        self.progress_bar['value'] = 0

        # Run in a separate thread to keep UI responsive
        thread = threading.Thread(target=self.run_processing_logic)
        thread.daemon = True # Allows main program to exit even if thread is running
        thread.start()

    def run_processing_logic(self):
        # This function will construct and run the command
        # For demonstration, it just simulates work
        # In a real app, this would call your Python script's main function or use subprocess

        command_to_run = self.command_preview_text.get("1.0", tk.END).strip().split() # Get from preview

        self.status_label.config(text=f"Status: Running command...")
        self.progress_bar.config(mode='indeterminate') # Indeterminate while subprocess runs
        self.progress_bar.start()

        try:
            # Using subprocess to run the command
            # Ensure the path to python and the script are correct
            # For development, you might need to specify the python interpreter explicitly
            # e.g., [sys.executable, get_script_path(), ...]

            # Determine Python executable
            python_exe = ""
            if hasattr(os, 'sys'): # If running in a context where sys is available (e.g. not frozen)
                python_exe = os.sys.executable
            else: # Fallback for environments like some frozen apps, might need adjustment
                python_exe = "python" # Or "python3"

            # Reconstruct command for subprocess, ensuring paths are handled correctly
            # The command_to_run from preview is good for display, but for subprocess,
            # we need to be careful with quotes, especially for paths.

            # Rebuild command more robustly for subprocess
            cmd_list_for_subprocess = [python_exe, get_script_path()]

            # Input and Output paths need careful handling
            if self.is_batch_mode.get():
                cmd_list_for_subprocess.append(self.input_folder.get())
                cmd_list_for_subprocess.append(self.output_path_var.get())
                cmd_list_for_subprocess.append("--batch")
            elif self.selected_videos_list:
                if len(self.selected_videos_list) == 1:
                    cmd_list_for_subprocess.append(self.selected_videos_list[0])
                    cmd_list_for_subprocess.append(self.output_path_var.get())
                else: # Multiple files, implies batch behavior
                    # For subprocess, we pass the *first* file and --batch
                    # The script's main.py should handle iterating through the folder of the first file if --batch is present
                    # Or, adjust main.py to accept multiple input files directly if that's preferred.
                    # For now, assuming --batch means "process folder of first input if multiple given to UI"
                    # This is a simplification. A robust CLI might handle multiple inputs directly.
                    # Let's assume the script will process all files in the dir of the first file if --batch is passed.
                    # This means the UI's "multiple files" is slightly different from CLI's direct multiple file args.
                    # A better approach might be to modify main.py to accept multiple input files.
                    # For now, we'll pass the *directory* of the first file if multiple are selected AND output is a dir.
                    first_file_dir = str(Path(self.selected_videos_list[0]).parent)
                    cmd_list_for_subprocess.append(first_file_dir)
                    cmd_list_for_subprocess.append(self.output_path_var.get()) # Should be a directory
                    cmd_list_for_subprocess.append("--batch")
            else: # Should not happen due to validation
                 raise ValueError("No input specified for processing.")


            # Add other arguments from the GUI state
            current_ratio = self.target_ratio.get()
            if current_ratio == "custom_ratio":
                cmd_list_for_subprocess.extend(["--ratio", f"{self.custom_ratio_w.get()}:{self.custom_ratio_h.get()}"])
            else:
                cmd_list_for_subprocess.extend(["--ratio", current_ratio])
            cmd_list_for_subprocess.extend(["--output_height", self.output_height.get()])

            if self.enable_padding.get():
                cmd_list_for_subprocess.append("--padding")
                pad_type = self.padding_type.get()
                cmd_list_for_subprocess.extend(["--padding_type", pad_type])
                if pad_type == "blur":
                    cmd_list_for_subprocess.extend(["--blur_amount", str(self.blur_amount.get())])
                elif pad_type == "color":
                    color_val_str = self.padding_color_preset.get()
                    if color_val_str == "custom":
                        hex_color = self.custom_padding_color.get()
                        try:
                            r = int(hex_color[1:3], 16); g = int(hex_color[3:5], 16); b = int(hex_color[5:7], 16)
                            cmd_list_for_subprocess.extend(['--padding_color_value', f"({r},{g},{b})"])
                        except ValueError: cmd_list_for_subprocess.extend(['--padding_color_value', 'black'])
                    else:
                        cmd_list_for_subprocess.extend(["--padding_color_value", color_val_str])

            cmd_list_for_subprocess.extend(["--interpolation", self.interpolation_method.get()])
            if self.content_opacity.get() < 1.0:
                 cmd_list_for_subprocess.extend(["--content_opacity", f"{self.content_opacity.get():.1f}"])
            cmd_list_for_subprocess.extend(['--object_weights', self.object_weights_string_var.get()])
            if self.test_mode.get(): cmd_list_for_subprocess.append("--test")
            if self.log_file_path.get(): cmd_list_for_subprocess.extend(['--log_file', self.log_file_path.get()])

            # Debug: print the command list for subprocess
            print("Running subprocess with command:", cmd_list_for_subprocess)

            process = subprocess.Popen(cmd_list_for_subprocess, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8')

            # Non-blocking way to read stdout/stderr if needed, or just wait
            stdout, stderr = process.communicate() # This will block until process finishes

            self.progress_bar.stop()
            self.progress_bar.config(mode='determinate')

            if process.returncode == 0:
                self.status_label.config(text="Status: Processing Complete!")
                self.progress_bar['value'] = 100
                messagebox.showinfo("Success", "Video processing completed successfully.")
                # Offer to open output folder/file
                output_location = self.output_path_var.get()
                if output_location:
                    if messagebox.askyesno("Open Output", "Open output location?"):
                        try:
                            if os.path.isdir(output_location):
                                webbrowser.open(os.path.realpath(output_location))
                            elif os.path.isfile(output_location):
                                webbrowser.open(os.path.realpath(Path(output_location).parent))
                        except Exception as e:
                            messagebox.showwarning("Open Output", f"Could not open location: {e}")

            else:
                self.status_label.config(text=f"Status: Error (Code: {process.returncode})")
                self.progress_bar['value'] = 0
                error_message = f"Processing failed with error code {process.returncode}.\n\n"
                error_message += "Stderr:\n" + (stderr[:1000] if stderr else "N/A") + ("..." if stderr and len(stderr) > 1000 else "")
                error_message += "\n\nStdout:\n" + (stdout[:1000] if stdout else "N/A") + ("..." if stdout and len(stdout) > 1000 else "")
                print("--- FrameShift Subprocess Error ---") # Log to console
                print("STDOUT:\n", stdout)
                print("STDERR:\n", stderr)
                print("--- End FrameShift Subprocess Error ---")
                messagebox.showerror("Error", error_message)

        except Exception as e:
            self.progress_bar.stop()
            self.progress_bar.config(mode='determinate')
            self.status_label.config(text=f"Status: GUI Error - {type(e).__name__}")
            messagebox.showerror("GUI Error", f"An error occurred in the GUI or while preparing the command: {e}")
            print(f"GUI/Subprocess preparation Error: {e}") # Log to console
            import traceback
            traceback.print_exc()


        finally:
            self.process_button.config(state="normal")
            if not self.status_label.cget("text").startswith("Status: Processing Complete!") and \
               not self.status_label.cget("text").startswith("Status: Error"):
                self.status_label.config(text="Status: Idle")


class CustomWeightDialog(tk.Toplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.transient(parent) # Make it a dialog for parent
        self.title("Add Custom Object Weight")
        self.parent = parent
        self.result = None

        # COCO classes (subset for brevity, can be expanded)
        # From README: bicycle, car, motorcycle, ..., toothbrush
        self.coco_classes = [
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
            "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird",
            "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
            "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis",
            "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
            "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
            "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
            "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
            "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
            "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
            "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
            "hair drier", "toothbrush", "custom" # Add custom option
        ]


        ttk.Label(self, text="Object Label:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.label_var = tk.StringVar()
        self.label_combobox = ttk.Combobox(self, textvariable=self.label_var, values=self.coco_classes)
        self.label_combobox.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        self.label_combobox.bind("<<ComboboxSelected>>", self._check_custom_label)
        self.label_combobox.bind("<KeyRelease>", self._check_custom_label_key)


        self.custom_label_entry = ttk.Entry(self, textvariable=self.label_var, state="disabled") # Enabled if 'custom'
        self.custom_label_entry.grid(row=1, column=1, padx=5, pady=2, sticky="ew")
        ttk.Label(self, text="(Enter if 'custom')").grid(row=1, column=0, padx=5, pady=2, sticky="w")


        ttk.Label(self, text="Weight (0-10):").grid(row=2, column=0, padx=5, pady=5, sticky="w")
        self.weight_var = tk.DoubleVar(value=5.0)
        ttk.Scale(self, variable=self.weight_var, from_=0, to=10, orient=tk.HORIZONTAL, length=200,
                  command=lambda v: self.weight_label.config(text=f"{float(v):.1f}")).grid(row=2, column=1, padx=5, pady=5, sticky="ew")
        self.weight_label = ttk.Label(self, text="5.0")
        self.weight_label.grid(row=2, column=2, padx=5)

        btn_frame = ttk.Frame(self)
        btn_frame.grid(row=3, column=0, columnspan=3, pady=10)
        ttk.Button(btn_frame, text="OK", command=self.on_ok).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Cancel", command=self.destroy).pack(side=tk.LEFT, padx=5)

        self.columnconfigure(1, weight=1)
        self.grab_set() # Make modal
        self.protocol("WM_DELETE_WINDOW", self.destroy)
        self.label_combobox.focus_set()

        # Center window
        self.update_idletasks()
        x = parent.winfo_x() + (parent.winfo_width() // 2) - (self.winfo_width() // 2)
        y = parent.winfo_y() + (parent.winfo_height() // 2) - (self.winfo_height() // 2)
        self.geometry(f"+{x}+{y}")

    def _check_custom_label(self, event=None):
        if self.label_var.get().lower() == "custom":
            self.custom_label_entry.config(state="normal")
            self.custom_label_entry.focus()
            self.label_var.set("") # Clear "custom" to allow typing
        else:
            self.custom_label_entry.config(state="disabled")

    def _check_custom_label_key(self, event=None):
        # If user types something not in list, and custom_label_entry is disabled,
        # it implies they might want to type a custom label.
        # This is a bit tricky with combobox behavior.
        # A simpler approach is just to rely on the "custom" selection.
        pass


    def on_ok(self):
        label = self.label_var.get().strip().lower()
        weight = self.weight_var.get()

        if not label:
            messagebox.showerror("Error", "Label cannot be empty.", parent=self)
            return
        if not (0 <= weight <= 10):
            messagebox.showerror("Error", "Weight must be between 0 and 10.", parent=self)
            return

        # Prevent adding existing default weights as "custom"
        if label in ["face", "person", "default"]:
            messagebox.showerror("Error", f"'{label}' is a reserved weight term. Adjust its specific slider instead.", parent=self)
            return

        self.result = (label, weight)
        self.destroy()


def start_gui():
    app = FrameShiftGUI()
    app.mainloop()

if __name__ == "__main__":
    # This allows running the GUI directly for testing
    # In production, main.py would call start_gui()
    start_gui()
