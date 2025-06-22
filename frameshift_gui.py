import tkinter as tk
from tkinter import ttk, filedialog, messagebox, colorchooser
import subprocess
import threading
import os
from pathlib import Path

# Prova ad importare le funzioni necessarie da frameshift.main
# Questo potrebbe richiedere aggiustamenti in frameshift.main per renderlo importabile
# o la duplicazione/adattamento di alcune logiche qui.
try:
    from frameshift.main import process_video, detect_scenes, mux_video_audio_with_ffmpeg, get_cv2_interpolation_flag, parse_color_to_bgr, map_blur_input_to_kernel
    from frameshift.weights_parser import parse_object_weights
    from frameshift.utils.detection import Detector
    FRAMESHIFT_AVAILABLE = True
except ImportError as e:
    print(f"Attenzione: Impossibile importare moduli da frameshift. Alcune funzionalità potrebbero non essere disponibili. Errore: {e}")
    FRAMESHIFT_AVAILABLE = False
    # Definizioni fallback o gestione errori se l'import fallisce
    def process_video(*args, **kwargs):
        messagebox.showerror("Errore", "Modulo 'frameshift.main.process_video' non trovato.")
        return None
    def parse_object_weights(value):
        # Semplice parser di fallback o gestione errore
        if not value: return {}
        try:
            return dict(item.split(':') for item in value.split(','))
        except Exception:
            messagebox.showerror("Errore", "Formato pesi oggetti non valido.")
            return {}
    def get_cv2_interpolation_flag(name): return 0 # Placeholder
    class Detector: # Placeholder
        def __init__(self):
            print("Warning: Using placeholder Detector class.")
    # Aggiungi altri placeholder se necessario


class FrameShiftGUI:
    def __init__(self, master):
        self.master = master
        master.title("FrameShift GUI")
        master.geometry("800x700") # Dimensioni iniziali

        # Variabili Tkinter
        self.input_path = tk.StringVar()
        self.output_path = tk.StringVar()
        self.is_batch = tk.BooleanVar(value=False)
        self.aspect_ratio = tk.StringVar(value="9:16")
        self.output_height = tk.StringVar(value="1080")
        self.interpolation_method = tk.StringVar(value="lanczos")
        self.enable_padding = tk.BooleanVar(value=False)
        self.padding_type = tk.StringVar(value="black")
        self.blur_amount = tk.IntVar(value=5) # Per slider/spinbox
        self.padding_color_value = tk.StringVar(value="black")
        self.padding_color_rgb = tk.StringVar(value="(0,0,0)") # Per il color chooser
        self.object_weights = tk.StringVar(value="face:1.0,person:0.8,default:0.5")
        self.content_opacity = tk.DoubleVar(value=1.0)

        # Stile ttk
        style = ttk.Style()
        style.theme_use('clam') # Prova un tema diverso se 'default' non è soddisfacente

        # Menu Bar
        menubar = tk.Menu(master)
        master.config(menu=menubar)

        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="Informazioni su FrameShift GUI", command=self.show_about_dialog)
        help_menu.add_command(label="Guida Rapida (Placeholder)", command=self.show_quick_guide_placeholder)
        menubar.add_cascade(label="Aiuto", menu=help_menu)

        # Frame Principale
        main_frame = ttk.Frame(master, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # --- Sezione Input/Output ---
        io_frame = ttk.LabelFrame(main_frame, text="Input/Output", padding="10")
        io_frame.pack(fill=tk.X, expand=False, pady=5)
        io_frame.columnconfigure(1, weight=1)

        ttk.Label(io_frame, text="Input Video/Cartella:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        ttk.Entry(io_frame, textvariable=self.input_path, state="readonly").grid(row=0, column=1, padx=5, pady=5, sticky=tk.EW)
        ttk.Button(io_frame, text="Sfoglia...", command=self.browse_input).grid(row=0, column=2, padx=5, pady=5)

        ttk.Label(io_frame, text="Output Video/Cartella:").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        ttk.Entry(io_frame, textvariable=self.output_path, state="readonly").grid(row=1, column=1, padx=5, pady=5, sticky=tk.EW)
        ttk.Button(io_frame, text="Sfoglia...", command=self.browse_output).grid(row=1, column=2, padx=5, pady=5)

        ttk.Checkbutton(io_frame, text="Processa in Batch", variable=self.is_batch, command=self.toggle_batch_mode).grid(row=2, column=0, columnspan=3, padx=5, pady=5, sticky=tk.W)


        # --- Frame per Impostazioni (diviso in due colonne) ---
        settings_outer_frame = ttk.Frame(main_frame)
        settings_outer_frame.pack(fill=tk.X, expand=False, pady=5)
        settings_outer_frame.columnconfigure(0, weight=1)
        settings_outer_frame.columnconfigure(1, weight=1)

        # --- Sezione Impostazioni Principali (Sinistra) ---
        main_settings_frame = ttk.LabelFrame(settings_outer_frame, text="Impostazioni Principali", padding="10")
        main_settings_frame.grid(row=0, column=0, padx=5, pady=5, sticky=tk.NSEW)

        ttk.Label(main_settings_frame, text="Rapporto d'Aspetto (es. 9:16):").pack(anchor=tk.W, padx=5, pady=(5,0))
        ttk.Entry(main_settings_frame, textvariable=self.aspect_ratio).pack(fill=tk.X, padx=5, pady=(0,5))

        ttk.Label(main_settings_frame, text="Altezza Output (pixel):").pack(anchor=tk.W, padx=5, pady=(5,0))
        ttk.Entry(main_settings_frame, textvariable=self.output_height).pack(fill=tk.X, padx=5, pady=(0,5))

        ttk.Label(main_settings_frame, text="Interpolazione:").pack(anchor=tk.W, padx=5, pady=(5,0))
        interpolation_options = ["lanczos", "cubic", "linear", "area", "nearest"]
        ttk.Combobox(main_settings_frame, textvariable=self.interpolation_method, values=interpolation_options, state="readonly").pack(fill=tk.X, padx=5, pady=(0,5))

        # --- Sezione Impostazioni Padding (Destra) ---
        padding_settings_frame = ttk.LabelFrame(settings_outer_frame, text="Impostazioni Padding", padding="10")
        padding_settings_frame.grid(row=0, column=1, padx=5, pady=5, sticky=tk.NSEW)

        self.padding_checkbox = ttk.Checkbutton(padding_settings_frame, text="Abilita Padding", variable=self.enable_padding, command=self.toggle_padding_options)
        self.padding_checkbox.pack(anchor=tk.W, padx=5, pady=5)

        self.padding_options_frame = ttk.Frame(padding_settings_frame) # Contenitore per opzioni padding
        self.padding_options_frame.pack(fill=tk.X, expand=True)

        ttk.Label(self.padding_options_frame, text="Tipo di Padding:").pack(anchor=tk.W, padx=5, pady=(5,0))
        padding_types = ["black", "blur", "color"]
        self.padding_type_combo = ttk.Combobox(self.padding_options_frame, textvariable=self.padding_type, values=padding_types, state="readonly")
        self.padding_type_combo.pack(fill=tk.X, padx=5, pady=(0,5))
        self.padding_type_combo.bind("<<ComboboxSelected>>", self.toggle_padding_details)

        # Opzioni per Padding 'blur'
        self.blur_options_frame = ttk.Frame(self.padding_options_frame)
        ttk.Label(self.blur_options_frame, text="Intensità Sfocatura (0-10):").pack(anchor=tk.W, padx=5, pady=(5,0))
        self.blur_slider = ttk.Scale(self.blur_options_frame, from_=0, to=10, variable=self.blur_amount, orient=tk.HORIZONTAL, command=lambda s: self.blur_amount.set(int(float(s)))) # command to update int var
        self.blur_slider.pack(fill=tk.X, padx=5, pady=(0,5))
        # Potresti aggiungere uno Spinbox o Entry per visualizzare/modificare il valore numerico del blur
        # self.blur_spinbox = ttk.Spinbox(self.blur_options_frame, from_=0, to=10, textvariable=self.blur_amount, width=5)
        # self.blur_spinbox.pack(side=tk.LEFT, padx=5)


        # Opzioni per Padding 'color'
        self.color_options_frame = ttk.Frame(self.padding_options_frame)
        ttk.Label(self.color_options_frame, text="Colore Padding:").pack(anchor=tk.W, padx=5, pady=(5,0))
        self.color_entry = ttk.Entry(self.color_options_frame, textvariable=self.padding_color_value)
        self.color_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5, pady=(0,5))
        self.color_button = ttk.Button(self.color_options_frame, text="Scegli...", command=self.choose_padding_color)
        self.color_button.pack(side=tk.LEFT, padx=(0,5), pady=(0,5))

        # --- Sezione Impostazioni Avanzate ---
        advanced_frame = ttk.LabelFrame(main_frame, text="Impostazioni Avanzate", padding="10")
        advanced_frame.pack(fill=tk.X, expand=False, pady=5)
        advanced_frame.columnconfigure(1, weight=1)

        ttk.Label(advanced_frame, text="Pesi Oggetti (es. face:1.0):").grid(row=0, column=0, padx=5, pady=5, sticky=tk.NW)
        self.object_weights_text = tk.Text(advanced_frame, height=3, width=40) # tk.Text per multiline
        self.object_weights_text.insert(tk.END, self.object_weights.get())
        self.object_weights_text.grid(row=0, column=1, padx=5, pady=5, sticky=tk.EW)
        # self.object_weights_text.bind("<KeyRelease>", lambda event: self.object_weights.set(self.object_weights_text.get("1.0", tk.END).strip()))


        ttk.Label(advanced_frame, text="Opacità Contenuto (0.0-1.0):").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        self.opacity_slider = ttk.Scale(advanced_frame, from_=0.0, to=1.0, variable=self.content_opacity, orient=tk.HORIZONTAL, command=lambda s: self.content_opacity.set(round(float(s),2)))
        self.opacity_slider.grid(row=1, column=1, padx=5, pady=5, sticky=tk.EW)
        # Potresti aggiungere un Entry per visualizzare/modificare il valore numerico dell'opacità
        # self.opacity_entry = ttk.Entry(advanced_frame, textvariable=self.content_opacity, width=5)
        # self.opacity_entry.grid(row=1, column=2, padx=5, pady=5)


        # --- Sezione Azioni e Stato ---
        action_frame = ttk.LabelFrame(main_frame, text="Azioni e Stato", padding="10")
        action_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        action_frame.columnconfigure(0, weight=1)
        action_frame.rowconfigure(1, weight=1) # Fa espandere il Text Log

        self.start_button = ttk.Button(action_frame, text="AVVIA PROCESSO", command=self.start_processing_thread)
        self.start_button.grid(row=0, column=0, columnspan=2, padx=5, pady=10)

        self.progress_bar = ttk.Progressbar(action_frame, orient=tk.HORIZONTAL, mode='determinate')
        self.progress_bar.grid(row=0, column=2, padx=5, pady=10, sticky=tk.EW) # Nascosta inizialmente o a fianco
        action_frame.columnconfigure(2, weight=1) # Fa espandere la progress bar


        self.log_text = tk.Text(action_frame, height=10, state="disabled", wrap=tk.WORD)
        log_scrollbar = ttk.Scrollbar(action_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scrollbar.set)
        self.log_text.grid(row=1, column=0, columnspan=3, padx=5, pady=5, sticky=tk.NSEW)
        log_scrollbar.grid(row=1, column=3, sticky=tk.NS)

        # Inizializza lo stato delle opzioni di padding
        self.toggle_padding_options()
        self.toggle_padding_details() # Chiama per nascondere/mostrare i dettagli specifici del tipo di padding

        # Inizializza il detector (se disponibile)
        self.detector = None
        if FRAMESHIFT_AVAILABLE:
            try:
                self.detector = Detector()
                self.log_message("Detector inizializzato con successo.")
            except Exception as e:
                self.log_message(f"Errore inizializzazione Detector: {e}", "ERROR")
                messagebox.showerror("Errore Detector", f"Impossibile inizializzare il detector: {e}\nFrameShift potrebbe non funzionare correttamente.")

        self._apply_tooltips() # Corretta indentazione


    def _apply_tooltips(self):
        # Tooltips per Input/Output
        ToolTip(self.io_frame.winfo_children()[1], "Percorso del video di input o della cartella contenente i video (se 'Processa in Batch' è attivo).")
        ToolTip(self.io_frame.winfo_children()[2], "Scegli il file video di input o la cartella.")
        ToolTip(self.io_frame.winfo_children()[4], "Percorso del video di output o della cartella dove salvare i video processati.")
        ToolTip(self.io_frame.winfo_children()[5], "Scegli dove salvare il video processato o la cartella di output.")
        ToolTip(self.io_frame.winfo_children()[6], "Spunta per processare tutti i video in una cartella di input e salvarli in una cartella di output.")

        # Tooltips per Impostazioni Principali
        ratio_entry = self.main_settings_frame.winfo_children()[1]
        ToolTip(ratio_entry, "Rapporto d'aspetto desiderato per il video finale. Esempi: '9:16' (verticale), '1:1' (quadrato), '16:9' (orizzontale), o un numero come '0.5625'.")
        height_entry = self.main_settings_frame.winfo_children()[3]
        ToolTip(height_entry, "Altezza del video di output in pixel (es. 1080, 720). La larghezza sarà calcolata automaticamente.")
        interpolation_combo = self.main_settings_frame.winfo_children()[5]
        ToolTip(interpolation_combo, "Algoritmo usato per ridimensionare i video. 'lanczos' o 'cubic' sono buoni per qualità, 'area' per rimpicciolire, 'linear' è più veloce.")

        # Tooltips per Impostazioni Padding
        ToolTip(self.padding_checkbox, "Se spuntato, aggiunge barre laterali se il video non riempie il nuovo formato, invece di tagliare i bordi.")
        ToolTip(self.padding_type_combo, "Scegli il tipo di barre da aggiungere: 'black' (nere), 'blur' (sfocate dal video stesso), 'color' (colore solido).")
        ToolTip(self.blur_slider, "Regola l'intensità della sfocatura per il padding 'blur'. 0 = minima, 10 = massima.")
        ToolTip(self.color_entry, "Inserisci un nome di colore (es. 'red', 'blue') o un codice RGB es. '(255,0,0)' per il padding 'color'.")
        ToolTip(self.color_button, "Scegli un colore di padding dalla palette.")

        # Tooltips per Impostazioni Avanzate
        ToolTip(self.object_weights_text, "Pesi per gli oggetti rilevati, formato 'etichetta:peso'. Es: 'face:1.0,person:0.8,default:0.2'. 'default' si applica a oggetti non specificati. Pesi più alti danno più importanza.")
        ToolTip(self.opacity_slider, "Opacità del contenuto video principale. Se < 1.0, il video viene fuso con uno sfondo sfocato del frame originale. 1.0 = completamente opaco.")

        # Tooltip per Bottone Azione
        ToolTip(self.start_button, "Avvia il processo di reframing del video con le impostazioni correnti.")
        ToolTip(self.log_text, "Area di log: mostra messaggi sullo stato dell'elaborazione, avvisi ed errori.")

# Classe Helper per Tooltip (semplice implementazione)
class ToolTip:
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tooltip_window = None
        self.widget.bind("<Enter>", self.show_tooltip)
        self.widget.bind("<Leave>", self.hide_tooltip)

    def show_tooltip(self, event=None):
        x, y, _, _ = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 25

        self.tooltip_window = tk.Toplevel(self.widget)
        self.tooltip_window.wm_overrideredirect(True)
        self.tooltip_window.wm_geometry(f"+{x}+{y}")

        label = ttk.Label(self.tooltip_window, text=self.text, background="#FFFFE0", relief="solid", borderwidth=1, wraplength=200)
        label.pack(ipadx=1)

    def hide_tooltip(self, event=None):
        if self.tooltip_window:
            self.tooltip_window.destroy()
        self.tooltip_window = None

    def show_about_dialog(self):
        messagebox.showinfo(
            "Informazioni su FrameShift GUI",
            "FrameShift GUI\n\n"
            "Un'interfaccia grafica per lo strumento FrameShift.\n"
            "FrameShift è ispirato a Google AutoFlip per il reframing automatico di video.\n\n"
            "Questa GUI permette di utilizzare le funzionalità di FrameShift senza usare la riga di comando."
        )

    def show_quick_guide_placeholder(self):
        messagebox.showinfo(
            "Guida Rapida",
            "Guida Rapida (Placeholder)\n\n"
            "1. Seleziona il video di input (o una cartella se usi 'Processa in Batch').\n"
            "2. Scegli dove salvare il video risultante.\n"
            "3. Modifica le impostazioni come rapporto d'aspetto, padding, ecc., secondo necessità.\n"
            "4. Controlla i tooltip (passando il mouse sopra le opzioni) per maggiori dettagli.\n"
            "5. Clicca 'AVVIA PROCESSO'.\n"
            "6. Monitora i messaggi nell'area di log in basso."
        )

    def log_message(self, message, level="INFO"):
        self.log_text.config(state="normal")
        self.log_text.insert(tk.END, f"[{level}] {message}\n")
        self.log_text.config(state="disabled")
        self.log_text.see(tk.END) # Auto-scroll
        if level == "ERROR":
            print(f"ERROR: {message}") # Anche su console per debug
        elif level == "WARNING":
            print(f"WARNING: {message}")

    def browse_input(self):
        if self.is_batch.get():
            path = filedialog.askdirectory()
        else:
            path = filedialog.askopenfilename(
                title="Seleziona File Video",
                filetypes=(("Video files", "*.mp4 *.mov *.avi *.mkv"), ("All files", "*.*"))
            )
        if path:
            self.input_path.set(path)
            self.log_message(f"Input selezionato: {path}")
            # Auto-imposta output se non ancora definito e se l'input è un file
            if not self.output_path.get() and not self.is_batch.get() and os.path.isfile(path):
                p = Path(path)
                suggested_output = p.parent / f"{p.stem}_reframed{p.suffix}"
                self.output_path.set(str(suggested_output))
                self.log_message(f"Output suggerito: {suggested_output}")


    def browse_output(self):
        if self.is_batch.get():
            path = filedialog.askdirectory(title="Seleziona Cartella di Output")
        else:
            # Pre-compila il nome del file basato sull'input se possibile
            default_name = ""
            if self.input_path.get() and os.path.isfile(self.input_path.get()):
                p = Path(self.input_path.get())
                default_name = f"{p.stem}_reframed{p.suffix}"

            path = filedialog.asksaveasfilename(
                title="Salva Video Reframato Come...",
                defaultextension=".mp4",
                initialfile=default_name,
                filetypes=(("MP4 files", "*.mp4"), ("MOV files", "*.mov"), ("All files", "*.*"))
            )
        if path:
            self.output_path.set(path)
            self.log_message(f"Output selezionato: {path}")

    def toggle_batch_mode(self):
        if self.is_batch.get():
            self.log_message("Modalità Batch attivata. Seleziona cartelle per input/output.")
        else:
            self.log_message("Modalità File Singolo attivata.")
        # Resetta i percorsi quando si cambia modalità per evitare confusione
        current_input = self.input_path.get()
        current_output = self.output_path.get()
        self.input_path.set("")
        self.output_path.set("")
        if self.is_batch.get():
            if os.path.isfile(current_input): self.input_path.set(os.path.dirname(current_input))
            else: self.input_path.set(current_input) # Mantieni se già dir
            if os.path.isfile(current_output): self.output_path.set(os.path.dirname(current_output))
            elif not os.path.isdir(current_output) and current_output: # Se era un file path non esistente
                 self.output_path.set(os.path.dirname(current_output))
            else: self.output_path.set(current_output)

        else: # Passando a file singolo
            if os.path.isdir(current_input): self.input_path.set("") # Richiede selezione file
            else: self.input_path.set(current_input)
            if os.path.isdir(current_output): self.output_path.set("") # Richiede selezione file
            else: self.output_path.set(current_output)


    def toggle_padding_options(self):
        if self.enable_padding.get():
            for child in self.padding_options_frame.winfo_children():
                if isinstance(child, (ttk.Label, ttk.Combobox, ttk.Scale, ttk.Entry, ttk.Button, ttk.Frame)):
                    try:
                        child.state(['!disabled'])
                    except tk.TclError: # Alcuni widget potrebbero non supportare lo stato disabled/enabled direttamente
                        pass
            self.toggle_padding_details() # Aggiorna la visibilità dei dettagli specifici
            self.log_message("Padding abilitato.")
        else:
            for child in self.padding_options_frame.winfo_children():
                 if isinstance(child, (ttk.Label, ttk.Combobox, ttk.Scale, ttk.Entry, ttk.Button, ttk.Frame)):
                    try:
                        child.state(['disabled'])
                    except tk.TclError:
                        pass
            # Nascondi anche i frame specifici quando il padding è disabilitato
            self.blur_options_frame.pack_forget()
            self.color_options_frame.pack_forget()
            self.log_message("Padding disabilitato.")

    def toggle_padding_details(self, event=None):
        # Questa funzione è chiamata quando il padding è abilitato E il tipo di padding cambia
        if not self.enable_padding.get():
            self.blur_options_frame.pack_forget()
            self.color_options_frame.pack_forget()
            return

        selected_padding_type = self.padding_type.get()
        if selected_padding_type == "blur":
            self.blur_options_frame.pack(fill=tk.X, expand=True, pady=(5,0))
            self.color_options_frame.pack_forget()
            self.log_message("Tipo padding: Blur. Configura intensità.")
        elif selected_padding_type == "color":
            self.blur_options_frame.pack_forget()
            self.color_options_frame.pack(fill=tk.X, expand=True, pady=(5,0))
            self.log_message("Tipo padding: Color. Configura colore.")
        else: # black o altro
            self.blur_options_frame.pack_forget()
            self.color_options_frame.pack_forget()
            self.log_message(f"Tipo padding: {selected_padding_type}.")


    def choose_padding_color(self):
        # Chiede il colore all'utente, il risultato è una tupla (R,G,B) e un esadecimale
        color_code = colorchooser.askcolor(title="Scegli il colore di padding")
        if color_code and color_code[1]: # color_code[1] è il valore esadecimale
            self.padding_color_value.set(color_code[1]) # Imposta l'esadecimale nel campo di testo
            # FrameShift si aspetta nomi o tuple "(R,G,B)", quindi convertiamo
            rgb_tuple_str = f"({int(color_code[0][0])},{int(color_code[0][1])},{int(color_code[0][2])})"
            self.padding_color_rgb.set(rgb_tuple_str) # Memorizza per l'uso
            # Potremmo anche aggiornare self.padding_color_value con rgb_tuple_str se preferito
            # self.padding_color_value.set(rgb_tuple_str)
            self.log_message(f"Colore padding selezionato: {color_code[1]} (RGB: {rgb_tuple_str})")

    def _validate_inputs(self):
        if not self.input_path.get():
            messagebox.showerror("Errore Input", "Seleziona un file o una cartella di input.")
            return False
        if not self.output_path.get():
            messagebox.showerror("Errore Output", "Seleziona un file o una cartella di output.")
            return False

        if self.is_batch.get():
            if not os.path.isdir(self.input_path.get()):
                messagebox.showerror("Errore Input", "Per il batch processing, l'input deve essere una cartella.")
                return False
            if not os.path.isdir(self.output_path.get()):
                 # Prova a creare la cartella di output se non esiste
                try:
                    os.makedirs(self.output_path.get(), exist_ok=True)
                    self.log_message(f"Cartella di output '{self.output_path.get()}' creata.")
                except Exception as e:
                    messagebox.showerror("Errore Output", f"Per il batch processing, l'output deve essere una cartella esistente o creabile.\nErrore: {e}")
                    return False
        else: # File singolo
            if not os.path.isfile(self.input_path.get()):
                messagebox.showerror("Errore Input", "Il file di input selezionato non è valido o non esiste.")
                return False
            # Per l'output di un singolo file, il percorso della directory deve esistere
            output_dir = os.path.dirname(self.output_path.get())
            if not os.path.isdir(output_dir):
                 try:
                    os.makedirs(output_dir, exist_ok=True)
                    self.log_message(f"Cartella parente dell'output '{output_dir}' creata.")
                 except Exception as e:
                    messagebox.showerror("Errore Output", f"La cartella per il file di output ('{output_dir}') non esiste e non può essere creata.\nErrore: {e}")
                    return False

        try:
            # Valida aspect ratio (semplice controllo, può essere migliorato)
            ar = self.aspect_ratio.get()
            if ":" in ar:
                w, h = map(float, ar.split(':'))
                if w <=0 or h <=0: raise ValueError("Aspect ratio components must be positive.")
            else:
                val = float(ar)
                if val <=0: raise ValueError("Aspect ratio must be positive.")
        except ValueError:
            messagebox.showerror("Errore Parametri", "Rapporto d'aspetto non valido. Usa formati come '16:9' o '1.77'.")
            return False

        try:
            height = int(self.output_height.get())
            if height <= 0: raise ValueError("Height must be positive.")
        except ValueError:
            messagebox.showerror("Errore Parametri", "Altezza Output non valida. Inserisci un numero intero positivo.")
            return False

        if self.enable_padding.get() and self.padding_type.get() == "color":
            # Valida il colore se necessario (FrameShift ha la sua validazione, ma un check base qui è utile)
            # Qui usiamo self.padding_color_rgb.get() se il color chooser è stato usato, altrimenti self.padding_color_value.get()
            # Per semplicità, lasciamo che sia la logica di FrameShift a validare il colore finale.
            pass

        try:
            opacity = float(self.content_opacity.get())
            if not (0.0 <= opacity <= 1.0):
                raise ValueError("Opacity must be between 0.0 and 1.0")
        except ValueError:
            messagebox.showerror("Errore Parametri", "Opacità non valida. Inserisci un numero tra 0.0 e 1.0.")
            return False

        # Valida Pesi Oggetti (semplice check, parse_object_weights farà il resto)
        # La validazione effettiva la farà parse_object_weights
        if not self.object_weights_text.get("1.0", tk.END).strip():
             messagebox.showerror("Errore Parametri", "I pesi degli oggetti non possono essere vuoti se specificati.")
             return False

        return True

    def start_processing_thread(self):
        if not FRAMESHIFT_AVAILABLE:
            messagebox.showerror("Errore", "I moduli principali di FrameShift non sono disponibili. Impossibile procedere.")
            return
        if not self.detector:
            messagebox.showerror("Errore Detector", "Il Detector non è stato inizializzato. Impossibile procedere.")
            return

        # Aggiorna il valore di object_weights dalla Text box
        self.object_weights.set(self.object_weights_text.get("1.0", tk.END).strip())

        if not self._validate_inputs():
            return

        self.start_button.config(state="disabled")
        self.progress_bar.config(mode='indeterminate') # o determinate se si può calcolare il progresso
        self.progress_bar.start()
        self.log_message("Avvio elaborazione...")

        # Esegui l'elaborazione in un thread separato per non bloccare la GUI
        thread = threading.Thread(target=self._process_videos_in_thread)
        thread.daemon = True # Il thread termina quando la finestra principale si chiude
        thread.start()

    def _process_videos_in_thread(self):
        try:
            input_val = self.input_path.get()
            output_val = self.output_path.get()

            # Preparazione argomenti per process_video o logica batch
            common_args = {
                "ratio": self.aspect_ratio.get(),
                "apply_padding_flag": self.enable_padding.get(),
                "padding_type_str": self.padding_type.get() if self.enable_padding.get() else "black",
                "padding_color_str": self.padding_color_rgb.get() if self.padding_type.get() == "color" and self.padding_color_rgb.get() else self.padding_color_value.get(),
                "blur_amount_param": self.blur_amount.get(),
                "output_target_height": int(self.output_height.get()),
                "interpolation_flag": get_cv2_interpolation_flag(self.interpolation_method.get()),
                "content_opacity": self.content_opacity.get(),
                "object_weights_map": parse_object_weights(self.object_weights.get()),
                "detector": self.detector # Passa l'istanza del detector
            }

            ffmpeg_path = self._find_ffmpeg()
            if not ffmpeg_path:
                self.log_message("ATTENZIONE: FFmpeg non trovato. L'audio non verrà processato.", "WARNING")


            if self.is_batch.get():
                in_dir = Path(input_val)
                out_dir = Path(output_val)
                out_dir.mkdir(parents=True, exist_ok=True)

                videos_to_process = [p for p in in_dir.iterdir() if p.suffix.lower() in {".mp4", ".mov", ".mkv", ".avi"}]
                num_videos = len(videos_to_process)
                self.master.after(0, lambda: self.progress_bar.config(mode='determinate', maximum=num_videos, value=0))

                for i, vid_path_obj in enumerate(videos_to_process):
                    self.log_message(f"Processando file {i+1}/{num_videos}: {vid_path_obj.name}")

                    # Assicura che l'output sia una directory per il batch
                    # out_path_for_video = out_dir / f"{vid_path_obj.stem}_reframed{vid_path_obj.suffix}"
                    # process_video aspetta un output_path *file*, non directory
                    final_output_file_path = out_dir / f"{vid_path_obj.stem}_reframed{vid_path_obj.suffix}"

                    temp_video_file = process_video(
                        input_path=str(vid_path_obj),
                        output_path=str(final_output_file_path), # Deve essere un file path per process_video
                        **common_args
                    )
                    self._handle_muxing_and_temp_file(str(vid_path_obj), temp_video_file, str(final_output_file_path), ffmpeg_path)
                    self.master.after(0, lambda current_val=i+1: self.progress_bar.config(value=current_val))
                self.log_message("Elaborazione batch completata.", "INFO")

            else: # File singolo
                self.master.after(0, lambda: self.progress_bar.config(mode='indeterminate')) # o 0-100 per fasi
                self.master.after(0, lambda: self.progress_bar.start(10))

                temp_video_file = process_video(
                    input_path=input_val,
                    output_path=output_val, # output_val è già un file path qui
                    **common_args
                )
                self._handle_muxing_and_temp_file(input_val, temp_video_file, output_val, ffmpeg_path)
                self.log_message("Elaborazione file singolo completata.", "INFO")

        except Exception as e:
            self.log_message(f"Errore durante l'elaborazione: {e}", "ERROR")
            import traceback
            self.log_message(traceback.format_exc(), "DEBUG") # Log stack trace for debugging
            self.master.after(0, lambda: messagebox.showerror("Errore Elaborazione", f"Si è verificato un errore: {e}"))
        finally:
            self.master.after(0, self._processing_finished) # Esegui sulla GUI thread

    def _find_ffmpeg(self):
        # Cerca ffmpeg nel PATH
        ffmpeg_exe = "ffmpeg"
        if os.name == 'nt': # Windows
            ffmpeg_exe = "ffmpeg.exe"

        ffmpeg_path = None
        for path_dir in os.environ["PATH"].split(os.pathsep):
            potential_path = os.path.join(path_dir, ffmpeg_exe)
            if os.path.isfile(potential_path) and os.access(potential_path, os.X_OK):
                ffmpeg_path = potential_path
                break

        if not ffmpeg_path:
             # Fallback: prova a chiamare `shutil.which` se disponibile (Python 3.3+)
            try:
                import shutil
                ffmpeg_path = shutil.which(ffmpeg_exe)
            except ImportError:
                pass # shutil.which non disponibile

        if ffmpeg_path:
            self.log_message(f"FFmpeg trovato in: {ffmpeg_path}", "DEBUG")
        return ffmpeg_path

    def _handle_muxing_and_temp_file(self, original_input_path, temp_video_file_path, final_output_path, ffmpeg_executable_path):
        if temp_video_file_path:
            if ffmpeg_executable_path:
                self.log_message(f"Tentativo di muxing audio per {Path(final_output_path).name}...", "INFO")
                success = mux_video_audio_with_ffmpeg(
                    original_input_path,
                    temp_video_file_path,
                    final_output_path,
                    ffmpeg_executable_path
                )
                if success:
                    self.log_message(f"Muxing audio completato per {Path(final_output_path).name}.", "INFO")
                    try:
                        os.remove(temp_video_file_path)
                        self.log_message(f"File video temporaneo rimosso: {temp_video_file_path}", "DEBUG")
                    except OSError as e:
                        self.log_message(f"Impossibile rimuovere il file video temporaneo {temp_video_file_path}: {e}", "WARNING")
                else:
                    self.log_message(f"Muxing audio fallito per {Path(final_output_path).name}. Il video verrà salvato senza audio.", "WARNING")
                    try:
                        # Rinomina/sposta il file temporaneo (senza audio) al percorso finale
                        if os.path.exists(final_output_path): os.remove(final_output_path) # Rimuovi se esiste già per evitare errore su move
                        os.replace(temp_video_file_path, final_output_path) # os.replace è atomico se possibile
                        self.log_message(f"Video (senza audio) salvato in: {final_output_path}", "INFO")
                    except OSError as e:
                        self.log_message(f"Impossibile rinominare il file video temporaneo {temp_video_file_path} in {final_output_path}: {e}", "ERROR")
            else: # FFmpeg non trovato
                self.log_message("FFmpeg non disponibile. Il video verrà salvato senza audio.", "WARNING")
                try:
                    if os.path.exists(final_output_path): os.remove(final_output_path)
                    os.replace(temp_video_file_path, final_output_path)
                    self.log_message(f"Video (senza audio) salvato in: {final_output_path}", "INFO")
                except OSError as e:
                    self.log_message(f"Impossibile rinominare il file video temporaneo {temp_video_file_path} in {final_output_path}: {e}", "ERROR")
        else:
            self.log_message(f"Elaborazione video fallita per {Path(original_input_path).name}, nessun file temporaneo creato.", "ERROR")


    def _processing_finished(self):
        self.progress_bar.stop()
        self.progress_bar.config(value=0, mode='determinate') # Resetta la progress bar
        self.start_button.config(state="normal")
        # messagebox.showinfo("Completato", "Elaborazione video terminata.") # Forse un po' troppo invadente

def main_gui():
    root = tk.Tk()
    # Verifica se i moduli di FrameShift sono disponibili
    if not FRAMESHIFT_AVAILABLE:
        messagebox.showwarning(
            "Dipendenze Mancanti",
            "Alcuni moduli di FrameShift non sono stati trovati. "
            "L'applicazione GUI potrebbe non funzionare correttamente o alcune funzionalità potrebbero essere disabilitate.\n"
            "Assicurati che FrameShift sia installato correttamente e che gli script siano nel PYTHONPATH.\n"
            "Consulta la console per messaggi di errore specifici."
        )
        # Si potrebbe decidere di non avviare la GUI o di avviarla in modalità limitata
        # Per ora, la avviamo comunque, ma l'utente è avvisato.

    app = FrameShiftGUI(root)
    root.mainloop()

if __name__ == '__main__':
    # Questo permette di eseguire la GUI direttamente per test
    # Assicurati che la directory 'frameshift' sia nel PYTHONPATH o che questo script
    # sia eseguito da una directory che permette l'import di 'frameshift.main'
    # Esempio: python frameshift_gui.py (dalla root del progetto)
    # O se frameshift è installato: python -m frameshift.frameshift_gui (se lo strutturi come modulo)

    # Per testare, potresti voler aggiungere la root del progetto al sys.path se non è installato
    # import sys
    # project_root = Path(__file__).resolve().parent.parent # Se frameshift_gui.py è in una sub-dir tipo 'gui'
    # sys.path.insert(0, str(project_root))
    # print(f"Aggiunto al sys.path per test: {project_root}")

    main_gui()
