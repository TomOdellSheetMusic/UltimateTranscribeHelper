"""Windows desktop interface for the Omnizart transcription tools."""
import math
import json
import hashlib
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import traceback
import webbrowser
import tkinter as tk
from tkinter import filedialog, messagebox, ttk


MODES = {
    "Piano and instruments": ("music", "Audio", "MIDI"),
    "Drums": ("drum", "Audio", "MIDI"),
    "Chords": ("chord", "Audio", "MIDI + CSV"),
    "Vocal notes": ("vocal", "Audio", "MIDI + CSV"),
    "Vocal pitch contour": ("vocal-contour", "Audio", "CSV + WAV"),
    "Beats from MIDI": ("beat", "MIDI", "MIDI + CSV"),
    "Split piano MIDI into hands": ("hands", "MIDI", "MIDI (left and right hands)"),
    "Separate stems (audio)": ("stems", "Audio", "WAV stems"),
}

BG = "#101421"
PANEL = "#1a2030"
TEXT = "#f1f3fa"
MUTED = "#aab4c8"
ACCENT = "#a49aff"
FIELD = "#252d40"
CUSTOM_MODEL = "Imported Omnizart model"
RESULT_SUFFIXES = (".mid", ".midi", ".csv", "_beat.csv", "_down_beat.csv", "_f0.csv",
                   "_trans.wav", "_accompaniment.wav", "_smoothed.mid", "_yourmt3_multitrack.mid", "_muscriptor_multitrack.mid", "_model_multitrack.mid", "_review.txt", "_comparison.txt", ".musicxml", "_simplified.musicxml", "_drums.wav",
                   "_bass.wav", "_other.wav", "_vocals.wav", "_guitar.wav", "_piano.wav", "_hands.mid", ".pdf")

SETTING_HELP = {
    "onset": "Controls how strong a note's start must be. Lower values may detect quieter notes, "
             "but can also add false notes. This is a model threshold, not a time in seconds.",
    "duration": "Controls how strongly a note must remain active. Lower values may preserve longer or "
                "quieter notes, but can merge notes or add noise. This is not a duration in seconds.",
    "instrument": "Filters instrument tracks by model confidence. Lower values allow weaker tracks through, "
                  "but can add incorrect instruments. Applies only to Pop and Stream.",
}
BP_SETTING_HELP = {
    "onset": "Basic Pitch onset confidence from 0 to 1. Lower values can recover quiet notes but may add false ones. Default: 0.5.",
    "duration": "Basic Pitch frame confidence from 0 to 1. Lower values may sustain more notes, with more noise. Default: 0.3.",
    "instrument": "Discard Basic Pitch notes shorter than this length in milliseconds. Default: 127.7 ms.",
}


class Tooltip:
    """Show a short explanation when the pointer rests over a control."""
    def __init__(self, widget, text):
        self.widget, self.text = widget, text
        self.timer = None
        self.window = None
        widget.bind("<Enter>", self.schedule, add="+")
        widget.bind("<Leave>", self.hide, add="+")
        widget.bind("<Destroy>", self.hide, add="+")

    def schedule(self, event=None):
        self.hide()
        self.timer = self.widget.after(500, self.show)

    def show(self):
        self.timer = None
        self.window = tk.Toplevel(self.widget)
        self.window.wm_overrideredirect(True)
        label = tk.Label(self.window, text=self.text, justify="left", wraplength=310, bg="#30394f", fg=TEXT,
                         font=("Segoe UI", 9), padx=12, pady=9, relief="solid", borderwidth=1)
        label.pack()
        self.window.update_idletasks()
        x = min(self.widget.winfo_rootx(), self.widget.winfo_screenwidth() - self.window.winfo_reqwidth() - 10)
        below = self.widget.winfo_rooty() + self.widget.winfo_height() + 6
        y = below if below + self.window.winfo_reqheight() < self.widget.winfo_screenheight() else (
            self.widget.winfo_rooty() - self.window.winfo_reqheight() - 6)
        self.window.geometry(f"+{max(0, x)}+{max(0, y)}")

    def hide(self, event=None):
        if self.timer is not None:
            self.widget.after_cancel(self.timer)
            self.timer = None
        if self.window is not None:
            self.window.destroy()
            self.window = None


def midi_note_count(directory, output_base=None):
    """Count MIDI note-on events, returning None when no MIDI was produced."""
    import mido
    paths = ([Path(str(output_base) + ".mid"), Path(str(output_base) + ".midi")]
             if output_base is not None else list(directory.glob("*.mid")) + list(directory.glob("*.midi")))
    paths = [path for path in paths if path.is_file()]
    paths = [path for path in paths if not (path.stem.endswith("_smoothed") and
             (path.with_name(path.stem[:-9] + ".mid").exists() or
              path.with_name(path.stem[:-9] + ".midi").exists()))]
    if not paths:
        return None
    return sum(1 for path in paths for track in mido.MidiFile(path).tracks
               for message in track if message.type == "note_on" and message.velocity > 0)


def reserve_result_name(folder, source_stem):
    """Reserve a result basename while retaining all files from earlier runs."""
    folder = Path(folder)
    if source_stem.lower().endswith((".mid", ".midi")):
        source_stem += "_result"
    for number in range(1, 10001):
        name = source_stem + ("" if number == 1 else f"_{number}")
        base = folder / name
        lock = folder / ("." + name + ".omnizart.lock")
        try:
            with lock.open("x", encoding="utf-8") as stream:
                stream.write("Omnizart result reservation\n")
        except FileExistsError:
            continue
        try:
            if (base.exists() or any((folder / (name + ending)).exists() for ending in RESULT_SUFFIXES) or
                    any(p.is_file() and p.name.casefold().startswith((name + '_').casefold()) and p.suffix.lower() == '.wav'
                        for p in folder.iterdir())):
                lock.unlink()
                continue
            return base, lock
        except Exception:
            lock.unlink(missing_ok=True)
            raise
    raise OSError("Too many results with this name in the output folder.")


def validate_model_folder(folder):
    """Check for a complete Omnizart TensorFlow SavedModel checkpoint."""
    folder = Path(folder)
    variables = folder / "variables"
    return (folder.is_dir() and (folder / "configurations.yaml").is_file() and
            (folder / "saved_model.pb").is_file() and (variables / "variables.index").is_file() and
            any(variables.glob("variables.data-*")))


def model_settings_path():
    return Path(os.environ.get("LOCALAPPDATA", str(Path.home() / ".local" / "share"))) / "Omnizart" / "custom-model.json"


def folder_settings_path():
    return model_settings_path().with_name("folders.json")


def preferences_path():
    return model_settings_path().with_name("preferences.json")


def read_preferences():
    try:
        data = json.loads(preferences_path().read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def read_folders():
    try:
        data = json.loads(folder_settings_path().read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return {}
        return {key: value for key in ("input", "output", "musescore", "custom_six_stem", "pymss_model", "pymss_source")
                if isinstance((value := data.get(key)), str)}
    except (OSError, ValueError):
        return {}


def remember_folders(input_folder=None, output_folder=None, musescore=None, custom_six_stem=None, pymss_model=None, pymss_source=None):
    settings = folder_settings_path()
    folders = read_folders()
    if input_folder is not None:
        folders["input"] = str(input_folder)
    if output_folder is not None:
        folders["output"] = str(output_folder)
    if musescore is not None:
        folders["musescore"] = str(musescore)
    if custom_six_stem is not None:
        folders["custom_six_stem"] = str(custom_six_stem)
    if pymss_model is not None:
        folders["pymss_model"] = str(pymss_model)
    if pymss_source is not None:
        folders["pymss_source"] = str(pymss_source)
    settings.parent.mkdir(parents=True, exist_ok=True)
    temporary = settings.with_suffix(".tmp")
    try:
        temporary.write_text(json.dumps(folders), encoding="utf-8")
        temporary.replace(settings)
    finally:
        if temporary.exists():
            temporary.unlink()


def read_imported_model():
    try:
        data = json.loads(model_settings_path().read_text(encoding="utf-8"))
        path = data.get("checkpoint")
        return path if isinstance(path, str) and validate_model_folder(path) else None
    except (OSError, ValueError, AttributeError):
        return None


def read_imported_models():
    """Read saved paths, including the original single-checkpoint format."""
    try:
        data = json.loads(model_settings_path().read_text(encoding="utf-8"))
        paths = data.get("checkpoints", [])
        if not isinstance(paths, list):
            paths = []
        previous = data.get("checkpoint")
        if isinstance(previous, str) and previous not in paths:
            paths.append(previous)
        return list(dict.fromkeys(path for path in paths if isinstance(path, str)))
    except (OSError, ValueError, AttributeError):
        return []


def imported_model_label(path):
    normalized = os.path.normcase(os.path.abspath(path))
    suffix = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:8]
    return f"Imported: {Path(path).name} [{suffix}]"


def remember_imported_model(path):
    settings = model_settings_path()
    settings.parent.mkdir(parents=True, exist_ok=True)
    paths = read_imported_models()
    if str(path) not in paths:
        paths.append(str(path))
    temporary = settings.with_suffix(".tmp")
    try:
        temporary.write_text(json.dumps({"checkpoint": str(path), "checkpoints": paths}), encoding="utf-8")
        temporary.replace(settings)
    finally:
        if temporary.exists():
            temporary.unlink()


def music_options(values, model="Piano"):
    """Validate optional model thresholds and produce CLI arguments."""
    if model in ("Transkun (piano)", "Transkun V2 Aug", "Pedal Piano", "YourMT3", "MuScriptor"):
        return []
    result = []
    specs = (("onset", "Note onset", 1, "--onset-th"),
             ("duration", "Frame confidence", 1, "--frame-th"),
             ("instrument", "Minimum note length", 2000, "--minimum-note-ms")) if model == "Basic Pitch" else (
             ("onset", "Note onset", 20, "--onset-th"),
             ("duration", "Note duration", 20, "--duration-th"),
             ("instrument", "Instrument confidence", 5, "--instrument-th"))
    for key, label, maximum, flag in specs:
        raw = values[key].strip()
        if not raw:
            continue
        try:
            number = float(raw)
        except ValueError as exc:
            raise ValueError(f"{label} must be a number between 0 and {maximum}.") from exc
        if not math.isfinite(number) or not 0 <= number <= maximum:
            raise ValueError(f"{label} must be a number between 0 and {maximum}.")
        result.extend([flag, str(number)])
    return result


class Application(ttk.Frame):
    def __init__(self, master):
        self._configure_style(master)
        super().__init__(master, style="Root.TFrame")
        self.master = master
        self.pack(fill="both", expand=True)
        self.mode = tk.StringVar(value=next(iter(MODES)))
        self.music_model = tk.StringVar(value="Transkun V2 Aug")
        self.custom_model_path = read_imported_model()
        self.custom_models = {imported_model_label(path): path for path in read_imported_models()}
        self.remove_vocals = tk.BooleanVar(value=False)
        self.piano_only = tk.BooleanVar(value=False)
        self.input_is_piano = tk.BooleanVar(value=False)
        self.save_separated_stem = tk.BooleanVar(value=False)
        self.include_vocals = tk.BooleanVar(value=False)
        self.separator_model = tk.StringVar(value="UVR MDX (recommended)")
        self.vocal_backend = tk.StringVar(value="Built-in")
        self.vocal_track_backend = tk.StringVar(value="Same as vocal reduction")
        self.vocal_track_model = tk.StringVar(value="UVR MDX (recommended)")
        self.vocal_track_pymss_model = tk.StringVar(value="")
        self.stem_model = tk.StringVar(value="Built-in six-stem")
        self.stem_choice = tk.StringVar(value="All")
        self.custom_stem_model = tk.StringVar()
        self.pymss_model = tk.StringVar(value="")
        self.vocal_pymss_model = tk.StringVar(value="")
        self.pymss_source = tk.StringVar(value="huggingface")
        self.use_gpu = tk.BooleanVar(value=False)
        self.gpu_index = tk.StringVar(value="0")
        self.compare_enabled = tk.BooleanVar(value=False)
        self.compare_path = tk.StringVar()
        self.score_grid = tk.StringVar(value="Off")
        self.score_tempo = tk.StringVar(value="Auto")
        self.score_key = tk.StringVar(value="Auto")
        self.align_bars = tk.BooleanVar(value=True)
        self.score_downbeat = tk.StringVar(value="Auto")
        self.score_meter = tk.StringVar(value="Auto")
        self.simplify_score = tk.BooleanVar(value=False)
        self.neural_staff = tk.BooleanVar(value=False)
        self.prefer_right_octave = tk.BooleanVar(value=False)
        self.move_left_chords = tk.BooleanVar(value=False)
        self.split_hands = tk.BooleanVar(value=False)
        self.musescore_export = tk.BooleanVar(value=False)
        self.pdf_export = tk.BooleanVar(value=False)
        self.musescore_path = tk.StringVar()
        self.lint_notes = tk.BooleanVar(value=False)
        self.smooth_notes = tk.BooleanVar(value=False)
        self.smooth_gap = tk.StringVar(value="30")
        self.remove_short_notes = tk.BooleanVar(value=False)
        self.minimum_short_note = tk.StringVar(value="40")
        self.remove_quiet_notes = tk.BooleanVar(value=False)
        self.quiet_velocity = tk.StringVar(value="15")
        self.shorten_pedal_tails = tk.BooleanVar(value=False)
        self.clarify_pedal_melody = tk.BooleanVar(value=False)
        self.keep_accompaniment = tk.BooleanVar(value=False)
        folders = read_folders()
        self.custom_stem_model.set(folders.get("custom_six_stem", ""))
        self.pymss_model.set(folders.get("pymss_model", ""))
        if self.pymss_model.get():
            self.stem_model.set('pymss model')
        if folders.get('pymss_source') in ('huggingface', 'modelscope', 'hf-mirror'):
            self.pymss_source.set(folders['pymss_source'])
        self.musescore_path.set(folders.get("musescore", ""))
        self.input_folder = folders.get("input", "")
        self.source = tk.StringVar()
        saved_output = folders.get("output", "")
        self.destination = tk.StringVar(value=saved_output if saved_output and Path(saved_output).is_dir()
                                        else str(Path.home() / "Documents"))
        self.status = tk.StringVar(value="Ready when you are.")
        self.description = tk.StringVar()
        self.defaults_description = tk.StringVar()
        self.thresholds = {key: tk.StringVar() for key in ("onset", "duration", "instrument")}
        preferences = read_preferences()
        preference_variables = {
            "mode": self.mode, "music_model": self.music_model,
            "remove_vocals": self.remove_vocals, "piano_only": self.piano_only,
            "input_is_piano": self.input_is_piano, "save_separated_stem": self.save_separated_stem,
            "include_vocals": self.include_vocals, "separator_model": self.separator_model,
            "vocal_backend": self.vocal_backend,
            "vocal_pymss_model": self.vocal_pymss_model,
            "vocal_track_backend": self.vocal_track_backend, "vocal_track_model": self.vocal_track_model,
            "vocal_track_pymss_model": self.vocal_track_pymss_model,
            "stem_model": self.stem_model, "stem_choice": self.stem_choice,
            "use_gpu": self.use_gpu, "gpu_index": self.gpu_index,
            "compare_enabled": self.compare_enabled, "compare_path": self.compare_path,
            "score_tempo": self.score_tempo, "score_key": self.score_key, "align_bars": self.align_bars,
            "score_downbeat": self.score_downbeat, "score_meter": self.score_meter,
            "simplify_score": self.simplify_score, "neural_staff": self.neural_staff,
            "prefer_right_octave": self.prefer_right_octave, "move_left_chords": self.move_left_chords,
            "split_hands": self.split_hands, "musescore_export": self.musescore_export,
            "pdf_export": self.pdf_export, "lint_notes": self.lint_notes,
            "smooth_notes": self.smooth_notes, "smooth_gap": self.smooth_gap,
            "remove_short_notes": self.remove_short_notes, "minimum_short_note": self.minimum_short_note,
            "remove_quiet_notes": self.remove_quiet_notes, "quiet_velocity": self.quiet_velocity,
            "shorten_pedal_tails": self.shorten_pedal_tails,
            "clarify_pedal_melody": self.clarify_pedal_melody,
            "keep_accompaniment": self.keep_accompaniment,
        }
        valid_preferences = {
            "mode": set(MODES),
            "vocal_backend": {"Built-in", "pymss"},
            "vocal_track_backend": {"Same as vocal reduction", "Same as selected stem", "Built-in", "pymss"},
            "music_model": {"Piano", "PianoV2", "Pop", "Stream", "Basic Pitch", "Transkun (piano)",
                            "Transkun V2 Aug", "Pedal Piano", "YourMT3", "MuScriptor", *self.custom_models,
                            CUSTOM_MODEL},
        }
        for name, variable in preference_variables.items():
            if name in preferences:
                try:
                    saved = preferences[name]
                    if name in valid_preferences and saved not in valid_preferences[name]:
                        continue
                    if isinstance(variable, tk.BooleanVar):
                        if isinstance(saved, bool):
                            variable.set(saved)
                    elif isinstance(saved, (str, int, float)):
                        variable.set(str(saved))
                except (tk.TclError, TypeError):
                    pass
        if self.separator_model.get() == "PyMSS (choose model below)":
            self.vocal_backend.set('pymss')
            self.separator_model.set('UVR MDX (recommended)')
        if not self.vocal_pymss_model.get() and self.vocal_backend.get() == 'pymss':
            self.vocal_pymss_model.set(self.pymss_model.get())
        for key, variable in self.thresholds.items():
            if isinstance(preferences.get("thresholds"), dict) and key in preferences["thresholds"]:
                variable.set(str(preferences["thresholds"][key]))
        self._preference_variables = preference_variables
        self.current_model = None
        self.model_thresholds = {}
        self.messages = queue.Queue()
        self.process = None
        self.optional_installing = False
        self.score_warnings = []
        self.running = False
        self.active_mode = None
        self.output_folder = None
        self.output_base = None
        self.output_lock = None
        self.advanced_open = False
        self.log_open = False
        self.jobs = []
        self.active_job = None
        self._job_sequence = 0

        master.title("Omnizart")
        master.geometry("1020x850")
        master.minsize(720, 650)
        master.configure(bg=BG)
        toolbar = ttk.Frame(self, style='Root.TFrame', padding=(24, 10))
        toolbar.pack(side='top', fill='x')
        ttk.Label(toolbar, text='OMNIZART', style='Eyebrow.TLabel').pack(side='left', padx=(0, 24))
        ttk.Button(toolbar, text='Input', style='RootLink.TButton', command=lambda: self._navigate('input')).pack(side='left', padx=(0, 16))
        ttk.Button(toolbar, text='Queue', style='RootLink.TButton', command=lambda: self._navigate('queue')).pack(side='left', padx=(0, 16))
        ttk.Button(toolbar, text='Details', style='RootLink.TButton', command=lambda: self._navigate('details')).pack(side='left')
        self.quick_add_button = ttk.Button(toolbar, text='Add to queue  +', style='Primary.TButton', command=self.enqueue_current)
        self.quick_add_button.pack(side='right')
        canvas = tk.Canvas(self, bg=BG, highlightthickness=0, borderwidth=0)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=canvas.yview, style="App.Vertical.TScrollbar")
        canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        body = ttk.Frame(canvas, style="Root.TFrame", padding=(28, 22))
        body.columnconfigure(0, weight=1)
        canvas_window = canvas.create_window((0, 0), window=body, anchor="nw")
        body.bind("<Configure>", lambda event: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda event: canvas.itemconfigure(canvas_window, width=event.width))
        self.canvas = canvas
        self.body = body
        canvas.bind_all("<MouseWheel>", self._mousewheel)

        header = ttk.Frame(body, style="Root.TFrame")
        header.grid(row=0, column=0, sticky="ew", pady=(0, 20))
        header.columnconfigure(0, weight=1)
        ttk.Label(header, text="Turn sound into music.", style="Title.TLabel").grid(row=1, column=0, sticky="w", pady=(0, 4))
        ttk.Label(header, text="Choose a recording, pick a model, and create an editable MIDI.",
                  style="Subtitle.TLabel").grid(row=2, column=0, sticky="w")

        card = ttk.Frame(body, style="Card.TFrame", padding=(22, 18))
        card.grid(row=1, column=0, sticky="ew")
        card.columnconfigure(0, weight=1)
        card.columnconfigure(1, weight=1)
        ttk.Label(card, text="01  /  CHOOSE YOUR INPUT", style="Section.TLabel").grid(row=0, column=0, columnspan=2, sticky="w")
        ttk.Label(card, text="Transcription type", style="CardLabel.TLabel").grid(row=1, column=0, sticky="w", pady=(16, 5))
        ttk.Label(card, text="Model  •  click to choose", style="CardLabel.TLabel").grid(row=1, column=1, sticky="w", padx=(14, 0), pady=(16, 5))
        modes = ttk.Combobox(card, textvariable=self.mode, values=list(MODES), state="readonly", style="App.TCombobox")
        modes.grid(row=2, column=0, sticky="ew")
        modes.bind("<<ComboboxSelected>>", self.mode_changed)
        self.model_picker = ttk.Combobox(card, textvariable=self.music_model,
                                        values=["Piano", "PianoV2", "Pop", "Stream", "Basic Pitch", "Transkun (piano)", "Transkun V2 Aug", "Pedal Piano", "YourMT3", "MuScriptor", *self.custom_models, CUSTOM_MODEL],
                                        state="readonly", style="App.TCombobox")
        self.model_picker.grid(row=2, column=1, sticky="ew", padx=(14, 0))
        self.model_picker.bind("<<ComboboxSelected>>", self.mode_changed)
        ttk.Label(card, textvariable=self.description, style="CardHint.TLabel", wraplength=500).grid(
            row=3, column=0, sticky="w", pady=(8, 0))
        self.models_button = ttk.Button(card, text="Models & import...", style="Link.TButton", command=self.show_models)
        self.models_button.grid(row=3, column=1, sticky="e", pady=(8, 0))
        input_entry = self._path_row(card, 4, "Input file (or drop a file)", self.source, self.browse_input)
        output_entry = self._path_row(card, 6, "Save results to", self.destination, self.browse_output)
        try:
            from tkinterdnd2 import DND_FILES, TkinterDnD
            TkinterDnD._require(master)
            for widget in (canvas, body, card, input_entry):
                widget.drop_target_register(DND_FILES)
                widget.dnd_bind('<<Drop>>', self.drop_input)
            output_entry.drop_target_register(DND_FILES)
            output_entry.dnd_bind('<<Drop>>', self.drop_output)
        except (ImportError, RuntimeError, tk.TclError):
            pass  # Browse remains available when running without the optional extension.
        self.sections = ttk.Notebook(body, style="App.TNotebook")
        self.sections.grid(row=2, column=0, sticky="ew", pady=(12, 0))
        separation_card = ttk.Frame(self.sections, style="Card.TFrame", padding=(20, 14))
        cleanup_card = ttk.Frame(self.sections, style="Card.TFrame", padding=(20, 14))
        score_card = ttk.Frame(self.sections, style="Card.TFrame", padding=(20, 14))
        for panel, title in ((separation_card, 'Separation'), (score_card, 'Score'), (cleanup_card, 'MIDI cleanup')):
            panel.columnconfigure(0, weight=1)
            panel.columnconfigure(1, weight=1)
            self.sections.add(panel, text=title)
        card = separation_card
        self.separation_option = ttk.Checkbutton(card, text="Reduce vocals before transcribing instruments",
                                                variable=self.remove_vocals, style="App.TCheckbutton",
                                                command=self.update_reduction_controls)
        self.separation_option.grid(row=8, column=0, sticky="w", pady=(16, 0))
        self.keep_option = ttk.Checkbutton(card, text="Keep accompaniment WAV", variable=self.keep_accompaniment,
                                           style="App.TCheckbutton", state="disabled")
        self.keep_option.grid(row=8, column=1, sticky="w", padx=(14, 0), pady=(16, 0))
        Tooltip(self.separation_option, "Uses the included CPU model to separate vocals. It can help with piano and voice, "
                "but can also remove some instruments or introduce artifacts. Compare both results.")
        self.piano_only_option = ttk.Checkbutton(card, text="Separate a stem before transcribing",
                                                variable=self.piano_only, style="App.TCheckbutton",
                                                command=self.update_stem_controls)
        self.piano_only_option.grid(row=9, column=0, sticky="w", pady=(4, 0))
        Tooltip(self.piano_only_option, "Runs the selected separation model, then transcribes the chosen stem. "
                "Select a six-stem or pymss model below. The vocal MIDI track can use its own separator.")
        self.input_piano_option = ttk.Checkbutton(card, text="Input is already piano only",
                                                  variable=self.input_is_piano, style="App.TCheckbutton",
                                                  command=self.update_input_piano_controls)
        self.input_piano_option.grid(row=9, column=1, sticky="w", padx=(14, 0), pady=(4, 0))
        Tooltip(self.input_piano_option, "Use an already isolated piano recording without separation. "
                "For multi-instrument models, predicted tracks are combined for a piano score and the original MIDI is kept.")
        self.vocal_option = ttk.Checkbutton(card, text="Add a separate vocal track to MIDI",
                                            variable=self.include_vocals, style="App.TCheckbutton",
                                            command=self.update_vocal_controls)
        self.vocal_option.grid(row=10, column=0, columnspan=2, sticky="w", pady=(10, 0))
        Tooltip(self.vocal_option, "Separates and transcribes the sung melody into its own MIDI track. "
                "This takes longer, and both tracks may need editing.")
        self.vocal_backend_label = ttk.Label(card, text="Vocal reduction / shared model backend", style="CardLabel.TLabel")
        self.vocal_backend_label.grid(row=11, column=0, sticky="w", pady=(10, 2))
        self.vocal_backend_picker = ttk.Combobox(card, textvariable=self.vocal_backend,
            values=['Built-in', 'pymss'], state='readonly', style='App.TCombobox')
        self.vocal_backend_picker.grid(row=11, column=1, sticky='ew', padx=(14, 0), pady=(10, 2))
        self.vocal_backend_picker.bind('<<ComboboxSelected>>', self.update_separation_controls)
        self.separator_picker = ttk.Combobox(card, textvariable=self.separator_model,
                                             values=["UVR MDX (recommended)", "UVR Vocal FT (CPU)",
                                                     "UVR Instrumental HQ4 (CPU)", "UVR Instrumental HQ5 (CPU)", "Spleeter (legacy)"],
                                             state="readonly", style="App.TCombobox")
        self.separator_picker.grid(row=12, column=0, sticky="ew")
        Tooltip(self.separator_picker, "All models run locally. The bundled UVR MDX model can optionally use a GPU. "
                "The other UVR models need a separate download and run on CPU.")
        self.separator_picker.bind("<<ComboboxSelected>>", self.update_separation_controls)
        self.separation_models_button = ttk.Button(card, text="Manage separation models...", style="Link.TButton",
                                                   command=self.show_separation_models)
        self.separation_models_button.grid(row=12, column=1, sticky="w", padx=(14, 0))
        self.vocal_pymss_button = ttk.Button(card, text='Choose pymss model...', style='Link.TButton',
                                            command=lambda: self.show_pymss_models('vocal'))
        self.vocal_pymss_button.grid(row=12, column=0, sticky='w')
        self.vocal_pymss_button.grid_remove()
        self.vocal_pymss_label = ttk.Label(card, text="", style="CardHint.TLabel")
        self.vocal_pymss_label.grid(row=14, column=0, columnspan=2, sticky="w")
        self.vocal_pymss_label.grid_remove()
        self.vocal_track_settings = ttk.Frame(card, style='Card.TFrame')
        self.vocal_track_settings.grid(row=15, column=0, columnspan=2, sticky='ew', pady=(10, 0))
        self.vocal_track_settings.columnconfigure(1, weight=1)
        ttk.Label(self.vocal_track_settings, text='Vocal MIDI separator', style='CardLabel.TLabel').grid(row=0, column=0, sticky='w')
        self.vocal_track_backend_picker = ttk.Combobox(self.vocal_track_settings, textvariable=self.vocal_track_backend,
            values=['Same as vocal reduction', 'Same as selected stem', 'Built-in', 'pymss'], state='readonly', style='App.TCombobox')
        self.vocal_track_backend_picker.grid(row=0, column=1, sticky='ew', padx=(14, 0))
        self.vocal_track_backend_picker.bind('<<ComboboxSelected>>', self.update_separation_controls)
        Tooltip(self.vocal_track_backend_picker, 'Choose a model for the vocal MIDI track. Reuse the selected stem model when it produces both the instrument and vocals; otherwise choose a separate vocal model.')
        self.vocal_track_picker = ttk.Combobox(self.vocal_track_settings, textvariable=self.vocal_track_model,
            values=['UVR MDX (recommended)', 'UVR Vocal FT (CPU)', 'UVR Instrumental HQ4 (CPU)',
                    'UVR Instrumental HQ5 (CPU)', 'Spleeter (legacy)'], state='readonly', style='App.TCombobox')
        self.vocal_track_picker.grid(row=1, column=0, sticky='ew', pady=(6, 0))
        self.vocal_track_picker.bind('<<ComboboxSelected>>', self.update_separation_controls)
        self.vocal_track_manage = ttk.Button(self.vocal_track_settings, text='Manage separation models...',
            style='Link.TButton', command=lambda: self.show_separation_models('track'))
        self.vocal_track_manage.grid(row=1, column=1, sticky='w', padx=(14, 0), pady=(6, 0))
        self.vocal_track_pymss_button = ttk.Button(self.vocal_track_settings, text='Choose pymss model...',
            style='Link.TButton', command=lambda: self.show_pymss_models('track'))
        self.vocal_track_pymss_button.grid(row=1, column=0, sticky='w', pady=(6, 0))
        self.vocal_track_pymss_label = ttk.Label(self.vocal_track_settings, text='', style='CardHint.TLabel')
        self.vocal_track_pymss_label.grid(row=2, column=0, columnspan=2, sticky='w')
        self.gpu_option = ttk.Checkbutton(card, text="Use GPU for UVR separation (DirectML)",
                                          variable=self.use_gpu, command=self.update_separation_controls,
                                          style="App.TCheckbutton")
        self.gpu_option.grid(row=13, column=0, sticky="w", pady=(10, 0))
        self.gpu_frame = ttk.Frame(card, style="Card.TFrame")
        self.gpu_frame.grid(row=13, column=1, sticky="w", padx=(14, 0), pady=(10, 0))
        ttk.Label(self.gpu_frame, text="GPU index", style="CardLabel.TLabel").pack(side="left", padx=(0, 6))
        self.gpu_entry = ttk.Entry(self.gpu_frame, textvariable=self.gpu_index, width=4, style="App.TEntry")
        self.gpu_entry.pack(side="left")
        Tooltip(self.gpu_option, "Optional GPU acceleration for separation. pymss uses NVIDIA CUDA; UVR and six-stem use DirectML. "
                "Transcription still uses its own device settings. CPU is the default.")
        Tooltip(self.gpu_entry, "GPU index, normally 0. CUDA and DirectML may number devices differently.")
        card = cleanup_card
        self.smooth_option = ttk.Checkbutton(card, text="Smooth fragmented notes (keep original MIDI)",
                                             variable=self.smooth_notes, style="App.TCheckbutton",
                                             command=self.update_smoothing_controls)
        self.smooth_option.grid(row=14, column=0, sticky="w", pady=(10, 0))
        self.smooth_frame = ttk.Frame(card, style="Card.TFrame")
        self.smooth_frame.grid(row=14, column=1, sticky="w", padx=(14, 0), pady=(10, 0))
        ttk.Label(self.smooth_frame, text="Max gap (ms)", style="CardLabel.TLabel").pack(side="left", padx=(0, 6))
        self.smooth_entry = ttk.Entry(self.smooth_frame, textvariable=self.smooth_gap, width=5, style="App.TEntry")
        self.smooth_entry.pack(side="left")
        Tooltip(self.smooth_option, "Creates an extra MIDI by joining nearby same-pitch fragments when at least one is short. "
                "It can merge intentional repeated notes. Vocal tracks, drums and passages with pitch bends or sustain are left alone.")
        Tooltip(self.smooth_entry, "Maximum silence between fragments, in milliseconds (0–200). Default 30 ms. "
                "Larger gaps may merge intentional repeats.")
        self.short_option = ttk.Checkbutton(card, text="Remove very short notes (keep original MIDI)",
                                            variable=self.remove_short_notes, style="App.TCheckbutton",
                                            command=self.update_smoothing_controls)
        self.short_option.grid(row=15, column=0, sticky="w", pady=(10, 0))
        self.short_frame = ttk.Frame(card, style="Card.TFrame")
        self.short_frame.grid(row=15, column=1, sticky="w", padx=(14, 0), pady=(10, 0))
        ttk.Label(self.short_frame, text="Shorter than (ms)", style="CardLabel.TLabel").pack(side="left", padx=(0, 6))
        self.short_entry = ttk.Entry(self.short_frame, textvariable=self.minimum_short_note, width=5, style="App.TEntry")
        self.short_entry.pack(side="left")
        Tooltip(self.short_option, "Removes short notes from a separate cleaned MIDI. Off by default. "
                "Real fast notes may also be removed; compare with the original.")
        Tooltip(self.short_entry, "Minimum note duration in milliseconds (greater than 0, up to 200). Default: 40 ms.")

        self.quiet_option = ttk.Checkbutton(card, text="Remove very quiet notes (keep original MIDI)",
                                            variable=self.remove_quiet_notes, style="App.TCheckbutton",
                                            command=self.update_smoothing_controls)
        self.quiet_option.grid(row=17, column=0, sticky="w", pady=(10, 0))
        quiet_frame = ttk.Frame(card, style="Card.TFrame")
        quiet_frame.grid(row=17, column=1, sticky="w", padx=(14, 0), pady=(10, 0))
        ttk.Label(quiet_frame, text="Velocity at most", style="CardLabel.TLabel").pack(side="left", padx=(0, 6))
        self.quiet_entry = ttk.Entry(quiet_frame, textvariable=self.quiet_velocity, width=5, style="App.TEntry")
        self.quiet_entry.pack(side="left")
        Tooltip(self.quiet_option, "Removes low-velocity notes before hand assignment and score export. A nearby quiet repeat of the same pitch is held instead when safe. Drums and vocal tracks are skipped.")
        Tooltip(self.quiet_entry, "Remove notes at or below this MIDI velocity (1–127). The default is 15; genuine soft notes may also be removed.")
        self.pedal_cleanup_option = ttk.Checkbutton(card, text="Shorten likely pedal-held notes and create MusicXML",
            variable=self.shorten_pedal_tails, style="App.TCheckbutton",
            command=self.update_pedal_cleanup_controls)
        self.pedal_cleanup_option.grid(row=18, column=0, columnspan=2, sticky="w", pady=(10, 0))
        Tooltip(self.pedal_cleanup_option, "Estimates shorter key holds at the next attack and writes pedal markings to MusicXML. Open the MusicXML for pedal notation; MIDI import may hide pedal markings. Requires MIDI pedal events; original MIDI is kept.")
        self.pedal_melody_option = ttk.Checkbutton(card, text="Clarify melodies sustained by pedal (experimental)",
            variable=self.clarify_pedal_melody, style="App.TCheckbutton", command=self.update_pedal_cleanup_controls)
        self.pedal_melody_option.grid(row=19, column=0, columnspan=2, sticky="w", pady=(10, 0))
        Tooltip(self.pedal_melody_option, "In overcrowded passages, shortens overlapping notes at the next attack in the same hand when the sustain pedal covers their original release. The pedal remains in the final MIDI and MusicXML. Intentional key holds can be shortened; review the result.")

        self.lint_option = ttk.Checkbutton(card, text="Create note review report", variable=self.lint_notes, style="App.TCheckbutton")
        self.lint_option.grid(row=16, column=0, sticky="w", pady=(10, 0))
        Tooltip(self.lint_option, "Writes a separate text report of possible short or overlapping notes. Suggestions can be intentional; MIDI is unchanged.")
        self.split_hands_option = ttk.Checkbutton(card, text='Save MIDI with left and right hand tracks', variable=self.split_hands, style='App.TCheckbutton')
        self.split_hands_option.grid(row=27, column=0, columnspan=2, sticky='w', pady=(10, 0))
        Tooltip(self.split_hands_option, 'Creates an additional MIDI for piano learning apps. The original MIDI is preserved. Requires one identifiable piano part; hand assignments are estimates.')

        card = score_card
        self.score_option = ttk.Checkbutton(card, text="Create formatted MusicXML score", variable=self.musescore_export, style="App.TCheckbutton")
        self.score_option.grid(row=17, column=0, sticky="w", pady=(10, 0))
        score_pdf = ttk.Frame(card, style="Card.TFrame")
        score_pdf.grid(row=17, column=1, sticky="w", padx=(14, 0), pady=(10, 0))
        self.pdf_option = ttk.Checkbutton(score_pdf, text="PDF", variable=self.pdf_export, style="App.TCheckbutton")
        self.pdf_option.pack(side="left")
        self.score_button = ttk.Button(score_pdf, text="Choose MuseScore...", style="Link.TButton", command=self.choose_musescore)
        self.score_button.pack(side="left", padx=(8, 0))
        Tooltip(self.score_option, "Creates an editable MusicXML score without changing the MIDI. Percussion is omitted from the score. Tempo and notation are estimates; review the result.")
        Tooltip(self.score_button, "Select MuseScore4.exe from your MuseScore installation.")

        self.compare_option = ttk.Checkbutton(card, text="Compare with another MIDI", variable=self.compare_enabled, style="App.TCheckbutton")
        self.compare_option.grid(row=18, column=0, sticky="w", pady=(10, 0))
        self.compare_button = ttk.Button(card, text="Choose comparison MIDI...", style="Link.TButton", command=self.choose_comparison)
        self.compare_button.grid(row=18, column=1, sticky="w", padx=(14, 0), pady=(10, 0))
        Tooltip(self.compare_option, "Lists pitch and onset differences from another MIDI made from the same audio. Does not prove which is correct.")

        self.score_grid_label = ttk.Label(card, text="MIDI / score tempo (BPM or Auto)", style="CardLabel.TLabel")
        self.score_grid_label.grid(row=19, column=0, sticky="w", pady=(10, 0))
        self.score_grid_picker = ttk.Entry(card, textvariable=self.score_tempo, style="App.TEntry", width=8)
        self.score_grid_picker.grid(row=19, column=1, sticky="w", padx=(14, 0), pady=(10, 0))
        Tooltip(self.score_grid_picker, "Auto estimates tempo from the notes. Enter a BPM to override it. Clear triplet patterns may use a triplet grid; other notes use sixteenths.")
        self.score_simplify_option = ttk.Checkbutton(card, text="Simplify MIDI and score when needed", variable=self.simplify_score, style="App.TCheckbutton")
        self.score_simplify_option.grid(row=20, column=0, columnspan=2, sticky="w", pady=(10, 0))
        Tooltip(self.score_simplify_option, "Simplifies the final hand MIDI and optional score. Held notes may be shortened and other notes omitted; the original transcription is preserved.")
        self.neural_staff_option = ttk.Checkbutton(card, text="Experimental: neural piano hand assignment", variable=self.neural_staff, style="App.TCheckbutton")
        self.neural_staff_option.grid(row=21, column=0, columnspan=2, sticky="w", pady=(10, 0))
        Tooltip(self.neural_staff_option, "Uses Piano SVSep on CPU for the final MIDI hand tracks and optional score, balanced with hand reach. First use installs the model. Results still require review.")
        self.right_octave_option = ttk.Checkbutton(card, text="Prefer right hand for nearby simultaneous notes",
            variable=self.prefer_right_octave, style="App.TCheckbutton", command=self.update_octave_controls)
        self.right_octave_option.grid(row=25, column=0, columnspan=2, sticky="w", pady=(10, 0))
        Tooltip(self.right_octave_option, "After hand assignment, move a simultaneous left-hand note up to an octave below the highest new right-hand note if the right hand can hold all overlapping notes within an octave and five fingers. This is an estimate; review the result.")
        self.move_left_chords_option = ttk.Checkbutton(card, text="Also consider notes from left-hand chords",
            variable=self.move_left_chords, style="App.TCheckbutton")
        self.move_left_chords_option.grid(row=26, column=0, columnspan=2, sticky="w", padx=(20, 0), pady=(6, 0))
        Tooltip(self.move_left_chords_option, "Allows the octave adjustment to split or move a simultaneous left-hand chord. This can also move intentional accompaniment notes.")
        self.align_bars_option = ttk.Checkbutton(card, text="Align bars to audio beats", variable=self.align_bars, style="App.TCheckbutton")
        self.align_bars_option.grid(row=22, column=0, sticky="w", pady=(10, 0))
        Tooltip(self.align_bars_option, "Detects beats and downbeats using Beat This. The final piano MIDI and optional score share the beat grid. First use installs the model; review the result.")
        self.downbeat_frame = ttk.Frame(card, style="Card.TFrame")
        self.downbeat_frame.grid(row=22, column=1, sticky="w", padx=(14, 0), pady=(10, 0))
        ttk.Label(self.downbeat_frame, text="First bar line (seconds)", style="CardLabel.TLabel").pack(side="left", padx=(0, 6))
        self.downbeat_entry = ttk.Entry(self.downbeat_frame, textvariable=self.score_downbeat, width=7, style="App.TEntry")
        self.downbeat_entry.pack(side="left")
        Tooltip(self.downbeat_entry, "Auto uses the detected downbeat. Enter its time in seconds to correct the bar alignment.")
        self.meter_frame = ttk.Frame(card, style="Card.TFrame")
        self.meter_frame.grid(row=23, column=0, columnspan=2, sticky="w", pady=(10, 0))
        ttk.Label(self.meter_frame, text="Beats per bar", style="CardLabel.TLabel").pack(side="left", padx=(0, 8))
        self.meter_picker = ttk.Combobox(self.meter_frame, textvariable=self.score_meter, values=["Auto", "3", "4"], state="readonly", style="App.TCombobox", width=6)
        self.meter_picker.pack(side="left")
        Tooltip(self.meter_picker, "Override the detected meter when you know the recording has three or four beats per bar.")
        key_frame = ttk.Frame(card, style="Card.TFrame")
        key_frame.grid(row=24, column=0, columnspan=2, sticky="w", pady=(10, 0))
        ttk.Label(key_frame, text="MIDI / score key", style="CardLabel.TLabel").pack(side="left", padx=(0, 8))
        from omnizart.desktop.score_key import KEYS
        key_picker = ttk.Combobox(key_frame, textvariable=self.score_key, values=["Auto", *KEYS], state="readonly", width=7)
        key_picker.pack(side="left")
        Tooltip(key_picker, "Auto estimates a key when there is enough evidence. Review key signatures and enharmonic spelling.")

        card = separation_card
        self.stem_settings = ttk.Frame(card, style="Card.TFrame")
        self.stem_settings.grid(row=25, column=0, columnspan=2, sticky="ew", pady=(10, 0))
        self.stem_settings.columnconfigure(1, weight=1)
        ttk.Label(self.stem_settings, text="Stem model", style="CardLabel.TLabel").grid(row=0, column=0, sticky='w', padx=(0, 6))
        self.stem_model_picker = ttk.Combobox(self.stem_settings, textvariable=self.stem_model,
            values=["Built-in six-stem", "Custom six-stem ONNX", "pymss model"], state="readonly", width=23)
        self.stem_model_picker.grid(row=0, column=1, sticky='ew', padx=(0, 8))
        self.stem_model_picker.bind('<<ComboboxSelected>>', self.update_separation_controls)
        self.custom_stem_button = ttk.Button(self.stem_settings, text="Choose ONNX...", style="Link.TButton", command=self.choose_custom_stem_model)
        self.custom_stem_button.grid(row=0, column=2, sticky='w', padx=(0, 8))
        self.pymss_button = ttk.Button(self.stem_settings, text="Browse pymss models...", style="Link.TButton", command=self.show_pymss_models)
        self.pymss_button.grid(row=1, column=1, sticky='w', pady=(6, 0))
        ttk.Label(self.stem_settings, text="Stem", style="CardLabel.TLabel").grid(row=2, column=0, sticky='w', padx=(0, 6), pady=(6, 0))
        self.stem_choice_picker = ttk.Combobox(self.stem_settings, textvariable=self.stem_choice,
            values=["All", "Piano", "Vocals", "Drums", "Bass", "Guitar", "Other", "Instrumental", "Accompaniment"], state="readonly", width=13)
        self.stem_choice_picker.grid(row=2, column=1, sticky='w', pady=(6, 0))
        Tooltip(self.custom_stem_button, "Choose a six-stem ONNX model with the same mix/stems interface and output order as the built-in HT-Demucs model. Other architectures are not supported.")
        Tooltip(self.stem_choice_picker, "Choose the stem to transcribe or save. For pymss you may type another stem name; All is available in standalone separation. Models may still compute unselected stems.")
        self.pymss_selected_label = ttk.Label(card, textvariable=self.pymss_model, style="CardHint.TLabel")
        self.pymss_selected_label.grid(row=26, column=0, sticky='w')
        self.save_stem_option = ttk.Checkbutton(card, text='Keep separated WAV', variable=self.save_separated_stem, style='App.TCheckbutton')
        self.save_stem_option.grid(row=26, column=1, sticky='w', padx=(14, 0))
        Tooltip(self.save_stem_option, 'Save a copy of the selected stem next to the MIDI. If off, the temporary stem is removed after transcription.')
        self.stem_settings.grid_remove()
        self.pymss_selected_label.grid_remove()
        self.save_stem_option.grid_remove()

        self.settings_card = ttk.Frame(self.sections, style="Card.TFrame", padding=(20, 14))
        self.sections.add(self.settings_card, text="Model settings")
        self.settings_card.columnconfigure(0, weight=1)
        self.advanced_button = ttk.Button(self.settings_card, text="Advanced settings   +", style="Link.TButton",
                                          command=self.toggle_advanced)
        self.advanced_button.grid(row=0, column=0, sticky="w")
        self.advanced = ttk.Frame(self.settings_card, style="Card.TFrame")
        self.advanced.grid(row=1, column=0, sticky="ew", pady=(8, 4))
        for col in range(3):
            self.advanced.columnconfigure(col, weight=1)
        self.threshold_entries = {}
        self.threshold_labels = {}
        self.setting_tooltips = {}
        for col, (key, title) in enumerate((("onset", "Note onset"), ("duration", "Note duration"),
                                             ("instrument", "Instrument confidence"))):
            label = ttk.Label(self.advanced, text=title + "  ⓘ", style="CardLabel.TLabel")
            label.grid(row=0, column=col, sticky="w", padx=(0, 12))
            self.threshold_labels[key] = label
            entry = ttk.Entry(self.advanced, textvariable=self.thresholds[key], style="App.TEntry")
            entry.grid(row=1, column=col, sticky="ew", padx=(0, 12), pady=(5, 3))
            self.threshold_entries[key] = entry
            self.setting_tooltips[key] = (Tooltip(label, SETTING_HELP[key]), Tooltip(entry, SETTING_HELP[key]))
        ttk.Label(self.advanced, textvariable=self.defaults_description, style="CardHint.TLabel",
                  wraplength=740).grid(row=2, column=0, columnspan=3, sticky="w", pady=(5, 0))
        self.advanced.grid_remove()

        queue_card = ttk.Frame(body, style="Card.TFrame", padding=(20, 14))
        self.queue_card = queue_card
        queue_card.grid(row=3, column=0, sticky="ew", pady=(12, 0))
        queue_card.columnconfigure(0, weight=1)
        ttk.Label(queue_card, text="QUEUE", style="Section.TLabel").grid(row=0, column=0, sticky='w')
        ttk.Label(queue_card, text="Each job keeps the settings selected when it was added.", style="CardHint.TLabel").grid(row=1, column=0, sticky='w', pady=(4, 8))
        queue_list = ttk.Frame(queue_card, style="Card.TFrame")
        queue_list.grid(row=2, column=0, sticky='ew')
        queue_list.columnconfigure(0, weight=1)
        self.queue_view = ttk.Treeview(queue_list, columns=('file', 'mode', 'state', 'folder'), show='headings', height=4, selectmode='browse', style='Queue.Treeview')
        for name, title, width in (('file', 'File', 270), ('mode', 'Task', 160), ('state', 'Status', 120), ('folder', 'Results', 110)):
            self.queue_view.heading(name, text=title, anchor='w')
            self.queue_view.column(name, width=width, minwidth=70, stretch=name != 'folder', anchor='w')
        self.queue_view.grid(row=0, column=0, sticky='ew')
        queue_scroll = ttk.Scrollbar(queue_list, orient='vertical', command=self.queue_view.yview, style='App.Vertical.TScrollbar')
        queue_scroll.grid(row=0, column=1, sticky='ns')
        def queue_scrolled(first, last):
            queue_scroll.set(first, last)
            self.after_idle(self._position_queue_buttons)
        self.queue_view.configure(yscrollcommand=queue_scrolled)
        self.queue_view.bind('<Configure>', lambda event: self.after_idle(self._position_queue_buttons))
        self.queue_view.bind('<MouseWheel>', lambda event: self.after_idle(self._position_queue_buttons), add='+')
        self.queue_view.bind('<<TreeviewSelect>>', self._show_selected_job)
        queue_actions = ttk.Frame(queue_card, style='Card.TFrame')
        queue_actions.grid(row=3, column=0, sticky='w', pady=(8, 0))
        ttk.Button(queue_actions, text='Add multiple files...', style='Link.TButton', command=self.browse_queue_files).pack(side='left', padx=(0, 18))
        ttk.Button(queue_actions, text='Remove selected', style='Link.TButton', command=self.remove_queued_job).pack(side='left', padx=(0, 18))
        ttk.Button(queue_actions, text='Clear finished', style='Link.TButton', command=self.clear_finished_jobs).pack(side='left')

        footer = ttk.Frame(body, style="Root.TFrame")
        self.footer = footer
        footer.grid(row=4, column=0, sticky="ew", pady=(18, 0))
        footer.columnconfigure(0, weight=1)
        ttk.Label(footer, text="READY TO PROCESS", style="Eyebrow.TLabel").grid(row=0, column=0, sticky="w")
        self.start_button = ttk.Button(footer, text="Add to queue  →", style="Primary.TButton", command=self.enqueue_current)
        self.start_button.grid(row=0, column=1, rowspan=2, sticky="e", padx=(12, 0))
        ttk.Label(footer, textvariable=self.status, style="Subtitle.TLabel", wraplength=550).grid(row=1, column=0, sticky="w", pady=(6, 0))
        self.progress = ttk.Progressbar(footer, mode="indeterminate", style="App.Horizontal.TProgressbar")
        self.progress.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(14, 0))
        self.progress.grid_remove()
        self.open_button = ttk.Button(footer, text="Open selected folder", style="Secondary.TButton", command=self.open_results, state="disabled")
        self.open_button.grid(row=3, column=1, sticky="e", pady=(12, 0))
        self.log_button = ttk.Button(footer, text="Show details  +", style="RootLink.TButton", command=self.toggle_log)
        self.log_button.grid(row=3, column=0, sticky="w", pady=(12, 0))
        self.log_frame = ttk.Frame(footer, style='Root.TFrame')
        self.log_frame.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        self.log_frame.columnconfigure(0, weight=1)
        self.log = tk.Text(self.log_frame, height=9, wrap="word", state="disabled", bg="#0b0f19", fg=MUTED,
                           insertbackground=TEXT, relief="flat", font=("Consolas", 9), padx=12, pady=10)
        self.log.grid(row=0, column=0, sticky="ew")
        log_scroll = ttk.Scrollbar(self.log_frame, orient='vertical', command=self.log.yview, style='App.Vertical.TScrollbar')
        log_scroll.grid(row=0, column=1, sticky='ns')
        self.log.configure(yscrollcommand=log_scroll.set)
        self.log_frame.grid_remove()
        self.mode_changed()
        for variable in self._preference_variables.values():
            variable.trace_add("write", self.save_preferences)
        for variable in self.thresholds.values():
            variable.trace_add("write", self.save_preferences)
        self.after(100, self.poll)
        master.protocol("WM_DELETE_WINDOW", self.close)

    def save_preferences(self, *_args):
        """Persist user-selected interface settings between launches."""
        if getattr(self, '_applying_job_snapshot', False):
            return
        data = {name: variable.get() for name, variable in self._preference_variables.items()}
        data["thresholds"] = {key: variable.get() for key, variable in self.thresholds.items()}
        settings = preferences_path()
        temporary = settings.with_suffix(".tmp")
        try:
            settings.parent.mkdir(parents=True, exist_ok=True)
            temporary.write_text(json.dumps(data), encoding="utf-8")
            temporary.replace(settings)
        except OSError:
            pass
        finally:
            if temporary.exists():
                temporary.unlink()

    @staticmethod
    def _configure_style(master):
        style = ttk.Style(master)
        style.theme_use("clam")
        style.configure("Root.TFrame", background=BG)
        style.configure("Card.TFrame", background=PANEL)
        style.configure('App.TNotebook', background=BG, borderwidth=0, tabmargins=(0, 0, 0, 0))
        style.configure('App.TNotebook.Tab', background=FIELD, foreground=MUTED, padding=(18, 10), font=('Segoe UI', 10, 'bold'), borderwidth=0)
        style.map('App.TNotebook.Tab', background=[('selected', PANEL), ('active', '#35415a')], foreground=[('selected', TEXT)])
        style.configure('Queue.Treeview', background=FIELD, fieldbackground=FIELD, foreground=TEXT, borderwidth=0, rowheight=29, font=('Segoe UI', 10))
        style.map('Queue.Treeview', background=[('selected', '#6357ad')], foreground=[('selected', TEXT)])
        style.configure('Queue.Treeview.Heading', background=PANEL, foreground=MUTED, relief='flat', font=('Segoe UI', 9, 'bold'))
        style.configure('QueueAction.TButton', background='#35415a', foreground=TEXT, borderwidth=0, padding=(4, 1), font=('Segoe UI', 9))
        style.map('QueueAction.TButton', background=[('active', '#586b8b')])
        for name, background, foreground, font in (
            ("Eyebrow.TLabel", BG, ACCENT, ("Segoe UI", 9, "bold")),
            ("Title.TLabel", BG, TEXT, ("Segoe UI", 25, "bold")),
            ("Subtitle.TLabel", BG, MUTED, ("Segoe UI", 10)),
            ("Section.TLabel", PANEL, ACCENT, ("Segoe UI", 9, "bold")),
            ("CardLabel.TLabel", PANEL, TEXT, ("Segoe UI", 10, "bold")),
            ("CardHint.TLabel", PANEL, MUTED, ("Segoe UI", 9)),
        ):
            style.configure(name, background=background, foreground=foreground, font=font)
        style.configure("App.TEntry", fieldbackground=FIELD, foreground=TEXT, borderwidth=0, padding=9, insertcolor=TEXT)
        style.configure("App.TCheckbutton", background=PANEL, foreground=TEXT, font=("Segoe UI", 10))
        style.map("App.TCheckbutton", background=[("active", PANEL)], foreground=[("disabled", MUTED)])
        style.configure("App.TCombobox", fieldbackground=FIELD, background=FIELD, foreground=TEXT, arrowcolor=TEXT,
                        borderwidth=0, padding=9)
        style.map("App.TCombobox", fieldbackground=[("readonly", FIELD), ("disabled", PANEL)],
                  foreground=[("readonly", TEXT), ("disabled", MUTED)], selectbackground=[("readonly", FIELD)],
                  selectforeground=[("readonly", TEXT)])
        style.configure("Primary.TButton", background="#9d8bff", foreground="#111426", font=("Segoe UI", 10, "bold"),
                        borderwidth=0, padding=(20, 11))
        style.map("Primary.TButton", background=[("active", "#b8aaff"), ("disabled", "#45495e")],
                  foreground=[("disabled", "#aab4c8")])
        style.configure("Secondary.TButton", background=FIELD, foreground=TEXT, borderwidth=0, padding=(13, 8))
        style.map("Secondary.TButton", background=[("active", "#35415a"), ("disabled", PANEL)],
                  foreground=[("disabled", MUTED)])
        for name, background in (("Link.TButton", PANEL), ("RootLink.TButton", BG)):
            style.configure(name, background=background, foreground=ACCENT, borderwidth=0, padding=(0, 4),
                            font=("Segoe UI", 10, "bold"))
            style.map(name, background=[("active", background)], foreground=[("disabled", MUTED)])
        style.configure("App.Horizontal.TProgressbar", troughcolor=FIELD, background=ACCENT, borderwidth=0, thickness=4)
        style.configure("App.Vertical.TScrollbar", troughcolor=BG, background=FIELD, arrowcolor=MUTED,
                        borderwidth=0, arrowsize=12)
        style.map("App.Vertical.TScrollbar", background=[("active", "#46516b")])
        master.option_add("*TCombobox*Listbox.background", FIELD)
        master.option_add("*TCombobox*Listbox.foreground", TEXT)
        master.option_add("*TCombobox*Listbox.selectBackground", "#6357ad")

    @staticmethod
    def _path_row(parent, row, label, variable, callback):
        ttk.Label(parent, text=label, style="CardLabel.TLabel").grid(row=row, column=0, columnspan=2, sticky="w", pady=(15, 5))
        frame = ttk.Frame(parent, style="Card.TFrame")
        frame.grid(row=row + 1, column=0, columnspan=2, sticky="ew")
        frame.columnconfigure(0, weight=1)
        entry = ttk.Entry(frame, textvariable=variable, style="App.TEntry")
        entry.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        ttk.Button(frame, text="Browse...", style="Secondary.TButton", command=callback).grid(row=0, column=1)
        return entry

    def _drop_path(self, event, directory=False):
        try:
            paths = self.master.tk.splitlist(event.data)
            if len(paths) > 1 and not directory:
                return 'copy' if self.enqueue_files(paths) else 'refuse_drop'
            if len(paths) != 1:
                raise ValueError('Drop one path at a time.')
            path = Path(paths[0])
            if not (path.is_dir() if directory else path.is_file()):
                raise ValueError('Drop an existing ' + ('folder.' if directory else 'file.'))
            if directory:
                self.destination.set(str(path))
                self.save_folders(output_folder=str(path))
            else:
                self.source.set(str(path))
                self.input_folder = str(path.parent)
                self.save_folders(input_folder=self.input_folder)
            self.status.set('Selected: ' + str(path))
            return 'copy'
        except (ValueError, tk.TclError) as exc:
            self.status.set(str(exc))
            return 'refuse_drop'

    def drop_input(self, event):
        return self._drop_path(event)

    def drop_output(self, event):
        return self._drop_path(event, directory=True)

    def mode_changed(self, event=None):
        mode, input_kind, output_kind = MODES[self.mode.get()]
        if mode == "stems" or (mode == "music" and self.piano_only.get()):
            self.stem_settings.grid()
            self.pymss_selected_label.grid()
        else:
            self.stem_settings.grid_remove()
            self.pymss_selected_label.grid_remove()
        self.start_button.configure(text="Add to queue  →")
        if mode == "music":
            selected_model = self.music_model.get()
            if selected_model != self.current_model:
                if self.current_model is not None:
                    self.model_thresholds[self.current_model] = {key: value.get() for key, value in self.thresholds.items()}
                saved = self.model_thresholds.get(selected_model, {})
                for key, variable in self.thresholds.items():
                    variable.set(saved.get(key, ""))
                self.current_model = selected_model
            self.model_picker.configure(state="readonly")
            self.models_button.configure(state="normal")
            self.separation_option.configure(state="normal")
            self.piano_only_option.configure(state="normal")
            self.input_piano_option.configure(state="normal")
            self.vocal_option.configure(state="normal")
            self.vocal_backend_picker.configure(state="readonly")
            self.separator_picker.configure(state="readonly")
            self.separation_models_button.configure(state="normal")
            self.compare_option.configure(state="normal")
            self.compare_button.configure(state="normal")
            self.score_grid_picker.configure(state="normal")
            self.score_option.configure(state="normal")
            self.score_simplify_option.configure(state="normal")
            self.neural_staff_option.configure(state="normal")
            self.right_octave_option.configure(state="normal")
            self.update_octave_controls()
            self.align_bars_option.configure(state="normal")
            self.downbeat_entry.configure(state="normal")
            self.meter_picker.configure(state="readonly")
            self.score_button.configure(state="normal")
            self.pdf_option.configure(state="normal")
            self.lint_option.configure(state="normal")
            self.split_hands_option.configure(state="normal")
            self.smooth_option.configure(state="normal")
            self.short_option.configure(state="normal")
            self.quiet_option.configure(state="normal")
            self.pedal_cleanup_option.configure(state="normal")
            self.pedal_melody_option.configure(state="normal")
            self.update_smoothing_controls()
            self.update_separation_controls()
            is_fixed_piano = self.music_model.get() in ("Transkun (piano)", "Transkun V2 Aug", "Pedal Piano", "YourMT3", "MuScriptor")
            self.advanced_button.configure(state="disabled" if is_fixed_piano else "normal")
            if is_fixed_piano:
                self.advanced.grid_remove()
                self.advanced_open = False
                self.advanced_button.configure(text="Advanced settings   +")
            hints = {"Piano": "For solo piano recordings. A good starting point for piano.",
                     "PianoV2": "Alternative solo piano model; compare it with Piano.",
                     "Pop": "For multiple instruments. Solo piano may produce no notes.",
                     "Stream": "Attempts to identify different instruments in a recording.",
                     "Basic Pitch": "A separate model for individual instruments and voice; outputs MIDI pitch bends.",
                     "Transkun (piano)": "A piano-specific model. Uses CPU and may take longer on full songs.",
                     "Transkun V2 Aug": "Alternative Transkun piano model. Select Models & import to download or import it.",
                     "Pedal Piano": "Piano model with sustain-pedal estimates. Download or import it in Models & import.",
                     "YourMT3": "Multi-instrument transcription model. Install the optional PyTorch backend; weights download on first use.",
                     "MuScriptor": "Multi-instrument transcription with MusicXML-ready output. Requires optional install and Hugging Face model access.",
                     CUSTOM_MODEL: (f"Imported checkpoint: {Path(self.custom_model_path).name}"
                                    if self.custom_model_path else
                                    "Select a compatible Omnizart checkpoint in Models & import.")}
            selected = self.music_model.get()
            if selected in self.custom_models:
                self.description.set(f"Imported checkpoint: {self.custom_models[selected]}")
            else:
                self.description.set(hints[selected])
            basic = self.music_model.get() == "Basic Pitch"
            custom = selected == CUSTOM_MODEL or selected in self.custom_models
            multi_instrument = selected in ("Pop", "Stream") or custom
            self.threshold_entries["instrument"].configure(state="normal" if multi_instrument or basic else "disabled")
            labels = {"onset": "Note onset", "duration": "Frame confidence" if basic else "Note duration",
                      "instrument": "Minimum note length (ms)" if basic else "Instrument confidence"}
            for key in labels:
                self.threshold_labels[key].configure(text=labels[key] + "  ⓘ")
                for tooltip in self.setting_tooltips[key]:
                    tooltip.text = (BP_SETTING_HELP if basic else SETTING_HELP)[key]
            if is_fixed_piano:
                self.defaults_description.set("This piano model uses its own model settings.")
            elif custom:
                self.defaults_description.set("Uses the imported checkpoint's defaults. Leave blank to use them.\n"
                                              "Lower thresholds may recover more notes but can add errors.")
            elif basic:
                self.defaults_description.set("Defaults: onset 0.5, frame 0.3, minimum note length 127.7 ms. "
                                              "Leave blank to use defaults.\nLower thresholds may recover "
                                              "more notes, but can add errors.")
            else:
                onset_default = "6" if multi_instrument else "4"
                confidence_default = ", instrument confidence 1.1" if multi_instrument else ""
                self.defaults_description.set(f"Defaults: onset {onset_default}, duration 0.5{confidence_default}. "
                                              "Leave blank to use defaults. Values are thresholds, not seconds.\n"
                                              "Lower values may recover more notes but can add errors. "
                                              "Instrument confidence is used by Pop and Stream.")
        else:
            self.model_picker.configure(state="disabled")
            self.models_button.configure(state="normal" if mode == "stems" else "disabled")
            self.separation_option.configure(state="disabled")
            self.piano_only_option.configure(state="disabled")
            self.input_piano_option.configure(state="disabled")
            self.vocal_option.configure(state="disabled")
            self.vocal_backend_picker.configure(state="disabled")
            self.separator_picker.configure(state="disabled")
            self.separation_models_button.configure(state="disabled")
            self.keep_option.configure(state="disabled")
            self.gpu_option.configure(state="disabled")
            self.gpu_entry.configure(state="disabled")
            self.compare_option.configure(state="disabled")
            self.compare_button.configure(state="disabled")
            self.score_grid_picker.configure(state="disabled")
            self.score_option.configure(state="disabled")
            self.score_simplify_option.configure(state="disabled")
            self.neural_staff_option.configure(state="disabled")
            self.right_octave_option.configure(state="disabled")
            self.move_left_chords_option.configure(state="disabled")
            self.align_bars_option.configure(state="disabled")
            self.downbeat_entry.configure(state="disabled")
            self.meter_picker.configure(state="disabled")
            self.score_button.configure(state="disabled")
            self.pdf_option.configure(state="disabled")
            self.lint_option.configure(state="disabled")
            self.split_hands_option.configure(state="disabled")
            self.smooth_option.configure(state="disabled")
            self.smooth_entry.configure(state="disabled")
            self.short_option.configure(state="disabled")
            self.short_entry.configure(state="disabled")
            self.quiet_option.configure(state="disabled")
            self.quiet_entry.configure(state="disabled")
            self.pedal_cleanup_option.configure(state="disabled")
            self.pedal_melody_option.configure(state="disabled")
            self.advanced_button.configure(state="disabled")
            self.advanced.grid_remove()
            self.advanced_open = False
            self.advanced_button.configure(text="Advanced settings   +")
            self.description.set(f"Input: {input_kind}  •  Output: {output_kind}. This mode uses its model defaults.")
            if mode == "stems":
                self.description.set("Choose a separation model and save all or selected audio stems as WAV files.")
                self.update_separation_controls()
            elif mode == 'hands':
                self.description.set('Choose a piano MIDI to create separate left- and right-hand tracks. The source remains unchanged.')
                self.neural_staff_option.configure(state='normal')

    def toggle_advanced(self):
        self.advanced_open = not self.advanced_open
        if self.advanced_open:
            self.advanced.grid()
        else:
            self.advanced.grid_remove()
        self.advanced_button.configure(text="Advanced settings   −" if self.advanced_open else "Advanced settings   +")

    def update_separation_controls(self, event=None):
        music_mode = self.mode.get() == "Piano and instruments"
        enabled = music_mode and self.remove_vocals.get()
        track_active = music_mode and self.include_vocals.get()
        track_same = self.vocal_track_backend.get() == 'Same as vocal reduction'
        show_vocal_settings = music_mode and (self.remove_vocals.get() or (track_active and track_same))
        self.keep_option.configure(state="normal" if enabled else "disabled")
        pymss_vocals = show_vocal_settings and self.vocal_backend.get() == 'pymss'
        pymss_track = track_active and self.vocal_track_backend.get() == 'pymss'
        use_pymss_backend = music_mode and self.vocal_backend.get() == 'pymss'
        self.vocal_backend_picker.configure(state='readonly' if music_mode else 'disabled')
        if show_vocal_settings:
            self.vocal_backend_label.grid()
            self.vocal_backend_picker.grid()
        else:
            self.vocal_backend_label.grid_remove()
            self.vocal_backend_picker.grid_remove()
        if show_vocal_settings and use_pymss_backend:
            self.separator_picker.grid_remove()
            self.separation_models_button.grid_remove()
            self.vocal_pymss_button.grid()
        elif show_vocal_settings:
            self.separator_picker.grid()
            self.separation_models_button.grid()
            self.vocal_pymss_button.grid_remove()
        else:
            self.separator_picker.grid_remove()
            self.separation_models_button.grid_remove()
            self.vocal_pymss_button.grid_remove()
        self.vocal_pymss_label.configure(text=("Selected pymss model: " + (self.vocal_pymss_model.get() or "none selected")) if use_pymss_backend else "")
        if show_vocal_settings and use_pymss_backend:
            self.vocal_pymss_label.grid()
        else:
            self.vocal_pymss_label.grid_remove()
        if track_active:
            self.vocal_track_settings.grid()
        else:
            self.vocal_track_settings.grid_remove()
        if track_active and self.vocal_track_backend.get() == 'Built-in':
            self.vocal_track_picker.grid()
            self.vocal_track_manage.grid()
        else:
            self.vocal_track_picker.grid_remove()
            self.vocal_track_manage.grid_remove()
        if pymss_track:
            self.vocal_track_pymss_button.grid()
            self.vocal_track_pymss_label.configure(text='Selected pymss model: ' + (self.vocal_track_pymss_model.get() or 'none selected'))
            self.vocal_track_pymss_label.grid()
        else:
            self.vocal_track_pymss_button.grid_remove()
            self.vocal_track_pymss_label.grid_remove()
        uvr = ((show_vocal_settings and self.vocal_backend.get() == 'Built-in' and self.separator_model.get() == 'UVR MDX (recommended)') or
               (track_active and self.vocal_track_backend.get() == 'Built-in' and self.vocal_track_model.get() == 'UVR MDX (recommended)'))
        stems = self.mode.get() == "Separate stems (audio)"
        music_piano = self.mode.get() == 'Piano and instruments' and self.piano_only.get()
        if music_piano:
            self.stem_settings.grid()
            self.pymss_selected_label.grid()
            self.save_stem_option.grid()
            if self.stem_choice.get() == 'All':
                self.stem_choice.set('Piano')
        elif self.mode.get() != 'Separate stems (audio)':
            self.stem_settings.grid_remove()
            self.pymss_selected_label.grid_remove()
            self.save_stem_option.grid_remove()
        else:
            self.save_stem_option.grid_remove()
        self.stem_choice_picker.configure(state='normal' if self.stem_model.get() == 'pymss model' else 'readonly')
        choices = ['All', 'Piano', 'Vocals', 'Drums', 'Bass', 'Guitar', 'Other']
        if self.stem_model.get() == 'pymss model':
            choices += ['Instrumental', 'Accompaniment']
        elif self.stem_choice.get() not in choices:
            self.stem_choice.set('All')
        self.stem_choice_picker.configure(values=choices)
        piano = self.mode.get() == "Piano and instruments" and self.piano_only.get()
        pymss_gpu = ((stems or piano) and self.stem_model.get() == 'pymss model') or pymss_vocals or pymss_track
        self.gpu_option.configure(state="normal" if uvr or stems or piano or pymss_vocals or pymss_track else "disabled",
                                  text="Use NVIDIA GPU for pymss / DirectML for built-in separation" if pymss_gpu and (uvr or ((stems or piano) and self.stem_model.get() != 'pymss model')) else
                                       "Use NVIDIA GPU for separation (CUDA)" if pymss_gpu else "Use GPU for separation (DirectML)" if stems or piano else
                                       "Use GPU for UVR separation (DirectML)")
        self.gpu_entry.configure(state="normal" if (uvr or stems or piano or pymss_vocals or pymss_track) and self.use_gpu.get() else "disabled")

    def update_vocal_controls(self):
        if self.include_vocals.get():
            self.input_is_piano.set(False)
            if self.piano_only.get() and self.vocal_track_backend.get() == 'Same as vocal reduction':
                self.vocal_track_backend.set('Same as selected stem')
        self.update_separation_controls()

    def update_reduction_controls(self):
        if self.remove_vocals.get():
            self.piano_only.set(False)
            self.input_is_piano.set(False)
            if self.vocal_track_backend.get() == 'Same as selected stem':
                self.vocal_track_backend.set('Same as vocal reduction')
        self.update_separation_controls()

    def update_stem_controls(self):
        if self.piano_only.get():
            self.remove_vocals.set(False)
            self.input_is_piano.set(False)
            if self.include_vocals.get() and self.vocal_track_backend.get() == 'Same as vocal reduction':
                self.vocal_track_backend.set('Same as selected stem')
        elif self.vocal_track_backend.get() == 'Same as selected stem':
            self.vocal_track_backend.set('Same as vocal reduction')
        self.update_separation_controls()

    def update_input_piano_controls(self):
        if self.input_is_piano.get():
            self.remove_vocals.set(False)
            self.piano_only.set(False)
            self.include_vocals.set(False)
        self.update_separation_controls()

    def update_smoothing_controls(self):
        enabled = self.mode.get() == "Piano and instruments" and self.smooth_notes.get()
        self.smooth_entry.configure(state="normal" if enabled else "disabled")
        short_enabled = self.mode.get() == "Piano and instruments" and self.remove_short_notes.get()
        self.short_entry.configure(state="normal" if short_enabled else "disabled")
        quiet_enabled = self.mode.get() == "Piano and instruments" and self.remove_quiet_notes.get()
        self.quiet_entry.configure(state="normal" if quiet_enabled else "disabled")

    def update_pedal_cleanup_controls(self):
        if self.clarify_pedal_melody.get():
            self.shorten_pedal_tails.set(True)
        if self.shorten_pedal_tails.get():
            self.musescore_export.set(True)

    def update_octave_controls(self):
        enabled = self.mode.get() == "Piano and instruments" and self.prefer_right_octave.get()
        self.move_left_chords_option.configure(state="normal" if enabled else "disabled")

    def toggle_log(self):
        self.log_open = not self.log_open
        if self.log_open:
            self.log_frame.grid()
        else:
            self.log_frame.grid_remove()
        self.log_button.configure(text="Hide details  −" if self.log_open else "Show details  +")

    def _mousewheel(self, event):
        # Text and queue widgets have their own scrolling; do not move the form too.
        widget = event.widget
        while widget is not None:
            if widget in (getattr(self, 'log', None), getattr(self, 'queue_view', None)):
                return
            widget = getattr(widget, 'master', None)
        self.canvas.yview_scroll(-int(event.delta / 120), 'units')

    def _navigate(self, destination):
        if destination == 'input':
            self.canvas.yview_moveto(0)
            return
        if destination == 'details' and not self.log_open:
            self.toggle_log()
        self.update_idletasks()
        target = self.queue_card if destination == 'queue' else self.footer
        self.canvas.yview_moveto(target.winfo_y() / max(1, self.body.winfo_height()))

    def _form_snapshot(self):
        excluded = {'status', 'description', 'defaults_description'}
        variables = {name: value.get() for name, value in vars(self).items()
                     if isinstance(value, tk.Variable) and name not in excluded}
        return {'variables': variables, 'thresholds': {key: value.get() for key, value in self.thresholds.items()}}

    def _apply_snapshot(self, snapshot):
        self._applying_job_snapshot = True
        try:
            for name, value in snapshot['variables'].items():
                getattr(self, name).set(value)
            for key, value in snapshot['thresholds'].items():
                self.thresholds[key].set(value)
        finally:
            self._applying_job_snapshot = False

    def enqueue_current(self):
        source = Path(self.source.get()).expanduser()
        destination = Path(self.destination.get()).expanduser()
        if not self.source.get().strip() or not source.is_file():
            messagebox.showerror('Input file', 'Choose an existing input file.')
            return
        if not self.destination.get().strip() or not destination.is_dir():
            messagebox.showerror('Output folder', 'Choose an existing output folder.')
            return
        self._job_sequence += 1
        job = {'id': str(self._job_sequence), 'snapshot': self._form_snapshot(), 'source': str(source),
               'mode': self.mode.get(), 'state': 'Waiting', 'log': '', 'output': None}
        self.jobs.append(job)
        self.queue_view.insert('', 'end', iid=job['id'], values=(source.name, job['mode'], job['state'], ''))
        job['folder_button'] = ttk.Button(self.queue_view, text='Open folder', style='QueueAction.TButton',
                                          command=lambda: self._open_job_folder(job))
        if self.active_job is None and not self.running and not self.optional_installing:
            self._start_next_job()
        else:
            self.status.set(f"Added {source.name} to the queue.")

    def browse_queue_files(self):
        midi = MODES[self.mode.get()][0] in ('beat', 'hands')
        kinds = [('MIDI files', '*.mid *.midi'), ('All files', '*.*')] if midi else [
            ('Audio files', '*.wav *.flac *.mp3 *.ogg *.aiff *.aif'), ('All files', '*.*')]
        paths = filedialog.askopenfilenames(filetypes=kinds, initialdir=self.input_folder or None)
        if paths:
            self.enqueue_files(paths)

    def enqueue_files(self, paths):
        paths = [Path(path) for path in paths]
        if not paths or any(not path.is_file() for path in paths):
            self.status.set('Drop existing files to add them to the queue.')
            return False
        if not self.destination.get().strip() or not Path(self.destination.get()).expanduser().is_dir():
            messagebox.showerror('Output folder', 'Choose an existing output folder.')
            return False
        for path in paths:
            self.source.set(str(path))
            self.enqueue_current()
        self.source.set(str(paths[0]))
        self.input_folder = str(paths[0].parent)
        self.save_folders(input_folder=self.input_folder)
        return True

    def _update_job(self, job, state):
        job['state'] = state
        if self.queue_view.exists(job['id']):
            self.queue_view.item(job['id'], values=(Path(job['source']).name, job['mode'], state, ''))
            self.after_idle(self._position_queue_buttons)

    def _position_queue_buttons(self):
        for job in self.jobs:
            button = job.get('folder_button')
            if button is None:
                continue
            bounds = self.queue_view.bbox(job['id'], 'folder') if self.queue_view.exists(job['id']) else ()
            if bounds and job['output'] and job['output'].is_dir():
                x, y, width, height = bounds
                button.place(x=x + 2, y=y + 2, width=max(1, width - 4), height=max(1, height - 4))
            else:
                button.place_forget()

    def _open_job_folder(self, job):
        folder = job['output']
        if folder and folder.is_dir():
            try:
                os.startfile(str(folder))
            except OSError as exc:
                messagebox.showerror('Cannot open results', str(exc))

    def _start_next_job(self):
        if self.active_job is not None or self.running or self.optional_installing:
            return
        job = next((item for item in self.jobs if item['state'] == 'Waiting'), None)
        if job is None:
            return
        self.active_job = job
        self._update_job(job, 'Starting')
        self.queue_view.selection_set(job['id'])
        self.queue_view.see(job['id'])
        self._show_selected_job()
        self._start_active_job()

    def _start_active_job(self):
        job = self.active_job
        if job is None:
            return
        current = self._form_snapshot()
        try:
            self._apply_snapshot(job['snapshot'])
            self.start()
        except Exception as exc:
            job['log'] += ''.join(traceback.format_exception(type(exc), exc, exc.__traceback__))
            self._update_job(job, 'Needs attention')
            self.status.set('Could not start the queued job. Open details for more information.')
            self.active_job = None
            self._show_selected_job()
            self.after_idle(self._start_next_job)
            return
        finally:
            self._apply_snapshot(current)
        if self.running:
            job['output'] = self.output_folder
            self._update_job(job, 'Running')
            self.start_button.configure(state='normal')
        elif self.optional_installing:
            self._update_job(job, 'Installing model')
            self.start_button.configure(state='normal')
        else:
            self._update_job(job, 'Needs attention')
            self.active_job = None
            self.after_idle(self._start_next_job)

    def _show_selected_job(self, event=None):
        selection = self.queue_view.selection()
        job = next((item for item in self.jobs if selection and item['id'] == selection[0]), None)
        if job is None:
            return
        self.log.configure(state='normal')
        self.log.delete('1.0', 'end')
        self.log.insert('end', job['log'])
        self.log.see('end')
        self.log.configure(state='disabled')
        self.open_button.configure(state='normal' if job['output'] and job['output'].is_dir() else 'disabled')

    def remove_queued_job(self):
        selection = self.queue_view.selection()
        job = next((item for item in self.jobs if selection and item['id'] == selection[0]), None)
        if job is None:
            return
        if job is self.active_job:
            self.status.set('The active job cannot be removed while it is running.')
            return
        self.queue_view.delete(job['id'])
        job['folder_button'].destroy()
        self.jobs.remove(job)
        self.status.set('Removed job from the queue.')

    def clear_finished_jobs(self):
        for job in self.jobs[:]:
            if job is not self.active_job and job['state'] != 'Waiting':
                self.queue_view.delete(job['id'])
                job['folder_button'].destroy()
                self.jobs.remove(job)

    def browse_input(self):
        midi = MODES[self.mode.get()][0] in ("beat", "hands")
        kinds = [("MIDI files", "*.mid *.midi"), ("All files", "*.*")] if midi else [
            ("Audio files", "*.wav *.flac *.mp3 *.ogg *.aiff *.aif"), ("All files", "*.*")]
        current = Path(self.source.get()).expanduser() if self.source.get().strip() else None
        initial = str(current.parent) if current and current.parent.is_dir() else (
            self.input_folder if self.input_folder and Path(self.input_folder).is_dir() else "")
        path = filedialog.askopenfilename(filetypes=kinds, initialdir=initial or None)
        if path:
            self.source.set(path)
            self.input_folder = str(Path(path).parent)
            self.save_folders(input_folder=self.input_folder)

    def show_models(self):
        if self.mode.get() == "Separate stems (audio)":
            self.show_six_stem_model()
            return
        if self.mode.get() == "Piano and instruments" and self.piano_only.get():
            self.show_six_stem_model()
            return
        if self.mode.get() == "Piano and instruments" and self.music_model.get() == "Transkun V2 Aug":
            self.show_transkun_aug_model()
            return
        dialog = tk.Toplevel(self.master)
        dialog.title("Models & import")
        dialog.configure(bg=BG)
        dialog.geometry("650x740")
        dialog.transient(self.master)
        frame = ttk.Frame(dialog, style="Root.TFrame", padding=24)
        frame.pack(fill="both", expand=True)
        ttk.Label(frame, text="Models & import", style="Title.TLabel").pack(anchor="w")
        ttk.Label(frame, text="Included: Piano, PianoV2, Pop, Stream, Basic Pitch and Transkun.\n"
                  "YourMT3 installs automatically; its first model download requires Git and Git LFS on PATH.\n"
                  "MuScriptor installs automatically; download its model file from Hugging Face into the folder below.\n"
                  "Select Transkun V2 Aug to manage its optional piano checkpoint.",
                  style="Subtitle.TLabel", wraplength=555, justify="left").pack(anchor="w", pady=(12, 16))
        ttk.Label(frame, text="MuScriptor small is CC BY-NC 4.0 (non-commercial use only). Accept the terms on its Hugging Face page, then save model.safetensors and config.json in this folder:",
                  style="Subtitle.TLabel", wraplength=555, justify="left").pack(anchor="w", pady=(0, 8))
        from omnizart.desktop.muscriptor import model_folder
        muscriptor_folder = model_folder()
        ttk.Label(frame, text=str(muscriptor_folder), style="Subtitle.TLabel", wraplength=555).pack(anchor="w")
        muscriptor_controls = ttk.Frame(frame, style="Root.TFrame")
        muscriptor_controls.pack(anchor="w", pady=(6, 12))
        ttk.Button(muscriptor_controls, text="Open MuScriptor folder", style="Secondary.TButton",
                   command=lambda: os.startfile(str(muscriptor_folder))).pack(side="left", padx=(0, 8))
        ttk.Button(muscriptor_controls, text="Open Hugging Face model page", style="RootLink.TButton",
                   command=lambda: webbrowser.open("https://huggingface.co/MuScriptor/muscriptor-small")).pack(side="left")
        ttk.Label(frame, text="Import a complete Omnizart music checkpoint folder containing configurations.yaml, "
                  "saved_model.pb and its variables. The release download supplies weights only; it does not include "
                  "the other files. Transkun and Basic Pitch models use different formats and cannot be imported here.",
                  style="Subtitle.TLabel", wraplength=555, justify="left").pack(anchor="w")
        ttk.Button(frame, text="Choose checkpoint folder...", style="Secondary.TButton",
                   command=lambda: self.import_model(dialog)).pack(anchor="w", pady=(12, 12))
        ttk.Label(frame, text="Pedal Piano (optional): estimates piano notes and sustain pedal. "
                  "Download the official 164 MiB checkpoint or import an existing copy. "
                  "The file is checked before use.", style="Subtitle.TLabel", wraplength=555,
                  justify="left").pack(anchor="w", pady=(0, 8))
        model_status = tk.StringVar(value="Pedal Piano checkpoint is stored in your local Omnizart models folder.")
        controls = ttk.Frame(frame, style="Root.TFrame")
        controls.pack(anchor="w")
        download = ttk.Button(controls, text="Download Pedal Piano", style="Secondary.TButton")
        import_button = ttk.Button(controls, text="Import checkpoint...", style="Secondary.TButton")
        download.pack(side="left", padx=(0, 8))
        import_button.pack(side="left")
        ttk.Label(frame, textvariable=model_status, style="Subtitle.TLabel", wraplength=555).pack(anchor="w", pady=(6, 10))

        def prepare_model(source=None):
            from omnizart.desktop.pedal_piano import install_checkpoint
            results = queue.Queue()
            download.configure(state="disabled")
            import_button.configure(state="disabled")
            model_status.set("Preparing model; please wait. The download can take several minutes.")

            def work():
                try:
                    results.put((True, str(install_checkpoint(source))))
                except Exception as exc:
                    results.put((False, str(exc)))

            def poll_model():
                try:
                    ok, result = results.get_nowait()
                except queue.Empty:
                    dialog.after(250, poll_model)
                    return
                download.configure(state="normal")
                import_button.configure(state="normal")
                if ok:
                    model_status.set("Pedal Piano is ready and selected.")
                    self.music_model.set("Pedal Piano")
                    self.mode_changed()
                else:
                    model_status.set("Model preparation failed: " + result)

            threading.Thread(target=work, daemon=True).start()
            dialog.after(250, poll_model)

        def import_pedal():
            path = filedialog.askopenfilename(parent=dialog, title="Choose the official Pedal Piano checkpoint",
                                              filetypes=[("PyTorch checkpoint", "*.pth"), ("All files", "*.*")])
            if path:
                prepare_model(path)

        download.configure(command=prepare_model)
        import_button.configure(command=import_pedal)
        ttk.Button(frame, text="Open Pedal Piano model page ↗", style="RootLink.TButton",
                   command=lambda: webbrowser.open("https://zenodo.org/records/4034264")).pack(anchor="w")
        ttk.Button(frame, text="Open Omnizart model downloads ↗", style="RootLink.TButton",
                   command=lambda: webbrowser.open("https://github.com/Music-and-Culture-Technology-Lab/omnizart/releases/tag/checkpoints-20211001")).pack(anchor="w")
        ttk.Button(frame, text="Open Basic Pitch project page ↗", style="RootLink.TButton",
                   command=lambda: webbrowser.open("https://github.com/spotify/basic-pitch")).pack(anchor="w")
        ttk.Button(frame, text="Open YourMT3 project page ↗", style="RootLink.TButton",
                   command=lambda: webbrowser.open("https://github.com/openmirlab/mt3-infer")).pack(anchor="w")
        ttk.Button(frame, text="Open Transkun project page ↗", style="RootLink.TButton",
                   command=lambda: webbrowser.open("https://github.com/Yujia-Yan/Transkun")).pack(anchor="w")

    def show_six_stem_model(self):
        dialog = tk.Toplevel(self.master)
        dialog.title("Six-stem model")
        dialog.configure(bg=BG)
        dialog.geometry("620x280")
        dialog.transient(self.master)
        frame = ttk.Frame(dialog, style="Root.TFrame", padding=24)
        frame.pack(fill="both", expand=True)
        ttk.Label(frame, text="Six-stem model", style="Title.TLabel").pack(anchor="w")
        ttk.Label(frame, text="Experimental HT-Demucs model: drums, bass, other, vocals, guitar and piano. "
                  "Download the 130 MiB model or import an existing copy. The file is verified before use.",
                  style="Subtitle.TLabel", wraplength=550, justify="left").pack(anchor="w", pady=(12, 12))
        state = tk.StringVar(value="The model is stored in your local Omnizart models folder.")
        controls = ttk.Frame(frame, style="Root.TFrame")
        controls.pack(anchor="w")
        download = ttk.Button(controls, text="Download six-stem model", style="Secondary.TButton")
        import_button = ttk.Button(controls, text="Import model...", style="Secondary.TButton")
        download.pack(side="left", padx=(0, 8))
        import_button.pack(side="left")
        ttk.Label(frame, textvariable=state, style="Subtitle.TLabel", wraplength=550).pack(anchor="w", pady=(10, 0))
        ttk.Button(frame, text="Open model page ↗", style="RootLink.TButton",
                   command=lambda: webbrowser.open("https://huggingface.co/StemSplitio/htdemucs-6s-onnx")).pack(anchor="w")

        def prepare(source=None):
            from omnizart.desktop.six_stems import install_model
            results = queue.Queue()
            download.configure(state="disabled")
            import_button.configure(state="disabled")
            state.set("Preparing model; please wait. Download can take several minutes.")
            def work():
                try:
                    results.put((True, str(install_model(source))))
                except Exception as exc:
                    results.put((False, str(exc)))
            def poll_model():
                try:
                    ok, result = results.get_nowait()
                except queue.Empty:
                    dialog.after(250, poll_model)
                    return
                download.configure(state="normal")
                import_button.configure(state="normal")
                state.set("Six-stem model is ready." if ok else "Model preparation failed: " + result)
            threading.Thread(target=work, daemon=True).start()
            dialog.after(250, poll_model)

        def import_file():
            path = filedialog.askopenfilename(parent=dialog, title="Choose the six-stem ONNX model",
                                              filetypes=[("ONNX model", "*.onnx"), ("All files", "*.*")])
            if path:
                prepare(path)
        download.configure(command=prepare)
        import_button.configure(command=import_file)

    def show_transkun_aug_model(self):
        dialog = tk.Toplevel(self.master)
        dialog.title("Transkun V2 Aug")
        dialog.configure(bg=BG)
        dialog.geometry("620x290")
        dialog.transient(self.master)
        frame = ttk.Frame(dialog, style="Root.TFrame", padding=24)
        frame.pack(fill="both", expand=True)
        ttk.Label(frame, text="Transkun V2 Aug", style="Title.TLabel").pack(anchor="w")
        ttk.Label(frame, text="An alternative piano checkpoint trained with augmentation. "
                  "Download the official 48 MiB archive or import an existing copy. "
                  "The archive and its files are verified before use.", style="Subtitle.TLabel",
                  wraplength=550, justify="left").pack(anchor="w", pady=(12, 12))
        state = tk.StringVar(value="The model is stored in your local Omnizart models folder.")
        controls = ttk.Frame(frame, style="Root.TFrame")
        controls.pack(anchor="w")
        download = ttk.Button(controls, text="Download Transkun V2 Aug", style="Secondary.TButton")
        import_button = ttk.Button(controls, text="Import model archive...", style="Secondary.TButton")
        download.pack(side="left", padx=(0, 8))
        import_button.pack(side="left")
        ttk.Label(frame, textvariable=state, style="Subtitle.TLabel", wraplength=550).pack(anchor="w", pady=(10, 0))
        ttk.Button(frame, text="Open model download page ↗", style="RootLink.TButton",
                   command=lambda: webbrowser.open("https://drive.google.com/file/d/1Hg5ua8vYdtg1Y-MnXD0mLyhRK9Srd7hm/view")).pack(anchor="w")

        def prepare(source=None):
            from omnizart.desktop.transkun_aug import install_model
            results = queue.Queue()
            download.configure(state="disabled")
            import_button.configure(state="disabled")
            state.set("Preparing model; please wait. Download can take several minutes.")
            def work():
                try:
                    results.put((True, str(install_model(source))))
                except Exception as exc:
                    results.put((False, str(exc)))
            def poll_model():
                try:
                    ok, result = results.get_nowait()
                except queue.Empty:
                    dialog.after(250, poll_model)
                    return
                download.configure(state="normal")
                import_button.configure(state="normal")
                state.set("Transkun V2 Aug is ready." if ok else "Model preparation failed: " + result)
            threading.Thread(target=work, daemon=True).start()
            dialog.after(250, poll_model)

        def import_file():
            path = filedialog.askopenfilename(parent=dialog, title="Choose the official Transkun V2 Aug archive",
                                              filetypes=[("Model archive", "*.zip"), ("All files", "*.*")])
            if path:
                prepare(path)
        download.configure(command=prepare)
        import_button.configure(command=import_file)

    def choose_comparison(self):
        path = filedialog.askopenfilename(parent=self.master, title="Choose a MIDI to compare", filetypes=[("MIDI", "*.mid *.midi"), ("All files", "*.*")])
        if path:
            self.compare_path.set(path)
            self.compare_enabled.set(True)

    def choose_musescore(self):
        path = filedialog.askopenfilename(parent=self.master, title="Choose MuseScore4.exe", filetypes=[("Executable", "*.exe"), ("All files", "*.*")])
        if path:
            self.musescore_path.set(path)
            self.musescore_export.set(True)
            self.pdf_export.set(True)
            try:
                remember_folders(musescore=path)
            except OSError:
                pass

    def show_separation_models(self, target='reduction'):
        dialog = tk.Toplevel(self.master)
        dialog.title("Separation models")
        dialog.configure(bg=BG)
        dialog.geometry("650x330")
        dialog.transient(self.master)
        frame = ttk.Frame(dialog, style="Root.TFrame", padding=24)
        frame.pack(fill="both", expand=True)
        ttk.Label(frame, text="Separation models", style="Title.TLabel").pack(anchor="w")
        ttk.Label(frame, text="Optional sherpa-onnx UVR models run on CPU. Download or import the selected "
                  "metadata-enabled model (about 60–65 MiB); the file is verified before use. "
                  "Different models can produce different results on the same recording.",
                  style="Subtitle.TLabel", wraplength=570, justify="left").pack(anchor="w", pady=(12, 12))
        labels = {"UVR Vocal FT (CPU)": "vocal-ft", "UVR Instrumental HQ4 (CPU)": "inst-hq4", "UVR Instrumental HQ5 (CPU)": "inst-hq5"}
        current = (self.vocal_track_model if target == 'track' else self.separator_model).get()
        selected = tk.StringVar(value=current if current in labels else next(iter(labels)))
        ttk.Combobox(frame, textvariable=selected, values=list(labels), state="readonly",
                     style="App.TCombobox").pack(anchor="w", fill="x", pady=(0, 8))
        state = tk.StringVar(value="The models are stored in your local Omnizart models folder.")
        controls = ttk.Frame(frame, style="Root.TFrame")
        controls.pack(anchor="w")
        download = ttk.Button(controls, text="Download selected model", style="Secondary.TButton")
        import_button = ttk.Button(controls, text="Import model...", style="Secondary.TButton")
        download.pack(side="left", padx=(0, 8))
        import_button.pack(side="left")
        ttk.Label(frame, textvariable=state, style="Subtitle.TLabel", wraplength=570).pack(anchor="w", pady=(10, 0))
        ttk.Button(frame, text="Open compatible model list ↗", style="RootLink.TButton",
                   command=lambda: webbrowser.open("https://k2-fsa.github.io/sherpa/onnx/source-separation/models.html#uvr")).pack(anchor="w")

        def prepare(source=None):
            from omnizart.desktop.sherpa_uvr import install_model
            label = selected.get()
            model = labels[label]
            results = queue.Queue()
            download.configure(state="disabled")
            import_button.configure(state="disabled")
            state.set("Preparing model; please wait. Download can take several minutes.")
            def work():
                try:
                    results.put((True, str(install_model(model, source))))
                except Exception as exc:
                    results.put((False, str(exc)))
            def poll_model():
                try:
                    ok, result = results.get_nowait()
                except queue.Empty:
                    dialog.after(250, poll_model)
                    return
                download.configure(state="normal")
                import_button.configure(state="normal")
                if ok:
                    (self.vocal_track_model if target == 'track' else self.separator_model).set(label)
                    self.update_separation_controls()
                    state.set(label + " is ready and selected.")
                else:
                    state.set("Model preparation failed: " + result)
            threading.Thread(target=work, daemon=True).start()
            dialog.after(250, poll_model)

        def import_file():
            path = filedialog.askopenfilename(parent=dialog, title="Choose the selected sherpa-onnx UVR model",
                                              filetypes=[("ONNX model", "*.onnx"), ("All files", "*.*")])
            if path:
                prepare(path)
        download.configure(command=prepare)
        import_button.configure(command=import_file)

    def import_model(self, dialog=None):
        path = filedialog.askdirectory(title="Choose an Omnizart music checkpoint folder", parent=dialog or self.master)
        if not path:
            return
        if not validate_model_folder(path):
            messagebox.showerror("Incomplete checkpoint", "Choose a complete Omnizart music checkpoint folder with "
                                 "configurations.yaml, saved_model.pb and the variables files.", parent=dialog or self.master)
            return
        self.custom_model_path = path
        try:
            remember_imported_model(path)
        except OSError:
            messagebox.showwarning("Model selected", "The model is available for this session, but its location "
                                   "could not be saved. You may need to select it again next time.", parent=dialog or self.master)
        label = imported_model_label(path)
        self.custom_models[label] = path
        current_values = list(self.model_picker.cget("values"))
        if label not in current_values:
            current_values.insert(current_values.index(CUSTOM_MODEL), label)
            self.model_picker.configure(values=current_values)
        self.music_model.set(label)
        self.mode_changed()
        if dialog is not None:
            dialog.destroy()

    def browse_output(self):
        current = self.destination.get().strip()
        path = filedialog.askdirectory(initialdir=current if current and Path(current).is_dir() else None)
        if path:
            self.destination.set(path)
            self.save_folders(output_folder=path)

    def choose_custom_stem_model(self):
        path = filedialog.askopenfilename(parent=self.master, title="Choose compatible six-stem ONNX model",
                                          filetypes=[("ONNX models", "*.onnx"), ("All files", "*.*")])
        if path:
            self.custom_stem_model.set(path)
            self.stem_model.set("Custom six-stem ONNX")
            try:
                remember_folders(custom_six_stem=path)
            except OSError:
                self.status.set("Model path could not be saved for next time.")

    def show_pymss_models(self, target='stem'):
        self.pymss_dialog_target = target
        from omnizart.desktop.optional_models import install, ready
        if not ready('pymss'):
            if self.optional_installing:
                return
            self.optional_installing = True
            self.status.set('Installing the optional pymss CPU runtime; this may take several minutes...')
            def install_pymss():
                try:
                    install('pymss')
                    self.messages.put(('pymss_install_done', ''))
                except Exception as exc:
                    self.messages.put(('pymss_install_done', str(exc)))
            threading.Thread(target=install_pymss, daemon=True).start()
            return
        from omnizart.desktop.pymss_backend import catalog
        try:
            entries = catalog()
        except (OSError, RuntimeError, ValueError) as exc:
            messagebox.showerror('pymss models', str(exc))
            return
        dialog = tk.Toplevel(self.master)
        dialog.title('pymss models')
        dialog.geometry('760x500')
        search = tk.StringVar()
        vocal_filter = tk.BooleanVar(value=target in ('vocal', 'track'))
        ttk.Label(dialog, text='Search models by name or category').pack(anchor='w', padx=12, pady=(12, 0))
        ttk.Entry(dialog, textvariable=search).pack(fill='x', padx=12, pady=6)
        source_row = ttk.Frame(dialog)
        source_row.pack(fill='x', padx=12, pady=(0, 6))
        ttk.Label(source_row, text='Download source').pack(side='left', padx=(0, 8))
        ttk.Combobox(source_row, textvariable=self.pymss_source, values=['huggingface', 'modelscope', 'hf-mirror'],
                     state='readonly', width=15).pack(side='left')
        if target in ('vocal', 'track'):
            ttk.Checkbutton(source_row, text='Show vocal models only', variable=vocal_filter).pack(side='left', padx=(16, 0))
        listing = tk.Listbox(dialog, exportselection=False)
        listing.pack(fill='both', expand=True, padx=12)
        visible = []
        def refresh(*_):
            visible.clear()
            listing.delete(0, 'end')
            query = search.get().casefold()
            for item in entries:
                name = str(item['name'])
                stem = str(item.get('target_stem', '')).casefold()
                if vocal_filter.get() and not any(word in stem for word in ('vocal', 'voice')):
                    continue
                detail = f"{name}  |  {item.get('primary_category', '')}  |  {item.get('target_stem', '')}"
                if query in detail.casefold():
                    visible.append(name)
                    listing.insert('end', detail)
        search.trace_add('write', refresh)
        vocal_filter.trace_add('write', refresh)
        refresh()
        def selected():
            selection = listing.curselection()
            return visible[selection[0]] if selection else None
        def choose():
            name = selected()
            if name:
                (self.vocal_track_pymss_model if target == 'track' else self.vocal_pymss_model if target == 'vocal' else self.pymss_model).set(name)
                if target == 'stem':
                    self.stem_model.set('pymss model')
                self.update_separation_controls()
                self.remember_pymss_choice()
                self.status.set(f'pymss model selected: {name}')
                dialog.destroy()
        def download():
            name = selected()
            if not name:
                return
            source = self.pymss_source.get()
            self.status.set(f'Downloading pymss model: {name}')
            def work():
                from omnizart.desktop.pymss_backend import download as download_model
                try:
                    download_model(name, source)
                    self.messages.put(('pymss_model_done', (name, '')))
                except Exception as exc:
                    self.messages.put(('pymss_model_done', (name, str(exc))))
            threading.Thread(target=work, daemon=True).start()
        buttons = ttk.Frame(dialog)
        buttons.pack(fill='x', padx=12, pady=12)
        ttk.Button(buttons, text='Use selected model', command=choose).pack(side='left')
        ttk.Button(buttons, text='Download selected', command=download).pack(side='left', padx=6)
        ttk.Button(buttons, text='Import custom model...', command=lambda: self.import_pymss_model(dialog, target)).pack(side='left')

    def import_pymss_model(self, parent, target='stem'):
        from tkinter import simpledialog
        from omnizart.desktop.pymss_backend import register
        weights = filedialog.askopenfilename(parent=parent, title='Choose pymss model weights')
        if not weights:
            return
        config = filedialog.askopenfilename(parent=parent, title='Choose the matching model YAML (cancel if not required)',
                                            filetypes=[('YAML', '*.yaml *.yml'), ('All files', '*.*')])
        name = simpledialog.askstring('Model name', 'Name for this model:', parent=parent, initialvalue=Path(weights).stem)
        if not name:
            return
        architecture = simpledialog.askstring('Model architecture', 'pymss architecture (for example bs_roformer, scnet, htdemucs or vr):', parent=parent)
        if not architecture:
            return
        try:
            register(name, architecture, weights, config or None)
        except (OSError, RuntimeError, ValueError) as exc:
            messagebox.showerror('Custom pymss model', str(exc), parent=parent)
            return
        (self.vocal_track_pymss_model if target == 'track' else self.vocal_pymss_model if target == 'vocal' else self.pymss_model).set(name)
        if target == 'stem':
            self.stem_model.set('pymss model')
        self.update_separation_controls()
        self.remember_pymss_choice()
        self.status.set(f'Custom pymss model registered: {name}')
        parent.destroy()

    def remember_pymss_choice(self):
        try:
            remember_folders(pymss_model=self.pymss_model.get(), pymss_source=self.pymss_source.get())
        except OSError:
            self.status.set('pymss model choice could not be saved for next time.')

    def save_folders(self, input_folder=None, output_folder=None):
        try:
            remember_folders(input_folder, output_folder)
        except OSError:
            self.status.set("Folders could not be saved for next time.")

    def start(self):
        source = Path(self.source.get()).expanduser()
        destination = Path(self.destination.get()).expanduser()
        if not self.source.get().strip() or not source.is_file():
            messagebox.showerror("Input file", "Choose an existing input file.")
            return
        if not self.destination.get().strip() or not destination.is_dir():
            messagebox.showerror("Output folder", "Choose an existing output folder.")
            return
        self.input_folder = str(source.parent.resolve())
        self.save_folders(input_folder=self.input_folder, output_folder=str(destination.resolve()))
        mode = MODES[self.mode.get()][0]
        if mode in ("beat", "hands") and source.suffix.lower() not in (".mid", ".midi"):
            messagebox.showerror("Input file", "This task requires a MIDI file.")
            return
        selected_model = self.music_model.get()
        piano_selected = (self.input_is_piano.get() or (self.piano_only.get() and self.stem_choice.get().strip().lower() == 'piano') or
                          selected_model in ('Piano', 'PianoV2', 'Transkun (piano)', 'Transkun V2 Aug', 'Pedal Piano'))
        final_hands = self.split_hands.get() or self.prefer_right_octave.get() or (piano_selected and (
            self.neural_staff.get() or self.prefer_right_octave.get() or self.align_bars.get() or self.simplify_score.get() or self.shorten_pedal_tails.get() or self.clarify_pedal_melody.get() or
            self.score_tempo.get().strip().lower() != 'auto' or self.score_key.get() != 'Auto' or
            self.score_meter.get() != 'Auto' or self.score_downbeat.get().strip().lower() != 'auto'))
        imported_path = self.custom_models.get(selected_model, self.custom_model_path if selected_model == CUSTOM_MODEL else None)
        if mode == "music" and self.input_is_piano.get() and (self.piano_only.get() or self.remove_vocals.get() or self.include_vocals.get()):
            messagebox.showerror("Piano input", "Already-isolated piano cannot be combined with separation or a vocal track. Turn those options off.")
            return
        if mode == 'music' and self.piano_only.get() and self.remove_vocals.get():
            messagebox.showerror('Separation options', 'Choose either vocal reduction or a single stem for instrument transcription.')
            return
        if mode == 'music' and self.include_vocals.get() and self.vocal_track_backend.get() == 'Same as selected stem' and not self.piano_only.get():
            messagebox.showerror('Vocal MIDI separator', 'Select a stem to transcribe, or choose another vocal MIDI separator.')
            return
        if mode == 'music' and self.piano_only.get() and self.stem_choice.get().strip().lower() in ('', 'all'):
            messagebox.showerror('Stem selection', 'Choose one stem to transcribe.')
            return
        if mode == "music" and selected_model in ("YourMT3", "MuScriptor"):
            from omnizart.desktop.optional_models import install, ready
            if not ready(selected_model):
                if self.optional_installing:
                    return
                self.optional_installing = True
                self.start_button.configure(state="disabled")
                self.status.set(f"Installing {selected_model} automatically. This may take several minutes...")

                def install_optional():
                    try:
                        install(selected_model)
                        self.messages.put(("optional_install_done", (selected_model, 0, "")))
                    except Exception as exc:
                        self.messages.put(("optional_install_done", (selected_model, 1, str(exc))))

                threading.Thread(target=install_optional, daemon=True).start()
                return
            if selected_model == "MuScriptor":
                from omnizart.desktop.muscriptor import model_checkpoint, model_folder
                if model_checkpoint() is None:
                    messagebox.showerror("MuScriptor model", f"Download the small model from its Hugging Face page and place exactly one .safetensors file and its config.json here:\n\n{model_folder()}")
                    return
        if ((mode == "music" and (self.musescore_export.get() or final_hands)) or mode == 'hands') and self.neural_staff.get():
            if mode == 'music' and not piano_selected:
                messagebox.showerror("Neural piano score", "Select a piano model, isolate piano, or mark the input as already piano only.")
                return
            from omnizart.desktop.optional_models import install, ready
            if not ready("Piano SVSep"):
                if self.optional_installing:
                    return
                self.optional_installing = True
                self.start_button.configure(state="disabled")
                self.status.set("Installing Piano SVSep and its model. This may take several minutes...")

                def install_score_model():
                    try:
                        install("Piano SVSep")
                        self.messages.put(("optional_install_done", ("Piano SVSep", 0, "")))
                    except Exception as exc:
                        self.messages.put(("optional_install_done", ("Piano SVSep", 1, str(exc))))

                threading.Thread(target=install_score_model, daemon=True).start()
                return
        if mode == "music" and (self.musescore_export.get() or final_hands) and self.align_bars.get():
            from omnizart.desktop.optional_models import install, ready
            if not ready("Beat This"):
                if self.optional_installing:
                    return
                self.optional_installing = True
                self.start_button.configure(state="disabled")
                self.status.set("Installing Beat This and its model for bar alignment. This may take several minutes...")

                def install_beat_model():
                    try:
                        install("Beat This")
                        self.messages.put(("optional_install_done", ("Beat This", 0, "")))
                    except Exception as exc:
                        self.messages.put(("optional_install_done", ("Beat This", 1, str(exc))))

                threading.Thread(target=install_beat_model, daemon=True).start()
                return
        if mode == "music" and (selected_model == CUSTOM_MODEL or selected_model in self.custom_models) and (
                not imported_path or not validate_model_folder(imported_path)):
            messagebox.showerror("Model", "Use Models & import to select a complete Omnizart music checkpoint.")
            return
        if mode == "music" and self.music_model.get() == "Pedal Piano":
            from omnizart.desktop.pedal_piano import checkpoint_path, CHECKPOINT_SIZE
            checkpoint = checkpoint_path()
            try:
                available = checkpoint.is_file() and checkpoint.stat().st_size == CHECKPOINT_SIZE
            except OSError:
                available = False
            if not available:
                messagebox.showerror("Pedal Piano model", "Open Models & import to download or import the Pedal Piano checkpoint.")
                return
        if mode == "music" and self.music_model.get() == "Transkun V2 Aug":
            from omnizart.desktop.transkun_aug import model_path, ARCHIVE_SIZE
            checkpoint = model_path()
            try:
                available = checkpoint.is_file() and checkpoint.stat().st_size == ARCHIVE_SIZE
            except OSError:
                available = False
            if not available:
                messagebox.showerror("Transkun V2 Aug", "Open Models & import to download or import the model archive.")
                return
        uses_stem_model = mode == 'stems' or (mode == 'music' and self.piano_only.get())
        if uses_stem_model and self.stem_model.get() == "pymss model":
            from omnizart.desktop.optional_models import ready
            if not ready('pymss') or not self.pymss_model.get().strip():
                messagebox.showerror('pymss model', 'Use Browse pymss models to install pymss and choose a model.')
                return
            if self.use_gpu.get() and not ready('pymss CUDA'):
                if self.optional_installing:
                    return
                from omnizart.desktop.optional_models import install
                self.optional_installing = True
                self.start_button.configure(state='disabled')
                self.status.set('Installing NVIDIA CUDA runtime for pymss; this may take several minutes...')
                def install_cuda():
                    try:
                        install('pymss CUDA')
                        self.messages.put(('optional_install_done', ('pymss CUDA', 0, '')))
                    except Exception as exc:
                        self.messages.put(('optional_install_done', ('pymss CUDA', 1, str(exc))))
                threading.Thread(target=install_cuda, daemon=True).start()
                return
        elif uses_stem_model and self.stem_model.get() == "Custom six-stem ONNX":
            custom = Path(self.custom_stem_model.get().strip())
            if not custom.is_file() or custom.suffix.lower() != '.onnx':
                messagebox.showerror("Custom six-stem model", "Choose an existing compatible six-stem ONNX file.")
                return
        elif uses_stem_model:
            from omnizart.desktop.six_stems import model_path, MODEL_SIZE, verify_model
            checkpoint = model_path()
            try:
                available = (verify_model(checkpoint) if mode == 'music' else
                             checkpoint.is_file() and checkpoint.stat().st_size == MODEL_SIZE)
            except OSError:
                available = False
            if not available:
                messagebox.showerror("Six-stem model", "Open Models & import to download or import the six-stem model.")
                return
        optional_separators = {"UVR Vocal FT (CPU)": "vocal-ft", "UVR Instrumental HQ4 (CPU)": "inst-hq4", "UVR Instrumental HQ5 (CPU)": "inst-hq5"}
        uses_reduction = mode == 'music' and (self.remove_vocals.get() or
            (self.include_vocals.get() and self.vocal_track_backend.get() == 'Same as vocal reduction'))
        uses_track_pymss = mode == 'music' and self.include_vocals.get() and self.vocal_track_backend.get() == 'pymss'
        if (uses_reduction and self.vocal_backend.get() == 'pymss') or uses_track_pymss:
            from omnizart.desktop.optional_models import install, ready
            missing_reduction = uses_reduction and self.vocal_backend.get() == 'pymss' and not self.vocal_pymss_model.get().strip()
            missing_track = uses_track_pymss and not self.vocal_track_pymss_model.get().strip()
            if not ready('pymss') or missing_reduction or missing_track:
                messagebox.showerror('PyMSS model', 'Choose a PyMSS model with Choose PyMSS model before starting.')
                return
            if self.use_gpu.get() and not ready('pymss CUDA'):
                if self.optional_installing:
                    return
                self.optional_installing = True
                self.start_button.configure(state='disabled')
                self.status.set('Installing NVIDIA CUDA runtime for PyMSS; this may take several minutes...')
                def install_vocal_pymss_cuda():
                    try:
                        install('pymss CUDA')
                        self.messages.put(('optional_install_done', ('pymss CUDA', 0, '')))
                    except Exception as exc:
                        self.messages.put(('optional_install_done', ('pymss CUDA', 1, str(exc))))
                threading.Thread(target=install_vocal_pymss_cuda, daemon=True).start()
                return
        builtin_models = []
        if uses_reduction and self.vocal_backend.get() == 'Built-in':
            builtin_models.append(self.separator_model.get())
        if mode == 'music' and self.include_vocals.get() and self.vocal_track_backend.get() == 'Built-in':
            builtin_models.append(self.vocal_track_model.get())
        for builtin_model in builtin_models:
            if builtin_model not in optional_separators:
                continue
            from omnizart.desktop.sherpa_uvr import model_path
            if not model_path(optional_separators[builtin_model]).is_file():
                messagebox.showerror("Separation model", "Use Manage separation models to download or import the selected model.")
                return
        try:
            values = {key: value.get() for key, value in self.thresholds.items()}
            if self.music_model.get() not in ("Pop", "Stream", "Basic Pitch", CUSTOM_MODEL) and self.music_model.get() not in self.custom_models:
                values["instrument"] = ""
            options = music_options(values, self.music_model.get()) if mode == "music" else []
            gpu_enabled = (self.use_gpu.get() and ((mode == "stems") or (mode == "music" and self.piano_only.get()) or
                           (uses_reduction and (self.vocal_backend.get() == 'pymss' or self.separator_model.get() == 'UVR MDX (recommended)')) or
                           (mode == 'music' and self.include_vocals.get() and
                            (self.vocal_track_backend.get() == 'pymss' or
                             (self.vocal_track_backend.get() == 'Built-in' and self.vocal_track_model.get() == 'UVR MDX (recommended)')))))
            gpu_index = self.gpu_index.get().strip()
            if gpu_enabled and (not gpu_index.isdecimal() or int(gpu_index) > 99):
                raise ValueError("GPU index must be a whole number between 0 and 99.")
            if mode == "music" and self.smooth_notes.get():
                try:
                    gap = float(self.smooth_gap.get())
                except ValueError as exc:
                    raise ValueError("Smoothing gap must be a number between 0 and 200 ms.") from exc
                if not math.isfinite(gap) or not 0 <= gap <= 200:
                    raise ValueError("Smoothing gap must be a number between 0 and 200 ms.")
            if mode == "music" and self.remove_short_notes.get():
                try:
                    minimum = float(self.minimum_short_note.get())
                except ValueError as exc:
                    raise ValueError("Minimum note length must be greater than 0 and at most 200 ms.") from exc
                if not math.isfinite(minimum) or not 0 < minimum <= 200:
                    raise ValueError("Minimum note length must be greater than 0 and at most 200 ms.")
            if mode == "music" and self.remove_quiet_notes.get():
                velocity = self.quiet_velocity.get().strip()
                if not velocity.isdecimal() or not 1 <= int(velocity) <= 127:
                    raise ValueError("Quiet-note velocity must be a whole number between 1 and 127.")
        except ValueError as exc:
            messagebox.showerror("Advanced settings", str(exc))
            return
        if mode == "music" and self.compare_enabled.get() and not Path(self.compare_path.get()).is_file():
            messagebox.showerror("MIDI comparison", "Choose an existing MIDI to compare first.")
            return
        score_tempo = self.score_tempo.get().strip()
        if mode == "music" and (self.musescore_export.get() or final_hands) and score_tempo.lower() != "auto":
            try:
                bpm = float(score_tempo)
                if not math.isfinite(bpm) or not 30 <= bpm <= 300:
                    raise ValueError()
            except ValueError:
                messagebox.showerror("Score tempo", "Enter Auto or a tempo between 30 and 300 BPM.")
                return
        score_downbeat = self.score_downbeat.get().strip()
        if mode == "music" and (self.musescore_export.get() or final_hands) and score_downbeat.lower() != "auto":
            try:
                value = float(score_downbeat)
                if not math.isfinite(value) or value < 0:
                    raise ValueError()
            except ValueError:
                messagebox.showerror("Score bar line", "Enter Auto or a non-negative time in seconds.")
                return
        if mode == "music" and self.musescore_export.get() and self.pdf_export.get() and not Path(self.musescore_path.get()).is_file():
            messagebox.showerror("MuseScore export", "The selected MuseScore executable no longer exists.")
            return
        target = destination / (source.stem + "_omnizart")
        try:
            target.mkdir(exist_ok=True)
            output_base, output_lock = reserve_result_name(target, source.stem)
        except OSError as exc:
            messagebox.showerror("Cannot prepare output", str(exc))
            return
        cmd = [sys.executable, "-u", "-m", "omnizart.desktop.worker", mode,
               "transcribe", str(source.resolve()), "--output", str(output_base.resolve())]
        if mode == "music":
            model = imported_path if imported_path is not None else self.music_model.get()
            cmd.extend(["--model-path", model, *options])
            if final_hands:
                cmd.append('--split-hands')
            if self.piano_only.get():
                selected_stem = self.stem_choice.get().strip().lower()
                if selected_stem == 'piano':
                    cmd.append('--piano-only')
                else:
                    cmd.extend(['--separate-stem', selected_stem])
                if self.save_separated_stem.get():
                    cmd.append('--save-separated-stem')
                if self.stem_model.get() == 'pymss model':
                    cmd.extend(['--piano-separator', 'pymss', '--pymss-model', self.pymss_model.get().strip(),
                                '--pymss-source', self.pymss_source.get()])
                elif self.stem_model.get() == 'Custom six-stem ONNX':
                    cmd.extend(['--custom-stem-model', self.custom_stem_model.get().strip()])
                if gpu_enabled:
                    cmd.extend(["--pymss-device", "cuda", "--gpu-index", gpu_index] if self.stem_model.get() == 'pymss model' else ["--uvr-device", "directml", "--gpu-index", gpu_index])
            if self.input_is_piano.get():
                cmd.append("--input-is-piano")
            if self.compare_enabled.get():
                cmd.extend(["--compare-midi", self.compare_path.get()])
            if self.musescore_export.get():
                cmd.append("--score")
            if self.musescore_export.get() or final_hands:
                if self.align_bars.get():
                    cmd.append("--align-bars")
                if score_downbeat.lower() != "auto":
                    cmd.extend(["--score-downbeat", score_downbeat])
                if self.score_meter.get() != "Auto":
                    cmd.extend(["--score-meter", self.score_meter.get()])
                if self.neural_staff.get():
                    cmd.append("--neural-staff")
                if self.prefer_right_octave.get():
                    cmd.append("--prefer-right-octave")
                    if self.move_left_chords.get():
                        cmd.append("--move-left-chords")
                if self.simplify_score.get():
                    cmd.append("--simplify-score")
                if self.score_key.get() != "Auto":
                    cmd.extend(["--score-key", self.score_key.get()])
                if score_tempo.lower() != "auto":
                    cmd.extend(["--score-tempo", score_tempo])
                if self.musescore_export.get() and self.pdf_export.get():
                    cmd.extend(["--musescore", self.musescore_path.get()])
            if self.lint_notes.get():
                cmd.append("--lint-notes")
            if self.smooth_notes.get():
                cmd.extend(["--smooth-notes", "--smooth-gap-ms", self.smooth_gap.get().strip()])
            if self.remove_short_notes.get():
                cmd.extend(["--remove-short-ms", self.minimum_short_note.get().strip()])
            if self.remove_quiet_notes.get():
                cmd.extend(["--remove-quiet-velocity", self.quiet_velocity.get().strip()])
            if self.shorten_pedal_tails.get():
                cmd.append("--shorten-pedal-tails")
            if self.clarify_pedal_melody.get():
                cmd.append("--clarify-pedal-melody")
            if self.include_vocals.get():
                cmd.append("--include-vocals")
            separator = None
            if uses_reduction:
                separator = ('pymss' if self.vocal_backend.get() == 'pymss' else
                             {"UVR MDX (recommended)": "uvr", "UVR Vocal FT (CPU)": "vocal-ft",
                              "UVR Instrumental HQ4 (CPU)": "inst-hq4", "UVR Instrumental HQ5 (CPU)": "inst-hq5", "Spleeter (legacy)": "spleeter"}[self.separator_model.get()])
                cmd.extend(["--separator", separator])
                if separator == 'pymss':
                    cmd.extend(['--reduction-pymss-model', self.vocal_pymss_model.get().strip()])
            track_separator = None
            if self.include_vocals.get() and self.vocal_track_backend.get() != 'Same as vocal reduction':
                track_separator = ('stem' if self.vocal_track_backend.get() == 'Same as selected stem' else
                    'pymss' if self.vocal_track_backend.get() == 'pymss' else
                    {"UVR MDX (recommended)": "uvr", "UVR Vocal FT (CPU)": "vocal-ft",
                     "UVR Instrumental HQ4 (CPU)": "inst-hq4", "UVR Instrumental HQ5 (CPU)": "inst-hq5", "Spleeter (legacy)": "spleeter"}[self.vocal_track_model.get()])
                cmd.extend(['--vocal-track-separator', track_separator])
                if track_separator == 'pymss':
                    cmd.extend(['--vocal-track-pymss-model', self.vocal_track_pymss_model.get().strip()])
            uses_pymss = (self.piano_only.get() and self.stem_model.get() == 'pymss model') or separator == 'pymss' or track_separator == 'pymss'
            if uses_pymss and '--pymss-source' not in cmd:
                cmd.extend(['--pymss-source', self.pymss_source.get()])
            if gpu_enabled:
                if uses_pymss and '--pymss-device' not in cmd:
                    cmd.extend(['--pymss-device', 'cuda'])
                if (separator == 'uvr' or track_separator == 'uvr') and '--uvr-device' not in cmd:
                    cmd.extend(['--uvr-device', 'directml'])
                if '--gpu-index' not in cmd:
                    cmd.extend(['--gpu-index', gpu_index])
            if self.remove_vocals.get():
                cmd.append("--remove-vocals")
                if self.keep_accompaniment.get():
                    cmd.append("--save-accompaniment")
        elif mode == "stems":
            if gpu_enabled:
                cmd.extend(["--device", "cuda" if self.stem_model.get() == 'pymss model' else "directml", "--gpu-index", gpu_index])
            if self.stem_model.get() == "Custom six-stem ONNX":
                cmd.extend(["--custom-model", self.custom_stem_model.get().strip()])
            elif self.stem_model.get() == "pymss model":
                cmd.extend(["--backend", "pymss", "--pymss-model", self.pymss_model.get().strip(),
                            '--pymss-source', self.pymss_source.get()])
            if self.stem_choice.get() != "All":
                cmd.extend(["--stem", self.stem_choice.get().lower()])
        elif mode == 'hands' and self.neural_staff.get():
            cmd.append('--neural-staff')
        self.active_stems = (None if mode == 'stems' and self.stem_model.get() == 'pymss model' and self.stem_choice.get() == 'All' else
                             (self.stem_choice.get().lower(),) if mode == "stems" and self.stem_choice.get() != "All"
                             else ("drums", "bass", "other", "vocals", "guitar", "piano"))
        self.score_warnings = []
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")
        self.start_button.configure(state="disabled")
        self.open_button.configure(state="disabled")
        self.running = True
        self.active_mode = mode
        self.output_folder = target
        self.output_base = output_base
        self.output_lock = output_lock
        self.status.set("Separating audio on your computer. This can take several minutes." if mode == "stems"
                        else "Transcribing on your computer. This can take several minutes.")
        self.progress.grid()
        self.progress.start(12)
        threading.Thread(target=self.run, args=(cmd, target), daemon=True).start()

    def run(self, cmd, target):
        try:
            flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            self.process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                            text=True, encoding="utf-8", errors="replace", creationflags=flags)
            for line in self.process.stdout:
                self.messages.put(("log", line))
            self.messages.put(("done", (self.process.wait(), target)))
        except Exception as exc:
            self.messages.put(("error", str(exc)))

    def poll(self):
        try:
            while True:
                kind, value = self.messages.get_nowait()
                if kind == "log":
                    if "UserWarning:" in value and any(text in value for text in
                            ("Piano hand spans", "Dense overlapping piano notes", "Percussion tracks are omitted", "Score simplification omitted", "Score simplification shortened", "Audio bar alignment", "Score key estimated")):
                        self.score_warnings.append(value.split("UserWarning:", 1)[1].strip())
                    if self.active_job is not None:
                        self.active_job['log'] += value
                    selection = self.queue_view.selection()
                    if self.active_job is None or not selection or selection[0] == self.active_job['id']:
                        self.log.configure(state="normal")
                        self.log.insert("end", value)
                        self.log.see("end")
                        self.log.configure(state="disabled")
                    continue
                if kind == "optional_install_done":
                    model, return_code, output = value
                    self.optional_installing = False
                    self.start_button.configure(state="normal")
                    if return_code == 0:
                        self.status.set(f"{model} is ready. Starting transcription...")
                        if self.active_job is not None:
                            self._start_active_job()
                        else:
                            self.start()
                    else:
                        self.status.set(f"Could not install {model}.")
                        messagebox.showerror(f"{model} setup failed", output or "Package installation failed.")
                        if self.active_job is not None:
                            self.active_job['log'] += output
                            self._update_job(self.active_job, 'Setup failed')
                            self.active_job = None
                            self.after_idle(self._start_next_job)
                    continue
                if kind == 'pymss_install_done':
                    self.optional_installing = False
                    if value:
                        self.status.set('Could not install pymss.')
                        messagebox.showerror('pymss setup failed', value)
                    else:
                        self.status.set('pymss is ready.')
                        self.show_pymss_models(getattr(self, 'pymss_dialog_target', 'stem'))
                    continue
                if kind == 'pymss_model_done':
                    name, error = value
                    if error:
                        self.status.set(f'Could not download {name}.')
                        messagebox.showerror('pymss model download failed', error)
                    else:
                        self.status.set(f'pymss model ready: {name}')
                    continue
                self.process = None
                if self.output_lock:
                    try:
                        self.output_lock.unlink(missing_ok=True)
                    except OSError:
                        pass
                    self.output_lock = None
                self.running = False
                self.progress.stop()
                self.progress.grid_remove()
                self.start_button.configure(state="normal")
                if self.output_folder and self.output_folder.is_dir():
                    self.open_button.configure(state="normal")
                if kind == "error":
                    self.status.set("Could not start transcription.")
                    messagebox.showerror("Transcription failed", value)
                elif value[0] == 0 and self.active_mode == "stems" and self.output_base and getattr(self, 'active_stems', None) is None:
                    if any(p.is_file() and p.name.casefold().startswith((self.output_base.name + '_').casefold()) and p.suffix.lower() == '.wav'
                           for p in self.output_base.parent.iterdir()):
                        self.status.set('Finished. Your results are ready.')
                    else:
                        self.status.set('No stem files were produced. Open details for more information.')
                elif value[0] == 0 and self.active_mode == "stems" and self.output_base and not all(
                        Path(str(self.output_base) + "_" + stem + ".wav").is_file()
                        for stem in getattr(self, "active_stems", ("drums", "bass", "other", "vocals", "guitar", "piano"))):
                    self.status.set("Some stem files are missing. Open details for more information.")
                elif value[0] == 0 and self.output_base and any(
                        Path(str(self.output_base) + suffix).is_file() for suffix in RESULT_SUFFIXES):
                    try:
                        notes = midi_note_count(value[1], self.output_base)
                    except (OSError, ValueError, EOFError):
                        self.status.set("Finished, but the MIDI could not be read. Open the results folder to inspect it.")
                    else:
                        if notes == 0:
                            self.status.set("No notes detected. Try Piano for solo piano, or adjust the thresholds.")
                        elif notes is None and self.active_mode not in ("vocal-contour", "stems"):
                            self.status.set("Finished without a MIDI file. Open details to inspect the result.")
                        elif notes is None:
                            self.status.set("Finished. Your results are ready.")
                        else:
                            self.status.set(f"Finished. {notes:,} notes detected.")
                elif value[0] == 0:
                    self.status.set("No output was created. Open details for more information.")
                else:
                    self.status.set("Transcription failed. Open details for more information.")
                    if not self.log_open:
                        self.toggle_log()
                if kind == "done" and value[0] == 0 and self.score_warnings and self.active_job is None:
                    messagebox.showwarning("Score needs review", "\n\n".join(self.score_warnings))
                if self.active_job is not None:
                    job = self.active_job
                    if kind == 'error':
                        job['log'] += str(value) + '\n'
                    if self.score_warnings:
                        job['log'] += '\nScore review:\n' + '\n'.join(self.score_warnings) + '\n'
                    good = kind == 'done' and value[0] == 0 and self.status.get().startswith('Finished.')
                    timing_review = good and any('Audio bar alignment unavailable' in warning or
                                                 'Audio bar alignment corrected' in warning for warning in self.score_warnings)
                    if timing_review:
                        self.status.set('Finished. Timing was estimated; review the bar alignment in details.')
                    self._update_job(job, 'Review timing' if timing_review else 'Finished' if good else 'Needs attention')
                    self.active_job = None
                    self._show_selected_job()
                self.after_idle(self._start_next_job)
        except queue.Empty:
            pass
        self.after(100, self.poll)

    def open_results(self):
        selection = self.queue_view.selection()
        selected = next((job for job in self.jobs if selection and job['id'] == selection[0]), None)
        folder = selected['output'] if selected else self.output_folder
        if folder and folder.is_dir():
            try:
                os.startfile(str(folder))
            except OSError as exc:
                messagebox.showerror("Cannot open results", str(exc))

    def close(self):
        if self.running or self.optional_installing:
            messagebox.showinfo("Transcription running", "Wait for transcription to finish before closing.")
            return
        if any(job['state'] == 'Waiting' for job in self.jobs) and not messagebox.askyesno(
                'Jobs waiting', 'Close Omnizart and discard the waiting jobs?'):
            return
        self.master.destroy()


def main():
    from omnizart.desktop.dpi import enable_windows_dpi_awareness, scale_tk_for_window
    enable_windows_dpi_awareness()
    root = tk.Tk()
    scale_tk_for_window(root)

    def report_callback_exception(exc_type, exc_value, exc_traceback):
        details = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
        messagebox.showerror("Omnizart error", "An unexpected error occurred:\n\n" + details[-4000:], parent=root)

    root.report_callback_exception = report_callback_exception
    Application(root)
    root.mainloop()


if __name__ == "__main__":
    main()
