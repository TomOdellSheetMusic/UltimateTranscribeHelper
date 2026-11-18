"""Configure native Windows scaling before creating the Tk interface."""
import ctypes
import sys


def enable_windows_dpi_awareness():
    """Ask Windows to render the process at the monitor's native resolution."""
    if sys.platform != 'win32':
        return
    try:
        # Per-monitor v2; this must run before the first Tk window is created.
        if ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4)):
            return
    except (AttributeError, OSError):
        pass
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except (AttributeError, OSError):
        pass


def scale_tk_for_window(root):
    """Set Tk's point scaling to the initial monitor's effective Windows DPI."""
    if sys.platform != 'win32':
        return
    try:
        dpi = ctypes.windll.user32.GetDpiForWindow(ctypes.c_void_p(root.winfo_id()))
        if dpi:
            root.tk.call('tk', 'scaling', dpi / 72.0)
    except (AttributeError, OSError):
        pass
