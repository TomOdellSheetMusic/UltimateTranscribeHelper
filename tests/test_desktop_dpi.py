import ctypes
from types import SimpleNamespace
from unittest.mock import Mock, patch

from omnizart.desktop import dpi


def test_windows_dpi_awareness_and_initial_tk_scaling():
    user32 = SimpleNamespace(SetProcessDpiAwarenessContext=Mock(return_value=1),
                             GetDpiForWindow=Mock(return_value=120))
    root = SimpleNamespace(winfo_id=lambda: 123, tk=SimpleNamespace(call=Mock()))
    with patch.object(dpi.sys, 'platform', 'win32'), patch.object(ctypes, 'windll', SimpleNamespace(user32=user32)):
        dpi.enable_windows_dpi_awareness()
        dpi.scale_tk_for_window(root)
    user32.SetProcessDpiAwarenessContext.assert_called_once()
    root.tk.call.assert_called_once_with('tk', 'scaling', 120 / 72)
