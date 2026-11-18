"""Input and output folders are saved independently across sessions."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from omnizart.desktop.app import read_folders, remember_folders


class FolderSettingsTests(unittest.TestCase):
    def test_pymss_selection_is_preserved_with_folders(self):
        with TemporaryDirectory() as temp:
            settings = Path(temp) / 'folders.json'
            with patch('omnizart.desktop.app.folder_settings_path', return_value=settings):
                remember_folders(pymss_model='example-model', pymss_source='modelscope')
                remember_folders(output_folder='D:/output')
                self.assertEqual(read_folders(), {'pymss_model': 'example-model',
                                                  'pymss_source': 'modelscope', 'output': 'D:/output'})

    def test_updating_each_folder_preserves_the_other(self):
        with TemporaryDirectory() as temp:
            settings = Path(temp) / "folders.json"
            with patch("omnizart.desktop.app.folder_settings_path", return_value=settings):
                remember_folders(input_folder="F:/music")
                remember_folders(output_folder="D:/midi")
                remember_folders(input_folder="G:/other")
                self.assertEqual(read_folders(), {"input": "G:/other", "output": "D:/midi"})

    def test_invalid_settings_do_not_prevent_saving(self):
        with TemporaryDirectory() as temp:
            settings = Path(temp) / "folders.json"
            settings.write_text("{broken", encoding="utf-8")
            with patch("omnizart.desktop.app.folder_settings_path", return_value=settings):
                self.assertEqual(read_folders(), {})
                remember_folders(output_folder="D:/midi")
                self.assertEqual(json.loads(settings.read_text(encoding="utf-8")), {"output": "D:/midi"})
