import subprocess
import sysconfig
import unittest
from pathlib import Path


class ConsoleScriptsTest(unittest.TestCase):
    def test_installed_commands_show_help(self) -> None:
        scripts_dir = Path(sysconfig.get_path("scripts"))
        commands = (
            "pgscen",
            "pgscen-load",
            "pgscen-wind",
            "pgscen-solar",
            "pgscen-load-solar",
            "pgscen-ny",
            "pgscen-ny-load",
            "pgscen-ny-wind",
            "pgscen-ny-solar",
            "pgscen-rts",
        )

        for command in commands:
            with self.subTest(command=command):
                script = scripts_dir / command
                self.assertTrue(script.is_file(), f"Missing installed script: {script}")
                result = subprocess.run(
                    [str(script), "--help"],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn(f"usage: {command} ", result.stdout)
