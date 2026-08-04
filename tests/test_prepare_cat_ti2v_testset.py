import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from scripts.prepare_cat_ti2v_testset import ACTION_PROMPT, DEFAULT_METADATA, prepare
from utils.dataset import ImagePromptDataset


class PrepareCatTI2VTestsetTest(unittest.TestCase):
    def test_prepared_fixture_matches_bf16_i2v_input_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "fixture"
            manifest = prepare(DEFAULT_METADATA, output_dir)

            image_path = output_dir / "images" / "russian_forest_jump_wand_yarn.png"
            prompt_path = output_dir / "prompts" / "russian_forest_jump_wand_yarn.txt"
            with Image.open(image_path) as image:
                self.assertEqual(image.mode, "RGB")
                self.assertEqual(image.size, (1280, 704))

            self.assertEqual(prompt_path.read_text(encoding="utf-8").splitlines(), [ACTION_PROMPT])
            self.assertEqual(manifest["decoded_frames"], 253)
            self.assertEqual(manifest["num_blocks"], 8)

            dataset = ImagePromptDataset(
                data_path=str(output_dir),
                image_size=(704, 1280),
                num_blocks=8,
            )
            sample = dataset[0]
            self.assertEqual(tuple(sample["image"].shape), (3, 704, 1280))
            self.assertEqual(len(sample["prompts"]), 8)
            self.assertEqual(set(sample["prompts"]), {ACTION_PROMPT})

            manifest_on_disk = json.loads(
                (output_dir / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest_on_disk["actions"], [
                "forward_jump",
                "teaser_wand",
                "yarn_ball",
            ])


if __name__ == "__main__":
    unittest.main()
