$c = Get-Content "LLaVA_ScienceQA_RTX5080.ipynb" -Raw

# Fix 3: Add os.makedirs("results") in Cell 8
$old = 'os.makedirs(OUTPUT_DIR, exist_ok=True)'
$new = 'os.makedirs(OUTPUT_DIR, exist_ok=True)\nos.makedirs(\"results\", exist_ok=True)'
$c = $c.Replace($old, $new)

# Fix 5 & 7: process_images list guard (Cell 5)
$old5 = '        img_t = process_images([img], image_processor, model.config)\n        img_t = img_t.to(model.device, dtype=torch.float16)'
$new5 = '        img_t = process_images([img], image_processor, model.config)\n        if isinstance(img_t, list):\n            img_t = img_t[0]\n        img_t = img_t.to(model.device, dtype=torch.float16)'
$c = $c.Replace($old5, $new5)

# Fix 5 & 7: process_images list guard (Cell 9 — indented deeper)
$old9 = '            img_t = process_images([img], image_processor, model.config)\n            img_t = img_t.to(model.device, dtype=torch.float16)'
$new9 = '            img_t = process_images([img], image_processor, model.config)\n            if isinstance(img_t, list):\n                img_t = img_t[0]\n            img_t = img_t.to(model.device, dtype=torch.float16)'
$c = $c.Replace($old9, $new9)

$c | Set-Content "LLaVA_ScienceQA_RTX5080.ipynb" -Encoding UTF8 -NoNewline
Write-Host "Done: fixes 3, 5, 7 applied"
