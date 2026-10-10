# Outcome of every [VERIFY] item

Checked on 2026-10-10 against the installed packages (Python 3.11.15; torch 2.14.1+cpu, timm 1.0.30,
grad-cam 1.5.7, shap 0.51.0, lime 0.2.0.1). "Open" items need the downloaded data or a model run.

## Libraries

| # | Item (DESIGN.md) | Outcome | Status |
|---|---|---|---|
| V1 | D1, F5, Sec. 8.1: `maxvit_tiny_tf_512` exists in the installed timm | `timm.list_models("maxvit_tiny_tf_512")` returns it. Pretrained tag: `maxvit_tiny_tf_512.in1k` (ImageNet-1k). 30.54 M backbone parameters, `num_features` = 512. | Verified |
| V2 | Sec. 7.10: `img_size` is accepted by `timm.create_model` | Accepted for `maxvit_tiny_tf_512`. **Rejected for `convnext_tiny`** (`TypeError: unexpected keyword argument 'img_size'`); ConvNeXt is size-agnostic, so `backbone.py` must pass `img_size` only to MaxViT. `convnext_tiny` `num_features` = 768. | Verified; affects M1 |
| V3 | F3, Sec. 7.17: `pytorch-grad-cam` class name | `pytorch_grad_cam.GradCAMPlusPlus(model, target_layers, reshape_transform=None)` exists (PyPI package name `grad-cam`). | Verified |
| V4 | Sec. 7.17: Grad-CAM++ works with timm MaxViT and ConvNeXt target layers | MaxViT's last block is `stages[-1].blocks[-1]` (`MaxxVitBlock`: `conv`, `attn_block`, `attn_grid`); ConvNeXt's is `stages[-1]`. Running the CAM on both, and whether MaxViT needs a `reshape_transform`, is checked when `gradcam_pp.py` is written. | Open (M2) |
| V5 | F3: does Captum implement Grad-CAM++? | Captum is not installed in this build (it is not used). Its documented API has `LayerGradCam` (original Grad-CAM) and `GuidedGradCam`, and no Grad-CAM++ class, which agrees with F3. The Researcher should cite `pytorch-grad-cam` and confirm against the Captum documentation before changing the paper. | Not re-checked on an installed copy |
| V6 | Sec. 7.18: shap image-masker API | `shap.maskers.Image(mask_value, shape=None)`, `shap.PartitionExplainer` and `shap.KernelExplainer` all exist in 0.51.0. Whether `"inpaint_telea"` runs with `opencv-python-headless` 5.0 is checked when `shap_explainer.py` is written. | Partly verified (M2) |
| V7 | Sec. 7.19: LIME API | `LimeImageExplainer.explain_instance(image, classifier_fn, labels, hide_color, top_labels, num_features, num_samples, batch_size, segmentation_fn, ..., random_seed)`. The seed is passed as `random_seed` here and as `random_state` to the `LimeImageExplainer` constructor. | Verified |
| V8 | Sec. 7.20: attention weights in timm MaxViT | The final attention module is `stages[-1].blocks[-1].attn_grid.attn` (`AttentionCl`) and it has `fused_attn = True`, so the weights are not materialised by default. Extraction means setting `fused_attn = False` and hooking `attn_drop`, as the design anticipated. Time-boxed to one day in M2. | Open (M2) |

## Data

None of the three datasets is on disk yet, so every item below is open. The adapters read these
names from the config and stop with a message naming the difference instead of guessing.

| # | Item (DESIGN.md) | What the code currently assumes | Status |
|---|---|---|---|
| V9 | Sec. 7.4: IDRiD image file names | Any image file under `grading/images` and `segmentation/images`, matched by stem. | Open |
| V10 | Sec. 7.4: IDRiD mask naming | `{image stem}_{MA|HE|EX|SE|OD}.tif`, anywhere under `segmentation/masks`. Masks are binarised as "any non-zero pixel". | Open |
| V11 | Sec. 7.4: IDRiD grading CSV columns | `Image name`, `Retinopathy grade` (header whitespace is stripped). | Open |
| V12 | Sec. 7.4: overlap between IDRiD segmentation and grading sets | Detected by SHA-256 of the image files; written to `data/manifests/idrid_overlap.csv`. Re-encoded copies would be missed. | Open |
| V13 | F1, Sec. 13: lesion counts 81 / 80 / 81 / 40 (MA / HE / EX / SE) | Logged by `prepare-data`; to be compared with F1. | Open |
| V14 | F8, Sec. 7.4: Messidor-2 label columns and grade scheme | Config defaults `image_id`, `adjudicated_dr_grade`; `grade_map` and `detection_map` are empty and the adapter refuses to run until they are declared. | Open, needs Researcher (F8) |
| V15 | F17: licence of the ten supplied sample photos | Unknown. They are used locally as smoke-test fixtures and are git-ignored. One of the eleven attached files was a byte-identical duplicate. | Open |
