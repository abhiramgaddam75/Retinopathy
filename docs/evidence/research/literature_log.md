# Researcher evidence log

Checked on 2026-10-11. Every entry is a page or record that was actually opened in this session.
"Paper" means the team's draft `Explainable AI_.docx` (read in full, 328 lines of extracted text).
Publisher pages for Springer and Nature redirect to a login step for automated readers, so those
items were read through Crossref, Europe PMC and PubMed Central instead. Quotes are kept short.

## A. Sources opened

| # | Item | URL | What it shows |
|---|---|---|---|
| L1 | Chilukoti et al. [48], record and abstract | https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=DOI:%2210.1186/s12911-024-02446-x%22&resultType=core&format=json | BMC Med Inform Decis Mak 24, 37 (2024), PMC11323616. Abstract QWK: EyePACS 0.901, APTOS 0.967, Messidor 0.944 |
| L2 | Chilukoti et al. [48], full text | https://pmc.ncbi.nlm.nih.gov/articles/PMC11323616/ | Table 6: 0.901 EyePACS (no DR pretraining, ensemble); 0.967 APTOS and 0.944 Messidor-2 (both pretrained on EyePACS, then fine-tuned, ensemble); 0.854 is Messidor-2 without DR pretraining and without ensemble. No model is tested on a dataset it was not fine-tuned on. APTOS and Messidor-2 splits are not reported; the paper does not say whether Table 6 is validation or test |
| L3 | Akhtar et al. [49], record and abstract | https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=DOI:%2210.1038/s41598-025-87171-9%22&resultType=core&format=json | Sci Rep 15, 3763 (2025), PMC11782675. "RSG-Net", Messidor-1 only. Own results: 4-class accuracy 99.36%, binary 99.37%. No kappa |
| L4 | Akhtar et al. [49], full text | https://pmc.ncbi.nlm.nih.gov/articles/PMC11782675/ | 98.50%, 99.46%, 89.60% and QWK 93.00% appear only in the related-work section, describing Shakibania et al. APTOS 2019 is not used. Augmentation precedes the split: "the augmented dataset is partitioned" |
| L5 | Shakibania et al. (the paper's own ref [27]), search listing | search S3 below | "Dual Branch Deep Learning Network for Detection and Stage Grading of Diabetic Retinopathy", APTOS 2019: binary accuracy 98.50, sensitivity 99.46; grading QWK 93.00, accuracy 89.60 |
| L6 | Van Craenendonck et al. [51], full text | https://pmc.ncbi.nlm.nih.gov/articles/PMC7774113/ | TVST 9(2), 64 (2020). "ECS = 0.51; 95% confidence interval [CI]: [0.46; 0.55]". ECS compares top-15 cells of a 10x10 grid; it is neither IoU nor Dice. 81 IDRiD images, 54 used for the main results. Models trained on EyePACS. Ten heatmap methods; Grad-CAM++ is not one of them |
| L7 | Captum layer attribution API | https://captum.ai/api/layer.html | The only CAM class in the first 100,000 characters is `LayerGradCam`: "Computes GradCAM attribution for chosen layer", citing arXiv 1610.02391 (original Grad-CAM). No Grad-CAM++ class seen. The last 23,798 characters of the page were not read |
| L8 | DOI of ref [4] | https://api.crossref.org/works/10.1038/s41598-026-58292-6 | Tashrif, Kundu, Bithee, Rahman, Farid, Karim, Miah. Scientific Reports, published 2026-06-23. This DOI belongs to ref [4] only |
| L9 | Ref [5], correct record | https://api.crossref.org/works?query.bibliographic=Synthetic+data-driven+diabetic+retinopathy+diagnosis+explainable+AI+clinically+interpretable+framework+Vasireddi&rows=5&select=DOI,title,container-title,author,published | Vasireddi, Gedela, Dasari, Devarapalli. Graefe's Archive for Clinical and Experimental Ophthalmology, 2026-05-22. DOI **10.1007/s00417-026-07284-3** |
| L10 | Refs [1], [3], [21], [22], [23], [24], [37] | https://api.crossref.org/works?filter=doi:10.1038/s41598-026-56569-4,doi:10.3390/diagnostics13193116,doi:10.18240/ijo.2023.09.05,doi:10.1038/s41598-025-34016-0,doi:10.3390/diagnostics13101816,doi:10.3390/diagnostics13030345,doi:10.1109/ACCESS.2024.3405570&rows=10&select=DOI,title,container-title,author,published,volume,issue,page,article-number | See table C |
| L11 | Dinpajhouh & Seyyedsalehi [50], record | https://api.crossref.org/works/10.1007/s00521-023-09001-1 | Neural Computing and Applications 35(33), 23959-23971 (2023). Bibliographic details match. No abstract in the record |
| L12 | arXiv 2608.28207 | https://arxiv.org/abs/2608.28207 | Verma et al., Aug 2026. Grad-CAM and HiResCAM scored against IDRiD lesion masks with Dice, IoU and pointing game; trained on ODIR, "externally evaluated on APTOS". No joint detection and grading |
| L13 | arXiv 2106.13497 | https://arxiv.org/abs/2106.13497 | Srinivasan et al., 2021. DR grading, robustness and interpretability. The abstract does not state lesion-level metrics |

## B. Searches

| # | Query | Mode | Result |
|---|---|---|---|
| S1 | Van Craenendonck Elen Gerrits De Boever "Explainability Consistency Score" heatmapping techniques diabetic retinopathy lesion detection VGG16 Grad-CAM 0.51 | standard | 9 results; led to L6 |
| S2 | Vasireddi Gedela Dasari Devarapalli "Synthetic data-driven diabetic retinopathy diagnosis with explainable AI" clinically interpretable framework Graefe's Archive DOI | standard | 9 results, none matching; resolved through Crossref (L9) |
| S3 | Dinpajhouh Seyyedsalehi "Automated detecting and severity grading of diabetic retinopathy using transfer learning and attention mechanism" abstract accuracy AUC quadratic weighted kappa APTOS | standard | 10 results; the target paper's abstract was not found; surfaced Shakibania et al. with 98.50 / 99.46 / 93.00 / 89.60 on APTOS 2019 (L5) |
| S4 | Dinpajhouh Seyyedsalehi 2023 Neural Computing and Applications diabetic retinopathy transfer learning attention mechanism APTOS "98.44" "83.69" | standard | 9 results; **no source found containing 98.44 or 83.69** |
| S5 | Messidor-2 adjudicated DR grades Krause Gulshan "messidor2" labels adjudicated_dr_grade 0-4 ICDR 1744 gradable images | standard | 10 results. Messidor-2 ships without labels; the widely used grades are the Krause et al. (2018) adjudicated grades on a five-level scale, with an `adjudicated_gradable` flag; 4 images ungradable, 1,744 usable. Exact file encoding to be checked on the downloaded file |
| S6 | APTOS 2019 blindness detection duplicate images training set different labels md5 hash | standard | 10 results; nothing authoritative |
| S7 | APTOS 2019 blindness detection kaggle discussion duplicate images train set "duplicates" different diagnosis labels | extended | Kaggle discussion https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/100489: participants discuss removing duplicates; one commenter counts 262. Several published solutions removed duplicates. Not an official count; conflicting labels not confirmed |
| S8 | IDRiD dataset Porwal 516 images 81 pixel-level annotations ... | standard | 9 results. Confirmed: 516 images (413 train, 103 test), one clinic in Nanded, Kowa VX-10 alpha, 4288x2848; "81 images" chosen for pixel-level annotation. Per-lesion counts and images-per-patient **not confirmed** (the data descriptor at mdpi.com returned HTTP 403) |
| S9 | Alwakid Gouda Humayun Jhanjhi "Enhancing diabetic retinopathy classification using deep learning" Digital Health retraction notice 10.1177/20552076231203676 | standard | Retraction notice found on a mirror: retracted "for reasons of redundant publication", overlapping with the Diagnostics paper (the paper's ref [19]). To be confirmed on journals.sagepub.com |
| G1 | diabetic retinopathy multi-task joint detection grading Grad-CAM quantitative IoU IDRiD lesion masks cross-dataset Messidor-2 explainability evaluation | standard | 9 candidates. Partial matches: Foo et al. AAAI 2020 (multi-task grading + lesion segmentation), Zhou et al. M2CNN. None meets all three criteria |
| G2 | diabetic retinopathy saliency maps quantitative evaluation lesion annotations pointing game IoU IDRiD cross-dataset generalization explainable deep learning grading | standard | 19 candidates. arXiv 2608.28207 (L12) meets two of three criteria; arXiv 2106.13497 (L13) partial |
| G3 | "diabetic retinopathy" "multi-task" "joint" detection and grading "explainable" "cross-dataset" lesion localization quantitative attribution evaluation 2024 2025 | standard | 10 candidates. Foo et al. 2020; Lesion-Net (Wei et al.). None meets all three |

Gap search limits: three web searches, English only, no database export (Scopus, PubMed, IEEE Xplore
were not queried). This is not a systematic search.

## C. Reference checks (Crossref, L8-L11)

| Ref | In the paper | Found | Action |
|---|---|---|---|
| [1] | Sci Rep 16, 3780 (2026) | Title, authors, volume 16, article 3780 match; Crossref gives the publication date as 2025 | Check the year on the journal page |
| [3] | No volume or article number | Scientific Reports 16, article 27354 (2026) | Add |
| [4] | DOI 10.1038/s41598-026-58292-6 | Correct for this paper; no volume or article number in the record yet | Add when assigned |
| [5] | Graefe's Archive, DOI 10.1038/s41598-026-58292-6 | Venue correct; **DOI wrong**. Correct DOI 10.1007/s00417-026-07284-3 | Replace DOI |
| [21] | No title | "RDS-DR: An Improved Deep Learning Model for Classifying Severity Levels of Diabetic Retinopathy" | Add title |
| [22] | No title | "A Hybrid Technique for Diabetic Retinopathy Detection Based on Ensemble-Optimized CNN and Texture Features" | Add title |
| [23] | No title; described in Sec. 2.3 as a classification study | "A Survey on Deep-Learning-Based Diabetic Retinopathy Classification": it is a survey | Add title; fix the sentence |
| [24] | No title, volume or pages | "Enhancing Early Detection of Diabetic Retinopathy Through the Integration of Deep Learning Models and Explainable Artificial Intelligence", IEEE Access 12, 73950-73969 | Add |
| [37] | No pages | Int J Ophthalmol 16(9), 1395-1405 (2023) | Add pages |
| [50] | NCA 35(33), 23959-23971 (2023) | Matches | none |
| [25]-[28] | No titles | Not checked | Add titles |
| [43], [46], [47] | Method papers | Not re-opened: judged from the citation text only (see review F-012) | see review |

## D. Targeted searches in the supplied papers (extracted text, pattern search only)

| Paper | Pattern | Finding |
|---|---|---|
| Alwakid et al. 2023, Diagnostics 13, 2375 (ref [19]) | augment, split, 80%, APTOS | APTOS 2019 has 3,662 images, yet "80% was used for training (9360 photos)". 9,360 is 80% of 11,700, so the split was made on the augmented set. Headline test accuracy 98.36% |
| Alavee et al. 2024, IEEE Access (ref [24]) | Grad-CAM, IoU, Dice, Messidor, IDRiD | Grad-CAM is shown on "random images from the test set" and used for "qualitative model analysis". No IoU, Dice or pointing metric appears. External accuracy is reported on Messidor-2 (93.86%) and IDRiD (91.18%) taken from Kaggle re-uploads |
| Bashir et al. 2023, Diagnostics 13, 3116 (ref [21]) | dataset, APTOS, split | 9,860 images merged from DR-Insight (5,000, private), APTOS (2,690), DiaRet0/1 (200 each) and PAK-HR (1,770, private); 70/30 split. Its test data therefore contain APTOS images |
| Ishtiaq et al. 2023, Diagnostics 13, 1816 (ref [22]) | dataset, EyePACS, APTOS | Kaggle EyePACS only; APTOS and IDRiD are not used |
