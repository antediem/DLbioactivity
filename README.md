Attention-based Multi-input Neural network
=============
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/attention-based-multi-input-deep-learning/drug-discovery-on-egfr-inh)](https://paperswithcode.com/sota/drug-discovery-on-egfr-inh?p=attention-based-multi-input-deep-learning)

<img src="https://i.ibb.co/jg5kzd5/egfr-architecture-new.jpg" width="700">

## How to install

Using `conda`:
```bash
conda env create -n egfr -f environment.yml
conda activate egfr
```

## Usage

The working folder is `egfr-att/egfr` for the below instruction.

#### To train with Train/Test scheme, use:
```bash
python single_run.py --mode train
```
The original data will be splitted into training/test parts with ratio 8:2. 
When training completed, to evaluate on test data, use:
```bash
python single_run.py --mode test --model_path <MODEL-IN-TRAINED_MODELS-FOLDER>
# For example:
python single_run.py --mode test --model_path data/trained_models/model_TEST_BEST
```
ROC curve plot for test data will be placed in egfr/vis folder.

#### To train with 5-fold cross validation scheme, use:
```bash
python cross_val.py --mode train
``` 
When training completed, to evaluate on test data, use:
```bash
python cross_val.py --mode test --model_path <MODEL-IN-TRAINED_MODELS-FOLDER>
# For example:
python cross_val.py --mode test --model_path data/trained_models/model_TEST_BEST
```
ROC curve plot for test data will be placed in `egfr/vis/` folder.

#### Attention weight visualization
To visualized attention weight of the model, use:
```bash
python weight_vis.py --dataset <PATH-TO-DATASET> --modelpath <PATH-TO-MODEL>
# For example:
python weight_vis.py --dataset data/egfr_10_full_ft_pd_lines.json --modelpath data/trained_models/model_TEST_BEST
```
By default, all data will be used to to extract attention weights. However, 
only samples with prediction output over a threshold (0.2) are chosen.

## Citation
Please cite our study:
```
Pham, H.N., & Le, T.H. (2019). Attention-based Multi-Input Deep Learning Architecture for Biological Activity Prediction: An Application in EGFR Inhibitors. ArXiv, abs/1906.05168.
```

https://arxiv.org/pdf/1906.05168.pdf

Bibtex:
```
@article{Pham2019AttentionbasedMD,
  title={Attention-based Multi-Input Deep Learning Architecture for Biological Activity Prediction: An Application in EGFR Inhibitors},
  author={Huy Ngoc Pham and Trung Hoang Le},
  journal={2019 11th International Conference on Knowledge and Systems Engineering (KSE)},
  year={2019},
  pages={1-9}
}
```

## Related papers

CNN+SMILES feature matrix:
https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-018-2523-5

Artificial Intelligence in Biological Activity Prediction:
https://sci-hub.do/https://doi.org/10.1007/978-3-030-23873-5_20

DB:
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6367519/

## Project deliverables

Here is what we expect from you  in two weeks by Friday, May 28th

Deliverables
1) Report (PDF). It should roughly follow the structure of research paper. It should thus have the intro part explaining the task, literature review (short in your case), method description, and experimental results section with quantitative and qualitative results/comparisons, +references

On  top of that, you should include two more sections:

1-a) a contribution section. The structure for that one is very rigid. You should have one paragraph per person, explaining the contribution of that person. Once again, one paragraph for one person. Do not spread one person contribution over multiple paragraphs, do not describe the contribution of multiple people in the same paragraph. Projects without this section or with incorrectly structured section will not be graded.

1-b) 3rd party code list. List all third-party code that was used to obtain project results (with links). 

2) Presentation video (Youtube upload -- it is your choice whether you make it listed or delisted). The duration should be 10-12 minutes, the structure should roughly follow the report, but feel free be creative. My favorite tool is PowerPoint (it has good tools for adding narrations and rendering the final video), but feel free to choose other tools.

3) Code (github/collab notebooks). We will not look carefully in every submitted code, but will have a quick look at all of them and will do more thorough checks for some.

Project grading
a) Report quality and clarity (8 points total): Intro+motivation (2 pts), Related work section (1 pt), Methods description clarity (2 pts), Experimental section + conclusions (3 pts).

b) Presentation quality: 5 pts

c) Code availability: 2 pts

d) Novelty of the idea/application: 3 pts

e) Interestingness of results (including negative results): 7 pts -- note that this part is left deliberately vague. Please do not ask us to clarify what interestingness is.

f) Quality of comparison with ablations and baselines: 3 pts

g) Presence of comparison with state-of-the-art: 2 pts

Important note: when grading each of these aspects, we will take into account the size of your project team. I.e. we expect much more in every aspect from the project with seven people than from the project with three people.

Overall, your project grade will be <= 30 points. Personal grades will be computed using the contribution section and the peer review evaluation that you will need to pass on Canvas. Each of you can get upto 100% of the project grade.
