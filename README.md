<div align="center">

<h1>Navigating the Nuances: A Fine-grained Evaluation of Vision-Language Navigation</h1>

<div>
    <a href="https://homes.esat.kuleuven.be/~zwang" target="_blank">Zehao Wang</a><sup>1</sup>,
    <a href="https://wuminye.github.io/" target="_blank">Minye Wu</a><sup>1</sup>,
    <a href="https://sites.google.com/view/yixin-homepage" target="_blank">Yixin Cao</a><sup>4</sup>,
    <a href="https://mayubo2333.github.io" target="_blank">Yubo Ma</a><sup>3</sup>,
    <a href="https://chenmeiqii.github.io" target="_blank">Meiqi Chen</a><sup>2</sup>,
    <a href="https://www.esat.kuleuven.be/psi/TT" target="_blank">Tinne Tuytelaars</a><sup>1</sup> 
</div>
<sup>1</sup>ESAT-PSI, KU Leuven, <sup>2</sup>Peking University, <br> <sup>3</sup>Nanyang Technological University, <sup>4</sup>Fudan University
<br>

</div>

## Abstract

This study presents a novel evaluation framework for the Vision-Language Navigation (VLN) task. It aims to diagnose current models for various instruction categories at a finer-grained level. The framework is structured around the context-free grammar (CFG) of the task. The CFG serves as the basis for the problem decomposition and the core premise of the instruction categories design. We propose a semi-automatic method for CFG construction with the help of Large-Language Models (LLMs). Then, we induct and generate data spanning five principal instruction categories (i.e. direction change, landmark recognition, region recognition, vertical movement, and numerical comprehension). Our analysis of different models reveals notable performance discrepancies and recurrent issues. The stagnation of numerical comprehension, heavy selective biases over directional concepts, and other interesting findings contribute to the development of future language-guided navigation systems. A brief introduction of the project is available [here](https://zehao-wang.github.io/navnuances).

## Environment Setup

To evaluate on R2R-style dataset, you need to set up the [Matterport3DSimulator](https://github.com/peteanderson80/Matterport3DSimulator). Please follow the instructions provided in its official repository for installation and configuration.


## NavNuances Dataset

Download NavNuances data v1 from [link](https://drive.google.com/file/d/1rVy6n5UC5072dW3-x7XykYhFT-CueIL2/view?usp=sharing)

### A. Trajectory Prediction
We follow the R2R naming convention. To include the trajectory predictions of the NavNuances splits, simply specify the split names in the validation code of standard VLN methods trained on R2R.

We provide an example of setting up the [DUET](https://github.com/cshizhe/VLN-DUET)  model to generate predictions for the NavNuances dataset. You can check the details in ```baselines/VLN-DUET```.

### B. Evaluation
The evaluator definitions are provided in the ```evaluation/evaluators``` directory. After generating the submission file in the standard R2R format for all NavNuances splits, modify the directories in ```evaluation/run_eval_template.sh```. Then run:
```bash
cd evaluation
sh run_eval_template.sh
```
This will generate the evaluation results.

## NavGPT4v Model
To predict using the NavGPT4v model, follow these steps:
1. Link Matterport3D scans ```v1/scans``` to ```baselines/navgpt4v/data/v1/scans```
2. Place all the evaluation splits into ```baselines/navgpt4v/data/R2R/annotations```
3. Set your OPENAI_API_KEY and OPENAI_ORGANIZATION environment variable. 
4. Add the evaluation splits in the line 17 of ```baselines/navgpt4v/NavGPT4v.py```.
5. Run the trajectory prediction script with the following command:

```bash
cd baselines/navgpt4v
DEBUG=1 sh run_pred.sh # dump intermediate observations as detailed at line 339 of navgpt4v/LLMs/openai4v.py
sh run_pred.sh  # no intermediate log
```

## Citation
If you're using NavNuances in your research, please cite using the following BibTeX:
```bibtex
@inproceedings{wang2024navigating,
  title={Navigating the Nuances: A Fine-grained Evaluation of Vision-Language Navigation},
  author={Wang, Zehao and Wu, Minye and Cao, Yixin and Ma, Yubo and Chen, Meiqi and Tuytelaars, Tinne},
  booktitle={Findings of the Association for Computational Linguistics: EMNLP 2024},
  year={2024},
  publisher={Association for Computational Linguistics},
}
```
