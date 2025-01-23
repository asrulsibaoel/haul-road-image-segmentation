
<h1 align="center">
    Haul Road Image Segmentation
</h1>

<p align="center">
    <strong>(Re)maps the Haul Road of Coal Mine Sites.</strong>
</p>

<p align="center">
    <a href="#" title="PyPi Version"><img src="https://img.shields.io/badge/PyPi-v.0.4.8-blue"></a>
    <a href="#" title="Python Version"><img src="https://img.shields.io/badge/Python-3.8%2B-green"></a>
    <!-- <a href="https://www.codacy.com/gh/ml-tooling/lazydocs/dashboard" title="Codacy Analysis"><img src="https://app.codacy.com/project/badge/Grade/1c8ad486ce9547b6b713cce7ca1d1ec3"></a> -->
    <!-- <a href="" title="Build status"><img src="https://img.shields.io/github/workflow/status/ml-tooling/lazydocs/build-pipeline?style=flat"></a> -->
    <a href="#" title="Project License"><img src="https://img.shields.io/badge/License-Apache_2.0-red"></a>
    <!-- <a href="https://gitter.im/ml-tooling/lazydocs" title="Chat on Gitter"><img src="https://badges.gitter.im/ml-tooling/lazydocs.svg"></a> -->
    <a href="https://linkedin.com/in/asrulsibaoel" title="My Linkedin Profile"><img src="https://img.shields.io/badge/LinkedIn-0077B5?logo=linkedin&logoColor=white"></a>
</p>

<p align="center">
  <a href="#installation">Getting Started</a> •
  <a href="/docs#tutorials-and-api-overview">Documentation</a> •
  <a href="/../../issues">Support</a> •
  <a href="#development">Contribution</a>
  <!-- <a href="https://github.com/ml-tooling/lazydocs/releases">Changelog</a> -->
</p>

This project aimed to map the haul road of coal mines using Machine Learning. The algorithm used in this project is a deep learning algorithm, especially [YOLO](https://docs.ultralytics.com/)  and UNet.

## Prerequisites  
- Seldon Core Installed on your Kubernetes cluster. Guidance [here](https://docs.seldon.io/projects/seldon-core/en/latest/workflow/install.html)  
- MLflow Tracking Server deployed. Recommended to use the scenario 5  [here](https://www.mlflow.org/docs/latest/tracking.html#scenario-5-mlflow-tracking-server-enabled-with-proxied-artifact-storage-access)  
- Local or remote storage such as Amazon S3 or Google Cloud Storage (GCS) or similar.

## Installation  
### From PyPI
Using pip is the best option if you want to get a stable version and an easier way to install.
```bash
$ pip install klops
```

## Basic Usages 
Klops consists of three modules. Versioning, Experiment and Deployment.


## Development  

We are open for anyone who wants to contribute! Please read the [CONTRIBUTING.md](CONTRIBUTING.md) guide.