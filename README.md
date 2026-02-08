# AI-Powered Medical Chatbot

An intelligent medical chatbot that provides context-aware medical responses using **LangChain**, **OpenAI GPT**, and **Pinecone** for semantic search. Built with **Flask** and deployable on **AWS** with CI/CD pipelines via GitHub Actions.

---

## 🧠 Overview

This project implements an AI-powered medical chatbot that answers user health queries by retrieving relevant information from indexed medical data and generating natural language responses using large language models (LLMs). It can be run locally or deployed to AWS with automated pipelines.

---

## 🚀 Features

- Conversational medical chatbot using modern NLP techniques  
- Semantic knowledge retrieval using vector embeddings  
- Flask-based REST API for real-time interaction  
- Easy deployment via Docker and AWS (ECR, EC2)  
- CI/CD integration with GitHub Actions  

---

## Snapshot

<img width="963" height="332" alt="Screenshot 2026-02-08 003046" src="https://github.com/user-attachments/assets/6a685607-9335-41c5-ae73-9a38e5431bb2" />

---

### Techstack Used:

- Python
- LangChain
- Flask
- GPT
- Pinecone
- AWS

---

## How to run?
### STEPS:

Clone the repository

```bash
git clone https://github.com/sk-moin/AI-Powered-Medical-Chatbot
```
### STEP 01- Create a conda environment after opening the repository

```bash
conda create -n venv python=3.10 -y
```

```bash
conda activate venv
```


### STEP 02- install the requirements
```bash
pip install -r requirements.txt
```


### Create a `.env` file in the root directory and add your Pinecone & openai credentials as follows:

```ini
PINECONE_API_KEY = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
OPENAI_API_KEY = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```


```bash
# run the following command to store embeddings to pinecone
python store_index.py
```

```bash
# Finally run the following command
python app.py
```

Now,
```bash
open up localhost:
```

---

# AWS-CICD-Deployment-with-Github-Actions

## 1. Login to AWS console.

## 2. Create IAM user for deployment

	#with specific access

	1. EC2 access : It is virtual machine

	2. ECR: Elastic Container registry to save your docker image in aws


	#Description: About the deployment

	1. Build docker image of the source code

	2. Push your docker image to ECR

	3. Launch Your EC2 

	4. Pull Your image from ECR in EC2

	5. Lauch your docker image in EC2

	#Policy:

	1. AmazonEC2ContainerRegistryFullAccess

	2. AmazonEC2FullAccess

	
## 3. Create ECR repo to store/save docker image
    - Save the URI: 412913487878.dkr.ecr.ap-south-1.amazonaws.com/medical-chatbot

	
## 4. Create EC2 machine (Ubuntu) 

## 5. Open EC2 and Install docker in EC2 Machine:
	
	
	#optinal

	sudo apt-get update -y

	sudo apt-get upgrade
	
	#required

	curl -fsSL https://get.docker.com -o get-docker.sh

	sudo sh get-docker.sh

	sudo usermod -aG docker ubuntu

	newgrp docker
	
## 6. Configure EC2 as self-hosted runner:
    setting>actions>runner>new self hosted runner> choose os> then run command one by one


## 7. Setup github secrets:

   - AWS_ACCESS_KEY_ID
   - AWS_SECRET_ACCESS_KEY
   - AWS_DEFAULT_REGION
   - ECR_REPO
   - PINECONE_API_KEY
   - OPENAI_API_KEY
