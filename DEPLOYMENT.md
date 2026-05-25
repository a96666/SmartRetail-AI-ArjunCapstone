# Smart Retail AI Platform Deployment Guide

## Project Overview

The Smart Retail AI Platform is an end-to-end Multi-Agent AI system developed using:

- FastAPI
- Machine Learning
- MongoDB
- Retrieval-Augmented Generation (RAG)
- Azure Blob Storage
- Power BI
- Docker
- GitHub Actions CI/CD

The platform provides:

- Sales Forecasting
- Anomaly Detection
- AI-powered Chat Assistant
- Analytics Dashboard
- Automated Data Pipelines

---

# Deployment Architecture

Frontend (Streamlit Dashboard)
↓
FastAPI Backend APIs
↓
Machine Learning Models
↓
MongoDB Database
↓
RAG Agent System
↓
Azure Blob Storage
↓
Power BI Dashboard

---

# Docker Deployment

## Build Docker Image

```bash
docker build -t smart-retail-ai .