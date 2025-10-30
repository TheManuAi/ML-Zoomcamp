# Week 5 Homework - Model Deployment

## Homework Link

[Link to homework questions](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/cohorts/2025/05-deployment/homework.md)

## Summary

This homework focuses on deploying a machine learning model for lead scoring using modern Python tools and containerization.

### What I Did

1. **Question 1:** Installed `uv` package manager
2. **Question 2:** Used `uv` to install Scikit-Learn 1.6.1 and generated lock file
3. **Question 3:** Loaded pre-trained pipeline with pickle and scored a sample lead
4. **Question 4:** Created FastAPI service (`serve_q6.py`) to serve the model via REST API
5. **Question 5:** Pulled base Docker image `agrigorev/zoomcamp-model:2025`
6. **Question 6:** Built custom Docker container with dependencies and deployed the FastAPI service

### Tech Stack

- **Python 3.13** - Programming language
- **uv** - Fast Python package manager
- **FastAPI** - Web framework for the API
- **Uvicorn** - ASGI server
- **Scikit-learn 1.6.1** - ML library
- **Docker** - Containerization
- **Pydantic** - Data validation

### Key Files

- `serve_q6.py` - FastAPI application
- `Dockerfile` - Container configuration
- `pyproject.toml` - Project dependencies
- `test_q4.py` - API testing script
