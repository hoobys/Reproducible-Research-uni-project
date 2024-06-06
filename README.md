# Reproducible Research Uni Project

Repository for the final project of the Reproducible Research university course.

## Authors
- Hubert Wojewoda
- Jakub Żmujdzin
- Jakub Wujec
- Adam Janczyszyn

## Table of Contents
- [Reproducible Research Uni Project](#reproducible-research-uni-project)
  - [Authors](#authors)
  - [Table of Contents](#table-of-contents)
  - [About](#about)
  - [Installation](#installation)
    - [Using Docker](#using-docker)
    - [Using Poetry](#using-poetry)
  - [Usage](#usage)
  - [Project Structure](#project-structure)
  - [Contributing](#contributing)

## About
This repository contains the final project for the Reproducible Research course. The project aims to demonstrate best practices in reproducible research, including the use of Jupyter Notebooks, version control, and containerization.

## Installation
To get started, clone the repository and set up the necessary dependencies.

```bash
git clone https://github.com/hoobys/Reproducible-Research-uni-project.git
cd Reproducible-Research-uni-project
```

### Using Docker
Build and run the Docker container to ensure a consistent environment.

```bash
docker-compose up --build
```

### Using Poetry
Additionally, when docker is not used, packages can be installed using Poetry.

```bash
poetry install
```

## Usage
Use docker to run specific files or connect to jupyter kernel inside the container.

```bash
docker-compose run rr python rr_project/data_wrangling.py
```
## Project Structure
- `data/`: Contains the datasets used in the project.
- `models/`: Includes machine learning models.
- `notebooks/`: Jupyter Notebooks with the analyses and experiments.
- `rr_project/`: Main project directory.
- `Dockerfile`: Dockerfile for containerizing the project.
- `docker-compose.yml`: Docker Compose configuration.
- `pyproject.toml`: Poetry configuration file.

## Contributing
Contributions are welcome! Please fork the repository and create a pull request with your changes.

1. Fork the repository
2. Create a new branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'Add some feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Create a new pull request


## Disclaimer
For writing the descriptions, we have used ChatGPT Turbo 3.5 (May 2024). For writing code, we have leveraged GitHub Copilot in version as of May 2024.
