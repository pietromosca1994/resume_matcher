import random
from faker import Faker
import pandas as pd
import argparse
from tqdm import tqdm
import json 
import os 

fake = Faker()

# Define possible skills for IT professionals
SKILLS = [
    # Programming
    "Python", "Java", "C++", "C#", "JavaScript", "TypeScript", "Go", "Rust", "Ruby", "PHP", "Scala",
    # Cloud & DevOps
    "AWS", "Azure", "GCP", "Docker", "Kubernetes", "Terraform", "Ansible", "Jenkins", "CI/CD",
    "Linux", "Linux Administration", "Shell Scripting", "Bash", "PowerShell",
    # Databases
    "SQL", "NoSQL", "PostgreSQL", "MySQL", "MongoDB", "Redis", "Cassandra",
    # Data & ML
    "Machine Learning", "Deep Learning", "Data Science", "MLOps", "TensorFlow", "PyTorch", "Scikit-learn",
    "Pandas", "NumPy", "Data Visualization", "Matplotlib", "Seaborn", "Big Data", "Spark",
    # Web & APIs
    "REST APIs", "GraphQL", "Microservices", "React", "Angular", "Node.js", "Express.js", "Flask", "Django",
    # Security
    "Cybersecurity", "Penetration Testing", "Network Security", "Cloud Security", "Ethical Hacking",
    # Other emerging tech
    "AI", "NLP", "Computer Vision", "Blockchain", "IoT", "Edge Computing", "AR/VR", "Robotics"
]

# Define possible roles for IT professionals
ROLES = [
    "Software Engineer", "Senior Software Engineer", "Junior Software Engineer",
    "Data Scientist", "Senior Data Scientist", "Machine Learning Engineer",
    "MLOps Engineer", "DevOps Engineer", "Cloud Architect", "Cloud Engineer",
    "Site Reliability Engineer", "Backend Developer", "Frontend Developer",
    "Full Stack Developer", "Data Engineer", "AI Researcher", "Security Engineer",
    "Cybersecurity Analyst", "Blockchain Developer", "IoT Developer", "Robotics Engineer"
]

# Define possible degrees for IT professionals
DEGREES = [
    "B.Sc. in Computer Science", "B.Eng. in Software Engineering", "B.Sc. in Information Technology",
    "M.Sc. in Computer Science", "M.Sc. in Data Science", "M.Sc. in Artificial Intelligence",
    "M.Sc. in Cybersecurity", "Ph.D. in Computer Science", "Ph.D. in Artificial Intelligence",
    "Ph.D. in Robotics", "Certificate in Cloud Computing", "Certificate in Machine Learning"
]

def generate_experience():
    """Generate a random list of experiences."""
    num_exp = random.randint(1, 3)
    experiences = []
    start_year = random.randint(2012, 2018)
    for _ in range(num_exp):
        end_year = start_year + random.randint(2, 3)
        experiences.append({
            "company": fake.company(),
            "role": random.choice(ROLES),
            "start_date": f"{start_year}-01-01",
            "end_date": f"{end_year}-12-31",
            "description": fake.text(max_nb_chars=120)
        })
        start_year = end_year + 1
    return experiences

def generate_education():
    """Generate a random list of educational records."""
    num_edu = random.randint(1, 2)
    educations = []
    start_year = random.randint(2008, 2014)
    for _ in range(num_edu):
        year_of_grad = start_year + random.randint(3, 5)
        educations.append({
            "institution": fake.company() + " University",
            "degree": random.choice(DEGREES),
            "year_of_graduation": year_of_grad,
            "description": fake.text(max_nb_chars=80)
        })
        start_year = year_of_grad
    return educations

def generate_person():
    """Generate a synthetic IT professional profile."""
    birthdate = fake.date_of_birth(minimum_age=22, maximum_age=55)
    skills = random.sample(SKILLS, k=random.randint(5, 10))
    return {
        "first_name": fake.first_name(),
        "last_name": fake.last_name(),
        "birthdate": birthdate.isoformat(),
        "age": (pd.Timestamp("today").year - birthdate.year),
        "email": fake.email(),
        "phone": fake.phone_number(),
        "address": fake.address().replace("\n", ", "),
        "skills": skills,
        "experiences": generate_experience(),
        "education": generate_education(),
    }

def main(num_people: int, otpout_folder: str):
    resumes = []
    for _ in tqdm(range(num_people), desc="Generating synthetic IT profiles"):
        person=generate_person()
        
        with open(f"{otpout_folder}/{person['first_name']}_{person['last_name']}_resume.json", "w") as f:
            json.dump(person, f, indent=4)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic IT profiles")
    parser.add_argument(
        "--num_people",
        type=int,
        default=50,
        help="Number of people to generate (default: 50)"
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default="./data",
        help="Output folder"
    )
    args = parser.parse_args()
    os.makedirs(args.output_folder, exist_ok=True)
    main(args.num_people, args.output_folder)