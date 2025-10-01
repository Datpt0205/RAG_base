from typing import Dict, List, Tuple

# ánh xạ synonym -> canonical
CANONICAL: Dict[str, str] = {
    "js": "JavaScript", "javascript": "JavaScript",
    "ts": "TypeScript",
    "postgres": "PostgreSQL", "postgresql": "PostgreSQL",
    "py": "Python", "python": "Python",
    "docker": "Docker",
    "k8s": "Kubernetes", "kubernetes": "Kubernetes",
    "nlp": "NLP", "ml": "Machine Learning", "recsys": "Recommender Systems",
    "etl": "ETL", "airflow": "Apache Airflow",
    "sql": "SQL",
    "react": "React", "next": "Next.js", "nextjs": "Next.js",
    "fastapi": "FastAPI",
    "aws": "AWS", "gcp": "GCP", "azure": "Azure",
}

# domain để build radar động
DOMAINS: Dict[str, List[str]] = {
    "Backend": ["Python","FastAPI","Node.js","Java","Go","PostgreSQL","SQL","Docker","Kubernetes","REST","GraphQL"],
    "Data": ["SQL","Python","ETL","Apache Airflow","Spark","Data Warehouse","dbt"],
    "DevOps": ["Docker","Kubernetes","CI/CD","Terraform","AWS","GCP","Azure"],
    "Frontend": ["JavaScript","TypeScript","React","Next.js","CSS","HTML"],
    "AI/ML": ["Machine Learning","NLP","Computer Vision","PyTorch","TensorFlow","Recommender Systems"],
}

# Link học nhanh (có thể mở rộng thêm)
COURSE_LINKS: Dict[str, str] = {
    "SQL": "https://www.coursera.org/specializations/learn-sql-basics-data-science",
    "Python": "https://www.coursera.org/specializations/python",
    "FastAPI": "https://www.udemy.com/course/fastapi-the-complete-course/",
    "Docker": "https://www.coursera.org/learn/docker",
    "Kubernetes": "https://www.coursera.org/specializations/google-kubernetes-engine",
    "Apache Airflow": "https://www.udemy.com/course/the-ultimate-hands-on-course-to-master-apache-airflow/",
    "Machine Learning": "https://www.coursera.org/learn/machine-learning",
    "NLP": "https://www.coursera.org/learn/language-processing",
    "React": "https://www.udemy.com/course/react-the-complete-guide-incl-redux/",
    "Next.js": "https://nextjs.org/learn",
}

def canonicalize(skills: List[str]) -> List[str]:
    out = []
    for s in skills or []:
        key = (s or "").strip().lower()
        out.append(CANONICAL.get(key, s.strip()))
    # remove dup giữ thứ tự
    seen = set(); uniq = []
    for s in out:
        if s not in seen:
            seen.add(s); uniq.append(s)
    return uniq

def score_domains(skills: List[str]) -> List[Tuple[str, int]]:
    """Return [(domain, score 0..100)]. Score = % kỹ năng của domain xuất hiện."""
    skills_set = set(skills or [])
    results = []
    for domain, vocab in DOMAINS.items():
        have = len([v for v in vocab if v in skills_set])
        score = int(round(100 * have / max(1, len(vocab))))
        results.append((domain, score))
    return results

def learning_link(skill: str) -> str | None:
    return COURSE_LINKS.get(skill)
