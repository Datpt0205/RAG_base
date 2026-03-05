import os
import pandas as pd
from sqlalchemy import create_engine, text

PG_HOST = os.getenv("PG_HOST", "mse_pg_dev")
PG_PORT = os.getenv("PG_PORT", "5432")
PG_DB = os.getenv("POSTGRES_DB", "mse_db_dev")
PG_USER = os.getenv("POSTGRES_USER", "mse_pg_dev")
PG_PASS = os.getenv("POSTGRES_PASSWORD", "mse_pg_dev")

ENGINE = create_engine(f"postgresql+psycopg2://{PG_USER}:{PG_PASS}@localhost:5555/{PG_DB}")

PARQUET_PATH ="C:\\Users\\USER\\mse\\outputs_jobfit_two_stage\\cache\\jobfit_merged.parquet"

UPSERT = """
INSERT INTO jobs (
  job_id,title,company_name,location,
  industries,skills,salary_bucket,benefits,
  employee_bucket,company_industries,company_specialities,
  job_description,source,updated_at
)
VALUES (
  :job_id,:title,:company_name,:location,
  :industries,:skills,:salary_bucket,:benefits,
  :employee_bucket,:company_industries,:company_specialities,
  :job_description,:source, now()
)
ON CONFLICT (job_id) DO UPDATE SET
  title=EXCLUDED.title,
  company_name=EXCLUDED.company_name,
  location=EXCLUDED.location,
  industries=EXCLUDED.industries,
  skills=EXCLUDED.skills,
  salary_bucket=EXCLUDED.salary_bucket,
  benefits=EXCLUDED.benefits,
  employee_bucket=EXCLUDED.employee_bucket,
  company_industries=EXCLUDED.company_industries,
  company_specialities=EXCLUDED.company_specialities,
  job_description=EXCLUDED.job_description,
  source=EXCLUDED.source,
  updated_at=now();
"""

def main():
    df = pd.read_parquet(PARQUET_PATH)

    rows = []
    for _, r in df.iterrows():
        rows.append({
            "job_id": int(r["job_id"]),
            "title": str(r.get("job_title") or ""),
            "company_name": str(r.get("company_name") or ""),
            "location": str(r.get("location") or ""),  # nếu không có thì bỏ
            "industries": str(r.get("industries") or ""),
            "skills": str(r.get("skills") or ""),
            "salary_bucket": str(r.get("salary_bucket") or ""),
            "benefits": str(r.get("benefits") or ""),
            "employee_bucket": str(r.get("employee_count_bucket") or ""),
            "company_industries": str(r.get("company_industries") or ""),
            "company_specialities": str(r.get("company_specialities") or ""),
            "job_description": str(r.get("job_description") or ""),
            "source": "jobfit",
        })

    with ENGINE.begin() as conn:
        for i in range(0, len(rows), 2000):
            conn.execute(text(UPSERT), rows[i:i+2000])

if __name__ == "__main__":
    main()
